import ctypes
import io
import logging
import os
import pickle
from datetime import timedelta
from enum import Enum
from typing import Any, List, Optional

import torch
import torch.distributed
from torch.distributed import ReduceOp
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import (
    Function,
    NCCLLibrary,
    buffer_type,
    ncclComm_t,
    ncclResult_t,
)
from vllm.distributed.utils import StatelessProcessGroup
from vllm.utils import current_stream


_pickler = pickle.Pickler
_unpickler = pickle.Unpickler


def _object_to_tensor(obj, device):
    f = io.BytesIO()
    _pickler(f).dump(obj)
    byte_storage = torch.ByteStorage._from_buffer(f.getvalue())
    byte_tensor = torch.ByteTensor(byte_storage).to(device)
    local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
    return byte_tensor, local_size


def _tensor_to_object(tensor, tensor_size):
    tensor = tensor.cpu()
    buf = tensor.numpy().tobytes()[:tensor_size]
    return _unpickler(io.BytesIO(buf)).load()


def _flatten_for_scatter_gather(tensor_list, copy=False):
    if not tensor_list:
        raise RuntimeError("Received an empty list.")
    t = tensor_list[0]
    buffer_shape = [len(tensor_list)] + list(t.shape)

    buffer = torch.empty(tuple(buffer_shape), dtype=t.dtype, device=t.device)
    if copy:
        for i, tensor in enumerate(tensor_list):
            buffer[i].copy_(tensor)
    return buffer


def _common_all_gather_object(comm, device, world_size, object_list, object):
    input_tensor, local_size = _object_to_tensor(object, device)
    object_sizes_tensor = torch.empty(world_size, dtype=torch.long, device=device)
    comm.all_gather(object_sizes_tensor, local_size)
    object_size_list = [object_sizes_tensor[i].unsqueeze(dim=0) for i in range(world_size)]
    max_object_size = int(max(object_size_list).item())
    input_tensor.resize_(max_object_size)
    coalesced_output_tensor = torch.empty(
        max_object_size * world_size, dtype=torch.uint8, device=device
    )

    comm.all_gather(coalesced_output_tensor, input_tensor)
    output_tensors = [
        coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
        for i in range(world_size)
    ]
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        tensor_size = object_size_list[i]
        object_list[i] = _tensor_to_object(tensor, tensor_size)


class ncclConfig_t(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("magic", ctypes.c_uint),
        ("version", ctypes.c_uint),
        ("blocking", ctypes.c_int),
        ("cgaClusterSize", ctypes.c_int),
        ("minCTAs", ctypes.c_int),
        ("maxCTAs", ctypes.c_int),
        ("netName", ctypes.c_char_p),
        ("splitShare", ctypes.c_int),
        ("trafficClass", ctypes.c_int),
        ("commName", ctypes.c_char_p),
        ("collnetEnable", ctypes.c_int),
        ("CTAPolicy", ctypes.c_int),
        ("shrinkShare", ctypes.c_int),
        ("nvlsCTAs", ctypes.c_int),
        ("nChannelsPerNetPeer", ctypes.c_int),
        ("nvlinkCentricSched", ctypes.c_int),
        ("graphUsageMode", ctypes.c_int),
        ("numRmdCtx", ctypes.c_int),
    ]

nccl_orig_exported_functions = NCCLLibrary.exported_functions
nccl_extended_functions = [
    # ncclResult_t ncclCommSplit(
    #   ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config
    # )
    Function(
        "ncclCommSplit",
        ncclResult_t,
        [
            ncclComm_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ncclComm_t),
            ctypes.POINTER(ncclConfig_t),
        ],
    ),
]


def nccl_comm_split(
    self,
    comm: ncclComm_t,
    color: int,
    key: int,
) -> ncclComm_t:
    newcomm = ncclComm_t()

    self.NCCL_CHECK(
        self._funcs["ncclCommSplit"](comm, color, key, ctypes.byref(newcomm), None)
    )
    return newcomm


# extend NCCLLibrary
NCCLLibrary.exported_functions = nccl_orig_exported_functions + nccl_extended_functions
NCCLLibrary.ncclCommSplit = nccl_comm_split


class PyNcclCommunicatorEx(PyNcclCommunicator):
    def destroy_comm(self, comm=None):
        if comm:
            self.nccl.ncclCommDestroy(comm)
        else:
            self.nccl.ncclCommDestroy(self.comm)

    def create_newcomm(self, ranks):
        if self.rank in ranks:
            color = 0
        else:
            color = -1 # NCCL_SPLIT_NOCOLOR
        newcomm = self.nccl.ncclCommSplit(self.comm, color, self.rank)
        return newcomm


class DistributedNccl:
    def __init__(self):
        self.pg = None
        self.pynccl = None
        self.sub_groups = {}

    def init_process_group(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        timeout: timedelta = timedelta(seconds=300),
        **kwargs,
    ):
        self._host = host
        self._port = port
        self._rank = rank
        self._world_size = world_size
        self._device = torch.device("cuda", rank)

        self.pg = StatelessProcessGroup.create(
            host, port, rank, world_size, store_timeout=int(timeout.total_seconds())
        )

        self.pynccl = PyNcclCommunicatorEx(group=self.pg, device=self._device)
        self._comm = self.pynccl.comm

    def destroy_process_group(self, group=None):
        if group in self.sub_groups:
            newcomm = ctypes.c_void_p(group)
            self.pynccl.destroy_comm(newcomm)
            del self.sub_groups[group]
            return

        self.pynccl.destroy_comm()

        self.pynccl = None
        self.pg = None

    def is_initialized(self) -> bool:
        return self.pynccl is not None

    def all_gather_object(self, object_list: list[Any], obj: Any, group=None):
        if group:
            assert group in self.sub_groups, "invalid sub_group"
            newcomm = ctypes.c_void_p(group)
            self.pynccl.comm = newcomm

        _common_all_gather_object(self.pynccl, self._device, self._world_size, object_list, obj)
        current_stream().synchronize()

        if group:
            self.pynccl.comm = self._comm

    def all_reduce(self, tensor: torch.Tensor, op=ReduceOp.SUM, group=None):
        if group:
            assert group in self.sub_groups, "invalid sub_group"
            newcomm = ctypes.c_void_p(group)
            self.pynccl.comm = newcomm

        out_tensor = self.pynccl.all_reduce(in_tensor=tensor, op=op)
        current_stream().synchronize()
        tensor.copy_(out_tensor)

        if group:
            self.pynccl.comm = self._comm

    def broadcast(self, tensor: torch.Tensor, src=None, group=None):
        if group:
            assert group in self.sub_groups, "invalid sub_group"
            assert src in self.sub_groups[group], "src rank not in group"
            newcomm = ctypes.c_void_p(group)
            self.pynccl.comm = newcomm
            # convert src rank id in default world to newcomm
            #src = self.sub_groups[group].index(src)

        self.pynccl.broadcast(tensor, src)
        current_stream().synchronize()

        if group:
            self.pynccl.comm = self._comm

    def barrier(self, group=None):
        if group:
            assert group in self.sub_groups, "invalid sub_group"
            newcomm = ctypes.c_void_p(group)
            self.pynccl.comm = newcomm

        data = torch.zeros(1, device=self._rank)
        self.pynccl.all_reduce(data)
        current_stream().synchronize()

        if group:
            self.pynccl.comm = self._comm

    def new_group(self, ranks):
        # ranks is None or []
        if not ranks:
            ranks = list(range(self._world_size))

        newcomm = self.pynccl.create_newcomm(ranks)
        value = 0
        if newcomm:
            value = newcomm.value
            self.sub_groups[value] = ranks
        return value
