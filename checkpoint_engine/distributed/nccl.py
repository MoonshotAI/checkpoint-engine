import ctypes
from datetime import timedelta
from typing import Any, List, Optional

import torch
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
from checkpoint_engine.distributed.base import Distributed, _common_all_gather_object


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
        ("numRmaCtx", ctypes.c_int),
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

    self.NCCL_CHECK(self._funcs["ncclCommSplit"](comm, color, key, ctypes.byref(newcomm), None))
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
            color = -1  # NCCL_SPLIT_NOCOLOR
        newcomm = self.nccl.ncclCommSplit(self.comm, color, self.rank)
        return newcomm


class DistributedNccl(Distributed):
    def __init__(self):
        self.pg: StatelessProcessGroup = None
        self.pynccl: PyNcclCommunicatorEx = None
        self.sub_groups: dict[int, list[int]] = {}
        self.comm: ncclComm_t = None

        self.host: str = None
        self.port: int = None
        self.rank: int = None
        self.world_size: int = None
        self.device: torch.device = None

        self.initialized: bool = False

    def init_process_group(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        timeout: timedelta = timedelta(seconds=300),
    ):
        assert not self.initialized, "already initialized"

        self.host = host
        self.port = port
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device("cuda", torch.cuda.current_device())

        self.pg = StatelessProcessGroup.create(
            host, port, rank, world_size, store_timeout=int(timeout.total_seconds())
        )

        self.pynccl = PyNcclCommunicatorEx(group=self.pg, device=self.device)
        self.comm = self.pynccl.comm
        self.initialized = True


    def destroy_process_group(
        self,
        group=None,
    ):
        assert self.initialized, "not initialized"

        if group in self.sub_groups:
            newcomm = ctypes.c_void_p(group)
            self.pynccl.destroy_comm(newcomm)
            del self.sub_groups[group]
            return

        self.pynccl.destroy_comm()
        self.pynccl = None
        self.pg = None
        self.initialized = False


    def is_initialized(self) -> bool:
        return self.initialized


    def all_gather_object(
        self,
        object_list: list[Any],
        obj: Any,
        group=None
    ):
        assert self.initialized, "not initialized"

        if group:
            assert group in self.sub_groups, "invalid sub_group"
            newcomm = ctypes.c_void_p(group)
            self.pynccl.comm = newcomm

        _common_all_gather_object(self.pynccl, self.device, self.world_size, object_list, obj)
        current_stream().synchronize()

        if group:
            self.pynccl.comm = self.comm


    def all_reduce(
        self,
        tensor: torch.Tensor,
        op=ReduceOp.SUM,
        group=None
    ):
        assert self.initialized, "not initialized"

        if group:
            assert group in self.sub_groups, "invalid sub_group"
            newcomm = ctypes.c_void_p(group)
            self.pynccl.comm = newcomm

        out_tensor = self.pynccl.all_reduce(in_tensor=tensor, op=op)
        current_stream().synchronize()
        tensor.copy_(out_tensor)

        if group:
            self.pynccl.comm = self.comm


    def broadcast(
        self,
        tensor: torch.Tensor,
        src=None,
        group=None
    ):
        assert self.initialized, "not initialized"

        if group:
            assert group in self.sub_groups, "invalid sub_group"
            assert src in self.sub_groups[group], "src rank not in group"
            newcomm = ctypes.c_void_p(group)
            self.pynccl.comm = newcomm
            # convert src rank id in default world to newcomm
            src = self.sub_groups[group].index(src)
            self.pynccl.rank = self.sub_groups[group].index(self.rank)

        self.pynccl.broadcast(tensor, src)
        current_stream().synchronize()

        if group:
            self.pynccl.comm = self.comm
            self.pynccl.rank = self.rank


    def barrier(
        self,
        group=None
    ):
        assert self.initialized, "not initialized"

        if group:
            assert group in self.sub_groups, "invalid sub_group"
            newcomm = ctypes.c_void_p(group)
            self.pynccl.comm = newcomm

        data = torch.zeros(1, device=self.device)
        self.pynccl.all_reduce(data)
        current_stream().synchronize()

        if group:
            self.pynccl.comm = self.comm


    def new_group(
        self,
        ranks
    ):
        assert self.initialized, "not initialized"

        # ranks is None or []
        if not ranks:
            ranks = list(range(self.world_size))

        newcomm = self.pynccl.create_newcomm(ranks)
        value = 0
        if newcomm:
            value = newcomm.value
            self.sub_groups[value] = ranks
        return value
