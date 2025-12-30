import ctypes
import io
import logging
import os
import pickle
from enum import Enum
from typing import Any, List, Optional

import torch
import torch.distributed
from torch.distributed import ReduceOp
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
        timeout: int = 300,
        **kwargs,
    ):
        self._host = host
        self._port = port
        self._rank = rank
        self._world_size = world_size
        self._device = torch.device("cuda", rank)

        self.pg = StatelessProcessGroup.create(host, port, rank, world_size, store_timeout=timeout)

        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        self.pynccl = PyNcclCommunicator(group=self.pg, device=self._device)

    def destroy_process_group(self, group=None):
        # PyNcclCommunicator does not provide destroy method
        if group in self.sub_groups:
            del self.sub_groups[group]
            return

    def is_initialized(self) -> bool:
        return self.pynccl is not None

    def all_gather_object(self, object_list: list[Any], obj: Any, group=None):
        if group:
            assert group in self.sub_groups, "invalid sub_group"
            pynccl = group.pynccl
        else:
            pynccl = self.pynccl

        _common_all_gather_object(pynccl, self._device, self._world_size, object_list, object)
        current_stream().synchronize()

    def all_reduce(self, tensor: torch.Tensor, op=ReduceOp.SUM, group=None):
        if group:
            assert group in self.sub_groups, "invalid sub_group"
            pynccl = group.pynccl
        else:
            pynccl = self.pynccl

        out_tensor = pynccl.all_reduce(in_tensor=tensor, op=op)
        current_stream().synchronize()
        tensor.copy_(out_tensor)

    def broadcast(self, tensor: torch.Tensor, src=None, group=None):
        if group:
            assert group in self.sub_groups, "invalid sub_group"
            pynccl = group.pynccl
        else:
            pynccl = self.pynccl

        pynccl.broadcast(tensor, src)
        current_stream().synchronize()

    def barrier(self, group=None):
        if group:
            assert group in self.sub_groups, "invalid sub_group"
            pynccl = group.pynccl
        else:
            pynccl = self.pynccl

        data = torch.zeros(1, device=self._rank)
        pynccl.all_reduce(data)
        current_stream().synchronize()

    def new_group(self, ranks):
        # ranks is None or []
        if not ranks:
            return self

        host = self._host
        port = self._port
        rank = self._rank

        if rank not in ranks:
            return

        new_rank = ranks.index(rank)
        new_world_size = len(ranks)

        new_dist = DistributedNccl()
        new_dist.init_process_group(
            host, port + 10, new_rank, new_world_size
        )  # todo host maybe incorrect
        self.sub_groups.append(new_dist)

        return new_dist


try:
    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator
    from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import (
        Function,
        HCCLLibrary,
        aclrtStream_t,
        buffer_type,
        hcclComm_t,
        hcclDataType_t,
        hcclDataTypeEnum,
        hcclRedOp_t,
        hcclRedOpTypeEnum,
        hcclResult_t,
        hcclUniqueId,
    )
    from vllm_ascend.utils import current_stream

    orig_exported_functions = HCCLLibrary.exported_functions
    extended_functions = [
        # HcclResult HcclAllGather(
        #   void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
        #   HcclComm comm, alcrtStream stream
        # )
        Function(
            "HcclAllGather",
            hcclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_uint64,
                hcclDataType_t,
                hcclComm_t,
                aclrtStream_t,
            ],
        ),
        # HcclResult HcclCreateSubCommConfig(
        #   HcclComm *comm, uin32_t rankNum, uint32_t *rankIds, uint64_t subCommId,
        #   uint32_t subCommRankId, HcclCommConfig *config, HcclComm *subComm
        # )
        Function(
            "HcclCreateSubCommConfig",
            hcclResult_t,
            [
                ctypes.POINTER(hcclComm_t),
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_uint64,
                ctypes.c_uint32,
                ctypes.POINTER(hcclUniqueId),
                ctypes.POINTER(hcclComm_t),
            ],
        ),
    ]

    def hccl_all_gather(self, send_buf, recv_buf, count, data_type, comm, stream):
        self.HCCL_CHECK(
            self._funcs["HcclAllGather"](send_buf, recv_buf, count, data_type, comm, stream)
        )

    class HcclCommConfig(ctypes.Structure):
        _fields_ = [
            ("size", ctypes.c_size_t),
            ("magic_word", ctypes.c_uint32),
            ("version", ctypes.c_uint32),
            ("reserved", ctypes.c_uint64),
            ("hccl_buffer_size", ctypes.c_uint32),
            ("hccl_deterministic", ctypes.c_uint32),
            ("hccl_comm_name", ctypes.c_char * 128),
            ("hccl_udi", ctypes.c_char * 128),
            ("hccl_op_expansion_mode", ctypes.c_uint32),
            ("hccl_rdma_traffic_class", ctypes.c_uint32),
            ("hccl_rdma_service_level", ctypes.c_uint32),
            ("hcll_world_rank_id", ctypes.c_uint32),
            ("hccl_job_id", ctypes.c_uint64),
            ("comm_engine", ctypes.c_int32),
            ("thread_num", ctypes.c_uint32),
            ("notify_num_per_thread", ctypes.c_uint32),
            ("acl_graph_zero_copy_enable", ctypes.c_uint8),
        ]

    def hccl_create_subcomm_config(
        self, comm, ranks_size, c_rank_ids, subcomm_id, subcomm_rank, comm_config
    ):
        subcomm = hcclComm_t()
        self.HCCL_CHECK(
            self._funcs["HcclCreateSubCommConfig"](
                ctypes.byref(comm),
                ranks_size,
                c_rank_ids,
                subcomm_id,
                subcomm_rank,
                ctypes.byref(comm_config),
                ctypes.byref(subcomm),
            )
        )
        return subcomm

    # extend HCCLLibrary
    HCCLLibrary.exported_functions = orig_exported_functions + extended_functions
    HCCLLibrary.hcclAllGather = hccl_all_gather
    HCCLLibrary.hcclCreateSubCommConfig = hccl_create_subcomm_config

    class PyHcclCommunicatorEx(PyHcclCommunicator):
        def __init__(self, group, device):
            super().__init__(group, device)
            self.subcomms = {}
            self.subcomm_id = 1

        def destroy_comm(self):
            self.hccl.hcclCommDestroy(self.comm)

        def all_reduce(
            self,
            in_tensor: torch.Tensor,
            op: ReduceOp = ReduceOp.SUM,
            stream=None,
        ) -> torch.Tensor:
            if self.disabled:
                return None
            assert in_tensor.device == self.device, (
                f"this hccl communicator is created to work on {self.device}, "
                f"but the input tensor is on {in_tensor.device}"
            )
            out_tensor = torch.empty_like(in_tensor)
            if stream is None:
                stream = current_stream()
            self.hccl.hcclAllReduce(
                buffer_type(in_tensor.data_ptr()),
                buffer_type(out_tensor.data_ptr()),
                in_tensor.numel(),
                hcclDataTypeEnum.from_torch(in_tensor.dtype),
                hcclRedOpTypeEnum.from_torch(op),
                self.comm,  # todo
                aclrtStream_t(stream.npu_stream),
            )
            return out_tensor

        def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
            if self.disabled:
                return None
            assert tensor.device == self.device, (
                f"this hccl communicator is created to work on {self.device}, "
                f"but the input tensor is on {tensor.device}"
            )
            if stream is None:
                stream = current_stream()
            self.hccl.hcclBroadcast(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                hcclDataTypeEnum.from_torch(tensor.dtype),
                src,
                self.comm,  # todo
                aclrtStream_t(stream.npu_stream),
            )

        def all_gather(self, out_tensor: torch.Tensor, in_tensor: torch.Tensor, stream=None):
            if self.disabled:
                return
            assert in_tensor.device == self.device, (
                f"this hccl communicator is created to work on {self.device}, "
                f"but the input tensor in on {in_tensor.device}"
            )
            if stream is None:
                stream = current_stream()
            self.hccl.hcclAllGather(
                buffer_type(in_tensor.data_ptr()),
                buffer_type(out_tensor.data_ptr()),
                in_tensor.numel(),
                hcclDataTypeEnum.from_torch(in_tensor.dtype),
                self.comm,  # todo
                aclrtStream_t(stream.npu_stream),
            )
            return out_tensor

        def create_subcomm(
            self,
            ranks,
        ):
            comm_config = HcclCommConfig(
                size=312,
                magic_word=0xF0F0F0F0,
                version=6,
                reserved=0,
                hccl_buffer_size=0xFFFFFFFF,
                hccl_deterministic=0xFFFFFFFF,
                hccl_comm_name=b"\0",
                hccl_udi=b"\0",
                hccl_op_expansize_mode=0,
                hccl_rdma_traffic_class=0xFFFFFFFF,
                hccl_rdma_service_level=0xFFFFFFFF,
                hccl_world_rank_id=0,
                hccl_job_id=0,
                comm_engine=-1,
                thread_num=0xFFFFFFFF,
                notify_num_per_thread=0xFFFFFFFF,
                acl_graph_zero_copy_enable=0,
            )
            uint32_array = ctypes.c_uint32 * len(ranks)
            c_rank_ids = uint32_array(*ranks)
            subcomm_rank = ranks.index(self.rank)
            ranks_size = len(ranks)
            subcomm_id = self.subcomm_id

            subcomm = self.hccl.hcclCreateSubCommConfig(
                self.comm, ranks_size, c_rank_ids, subcomm_id, subcomm_rank, comm_config
            )
            self.subcomms[subcomm_id] = subcomm
            self.subcomm_id += 1
            return subcomm

    class DistributedHccl:
        def __init__(self):
            self.pg = None
            self.pyhccl = None
            self.sub_groups = {}

        def init_process_group(
            self,
            host: str,
            port: int,
            rank: int,
            world_size: int,
            timeout: int = 300,
            **kwargs,
        ):
            self._host = host
            self._port = port
            self._rank = rank
            self._world_size = world_size
            self._device = torch.device("npu", rank)

            self.pg = StatelessProcessGroup.create(
                host, port, rank, world_size, store_timeout=timeout
            )
            self.pyhccl = PyHcclCommunicatorEx(group=self.pg, device=self._device)

        def destroy_process_group(self, group=None):
            if group in self.sub_groups:
                group.pyhccl.destroy_comm()
                del self.sub_groups[group]
                return

            self.pyhccl.destroy_comm()

            self.pyhccl = None
            self.pg = None

        def is_initialized(self) -> bool:
            return self.pyhccl is not None

        def all_gather_object(self, object_list: list[Any], obj: Any, group=None):
            if group:
                assert group in self.sub_groups, "invalid sub_group"
                pyhccl = group.pyhccl
            else:
                pyhccl = self.pyhccl
            _common_all_gather_object(pyhccl, self._device, self._world_size, object_list, obj)
            current_stream().synchronize()

        def all_reduce(self, tensor: torch.Tensor, op=ReduceOp.SUM, group=None):
            if group:
                assert group in self.sub_groups, "invalid sub_group"
                pyhccl = group.pyhccl
            else:
                pyhccl = self.pyhccl

            out_tensor = pyhccl.all_reduce(tensor, op)
            current_stream().synchronize()
            tensor.copy_(out_tensor)

        def broadcast(self, tensor: torch.Tensor, src=None, group=None):
            if group:
                assert group in self.sub_groups, "invalid sub_group"
                assert src in self.sub_groups[group], "src rank not in group"
                pyhccl = group.pyhccl
                # src is rank id in global world
                src = self.sub_groups[group].index(src)
            else:
                pyhccl = self.pyhccl

            pyhccl.broadcast(tensor, src)
            current_stream().synchronize()

        def barrier(self, group=None):
            if group:
                assert group in self.sub_groups, "invalid sub_group"
                pyhccl = group.pyhccl
            else:
                pyhccl = self.pyhccl

            data = torch.zeros(1, device=self._rank)
            pyhccl.all_reduce(data)
            current_stream().synchronize()

        def new_group(self, ranks):
            # ranks is None or []
            if not ranks:
                return self

            host = self._host
            port = self._port
            rank = self._rank

            if rank not in ranks:
                return

            new_rank = ranks.index(rank)
            new_world_size = len(ranks)

            new_dist = DistributedHccl()
            new_dist.init_process_group(
                host, port + 10, new_rank, new_world_size
            )  # todo host maybe incorrect
            self.sub_groups[new_dist] = ranks

            return new_dist

except ImportError as e:
    pass
