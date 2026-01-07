import ctypes
from datetime import timedelta
from typing import Any, List, Optional

import torch
from torch.distributed import ReduceOp

from vllm.distributed.utils import StatelessProcessGroup
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

from .distributed_nccl import _common_all_gather_object

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
            ctypes.POINTER(HcclCommConfig),
            ctypes.POINTER(hcclComm_t),
        ],
    ),
]

def hccl_all_gather(self, send_buf, recv_buf, count, data_type, comm, stream):
    self.HCCL_CHECK(
        self._funcs["HcclAllGather"](send_buf, recv_buf, count, data_type, comm, stream)
    )

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
        self.subcomm_id = 1

    def destroy_comm(self, comm=None):
        if comm:
            self.hccl.hcclCommDestroy(comm)
        else:
            self.hccl.hcclCommDestroy(self.comm)

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

    def create_subcomm(self, ranks):
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
        self.subcomm_id += 1
        return subcomm

class DistributedHccl:
    pg: StatelessProcessGroup
    pyhccl: PyHcclCommunicatorEx
    sub_groups: dict[int, list[int]]
    comm: hcclComm_t

    host: str
    port: int
    rank: int
    world_size: int
    device: torch.device

    initialized: bool = False


dist = DistributedHccl()

def init_process_group(
    host: str,
    port: int,
    rank: int,
    world_size: int,
    timeout: timedelta = timedelta(seconds=300),
    **kwargs,
):
    assert not dist.initialized, "already initialized"

    dist.host = host
    dist.port = port
    dist.rank = rank
    dist.world_size = world_size
    dist.device = torch.device("npu", rank)

    dist.pg = StatelessProcessGroup.create(
        host, port, rank, world_size, store_timeout=int(timeout.total_seconds())
    )
    dist.pyhccl = PyHcclCommunicatorEx(group=dist.pg, device=dist.device)
    dist.comm = dist.pyhccl.comm
    dist.initialized = True

def destroy_process_group(group=None):
    assert dist.initialized, "not initialized"

    if group in dist.sub_groups:
        subcomm = ctypes.c_void_p(group)
        dist.pyhccl.destroy_comm(subcomm)
        del dist.sub_groups[group]
        return

    dist.pyhccl.destroy_comm()

    dist.pyhccl = None
    dist.pg = None
    dist.initialized = False

def is_initialized() -> bool:
    return dist.initialized

def all_gather_object(object_list: list[Any], obj: Any, group=None):
    assert dist.initialized, "not initialized"

    if group:
        assert group in dist.sub_groups, "invalid sub_group"
        subcomm = ctypes.c_void_p(group)
        dist.pyhccl.comm = subcomm

    _common_all_gather_object(dist.pyhccl, dist.device, dist.world_size, object_list, obj)
    current_stream().synchronize()

    if group:
        dist.pyhccl.comm = dist.comm

def all_reduce(tensor: torch.Tensor, op=ReduceOp.SUM, group=None):
    assert dist.initialized, "not initialized"

    if group:
        assert group in dist.sub_groups, "invalid sub_group"
        subcomm = ctypes.c_void_p(group)
        dist.pyhccl.comm = subcomm

    out_tensor = dist.pyhccl.all_reduce(tensor, op)
    current_stream().synchronize()
    tensor.copy_(out_tensor)

    if group:
        dist.pyhccl.comm = dist.comm

def broadcast(tensor: torch.Tensor, src=None, group=None):
    assert dist.initialized, "not initialized"

    if group:
        assert group in dist.sub_groups, "invalid sub_group"
        assert src in dist.sub_groups[group], "src rank not in group"
        subcomm = ctypes.c_void_p(group)
        dist.pyhccl.comm = subcomm
        # convert src rank id in default world to subcomm
        src = dist.sub_groups[group].index(src)
        dist.pyhccl.rank = dist.sub_groups[group].index(dist.rank)

    dist.pyhccl.broadcast(tensor, src)
    current_stream().synchronize()

    if group:
        dist.pyhccl.comm = dist.comm
        dist.pyhccl.rank = dist.rank

def barrier(group=None):
    assert dist.initialized, "not initialized"

    if group:
        assert group in dist.sub_groups, "invalid sub_group"
        subcomm = ctypes.c_void_p(group)
        dist.pyhccl.comm = subcomm

    data = torch.zeros(1, device=dist.rank)
    dist.pyhccl.all_reduce(data)
    current_stream().synchronize()

    if group:
        dist.pyhccl.comm = dist.comm

def new_group(ranks):
    assert dist.initialized, "not initialized"

    # if ranks is None or [], using the world instead
    if not ranks:
        ranks = list(range(dist.world_size))

    if dist.rank not in ranks:
        return

    subcomm = dist.pyhccl.create_subcomm(ranks)
    value = 0
    if subcomm:
        value = subcomm.value
        dist.sub_groups[value] = ranks
    return value
