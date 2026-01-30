import ctypes
from contextlib import contextmanager
from typing import Any, ClassVar

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
    hcclResult_t,
)
from vllm_ascend.utils import current_stream

from checkpoint_engine.distributed.base import CommGroup, Distributed, _common_all_gather_object


class HcclCommConfig(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [
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


def hccl_all_gather(
    self,  # noqa: ANN001
    send_buf: buffer_type,
    recv_buf: buffer_type,
    count: ctypes.c_uint64,
    data_type: hcclDataType_t,
    comm: hcclComm_t,
    stream: aclrtStream_t,
):
    self.HCCL_CHECK(
        self._funcs["HcclAllGather"](send_buf, recv_buf, count, data_type, comm, stream)
    )


def hccl_create_subcomm_config(
    self,  # noqa: ANN001
    comm: hcclComm_t,
    ranks_size: ctypes.c_uint32,
    c_rank_ids: ctypes.POINTER(ctypes.c_uint32),
    subcomm_id: ctypes.c_uint64,
    subcomm_rank: ctypes.c_uint64,
    comm_config: HcclCommConfig,
) -> hcclComm_t:
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
    def __init__(self, group: StatelessProcessGroup, device: torch.device):
        super().__init__(group, device)
        self.subcomm_id = 1

    def destroy_comm(self, comm: hcclComm_t = None):
        if comm:
            self.hccl.hcclCommDestroy(comm)
        else:
            self.hccl.hcclCommDestroy(self.comm)

    def all_gather(
        self, out_tensor: torch.Tensor, in_tensor: torch.Tensor, stream: torch.npu.Stream = None
    ) -> torch.Tensor:
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

    def create_subcomm(self, ranks: list[int]) -> hcclComm_t:
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


class DistributedHccl(Distributed):
    def __init__(self):
        self.pg: StatelessProcessGroup = None
        self.pyhccl: PyHcclCommunicatorEx = None
        self.sub_groups: dict[int, CommGroup] = {}
        self.comm: hcclComm_t = None

        self.host: str = None
        self.port: int = None
        self.rank: int = None
        self.world_size: int = None
        self.device: torch.device = None

        self.initialized: bool = False

    @contextmanager
    def _use_group(self, group: CommGroup | None, src: int | None = None):
        active_src = src
        if group:
            assert group.handle in self.sub_groups, "invalid sub_group"
            newcomm = ctypes.c_void_p(group.handle)
            self.pyhccl.comm = newcomm

            if src is not None:
                assert src in group.ranks, "src rank not in group"
                # convert src rank id in default world to newcomm
                active_src = group.ranks.index(src)
                self.pyhccl.rank = group.ranks.index(self.rank)

        try:
            yield active_src
        finally:
            if group:
                self.pyhccl.comm = self.comm
                if src is not None:
                    self.pyhccl.rank = self.rank

    def init_process_group(
        self,
        rank: int,
        world_size: int,
        store: torch.distributed.TCPStore,
        **kwargs,
    ):
        assert not self.initialized, "already initialized"

        self.rank = rank
        self.world_size = world_size
        self.device = torch.device("npu", torch.npu.current_device())

        self.pg = StatelessProcessGroup(rank=rank, world_size=world_size, store=store, socket=None)
        self.pyhccl = PyHcclCommunicatorEx(group=self.pg, device=self.device)
        self.comm = self.pyhccl.comm
        self.initialized = True

    def destroy_process_group(
        self,
        group: CommGroup | None = None,
    ):
        assert self.initialized, "not initialized"

        if group and group.handle in self.sub_groups:
            subcomm = ctypes.c_void_p(group.handle)
            self.pyhccl.destroy_comm(subcomm)
            del self.sub_groups[group.handle]
            return

        self.pyhccl.destroy_comm()
        self.pyhccl = None
        self.pg = None
        self.initialized = False

    def is_initialized(self) -> bool:
        return self.initialized

    def all_gather_object(self, object_list: list[Any], obj: Any, group: CommGroup | None = None):
        assert self.initialized, "not initialized"

        with self._use_group(group):
            _common_all_gather_object(self.pyhccl, self.device, self.world_size, object_list, obj)
            current_stream().synchronize()

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp.RedOpType = ReduceOp.SUM,
        group: CommGroup | None = None,
        **kwargs,
    ):
        assert self.initialized, "not initialized"

        with self._use_group(group):
            out_tensor = self.pyhccl.all_reduce(tensor, op)
            current_stream().synchronize()
            tensor.copy_(out_tensor)

    def broadcast(
        self, tensor: torch.Tensor, src: int | None = None, group: CommGroup | None = None, **kwargs
    ):
        assert self.initialized, "not initialized"

        with self._use_group(group, src) as local_rank:
            self.pyhccl.broadcast(tensor, local_rank)
            current_stream().synchronize()

    def barrier(self, group: CommGroup | None = None, **kwargs):
        assert self.initialized, "not initialized"

        with self._use_group(group):
            data = torch.zeros(1, device=self.device)
            self.pyhccl.all_reduce(data)
            current_stream().synchronize()

    def new_group(self, ranks: list[int], **kwargs) -> CommGroup | None:
        assert self.initialized, "not initialized"

        # ranks is None or []
        if not ranks:
            ranks = list(range(self.world_size))
        else:
            ranks.sort()

        group: CommGroup = None
        if self.rank not in ranks:
            return group

        subcomm = self.pyhccl.create_subcomm(ranks)
        if subcomm:
            group = CommGroup(subcomm.value, ranks)
            self.sub_groups[subcomm.value] = group
        return group
