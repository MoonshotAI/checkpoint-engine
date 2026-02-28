import ctypes
from contextlib import contextmanager
from typing import Any, ClassVar

import torch
from torch.distributed import ReduceOp
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import (
    Function,
    NCCLLibrary,
    ncclComm_t,
    ncclResult_t,
)
from vllm.distributed.utils import StatelessProcessGroup

from checkpoint_engine.distributed.base import CommGroup, Distributed, _common_all_gather_object


try:
    from vllm.utils.torch_utils import current_stream
except ImportError:
    try:
        from vllm.utils import current_stream
    except ImportError:
        raise ImportError(
            "Could not find 'current_stream' in vllm. Please check your vllm version."
        ) from None


class NcclConfigT(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [
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
    # ncclComm_t comm, int color, int key, ncclComm_t *newcomm, NcclConfigT *config
    # )
    Function(
        "ncclCommSplit",
        ncclResult_t,
        [
            ncclComm_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ncclComm_t),
            ctypes.POINTER(NcclConfigT),
        ],
    ),
]


def nccl_comm_split(
    self,  # noqa: ANN001
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
    def destroy_comm(self, comm: ncclComm_t = None):
        if comm:
            self.nccl.ncclCommDestroy(comm)
        else:
            self.nccl.ncclCommDestroy(self.comm)

    def create_newcomm(self, ranks: list[int]) -> ncclComm_t:
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

    @contextmanager
    def _use_group(self, group: CommGroup | None, src: int | None = None):
        active_src = src
        if group:
            assert group.handle in self.sub_groups, "invalid sub_group"
            newcomm = ctypes.c_void_p(group.handle)
            self.pynccl.comm = newcomm

            if src is not None:
                assert src in group.ranks, "src rank not in group"
                # convert src rank id in default world to newcomm
                active_src = group.ranks.index(src)
                self.pynccl.rank = group.ranks.index(self.rank)

        try:
            yield active_src
        finally:
            if group:
                self.pynccl.comm = self.comm
                if src is not None:
                    self.pynccl.rank = self.rank

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
        self.device = torch.device("cuda", torch.cuda.current_device())

        self.pg = StatelessProcessGroup(rank=rank, world_size=world_size, store=store, socket=None)
        self.pynccl = PyNcclCommunicatorEx(group=self.pg, device=self.device)
        self.comm = self.pynccl.comm
        self.initialized = True

    def destroy_process_group(
        self,
        group: CommGroup | None = None,
    ):
        assert self.initialized, "not initialized"

        if group and group.handle in self.sub_groups:
            newcomm = ctypes.c_void_p(group.handle)
            self.pynccl.destroy_comm(newcomm)
            del self.sub_groups[group.handle]
            return

        self.pynccl.destroy_comm()
        self.pynccl = None
        self.pg = None
        self.initialized = False

    def is_initialized(self) -> bool:
        return self.initialized

    def all_gather_object(self, object_list: list[Any], obj: Any, group: CommGroup | None = None):
        assert self.initialized, "not initialized"

        with self._use_group(group):
            _common_all_gather_object(self.pynccl, self.device, self.world_size, object_list, obj)
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
            out_tensor = self.pynccl.all_reduce(in_tensor=tensor, op=op)
            current_stream().synchronize()
            tensor.copy_(out_tensor)

    def broadcast(
        self, tensor: torch.Tensor, src: int | None = None, group: CommGroup | None = None, **kwargs
    ):
        assert self.initialized, "not initialized"

        with self._use_group(group, src) as local_src:
            self.pynccl.broadcast(tensor, local_src)
            current_stream().synchronize()

    def barrier(self, group: CommGroup | None = None, **kwargs):
        assert self.initialized, "not initialized"

        with self._use_group(group):
            data = torch.zeros(1, device=self.device)
            self.pynccl.all_reduce(data)
            current_stream().synchronize()

    def new_group(self, ranks: list[int], **kwargs) -> CommGroup | None:
        assert self.initialized, "not initialized"

        # ranks is None or []
        if not ranks:
            ranks = list(range(self.world_size))
        else:
            ranks.sort()

        group: CommGroup = None
        newcomm = self.pynccl.create_newcomm(ranks)
        if newcomm:
            group = CommGroup(newcomm.value, ranks)
            self.sub_groups[newcomm.value] = group
        return group
