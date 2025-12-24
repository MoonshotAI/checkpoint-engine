import base64
import ctypes
import datetime
import io
import logging
import os
import pickle
from enum import Enum
from typing import Any, List

import torch
import torch_npu


_pickler = pickle.Pickler
_unpickler = pickle.Unpickler


class ReduceOp(Enum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3


logger = logging.getLogger(__name__)
libhccl = None
try:
    libhccl = ctypes.CDLL("libhccl.so")
except OSError:
    raise ImportError


class HcclRootInfo(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 4108)]


buffer_type = ctypes.c_void_p
npuStream_t = ctypes.c_void_p
hcclComm_t = ctypes.c_void_p


class HcclDataTypeEnum:
    HCCL_DATA_TYPE_INT8 = 0
    HCCL_DATA_TYPE_INT16 = 1
    HCCL_DATA_TYPE_INT32 = 2
    HCCL_DATA_TYPE_FP16 = 3
    HCCL_DATA_TYPE_FP32 = 4
    HCCL_DATA_TYPE_INT64 = 5
    HCCL_DATA_TYPE_UINT8 = 7
    HCCL_DATA_TYPE_FP64 = 10
    HCCL_DATA_TYPE_BFP16 = 11

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        _DTYPE_MAP = {
            torch.int8: cls.HCCL_DATA_TYPE_INT8,
            torch.int16: cls.HCCL_DATA_TYPE_INT16,
            torch.int32: cls.HCCL_DATA_TYPE_INT32,
            torch.float16: cls.HCCL_DATA_TYPE_FP16,
            torch.float32: cls.HCCL_DATA_TYPE_FP32,
            torch.int64: cls.HCCL_DATA_TYPE_INT64,
            torch.uint8: cls.HCCL_DATA_TYPE_UINT8,
            torch.float64: cls.HCCL_DATA_TYPE_FP64,
            torch.bfloat16: cls.HCCL_DATA_TYPE_BFP16,
        }
        hccl_dtype = _DTYPE_MAP.get(dtype)
        if hccl_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return hccl_dtype


class HcclRedOpTypeEnum:
    HCCL_REDUCE_SUM = 0
    HCCL_REDUCE_PROD = 1
    HCCL_REDUCE_MAX = 2
    HCCL_REDUCE_MIN = 3

    @classmethod
    def from_base(cls, op: ReduceOp) -> int:
        _OP_MAP = {
            ReduceOp.SUM: cls.HCCL_REDUCE_SUM,
            ReduceOp.PRODUCT: cls.HCCL_REDUCE_PROD,
            ReduceOp.MAX: cls.HCCL_REDUCE_MAX,
            ReduceOp.MIN: cls.HCCL_REDUCE_MIN,
        }
        hccl_op = _OP_MAP.get(op)
        if hccl_op is None:
            raise ValueError(f"Unsupported op: {op}")
        return hccl_op


_name_map = {}


def is_group_exist(group_name: str = "default_group") -> bool:
    return group_name in _name_map


def create_group(
    group_size: int,
    rank: int,
    device_index: int,
    group_name: str = "default_group",
    master_addr: str | None = None,
    master_port: int | None = None,
    store: torch.distributed.TCPStore | None = None,
):
    if group_name in _name_map:
        return _name_map[group_name]

    g = HCCLGroup(group_size, rank, group_name, device_index, master_addr, master_port, store)
    _name_map[group_name] = g
    return g


def destroy_group(group_name: str = "default_group"):
    assert isinstance(group_name, str)
    if group_name not in _name_map:
        return

    g = _name_map[group_name]
    g.destroy()
    del _name_map[group_name]


def get_handle_by_name(group_name: str):
    assert group_name in _name_map, f"{group_name} not in _name_map"
    return _name_map[group_name]


def get_default_handle():
    return get_handle_by_name("default_group")


def get_default_store():
    return get_handle_by_name("default_group").get_tcp_store()


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


class HCCLGroup:
    def __init__(
        self,
        group_size: int,
        rank: int,
        group_name: str,
        device_index: int,
        master_addr: str | None = None,
        master_port: int | None = None,
        store: torch.distributed.TCPStore | None = None,
    ):
        """Init an HCCL collective group."""

        self.group_size = group_size
        self.rank = rank
        self.group_name = group_name
        self.libhccl = libhccl
        self.device = torch.device("npu", device_index)
        self.store = store
        torch.npu.set_device(self.device)

        self.rank_table_file = os.environ.get("RANK_TABLE_FILE", None)

        master_addr = master_addr or os.environ["MASTER_ADDR"]
        master_port = master_port or int(os.environ["MASTER_PORT"]) + 100
        if self.store is None:
            self.store = torch.distributed.TCPStore(
                master_addr,
                master_port,
                group_size,
                is_master=rank == 0,
                timeout=datetime.timedelta(seconds=180),
            )
        if rank == 0:
            root_info = self._generate_hccl_root_info()
            root_info_b64 = base64.b64encode(bytes(root_info)).decode("utf-8")
            self.store.set(group_name, root_info_b64)
        else:
            root_info_b64 = self.store.get(group_name)
            root_info_bytes = base64.b64decode(root_info_b64)
            root_info = HcclRootInfo.from_buffer_copy(bytearray(root_info_bytes))

        self.comm = self._create_hccl_comm(root_info)
        self.stream = torch.npu.Stream()
        self.subcomm_id = 1
        self.subcomms = {}
        self.initialized = True

    def create_subcomm(self, ranks: list[int] | None) -> int:
        assert self.initialized, "Not initialied, maybe destroyed"

        if ranks and self.rank not in ranks:
            return 0

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
        subcomm = hcclComm_t()
        subcomm_id = self.subcomm_id

        if ranks:
            uint32_array = ctypes.c_uint32 * len(ranks)
            c_rank_ids = uint32_array(*ranks)
            subcomm_rank = ranks.index(self.rank)
        else:
            uint32_array = ctypes.c_uint32 * self.group_size
            c_rank_ids = uint32_array(*list(range(self.group_size)))
            subcomm_rank = self.rank

        ranks_size = len(ranks) if ranks else self.group_size
        exec_result = self.libhccl.HcclCreateSubCommConfig(
            ctypes.byref(self.comm),
            ranks_size,
            c_rank_ids,
            subcomm_id,
            subcomm_rank,
            ctypes.byref(comm_config),
            ctypes.byref(subcomm),
        )
        assert exec_result == 0, (
            f"Failed to execute 'HcclCreateSubCommConfig'. Error code: {exec_result}"
        )
        self.subcomms[subcomm_id] = subcomm
        self.subcomm_id += 1
        return subcomm_id

    def destroy(self, subcomm_id=None):
        if subcomm_id:
            assert subcomm_id in self.subcomms, f"{subcomm_id} not in subcomms"
            exec_result = self.libhccl.HcclCommDestroy(self.subcomms[subcomm_id])
            assert exec_result == 0, (
                f"Failed to execute 'HcclCommDestroy'. Error code: {exec_result}"
            )
            del self.subcomms[subcomm_id]
            return

        for _, subcomm in self.subcomms.items():
            exec_result = self.libhccl.HcclCommDestroy(subcomm)
            assert exec_result == 0, (
                f"Failed to execute 'HcclCommDestroy'. Error code: {exec_result}"
            )

        exec_result = self.libhccl.HcclCommDestroy(self.comm)
        assert exec_result == 0, f"Failed to execute 'HcclCommDestroy'. Error code: {exec_result}"
        if self.rank == 0:
            self.store.delete_key(self.group_name)

        self.store = None
        self.comm = None
        self.stream = None
        self.subcomm_id = 1
        self.subcomms = {}
        self.initialized = False

    def broadcast(self, tensor: torch.Tensor, src: int = 0, subcomm_id=None):
        """Broadcast tensors to all other npus following options.

        Args:
            tensor: tensor to be broadcast or received.
            src: source rank on group.

        Returns:
            None
        """

        assert self.initialized, "Not initialied, maybe destroyed"

        if subcomm_id:
            assert subcomm_id in self.subcomms, f"{subcomm_id} not in subcomms"
            comm = self.subcomms[subcomm_id]
        else:
            comm = self.comm

        with torch.npu.device(self.device):
            exec_result = self.libhccl.HcclBroadcast(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                HcclDataTypeEnum.from_torch(tensor.dtype),
                src,
                comm,
                npuStream_t(self.stream.npu_stream),
            )
            self.stream.synchronize()

        assert exec_result == 0, f"Failed to execute 'HcclBroadcast'. Error code: {exec_result}."

    def all_gather(self, tensor_list: list[torch.Tensor], tensor: torch.Tensor, subcomm_id=None):
        """Allgather tensors across npus into a list of tensors.

        Args:
            tensor_list (List[Tensor]): allgathered tensors.
            tensor (torch.Tensor): Tensor to be gathered from current process.

        Returns:
            None
        """
        assert self.initialized, "Not initialied, maybe destroyed"

        if subcomm_id:
            assert subcomm_id in self.subcomms, f"{subcomm_id} not in subcomms"
            comm = self.subcomms[subcomm_id]
        else:
            comm = self.comm

        output_flattened = _flatten_for_scatter_gather(tensor_list)

        with torch.npu.device(self.device):
            exec_result = self.libhccl.HcclAllGather(
                buffer_type(tensor.data_ptr()),
                buffer_type(output_flattened.data_ptr()),
                tensor.numel(),
                HcclDataTypeEnum.from_torch(tensor.dtype),
                comm,
                npuStream_t(self.stream.npu_stream),
            )
            self.stream.synchronize()
        assert exec_result == 0, f"Failed to execute 'HcclAllGather'. Error code: {exec_result}."

        for i, x in enumerate(tensor_list):
            x.copy_(output_flattened[i])

    def all_gather_object(self, object_list: list[Any], object: Any, subcomm_id=None):
        """Allgather python objects across npus into a list of objects.

        Args:
            tensor_list (List[Any]): allgathered python objects.
            tensor (Any): python object to be gathered from current process.

        Returns:
            None
        """
        assert self.initialized, "Not initialied, maybe destroyed"

        input_tensor, local_size = self._object_to_tensor(object, self.device)
        object_sizes_tensor = torch.zeros(self.group_size, dtype=torch.long, device=self.device)
        object_size_list = [object_sizes_tensor[i].unsqueeze(dim=0) for i in range(self.group_size)]
        self.all_gather(object_size_list, local_size, subcomm_id)
        max_object_size = int(max(object_size_list).item())
        input_tensor.resize_(max_object_size)
        coalesced_output_tensor = torch.empty(
            max_object_size * self.group_size, dtype=torch.uint8, device=self.device
        )
        output_tensors = [
            coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
            for i in range(self.group_size)
        ]
        self.all_gather(output_tensors, input_tensor, subcomm_id)
        for i, tensor in enumerate(output_tensors):
            tensor = tensor.type(torch.uint8)
            tensor_size = object_size_list[i]
            object_list[i] = self._tensor_to_object(tensor, tensor_size)

    def all_reduce(self, tensor, op=ReduceOp.SUM, subcomm_id=None):
        """AllReduce tensor across the collective group following options.

        Args:
            tensor: Input and output of the collective. Each tensor must reside on one NPU of the current process.
            reduce_op: reduce options.

        Returns:
            None
        """
        assert self.initialized, "Not initialied, maybe destroyed"

        if subcomm_id:
            assert subcomm_id in self.subcomms, f"{subcomm_id} not in subcomms"
            comm = self.subcomms[subcomm_id]
        else:
            comm = self.comm

        with torch.npu.device(self.device):
            exec_result = self.libhccl.HcclAllReduce(
                buffer_type(tensor.data_ptr()),
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                HcclDataTypeEnum.from_torch(tensor.dtype),
                HcclRedOpTypeEnum.from_base(op),
                comm,
                npuStream_t(self.stream.npu_stream),
            )
            self.stream.synchronize()
        assert exec_result == 0, f"Failed to execute 'HcclAllReduce'. Error code: {exec_result}."

    def barrier(self, subcomm_id=None):
        """Blocks until all processes reach this barrier.

        Returns:
            None
        """
        assert self.initialized, "Not initialied, maybe destroyed"

        tensor = torch.empty(1, dtype=torch.int8, device=self.device)
        self.all_reduce(tensor, subcomm_id=subcomm_id)

    def get_tcp_store(self):
        return self.store

    def _generate_hccl_root_info(self, dev=0):
        root_info = HcclRootInfo()

        with torch.npu.device(f"npu:{dev}"):
            exec_result = self.libhccl.HcclGetRootInfo(ctypes.byref(root_info))
        assert exec_result == 0, f"Failed to execute 'HcclGetRootInfo'. Error code: {exec_result}."

        return root_info

    def _create_hccl_comm(self, root_info):
        comm = hcclComm_t()

        with torch.npu.device(self.device):
            if self.rank_table_file is not None:
                exec_result = self.libhccl.HcclCommInitClusterInfo(
                    self.rank_table_file.encode("utf-8"),
                    self.rank,
                    ctypes.byref(comm),
                )
            else:
                exec_result = self.libhccl.HcclCommInitRootInfo(
                    self.group_size,
                    ctypes.byref(root_info),
                    self.rank,
                    ctypes.byref(comm),
                )
            assert exec_result == 0, (
                f"Failed to execute 'HcclCommInitRootInfo'. Error code: {exec_result}"
            )

        return comm

    def _object_to_tensor(self, obj, device):
        f = io.BytesIO()
        _pickler(f).dump(obj)
        byte_storage = torch.ByteStorage._from_buffer(f.getvalue())  # type: ignore[attr-defined]
        byte_tensor = torch.ByteTensor(byte_storage).to(device)
        local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
        return byte_tensor, local_size

    def _tensor_to_object(self, tensor, tensor_size):
        tensor = tensor.cpu()
        buf = tensor.numpy().tobytes()[:tensor_size]
        return _unpickler(io.BytesIO(buf)).load()


def _flatten_for_scatter_gather(tensor_list, copy=False):
    """Flatten the tensor for gather/scatter operations.

    Args:
        tensor_list: the list of tensors to be scattered/gathered.
        copy: whether to copy the tensors in tensor_list into the buffer.

    Returns:
        The flattened tensor buffer.
    """
    if not tensor_list:
        raise RuntimeError("Received an empty list.")
    t: torch.Tensor = tensor_list[0]
    buffer_shape = [len(tensor_list)] + list(t.shape)

    buffer = torch.empty(tuple(buffer_shape), dtype=t.dtype, device=t.device)
    if copy:
        for i, tensor in enumerate(tensor_list):
            buffer[i].copy_(tensor)
    return buffer


def _check_inputs_compatibility_for_scatter_gather(
    tensors: List[torch.Tensor], tensor_lists: List[List[torch.Tensor]]
) -> None:
    """Check the compatibility between tensor input and tensor list input."""
    if not tensors or not isinstance(tensors, list):
        raise RuntimeError("The first argument 'tensors' expects a list of tensors.")
    if not tensor_lists or not isinstance(tensor_lists, list):
        raise RuntimeError("The second argument 'tensor_lists' expects a list of tensor list.")
    dtype = tensors[0].dtype
    shape = list(tensors[0].shape)
    for i, tensor_list in enumerate(tensor_lists):
        # check all tensor in `tensors` match.
        dt = tensors[i].dtype
        if dt != dtype:
            raise RuntimeError(
                "All tensor operands to scatter/gather must "
                f"have the same dtype. Got '{dt}' and '{dtype}'."
            )
        s = list(tensors[i].shape)
        if s != shape:
            raise RuntimeError(
                "All tensor operands to scatter/gather must "
                f"have the same shape. Got '{s}' and '{shape}'."
            )
        # check all tensors in `tensor_lists` match.
        for t in tensor_lists[i]:
            # check dtype
            dtl = t.dtype
            if dtl != dtype:
                raise RuntimeError(
                    "All tensor operands to scatter/gather must "
                    f"have the same dtype. Got '{dtl}' and '{dtype}'."
                )
            sl = list(t.shape)
            if sl != shape:
                raise RuntimeError(
                    "All tensor operands to scatter/gather must "
                    f"have the same shape. Got '{sl}' and '{shape}'."
                )
