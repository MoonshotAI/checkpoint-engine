import importlib
import io
import pickle
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Protocol

import torch
import torch.distributed as torch_dist


class CommunicatorProtocol(Protocol):
    def all_gather(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...


class CommGroup:
    def __init__(self, comm_handle: int, ranks: list[int]):
        self._comm = comm_handle
        self._ranks = ranks

    @property
    def handle(self) -> int:
        return self._comm

    @property
    def ranks(self) -> list[int]:
        return self._ranks


DistributedProcessGroup = torch_dist.ProcessGroup | CommGroup


class Distributed(ABC):
    @abstractmethod
    def init_process_group(
        self,
        rank: int,
        world_size: int,
        store: torch_dist.TCPStore,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def destroy_process_group(
        self,
        group: DistributedProcessGroup | None = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def is_initialized(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def all_gather_object(
        self,
        object_list: list[Any],
        obj: Any,
        group: DistributedProcessGroup | None = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: torch_dist.ReduceOp.RedOpType,
        group: DistributedProcessGroup | None = None,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int,
        group: DistributedProcessGroup | None = None,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def barrier(
        self,
        group: DistributedProcessGroup | None = None,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def new_group(
        self,
        ranks: list[int],
        **kwargs,
    ):
        raise NotImplementedError


class TorchBackend(Distributed):
    def init_process_group(
        self,
        rank: int,
        world_size: int,
        store: torch_dist.TCPStore,
        **kwargs,
    ):
        backend = kwargs.get("backend", "nccl")
        timeout = kwargs.get("timeout", timedelta(minutes=10))

        torch_dist.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
            timeout=timeout,
            store=store,
        )

    def destroy_process_group(self, group: DistributedProcessGroup | None = None):
        torch_dist.destroy_process_group(group)

    def is_initialized(self) -> bool:
        return torch_dist.is_initialized()

    def all_gather_object(
        self, object_list: list[Any], obj: Any, group: DistributedProcessGroup | None = None
    ):
        torch_dist.all_gather_object(object_list, obj, group)

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: torch_dist.ReduceOp.RedOpType = torch_dist.ReduceOp.SUM,
        group: DistributedProcessGroup | None = None,
        **kwargs,
    ):
        torch_dist.all_reduce(tensor, op, group, **kwargs)

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
        group: DistributedProcessGroup | None = None,
        **kwargs,
    ):
        torch_dist.broadcast(tensor, src, group, **kwargs)

    def barrier(self, group: DistributedProcessGroup | None = None, **kwargs):
        torch_dist.barrier(group, **kwargs)

    def new_group(self, ranks: list[int], **kwargs) -> DistributedProcessGroup | None:
        return torch_dist.new_group(ranks, **kwargs)


# specific device instance
_BACKEND_INSTANCE: Distributed = TorchBackend()

_pickler = pickle.Pickler
_unpickler = pickle.Unpickler


def _object_to_tensor(obj: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    f = io.BytesIO()
    _pickler(f).dump(obj)
    byte_storage = torch.ByteStorage._from_buffer(f.getvalue())
    byte_tensor = torch.ByteTensor(byte_storage).to(device)
    local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
    return byte_tensor, local_size


def _tensor_to_object(tensor: torch.Tensor, tensor_size: int) -> Any:
    tensor = tensor.cpu()
    buf = tensor.numpy().tobytes()[:tensor_size]
    return _unpickler(io.BytesIO(buf)).load()


def _flatten_for_scatter_gather(
    tensor_list: list[torch.Tensor], copy: bool = False
) -> torch.Tensor:
    if not tensor_list:
        raise RuntimeError("Received an empty list.")
    t = tensor_list[0]
    buffer_shape = [len(tensor_list)] + list(t.shape)

    buffer = torch.empty(tuple(buffer_shape), dtype=t.dtype, device=t.device)
    if copy:
        for i, tensor in enumerate(tensor_list):
            buffer[i].copy_(tensor)
    return buffer


def _common_all_gather_object(
    comm: CommunicatorProtocol,
    device: torch.device,
    world_size: int,
    object_list: list[Any],
    object: Any,
):
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


def use_backend(backend: str | None):
    global _BACKEND_INSTANCE

    if not backend:
        return

    mapping = {
        "vllm_nccl": ".vllm_nccl.DistributedNccl",
        "vllm_hccl": ".vllm_hccl.DistributedHccl",
    }
    if backend not in mapping:
        raise ValueError(f"Unsupported custom backend: {backend}")

    module_path, class_name = mapping[backend].rsplit(".", 1)
    module = importlib.import_module(module_path, "checkpoint_engine.distributed")
    backend_class = getattr(module, class_name)
    _BACKEND_INSTANCE = backend_class()


def init_process_group(
    rank: int,
    world_size: int,
    store: torch_dist.TCPStore,
    **kwargs,
):
    _BACKEND_INSTANCE.init_process_group(rank, world_size, store, **kwargs)


def destroy_process_group(group: DistributedProcessGroup | None = None):
    _BACKEND_INSTANCE.destroy_process_group(group)


def is_initialized() -> bool:
    return _BACKEND_INSTANCE.is_initialized()


def all_gather_object(
    object_list: list[Any],
    obj: Any,
    group: DistributedProcessGroup | None = None,
):
    _BACKEND_INSTANCE.all_gather_object(object_list, obj, group)


def all_reduce(
    tensor: torch.Tensor,
    op: torch_dist.ReduceOp.RedOpType = torch_dist.ReduceOp.SUM,
    group: DistributedProcessGroup | None = None,
    **kwargs,
):
    _BACKEND_INSTANCE.all_reduce(tensor, op, group, **kwargs)


def broadcast(
    tensor: torch.Tensor,
    src: int = 0,
    group: DistributedProcessGroup | None = None,
    **kwargs,
):
    _BACKEND_INSTANCE.broadcast(tensor, src, group, **kwargs)


def barrier(group: DistributedProcessGroup | None = None, **kwargs):
    _BACKEND_INSTANCE.barrier(group, **kwargs)


def new_group(ranks: list[int], **kwargs) -> DistributedProcessGroup | None:
    return _BACKEND_INSTANCE.new_group(ranks, **kwargs)
