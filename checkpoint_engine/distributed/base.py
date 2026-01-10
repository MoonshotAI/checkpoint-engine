from abc import ABC, abstractmethod
import io
import pickle
from datetime import timedelta
from typing import Any, List
import importlib

import torch
from torch.distributed import ReduceOp


class Distributed(ABC):
    @abstractmethod
    def init_process_group(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        timeout: timedelta,
    ):
        raise NotImplementedError

    @abstractmethod
    def destroy_process_group(
        self,
        group,
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
        group,
    ):
        raise NotImplementedError

    @abstractmethod
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op :ReduceOp,
        group,
    ):
        raise NotImplementedError

    @abstractmethod
    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int,
        group,
    ):
        raise NotImplementedError

    @abstractmethod
    def barrier(
        self,
        group,
    ):
        raise NotImplementedError

    @abstractmethod
    def new_group(
        self,
        ranks: list[int],
    ):
        raise NotImplementedError


# specific device instance
_BACKEND_INSTANCE = None

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


def init_process_group(
    host: str,
    port: int,
    rank: int,
    world_size: int,
    device_type: str,
    timeout: timedelta = timedelta(seconds=300),
):
    global _BACKEND_INSTANCE

    mapping = {
        "cuda": ".nccl.DistributedNccl",
        "npu": ".hccl.DistributedHccl",
    }

    if device_type not in mapping:
        raise ValueError(f"Unsupported device type: {device_type}")

    module_path, class_name = mapping[device_type].rsplit(".", 1)
    module = importlib.import_module(module_path, ".checkpoint_engine.distributed")
    backend_class = getattr(module, class_name)

    _BACKEND_INSTANCE = backend_class()
    _BACKEND_INSTANCE.init_process_group(host, port, rank, world_size, timeout)


def destroy_process_group(group=None):
    if _BACKEND_INSTANCE is None:
        raise RuntimeError("distribute module not initialized")
    _BACKEND_INSTANCE.destroy_process_group(group)


def is_initialized() -> bool:
    if _BACKEND_INSTANCE is None:
        return False
    _BACKEND_INSTANCE.is_initialized()

def all_gather_object(
    object_list: list[Any],
    obj: Any,
    group=None,
):
    if _BACKEND_INSTANCE is None:
        raise RuntimeError("distribute module not initialized")
    _BACKEND_INSTANCE.all_gather_object(object_list, obj, group)


def all_reduce(
    tensor: torch.Tensor,
    op=ReduceOp.SUM,
    group=None,
):
    if _BACKEND_INSTANCE is None:
        raise RuntimeError("distribute module not initialized")
    _BACKEND_INSTANCE.all_reduce(tensor, op, group)


def broadcast(
    tensor: torch.Tensor,
    src= None,
    group=None,
):
    if _BACKEND_INSTANCE is None:
        raise RuntimeError("distribute module not initialized")
    _BACKEND_INSTANCE.all_reduce(tensor, src, group)


def barrier(group=None):
    if _BACKEND_INSTANCE is None:
        raise RuntimeError("distribute module not initialized")
    _BACKEND_INSTANCE.barrier(group)


def new_group(ranks: list[int]):
    if _BACKEND_INSTANCE is None:
        raise RuntimeError("distribute module not initialized")
    _BACKEND_INSTANCE.new_group(ranks)
