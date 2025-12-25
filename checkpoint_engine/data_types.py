from typing import TYPE_CHECKING, Annotated, Any, NamedTuple

import torch
from pydantic import BaseModel, PlainSerializer, PlainValidator, WithJsonSchema


if TYPE_CHECKING:
    from typing import TypeVar

    from typing_extensions import TypedDict

    class FileMeta(TypedDict):
        key: str  # parameter name
        dtype: torch.dtype
        shape: torch.Size
        type: type
        tp_concat_dim: int

    T = TypeVar("T")


def _dt_validate(value: Any) -> torch.dtype:
    if isinstance(value, str):
        if not value.startswith("torch."):
            raise ValueError(f"dtype {value} should start with torch.")
        try:
            value = getattr(torch, value.split(".")[1])
        except AttributeError as e:
            raise ValueError(f"unknown dtype: {value}") from e
    if not isinstance(value, torch.dtype):
        raise TypeError(f"dtype {value} should be torch.dtype, got {type(value)}")
    return value


_TorchDtype = Annotated[
    torch.dtype,
    PlainValidator(_dt_validate),
    PlainSerializer(lambda x: str(x), return_type=str),
    WithJsonSchema({"type": "string"}, mode="serialization"),
]


def _size_validate(value: Any) -> torch.Size:
    if isinstance(value, list | tuple):
        return torch.Size(value)
    if not isinstance(value, torch.Size):
        raise TypeError(f"size {value} should be torch.Size, got {type(value)}")
    return value


_TorchSize = Annotated[
    torch.Size,
    PlainValidator(_size_validate),
    PlainSerializer(lambda x: tuple(x), return_type=tuple),
    WithJsonSchema({"type": "array", "items": {"type": "integer"}}, mode="serialization"),
]


def _tensor_validate(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    raise TypeError(f"tensor {value} should be torch.Tensor, got {type(value)}")


_TorchTensor = Annotated[
    torch.Tensor,
    PlainValidator(_tensor_validate),
]


class ParameterMeta(BaseModel):
    name: str
    dtype: _TorchDtype
    shape: _TorchSize
    aligned_size: int


class BucketRange(NamedTuple):
    idx: int  # bucket_idx of MemoryBucket in memory_pool
    offset: int
    size: int


class H2DBucket(BaseModel):
    size: int
    ranges: list[BucketRange]
    items: list[ParameterMeta]


class MemoryBufferMetas(BaseModel):
    metas: list[ParameterMeta]
    ptr: int
    size: int


class MemoryBuffer(BaseModel):
    buffer: _TorchTensor
    size: int
    metas: list[ParameterMeta]
    manually_pinned: bool = False


class MemoryBufferMetaList(BaseModel):
    p2p_store_addr: str | None
    memory_buffer_metas_list: list[MemoryBufferMetas]
    rdma_device: str


class DataToGather(MemoryBufferMetaList):
    host_ip: str
    device_uuid: str
