import argparse
import concurrent.futures
import ctypes
import json
import os
import pickle
import random
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING, Annotated, Any, BinaryIO, NamedTuple

import httpx
import numpy as np
import torch
import torch.distributed as dist
import zmq
from loguru import logger
from pydantic import BaseModel, PlainSerializer, PlainValidator, WithJsonSchema
from safetensors.torch import _getdtype, safe_open
from torch.multiprocessing.reductions import reduce_tensor

from checkpoint_engine.device_utils import DeviceManager, get_ip, npu_generate_uuid


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


# 256 bytes alignment when flatten torch tensors to uint8 buffer
_ALIGN_SIZE = 256


def _align_size(dtype: torch.dtype, shape: torch.Size) -> int:
    return (dtype.itemsize * shape.numel() + _ALIGN_SIZE - 1) // _ALIGN_SIZE * _ALIGN_SIZE


def _to_named_tensor(metas: list[ParameterMeta], offset: int = 0) -> list[dict]:
    ret = []
    for meta in metas:
        size = meta.aligned_size
        ret.append(
            {
                "name": meta.name,
                "dtype": meta.dtype,
                "shape": meta.shape,
                "offset": offset,
            }
        )
        offset += size
    return ret


def _load_checkpoint_file(file_path: str) -> tuple[int, dict[str, tuple["FileMeta", torch.Tensor]]]:
    def _safetensors_load(fn: str) -> dict[str, tuple["FileMeta", torch.Tensor]]:
        ret = {}
        with safe_open(fn, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                weight = f.get_tensor(name)
                meta = {
                    "key": name,
                    "dtype": weight.dtype,
                    "shape": weight.shape,
                    "type": type(weight),
                    "tp_concat_dim": -1,  # safetensors does not support tp_concat_dim
                }
                ret[name] = (meta, weight)
        return ret

    # deprecated, will be removed in the future
    def _fast_np_load(fn: str) -> dict[str, tuple["FileMeta", torch.Tensor]]:
        """load *.np file and return memmap and related tensor meta"""

        def parse_npy_header(fin: BinaryIO) -> dict[str, Any]:
            start = fin.tell()
            major, minor = np.lib.format.read_magic(fin)
            if major == 1 and minor == 0:
                read_header_fn = np.lib.format.read_array_header_1_0
            elif major == 2 and minor == 0:
                read_header_fn = np.lib.format.read_array_header_2_0
            else:
                raise ValueError(
                    f"unknown version {major}.{minor} when parsing npy header from {fn}"
                )
            shape, is_fortran, dtype = read_header_fn(fin)
            return {
                "shape": shape,
                "is_fortran": is_fortran,
                "dtype": dtype,
                "header_length": fin.tell() - start,
            }

        meta_fn = fn + ".meta"
        with open(meta_fn, "rb") as fin:
            meta_lst = pickle.load(fin)

        tensors = []
        offset = 0
        with open(fn, "rb") as fin:
            fin.seek(0, os.SEEK_END)
            filesize = fin.tell()
            fin.seek(0)
            while fin.tell() < filesize:
                tensor_meta = parse_npy_header(fin)
                tensor = np.memmap(
                    fn,
                    dtype=tensor_meta["dtype"],
                    mode="c",
                    offset=offset + tensor_meta["header_length"],
                    shape=tensor_meta["shape"],
                )
                offset += tensor_meta["header_length"] + tensor.nbytes
                fin.seek(offset)
                tensors.append(tensor)

        assert len(meta_lst) == len(tensors)
        ret = {}
        for meta, tensor in zip(meta_lst, tensors):
            if meta["type"] == torch.Tensor:
                tensor = torch.from_numpy(tensor)
            tensor = tensor.view(dtype=meta["dtype"]).view(*meta["shape"])
            ret[meta["key"]] = (meta, tensor)
        return ret

    tp_rank = 0
    if file_path.endswith(".npy"):
        logger.warning("numpy model file is deprecated, will be removed in the future")
        filename_split = os.path.basename(file_path).split(".")
        # if using numpy and want to specify tp rank
        # file should be in model.{layer}.{tp}[.{ep}].npy format
        tp_rank = int(filename_split[2]) if len(filename_split) > 3 else 0
        ret = _fast_np_load(file_path)
    elif file_path.endswith(".safetensors"):
        ret = _safetensors_load(file_path)
    else:
        raise ValueError(f"unsupported file format: {file_path}")
    return tp_rank, ret


def _concat_tp_weights(
    tp_weights: list[torch.Tensor], tp_concat_dim: int, tp_size: int
) -> torch.Tensor:
    """Concat tp weights with meta info.
    If meta.concat_dim is -1, meas this is shared tp weights, just use the first weights.
    Else we will cat weights in concat_dim.
    """
    if tp_concat_dim == -1:
        return tp_weights[0]
    assert tp_size == len(tp_weights)
    if len(tp_weights) == 1:
        return tp_weights[0]
    return torch.cat([w for w in tp_weights], dim=tp_concat_dim)


def _get_physical_gpu_id(device_manager: DeviceManager, device_index: int | None = None) -> str:
    try:
        if device_manager.device_type == "npu":
            return f"NPU-{npu_generate_uuid()}"
        else:
            return f"GPU-{device_manager.device_module.get_device_properties(device_index).uuid!s}"
    except AssertionError as e:
        raise ValueError(f"fail to get physical gpu id {device_index}") from e


def _ibv_get_device_list() -> list[str]:
    lib = ctypes.CDLL("libibverbs.so.1")
    lib.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]  # int *num_devices
    lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)  # struct ibv_device **

    lib.ibv_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.ibv_get_device_name.argtypes = [ctypes.c_void_p]  # struct ibv_device *
    lib.ibv_get_device_name.restype = ctypes.c_char_p  # const char *

    num = ctypes.c_int()
    dev_array = lib.ibv_get_device_list(ctypes.byref(num))
    if not dev_array or num.value <= 0:
        return []

    devices = []
    for i in range(num.value):
        dev_ptr = dev_array[i]  # struct ibv_device *
        name = lib.ibv_get_device_name(dev_ptr)  # const char *
        devices.append(name.decode())
    lib.ibv_free_device_list(dev_array)
    return devices


def _get_rdma_devices() -> list[str]:
    """
    use _ibv_get_device_list to get RDMA devices, if NCCL_IB_HCA has multiple values, just return
    """
    devices_str = os.getenv("PS_P2P_STORE_RDMA_DEVICES")
    if devices_str:
        return devices_str.split(",")
    # if PS_P2P_STORE_RDMA_DEVICES is not set, try to use NCCL_IB_HCA to get RDMA devices
    hca = os.getenv("NCCL_IB_HCA", None)
    return _parse_NCCL_IB_HCA(hca or "", _ibv_get_device_list()) or _ibv_get_device_list()


def _get_my_rdma_device(local_rank: int, gpu_count: int, devices: list[str]) -> str:
    """
    implement network card device allocation, if network card is "mlx5_0,mlx5_1", then 0-3 will share mlx5_0, 4-7 will share mlx5_1, etc.
    """
    if not devices:
        raise RuntimeError("no rdma devices found")
    try:
        assert len(devices) <= gpu_count, (
            f"rdma devices count {len(devices)} should be less than or equal to gpu count {gpu_count}"
        )
        assert gpu_count % len(devices) == 0, (
            f"gpu count {gpu_count} should be divisible by rdma devices count {len(devices)}"
        )
        return devices[local_rank // (gpu_count // len(devices))]
    except AssertionError:
        logger.error(
            "Please set 'NCCL_IB_HCA' or 'PS_P2P_STORE_RDMA_DEVICES' environment variable to choose proper number of RDMA devices."
            "The number of RDMA devices should be less than or equal to GPU count, and GPU count should be divisible by the number of RDMA devices."
            "The acceptable value by NCCL_IB_HCA is documented in 'https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#id8'."
        )
        raise


def _parse_NCCL_IB_HCA(value: str, available_devices: list[str]) -> list[str]:
    """
    The acceptable value by NCCL_IB_HCA is documented in https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#id8.
    The Python version parser is referred to the CPP parser in NCCL: https://github.com/NVIDIA/nccl/blob/v2.28.3-1/src/transport/net_ib.cc#L658-L662.

    The list is comma-separated; port numbers are NOT supported yet.
    An optional prefix '^' indicates the list is an exclude list.
    A second optional prefix '=' indicates that the tokens are exact names, otherwise by default NCCL would treat each token as a prefix.
    Please note that when '^' and '=' appear together, only '^=' is allowed, '=^' is not supported.

    Examples:
    - `NCCL_IB_HCA="mlx5"`: Use all cards starting with `mlx5`.
    - `NCCL_IB_HCA="=mlx5_0,mlx5_1"`: Use specific cards `mlx5_0` and `mlx5_1`.
    - `NCCL_IB_HCA="^mlx5"`: Use all cards except those starting with `mlx5`.
    - `NCCL_IB_HCA="^=mlx5_0,mlx5_1"`: Use all cards except `mlx5_0` and `mlx5_1`.
    """
    max_hcas = 32
    if not value or value.strip() == "":
        return available_devices[:max_hcas]

    value = value.strip()
    result = []
    is_exclude = value.startswith("^")
    if is_exclude:
        value = value.removeprefix("^")
    is_exact_match = value.startswith("=")
    if is_exact_match:
        value = value.removeprefix("=")

    device_specs = [spec.strip() for spec in value.split(",") if spec.strip()]

    result = _resolve_device_specs(device_specs, is_exact_match, available_devices)
    if is_exclude:
        result = [dev for dev in available_devices if dev not in result]
    if len(result) > max_hcas:
        result = result[:max_hcas]

    logger.info(f"RDMA Devices from 'NCCL_IB_HCA': {result}")

    return result


def _resolve_device_specs(
    device_specs: list[str], is_exact_match: bool, available_devices: list[str]
) -> list[str]:
    devices = set()
    for spec in device_specs:
        parts = spec.split(":", 1)
        device_name = parts[0].strip()
        # HACK: mooncake transfer engine does not support port specification yet, so we ignore it
        # port = parts[1].strip() if len(parts) > 1 else None
        base_devices = (
            [device_name]
            if device_name in available_devices
            else []
            if is_exact_match
            else [dev for dev in available_devices if dev.startswith(device_name)]
        )

        if not base_devices:
            logger.warning(f"No RDMA device match {device_name=} where {is_exact_match=}.")
            continue

        for base_dev in base_devices:
            devices.add(base_dev)

    return sorted(devices)


def _load_checkpoint(files: list[str]) -> dict[str, torch.Tensor]:
    class TPMeta(BaseModel):
        concat_dim: int
        size: int

    parameters: dict[str, torch.Tensor] = {}
    parameter_metas: dict[str, ParameterMeta] = {}
    tp_metas: dict[str, TPMeta] = {}
    parameters_with_tp: dict[str, dict[int, torch.Tensor]] = {}
    for file in files:
        tp_rank, ret = _load_checkpoint_file(file)
        for parameter_name, (meta, weight) in ret.items():
            if parameter_name not in parameters_with_tp:
                parameters_with_tp[parameter_name] = {}
            parameters_with_tp[parameter_name][tp_rank] = weight
            if parameter_name not in tp_metas:
                tp_metas[parameter_name] = TPMeta(
                    concat_dim=meta["tp_concat_dim"],
                    size=1,
                )
            if parameter_name not in parameter_metas:
                assert isinstance(meta["dtype"], torch.dtype), (
                    f"meta {meta} dtype should be torch.dtype"
                )
                assert isinstance(meta["shape"], torch.Size), (
                    f"meta {meta} shape should be torch.Size"
                )
                parameter_metas[parameter_name] = ParameterMeta(
                    name=parameter_name,
                    shape=meta["shape"],
                    dtype=meta["dtype"],
                    aligned_size=_align_size(meta["dtype"], meta["shape"]),
                )
            tp_meta = tp_metas[parameter_name]
            if tp_meta.concat_dim != -1:
                tp_meta.size = max(tp_meta.size, tp_rank + 1)
    for name, tp_meta in tp_metas.items():
        if tp_meta.concat_dim != -1:
            shape = list(parameter_metas[name].shape)
            shape[tp_meta.concat_dim] = shape[tp_meta.concat_dim] * tp_meta.size
            parameter_metas[name] = ParameterMeta(
                name=name,
                shape=torch.Size(shape),
                dtype=parameter_metas[name].dtype,
                aligned_size=_align_size(parameter_metas[name].dtype, torch.Size(shape)),
            )
        weights_in_cpu = [parameters_with_tp[name][key] for key in sorted(parameters_with_tp[name])]
        # TODO: here concat is serial, which may be slow
        # but since tp storage is not used in the future
        # we ignore this performance issue for now
        parameters[name] = _concat_tp_weights(weights_in_cpu, tp_meta.concat_dim, tp_meta.size)
    for name, parameter in parameters.items():
        assert name in parameter_metas, f"parameter {name} not found in parameter_metas"
        assert parameter_metas[name].shape == parameter.shape, (
            f"parameter {name} shape mismatch, {parameter_metas[name].shape} != {parameter.shape}"
        )
        assert parameter_metas[name].dtype == parameter.dtype, (
            f"parameter {name} dtype mismatch, {parameter_metas[name].dtype} != {parameter.dtype}"
        )
    return parameters


def _inplace_pin_memory(files: list[str], rank: int | None = None) -> list[MemoryBuffer]:
    def _parse_and_pin_from_safetensors(file_path: str) -> MemoryBuffer:
        """
        safetensors format see https://huggingface.co/docs/safetensors/en/index#format.
        We load the safetensors file as bytes, then parse the header manually to get parameter metas.
        The actual tensor data is in the remaining bytes and is naturally aligned.
        We pin the remaining bytes as the buffer, making pinning faster.
        """

        def _pin(t: torch.Tensor):
            """
            Pin the memory of tensor in-place.
            See: https://github.com/pytorch/pytorch/issues/32167
            """
            cudart = torch.cuda.cudart()
            r = cudart.cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
            assert r == 0, f"pin memory error, error code: {r}"

        # TODO: should only support /dev/shm? but we found files in disk also work?
        size = os.stat(file_path).st_size
        flag_size = 8
        t = torch.from_file(file_path, True, size, dtype=torch.uint8)
        assert t.nbytes > flag_size, (
            f"tensor nbytes {t.nbytes} should be greater than flag_size {flag_size}"
        )
        start_pos = (
            int.from_bytes(t[0:flag_size].numpy().tobytes(), byteorder="little", signed=False)
            + flag_size
        )
        header_tensor = t[flag_size:start_pos]
        header = json.loads(header_tensor.numpy().tobytes())
        if "__metadata__" in header:
            header.pop("__metadata__")

        metas: list[ParameterMeta] = []
        offset = 0
        try:
            for name, meta in sorted(header.items(), key=lambda x: x[1]["data_offsets"]):
                start, end = meta["data_offsets"]
                # safetensors format ensures offsets are aligned
                assert offset == start, f"offset {offset} should be equal to start {start}"
                metas.append(
                    ParameterMeta(
                        name=name,
                        dtype=_getdtype(meta["dtype"]),
                        shape=torch.Size(meta["shape"]),
                        aligned_size=end - start,
                    )
                )
                offset = end
        except Exception as e:
            logger.error(f"fail to parse safetensors header from {file_path}: {e}")
            raise

        buffer = t[start_pos:]
        assert offset == buffer.nbytes, (
            f"offset {offset} should be equal to buffer.nbytes {buffer.nbytes}"
        )
        # Remove the file after successfully loading. This will avoid doubling the memory usage.
        # We assume files in /dev/shm/ are temporary files. So it's safe to remove them after loading.
        os.remove(file_path)
        _pin(buffer)
        logger.info(
            f"[rank{rank}] inplace pin memory for file {file_path} finished, size {buffer.nbytes / 1024 / 1024:.2f}MiB"
        )
        return MemoryBuffer(buffer=buffer, size=buffer.nbytes, metas=metas, manually_pinned=True)

    memory_buffers: list[MemoryBuffer] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        memory_buffers = list(executor.map(_parse_and_pin_from_safetensors, files))
    return memory_buffers


def _normal_pin_memory(
    files: list[str],
    named_tensors: dict[str, torch.Tensor],
    rank: int | None = None,
    shared_pin_memory: list[MemoryBuffer] | None = None,
) -> list[MemoryBuffer]:
    parameters = _load_checkpoint(files)
    if named_tensors:
        parameters.update(named_tensors)
    bucket_size = max(4 << 30, max(_align_size(x.dtype, x.shape) for x in parameters.values()))

    class MemoryBucket(BaseModel):
        size: int
        metas: list[ParameterMeta]

    buckets: list[MemoryBucket] = []
    buckets.append(MemoryBucket(size=0, metas=[]))
    for name, tensor in sorted(parameters.items()):
        size = _align_size(tensor.dtype, tensor.shape)
        if buckets[-1].size + size > bucket_size:
            assert buckets[-1], f"buckets[{len(buckets) - 1}] should not be empty"
            buckets.append(MemoryBucket(size=0, metas=[]))
        buckets[-1].metas.append(
            ParameterMeta(name=name, shape=tensor.shape, dtype=tensor.dtype, aligned_size=size)
        )
        buckets[-1].size += size

    memory_buffers = [
        MemoryBuffer(buffer=torch.empty(0), size=bucket.size, metas=bucket.metas)
        for bucket in buckets
    ]

    def register_pin_memory(
        idx: int, size: int, shared_pin_memory: list[MemoryBuffer] | None = None
    ) -> tuple[int, torch.Tensor]:
        if shared_pin_memory:
            # If shared_pin_memory is provided, reuse the pin memory buffer, do not allocate new one
            # Reusing pin memory only support fixed shape of checkpoints, which is registered the first time
            assert idx < len(shared_pin_memory), (
                f"idx {idx} should be less than shared_pin_memory length {len(shared_pin_memory)}"
            )
            assert shared_pin_memory[idx].size == size, (
                f"shared_pin_memory[{idx}].size {shared_pin_memory[idx].size} should be equal to {size}"
            )
            return idx, shared_pin_memory[idx].buffer
        else:
            buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            return idx, buffer

    def register_tensor(buffer: torch.Tensor, offset: int, tensor: torch.Tensor):
        buffer[offset : offset + tensor.nbytes] = tensor.view(-1).view(dtype=torch.uint8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(
                register_pin_memory,
                idx,
                bucket.size,
                shared_pin_memory,
            )
            for idx, bucket in enumerate(buckets)
        ]
        new_futures = []
        for future in concurrent.futures.as_completed(futures):
            idx, buffer = future.result()
            assert buffer.numel() == buckets[idx].size, (
                f"buffer numel {buffer.numel()} should be equal to bucket size {buckets[idx].size}"
            )
            memory_buffers[idx].buffer = buffer
            logger.info(
                f"[rank{rank}] register pin_memory for bucket {idx + 1}/{len(buckets)} finished, "
                f"size {buffer.numel() / 1024 / 1024:.2f}MiB, start to copy tensors to buffer"
            )
            offset = 0
            for meta in buckets[idx].metas:
                name = meta.name
                tensor = parameters[name]
                size = _align_size(tensor.dtype, tensor.shape)
                assert size == _align_size(meta.dtype, meta.shape), (
                    f"tensor {name} size {size} should be equal to meta size {_align_size(meta.dtype, meta.shape)}"
                )
                new_futures.append(executor.submit(register_tensor, buffer, offset, tensor))
                offset += size
        for future in concurrent.futures.as_completed(new_futures):
            future.result()
        return memory_buffers


def _register_checkpoint(
    *,
    files: list[str],
    named_tensors: dict[str, torch.Tensor],
    rank: int | None = None,
    shared_pin_memory: list[MemoryBuffer] | None = None,
    inplace_pin: bool = False,
) -> list[MemoryBuffer]:
    logger.info(
        f"[rank{rank}] start to register checkpoint with {len(files)} files and {len(named_tensors)} named_tensors"
    )
    if not files and not named_tensors:
        return []
    memory_buffers: list[MemoryBuffer] = []
    if inplace_pin:
        logger.info(f"[rank{rank}] allow inplace pin memory for /dev/shm/ safetensors files")
        files_to_inplace_pin = [
            file
            for file in files
            if file.startswith("/dev/shm/") and file.endswith(".safetensors")  # noqa: S108
        ]
        files_to_normal_pin = [file for file in files if file not in files_to_inplace_pin]
    else:
        files_to_normal_pin = files
        files_to_inplace_pin = []
    if files_to_normal_pin or named_tensors:
        memory_buffers.extend(
            _normal_pin_memory(
                files=files_to_normal_pin,
                named_tensors=named_tensors,
                rank=rank,
                shared_pin_memory=shared_pin_memory,
            )
        )
    if files_to_inplace_pin:
        memory_buffers.extend(_inplace_pin_memory(files_to_inplace_pin, rank=rank))
    return memory_buffers


def request_inference_to_update(
    url: str,
    socket_paths: dict[str, str],
    timeout: float = 300.0,
    uds: str | None = None,
):
    """Send an inference update request to inference server via HTTP or Unix socket.

    Args:
        url (str): The HTTP URL or request path (e.g., "http://localhost:19730/inference") to send the request to.
        socket_paths (dict[str, str]): A dictionary containing device uuid and IPC socket paths for updating weights.
        timeout (float, optional): Request timeout in seconds. Defaults to 300.0.
        uds (str, optional): Path to a Unix domain socket. If provided, the request
            will be sent via the Unix socket instead of HTTP. Defaults to None.

    Raises:
        httpx.HTTPStatusError: If the response contains an HTTP error status.
        httpx.RequestError: If there was an issue while making the request.
    """
    resp = httpx.Client(transport=httpx.HTTPTransport(uds=uds)).post(
        url,
        json={
            "method": "update_weights_from_ipc",
            "args": [socket_paths],
            "timeout": timeout,
        },
        timeout=timeout,
    )
    resp.raise_for_status()


def _gen_h2d_buckets(
    global_metas: dict[int, MemoryBufferMetaList],
    bucket_size: int,
    local_topo: dict[str, set[int]],
    remote_topo: dict[str, set[int]],
    ranks: list[int] | None = None,
) -> list[tuple[int, int, H2DBucket]]:
    buckets: list[tuple[int, H2DBucket]] = []

    for owner_rank, items in global_metas.items():
        buckets.append((owner_rank, H2DBucket(size=0, ranges=[], items=[])))
        for idx, metas in enumerate(items.memory_buffer_metas_list):
            start_offset, offset = 0, 0
            for meta in metas.metas:
                s = meta.aligned_size
                if buckets[-1][1].size + s > bucket_size:
                    if offset - start_offset > 0:
                        buckets[-1][1].ranges.append(
                            BucketRange(idx, start_offset, offset - start_offset)
                        )
                    start_offset = offset
                    buckets.append((owner_rank, H2DBucket(size=0, ranges=[], items=[])))
                offset += s
                buckets[-1][1].size += s
                buckets[-1][1].items.append(meta)
            buckets[-1][1].ranges.append(BucketRange(idx, start_offset, offset - start_offset))
        assert buckets[-1][1].size > 0, (
            f"buckets[-1][1].size {buckets[-1][1].size} should be greater than 0"
        )
    ranks_set = set(ranks) if ranks else set()
    actual_local_topo = (
        {k: v & ranks_set for k, v in local_topo.items() if v & ranks_set} if ranks else local_topo
    )
    # if ranks is empty, assign the owner_rank as receiver_rank, this is used for colocate architecture
    if not ranks:
        return [(owner_rank, owner_rank, bucket) for owner_rank, bucket in buckets]
    else:
        return _assign_receiver_ranks(buckets, actual_local_topo, remote_topo)


def _assign_receiver_ranks(
    buckets: list[tuple[int, "T"]],
    local_topo: dict[str, set[int]],
    remote_topo: dict[str, set[int]],
) -> list[tuple[int, int, "T"]]:
    """
    (owner_rank, bucket) -> (receiver_rank, owner_rank, bucket)

    Assign receiver ranks to buckets. If ranks is empty, assign the owner_rank as receiver_rank.
    GPU-rdma_device topology will be considered to make full use of the bandwidth.
    """
    if not buckets:
        logger.warning("bucket list is empty, no need to assign receiver ranks")
        return []
    rank_to_rdma_device = {
        rank: rdma_device for rdma_device, ranks in remote_topo.items() for rank in ranks
    }

    # group buckets by owner RDMA devices
    buckets_by_rdma_device = defaultdict(list)
    for owner_rank, bucket in buckets:
        owner_rdma_device = rank_to_rdma_device[owner_rank]
        buckets_by_rdma_device[owner_rdma_device].append((owner_rank, bucket))

    buckets_matrix = list(buckets_by_rdma_device.values())
    assert buckets_matrix, "buckets_matrix should not be empty"

    # Select receiver ranks. We use the minimum rank in each local RDMA device group as receiver rank
    num_receivers = min(len(local_topo), len(buckets_by_rdma_device))
    receiver_list = [min(ranks) for ranks in list(local_topo.values())[:num_receivers]]

    flattened_buckets = [
        buckets_matrix[row][col]
        for col in range(
            max(len(matrix_row) for matrix_row in buckets_matrix) if buckets_matrix else 0
        )
        for row in range(len(buckets_matrix))
        if col < len(buckets_matrix[row])
    ]

    buckets_with_receiver = []
    assigned_cnt = 0
    while assigned_cnt < len(flattened_buckets):
        occupied_devices = set()
        for receiver_rank in receiver_list:
            if assigned_cnt >= len(flattened_buckets):
                break
            owner_rank, bucket = flattened_buckets[assigned_cnt]
            rdma_device = rank_to_rdma_device[owner_rank]
            if rdma_device in occupied_devices:
                break
            buckets_with_receiver.append((receiver_rank, owner_rank, bucket))
            occupied_devices.add(rdma_device)
            assigned_cnt += 1

    return buckets_with_receiver


def _get_master_port(master_port: int | None = None) -> int:
    if master_port is None:
        # HACK: use MASTER_PORT + 1 as master_port, avoid conflict with torchrun's rendezvous port
        # TODO: check whether master_port is available or use a more elegant way
        master_port = int(os.getenv("MASTER_PORT")) + 1
    return master_port


class P2PStore:
    def __init__(self, device_manager: DeviceManager):
        from mooncake.engine import TransferEngine

        self.rank = int(os.getenv("RANK"))
        gpu_count = device_manager.device_module.device_count()
        local_rank = self.rank % gpu_count
        device_type = device_manager.device_type
        if device_type == "npu" and os.getenv("PS_P2P_STORE_RDMA_DEVICES") is None:
            self.device = ""
        else:
            self.device = _get_my_rdma_device(local_rank, gpu_count, _get_rdma_devices())
        self.ip = get_ip()

        # we will start at most 8 ps processes, so we use 8 retries to avoid port conflicts in extreme cases
        retry_count = 8
        for i in range(retry_count):
            self.engine = TransferEngine()
            ret = self.engine.initialize(
                self.ip,
                "P2PHANDSHAKE",
                "ascend_direct" if device_type == "npu" else "rdma",
                self.device,
            )
            if ret == 0:
                break
            # sleep 0.5 ~ 2.0s, to avoid port conflicts when two processes retry at the same time
            sleep_ms = random.randint(500, 2000)
            logger.warning(
                f"[rank{self.rank}] fail to initialize transfer engine, ret {ret}, retry {i + 1}/{retry_count} in {sleep_ms}ms"
            )
            time.sleep(sleep_ms / 1000)
        else:
            raise RuntimeError(f"[rank{self.rank}] fail to initialize transfer engine")
        self.port = self.engine.get_rpc_port()
        self.named_tensors: dict[str, torch.Tensor] = {}
        logger.info(
            f"[rank{self.rank}] p2p store initialized, addr is {self.addr}, rdma device is {self.device}"
        )

    @property
    def addr(self) -> str:
        return f"{self.ip}:{self.port}"

    def register_named_tensors(self, named_tensors: dict[str, torch.Tensor]):
        buffer_addresses = [tensor.data_ptr() for tensor in named_tensors.values()]
        capacities = [tensor.nbytes for tensor in named_tensors.values()]
        self.named_tensors.update(named_tensors)
        for i, name in enumerate(named_tensors.keys()):
            logger.info(
                f"[rank{self.rank}] p2p store register tensor {name} with addr {hex(buffer_addresses[i])} and capacity {capacities[i]}"
            )
        assert self.engine.batch_register_memory(buffer_addresses, capacities) == 0

    def unregister_named_tensors(self, names: list[str]) -> int:
        buffer_addresses = [self.named_tensors[name].data_ptr() for name in names]
        assert self.engine.batch_unregister_memory(buffer_addresses) == 0
        num_unregistered = 0
        for i, name in enumerate(names):
            del self.named_tensors[name]
            logger.info(
                f"[rank{self.rank}] p2p store unregister tensor {name} with addr {hex(buffer_addresses[i])}"
            )
            num_unregistered += 1
        return num_unregistered

    def batch_transfer_sync_read(
        self, target_hostname: str, buf_ptrs: list[int], remote_ptrs: list[int], lens: list[int]
    ):
        assert (
            self.engine.batch_transfer_sync_read(target_hostname, buf_ptrs, remote_ptrs, lens) == 0
        )


class ParameterServer:
    shared_memory_pool_name = "__shared_memory_pool__"

    def __init__(
        self,
        *,
        rank: int | None = None,
        world_size: int | None = None,
        auto_pg: bool = True,
        gpu_count: int | None = None,
        mem_fraction: float | None = None,
    ):
        """
        Initialize the parameter server. env RANK, WORLD_SIZE and MASTER_ADDR must be set.

        Args:
            auto_pg: Whether to automatically initialize the process group.
                Notice that if auto_pg is True, will destroy the process group after update. It is recommended to set auto_pg to True!
            mem_fraction: The proportion (as a fraction) of the current free device memory for allocation.
        """
        self._rank = rank or int(os.environ.get("RANK", None))
        self._world_size = world_size or int(os.environ.get("WORLD_SIZE", None))
        self.device_manager = DeviceManager()
        self._gpu_count = gpu_count or self.device_manager.device_module.device_count()
        self._local_rank = self._rank % self._gpu_count
        self._auto_pg = auto_pg
        self._all_hosts = []
        self._global_device_uuids: list[str] = []
        self._local_rdma_devices: dict[str, set[int]] = defaultdict(set)
        self._remote_rdma_devices: dict[str, set[int]] = defaultdict(set)
        self._mem_fraction = mem_fraction or 0.9

        assert self._rank is not None and self._rank >= 0, self._rank
        assert self._world_size and self._world_size > 0, self._world_size
        assert (
            self._gpu_count is not None
            and self._gpu_count > 0
            and self._gpu_count <= self.device_manager.device_module.device_count()
        ), self._gpu_count
        assert (
            self._mem_fraction is not None and self._mem_fraction > 0 and self._mem_fraction <= 1
        ), self._mem_fraction

        self._zmq_ctx = zmq.Context()
        self._zmq_addr_counter = 0

        # stores the name of the checkpoint currently using the shared memory pool, or empty string if none
        self._current_shared_memory_pool_user: str = ""
        self._memory_pool: dict[str, list[MemoryBuffer]] = {}
        self._memory_pool[self.shared_memory_pool_name] = []
        # dict key is owner_rank, value is a bucket metas list in owner_rank
        self._current_global_parameter_metas: dict[int, MemoryBufferMetaList] = {}
        # NPU transfer engine initialization requires prior set_device.
        device_index = self._local_rank
        self.device_manager.device_module.set_device(device_index)
        try:
            self._p2p_store = P2PStore(self.device_manager)
        except ImportError as e:
            logger.warning(f"[rank{self._rank}] fail to initialize p2p store due to {e}")
            self._p2p_store = None

        self._device_uuid = _get_physical_gpu_id(self.device_manager, device_index)
        self._rdma_device = None if self._p2p_store is None else self._p2p_store.device

    def _get_memory_pool(self, checkpoint_name: str) -> list[MemoryBuffer]:
        if checkpoint_name == self._current_shared_memory_pool_user:
            assert self._memory_pool[self.shared_memory_pool_name], (
                f"shared memory pool is not initialized, but checkpoint {checkpoint_name} is using it"
            )
            return self._memory_pool[self.shared_memory_pool_name]
        elif checkpoint_name in self._memory_pool:
            return self._memory_pool[checkpoint_name]
        else:
            raise RuntimeError(f"checkpoint {checkpoint_name} is not registered")

    def _logger_rank0(self, msg: str):
        if self._local_rank == 0:
            logger.info(msg)

    def get_metas(self) -> dict[int, MemoryBufferMetaList]:
        return self._current_global_parameter_metas

    def load_metas(self, metas: dict[int, MemoryBufferMetaList]):
        self._current_global_parameter_metas = metas
        self._remote_rdma_devices = defaultdict(set)
        for i, meta in self._current_global_parameter_metas.items():
            assert meta.rdma_device is not None, "meta.rdma_device should not be None"
            assert meta.p2p_store_addr is not None, "meta.p2p_store_addr should not be None"
            self._remote_rdma_devices[
                meta.rdma_device + "@" + meta.p2p_store_addr.split(":")[0]
            ].add(i)

    def register_checkpoint(
        self,
        checkpoint_name: str,
        *,
        files: list[str] | None = None,
        named_tensors: dict[str, torch.Tensor] | None = None,
        use_shared_memory_pool: bool = False,
        use_inplace_pin_memory: bool = True,
    ) -> None:
        """
        Register a checkpoint to the parameter server. Both files and named_tensors will be registered together.
        Warning: if `use_inplace_pin_memory` is True, .safetensors files in /dev/shm/ will be pinned in-place, and the files will be REMOVED after pinning.
        Please make sure to copy the files to disks if you need to keep them. NPU does not support inplace pin memory.

        Args:
            checkpoint_name: The name of the checkpoint.
            files: The safetensors files to register.
            named_tensors: The named tensors to register.
            use_shared_memory_pool: If True, uses a reusable shared pin memory pool instead of allocating new memory.
                Only one checkpoint can use the shared pool at a time. The pool's shape is fixed on first use and
                cannot accommodate checkpoints with different memory requirements.
                To free the actual memory of the shared pool or to modify its shape,
                please unregister the current user of the shared memory pool using `unregister_checkpoint` with `force=True`.
            use_inplace_pin_memory: If True (default), allows inplace pin memory for /dev/shm/ safetensors files.
                This option is ignored when ``use_shared_memory_pool`` is True.
        """
        if self.device_manager.device_type != "cuda" and use_inplace_pin_memory:
            logger.warning(
                f"[rank{self._rank}] Only cuda devices support in-place pin memory, set use_inplace_pin_memory to False"
            )
            use_inplace_pin_memory = False
        try:
            if use_shared_memory_pool:
                logger.info(
                    f"[rank{self._rank}] checkpoint {checkpoint_name} use shared memory pool"
                )
                assert self._current_shared_memory_pool_user == "", (
                    f"cannot register checkpoint {checkpoint_name} to shared memory pool, "
                    f"since checkpoint {self._current_shared_memory_pool_user} is already using shared memory pool. "
                    f"This registration may cause unexpected conflicts."
                )
                # Since we set the uninitialized shared memory pool to empty list,
                # we can check whether this is the first time to use shared memory pool
                _is_first_time = not self._memory_pool[self.shared_memory_pool_name]
                self._memory_pool[self.shared_memory_pool_name] = _register_checkpoint(
                    files=files or [],
                    named_tensors=named_tensors or {},
                    rank=self._rank,
                    shared_pin_memory=self._memory_pool[self.shared_memory_pool_name],
                    inplace_pin=False,  # inplace pin memory is not compatible with shared memory pool
                )
                self._current_shared_memory_pool_user = checkpoint_name
                if self._p2p_store is not None and _is_first_time:
                    self._register_parameters_to_p2p_store(checkpoint_name)
            else:
                assert checkpoint_name not in self._memory_pool, (
                    f"checkpoint {checkpoint_name} already registered"
                )
                self._memory_pool[checkpoint_name] = _register_checkpoint(
                    files=files or [],
                    named_tensors=named_tensors or {},
                    rank=self._rank,
                    inplace_pin=use_inplace_pin_memory,
                )
                if self._p2p_store is not None:
                    self._register_parameters_to_p2p_store(checkpoint_name)
        except Exception:
            logger.exception(
                f"[rank{self._rank}] fail to register checkpoint {checkpoint_name} with files {files}"
            )
            if self._p2p_store is not None and not use_shared_memory_pool:
                self._unregister_parameters_from_p2p_store(checkpoint_name)
            self.unregister_checkpoint(checkpoint_name)
            raise

    def unregister_checkpoint(self, checkpoint_name: str, force: bool = False) -> None:
        """
        Unregister a checkpoint from the parameter server. This function will also unregister the checkpoint
        from p2p store if p2p store is initialized.
        Args:
            checkpoint_name: The name of the checkpoint.
            force: This flag is designed for shared memory pool user. If True, the memory for shared memory pool itself will be freed.
                    If False, only the checkpoint name will be unregistered, and the shared memory pool will be kept for future use.
        """
        if (
            checkpoint_name not in self._memory_pool
            and checkpoint_name != self._current_shared_memory_pool_user
        ):
            logger.warning(
                f"[rank{self._rank}] unregister checkpoint name {checkpoint_name} not found"
            )
            return

        if checkpoint_name == self._current_shared_memory_pool_user and not force:
            self._current_shared_memory_pool_user = ""
            return

        if self._p2p_store is not None:
            num_unregistered = self._unregister_parameters_from_p2p_store(checkpoint_name)
            logger.info(
                f"[rank{self._rank}] unregister {num_unregistered} parameters from p2p store for checkpoint {checkpoint_name}"
            )

        if checkpoint_name == self._current_shared_memory_pool_user:
            self._current_shared_memory_pool_user = ""
            del self._memory_pool[self.shared_memory_pool_name]
            self._memory_pool[self.shared_memory_pool_name] = []
        else:

            def _unpin(t: torch.Tensor):
                """
                Un-pin the pinned memory.
                """
                p_flags = ctypes.c_uint()
                try:
                    libc = ctypes.CDLL(None)  # get all symbols from the current process
                    cuda_host_get_flags = libc.cudaHostGetFlags
                    cuda_host_get_flags.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
                    cuda_host_get_flags.restype = ctypes.c_int
                except AttributeError:
                    logger.error("cudaHostGetFlags not found in libc, cannot unpin memory manually")
                    raise
                r = cuda_host_get_flags(ctypes.byref(p_flags), ctypes.c_void_p(t.data_ptr()))
                assert r == 0, f"get pin flags error, error code: {r}"
                # p_flags value meaning from cuda/include/driver_types.h
                # cudaHostRegisterDefault             0x00  /**< Default host memory registration flag */
                # cudaHostRegisterPortable            0x01  /**< Pinned memory accessible by all CUDA contexts */
                # cudaHostRegisterMapped              0x02  /**< Map registered memory into device space */
                # cudaHostRegisterIoMemory            0x04  /**< Memory-mapped I/O space */
                # cudaHostRegisterReadOnly            0x08  /**< Memory-mapped read-only */
                assert p_flags.value == 0x02, (
                    f"pin memory flag error, expected: 0x02 (cudaHostRegisterMapped), got flag: {p_flags.value}"
                )
                cudart = torch.cuda.cudart()
                r = cudart.cudaHostUnregister(t.data_ptr())
                assert r == 0, f"unpin memory error, error code: {r}"

            # if the checkpoint is pinned by cudaHostRegister manually, we need to unpin it manually
            try:
                for memory_buffer in self._memory_pool.get(checkpoint_name, []):
                    if memory_buffer.manually_pinned:
                        _unpin(memory_buffer.buffer)
            except Exception as e:
                logger.error(
                    f"[rank{self._rank}] fail to unpin memory for checkpoint {checkpoint_name}: {e}"
                )
                raise
            # we won't delete the memory pool if unpinning fails.
            del self._memory_pool[checkpoint_name]
        # see https://github.com/pytorch/pytorch/blob/31d5c675394705f8a6bc767f80ae14bf4f01246b/torch/csrc/cuda/Module.cpp#L2018
        # this works by using torch>=2.5.0
        torch._C._host_emptyCache()

    def gather_metas(self, checkpoint_name: str):
        """
        Gather the parameter metas from all ranks. This will gather memory_buffer, and other metadatas.
        This function should be called before update and init a new value to `self._current_global_parameter_metas`,
        which can be exported by using `self.get_metas` function.
        """
        if self._auto_pg and not dist.is_initialized():
            self.init_process_group()
        assert dist.is_initialized(), "process group is not initialized"
        metas_lst: list[DataToGather | None] = [None for _ in range(self._world_size)]  # type: ignore
        try:
            memory_pool = self._get_memory_pool(checkpoint_name)
        except RuntimeError:
            memory_pool = []
        metas = DataToGather(
            memory_buffer_metas_list=[
                MemoryBufferMetas(
                    metas=x.metas,
                    ptr=x.buffer.data_ptr(),
                    size=x.size,
                )
                for x in memory_pool
            ],
            p2p_store_addr=None if self._p2p_store is None else self._p2p_store.addr,
            host_ip=get_ip(),
            device_uuid=self._device_uuid,
            rdma_device=self._rdma_device or "",
        )

        dist.all_gather_object(metas_lst, metas)

        self._current_global_parameter_metas = {}

        num_parameters = 0
        all_hosts: list[str] = []
        global_device_uuids: list[str] = []
        for i, metas_buckets in enumerate(metas_lst):
            assert metas_buckets is not None, f"metas_buckets {i} should not be None"
            if i % self._gpu_count == 0 and not self._all_hosts:
                all_hosts.append(metas_buckets.host_ip)
            if not self._global_device_uuids:
                global_device_uuids.append(metas_buckets.device_uuid)
            if metas_buckets.memory_buffer_metas_list:
                self._current_global_parameter_metas[i] = MemoryBufferMetaList(
                    memory_buffer_metas_list=metas_buckets.memory_buffer_metas_list,
                    p2p_store_addr=metas_buckets.p2p_store_addr,
                    rdma_device=metas_buckets.rdma_device,
                )
                num_parameters += sum(len(x.metas) for x in metas_buckets.memory_buffer_metas_list)
            self._local_rdma_devices[
                metas_buckets.rdma_device + "@" + metas_buckets.p2p_store_addr.split(":")[0]
                if metas_buckets.p2p_store_addr
                else metas_buckets.host_ip
            ].add(i)
        if not self._all_hosts:
            self._all_hosts = all_hosts
        if not self._global_device_uuids:
            self._global_device_uuids = global_device_uuids
        # Sender node and Receiver node have the same GPU-rdma_device topology is considered as default.
        # Rewrite the sender's topology (_remote_rdma_devices) by calling load_metas.
        self._remote_rdma_devices = self._local_rdma_devices.copy()
        logger.info(
            f"[rank{self._rank}] gather parameter metas finished, num_parameters: {num_parameters}"
        )

    def init_process_group(
        self,
        *,
        master_addr: str | None = None,
        master_port: int | None = None,
        timeout: timedelta = timedelta(minutes=10),
    ):
        """
        Initialize the process group for the ranks. This global group can be easily destroyed by calling dist.destroy_process_group.

        Args:
            master_port: The specified port of the master node. If not set, will use _get_master_port to get the port.
            timeout: The timeout of the process group.
        """
        master_addr = master_addr or os.getenv("MASTER_ADDR")
        assert master_addr, "master_addr is required"
        store = dist.TCPStore(
            master_addr,
            _get_master_port(master_port),
            self._world_size,
            timeout=timeout,
            is_master=self._rank == 0,
        )
        dist.init_process_group(
            backend=self.device_manager.backend,
            world_size=self._world_size,
            rank=self._rank,
            timeout=timeout,
            store=store,
        )
        logger.info(f"[rank{self._rank}] init process group successfully.")

    def store_based_barrier(
        self, store: dist.TCPStore, timeout: timedelta = timedelta(minutes=5)
    ) -> None:
        """
        Perform a store-based barrier synchronization across all ranks.

        This barrier uses a TCP store directly rather than a process group,
        allowing all ranks to synchronize regardless of which process group
        they belong to.

        Args:
            store: The TCPStore instance to use for synchronization.
        """
        dist.distributed_c10d._store_based_barrier(
            rank=self._rank,
            store=store,
            group_name="parameter_server_barrier",
            rendezvous_count=self._world_size,
            timeout=timeout,
        )

    def update(
        self,
        checkpoint_name: str,
        req_func: Callable[[list[tuple[str, str]]], None],
        *,
        timeout: timedelta = timedelta(minutes=10),
        ranks: list[int] | None = None,
        master_addr: str | None = None,
        master_port: int | None = None,
    ) -> None:
        """
        Update the checkpoint to inference engine. This function should be called after gather_metas.
        Warning: if _auto_pg is False when initializing ParameterServer, please make sure ALL ranks in the WORLD_SIZE call `update` function,
        otherwise, it will hang.

        Args:
            checkpoint_name: The name of the checkpoint.
            req_func: The function to request the inference of inference engine.
            ranks: The ranks to update. If not set, will use fully broadcast to update to all ranks,
                which is the fastest way to update weights, especially in colocated architecture.
                If set, will use p2p to update to the ranks, this is flexible to update to a group of ranks,
                which is useful in disaggregated architecture.
            master_addr: The master address for process group initialization. If not set, will use env MASTER_ADDR.
            master_port: The master port for process group initialization. If not set, will use _get_master_port to get the port, which will use MASTER_PORT+1.
            timeout: The timeout of the barrier operation.
        """
        assert req_func is not None, "req_func is required"
        ranks_group = None
        try:
            master_addr = os.getenv("MASTER_ADDR") or master_addr
            assert master_addr, "master_addr is required"
            if self._auto_pg:
                if not dist.is_initialized():
                    self.init_process_group(
                        timeout=timeout, master_addr=master_addr, master_port=master_port
                    )
                manager_store = dist.distributed_c10d._get_default_store()
            else:
                # HACK: MASTER_PORT+2 for barrier store if master_port is not provided, _get_master_port() returns MASTER_PORT+1
                # If master_port is provided, use master_port+1 for barrier store
                manager_store = dist.TCPStore(
                    master_addr,
                    _get_master_port(master_port) + 1,
                    self._world_size,
                    timeout=timeout,
                    is_master=self._rank == 0,
                )
            # if ranks is None or [], it will use fully broadcast to update to all ranks
            ranks_group = dist.new_group(ranks) if ranks else None
            self._update_per_bucket(checkpoint_name, req_func, ranks_group, ranks)
            self.store_based_barrier(manager_store)
        except Exception as e:
            logger.exception(
                f"[rank{self._rank}] update checkpoint {checkpoint_name} with ranks {ranks} error {e}"
            )
            raise
        finally:
            if ranks_group:
                dist.destroy_process_group(ranks_group)
            if self._auto_pg and dist.is_initialized():
                dist.destroy_process_group()
            self.device_manager.device_module.empty_cache()
            logger.info(
                f"[rank{self._rank}] update checkpoint {checkpoint_name} with ranks {ranks} done. "
                f"Current device allocated {self.device_manager.device_module.memory_allocated() / 1024 / 1024} MB, "
                f"reserved {self.device_manager.device_module.memory_reserved() / 1024 / 1024} MB."
            )

    def _bind_zmq_socket(self) -> tuple[zmq.Socket, list[tuple[str, str]]]:
        def zmq_handle(device_uuid: str) -> str:
            return f"ipc://@checkpoint-engine-{device_uuid}-{self._zmq_addr_counter}.sock"

        socket_paths = [(uid, zmq_handle(uid)) for uid in self._global_device_uuids]
        socket = self._zmq_ctx.socket(zmq.REQ)
        socket.bind(zmq_handle(self._device_uuid))
        self._zmq_addr_counter += 1
        return socket, socket_paths

    def _detect_bucket_size(
        self, ranks_group: dist.ProcessGroup | None, *, disable_h2d_buffer: bool = False
    ) -> tuple[int, bool]:
        GiB = 1 << 30  # noqa: N806
        # auto detect bucket size
        tensor = torch.tensor(
            [
                # proportion of current device free memory bytes
                int(
                    float(self.device_manager.device_module.mem_get_info()[0]) * self._mem_fraction
                ),
                # we use negative value to reuse allreduce min operation
                # for getting the max value of zmq_addr_counter in all ranks
                -self._zmq_addr_counter,
            ],
            dtype=torch.int64,
            device=self.device_manager.device_type,
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=ranks_group)
        tensor = tensor.cpu()
        free_bytes, self._zmq_addr_counter = tensor[0].item(), -tensor[1].item()
        max_tensor_bytes = 0
        for items in self._current_global_parameter_metas.values():
            for metas_list in items.memory_buffer_metas_list:
                for meta in metas_list.metas:
                    max_tensor_bytes = max(max_tensor_bytes, meta.aligned_size)
        free_bytes_divided_3 = free_bytes // (3 * _ALIGN_SIZE) * _ALIGN_SIZE
        if max_tensor_bytes <= free_bytes_divided_3 and not disable_h2d_buffer:
            self._logger_rank0(f"[rank{self._rank}] use h2d buffer")
            # using h2d_buffer can make all ranks' h2d parallel execution
            # the cost is that we need to allocate extra h2d_buffer's GPU memory
            free_bytes = free_bytes_divided_3
        else:
            # if the memory is not enough, it will fallback to disable_h2d_buffer mode,
            # at this time, the bandwidth will be limited by the h2d of a single machine,
            # but we can save GPU memory
            self._logger_rank0(
                f"[rank{self._rank}] disable h2d buffer when max_tensor_bytes {max_tensor_bytes} is larger than free_bytes {free_bytes} // 3"
            )
            free_bytes = free_bytes // (2 * _ALIGN_SIZE) * _ALIGN_SIZE
            assert max_tensor_bytes <= free_bytes, (
                f"max_tensor_bytes {max_tensor_bytes} should be less than free_bytes {free_bytes}"
            )
            disable_h2d_buffer = True
        max_bytes = int(os.getenv("PS_MAX_BUCKET_SIZE_GB", 8)) * GiB
        bucket_size = min(max(max_bytes, max_tensor_bytes), free_bytes)
        logger.info(f"[rank{self._rank}] auto detect bucket size {bucket_size / GiB:.2f} GiB")
        return bucket_size, disable_h2d_buffer

    def _copy_to_buffer(
        self,
        checkpoint_name: str,
        bucket: H2DBucket,
        buffer: torch.Tensor,
        owner_rank: int | None = None,
    ):
        offset = 0
        if owner_rank is not None:
            buf_ptrs, remote_ptrs, lens = [], [], []
            ptr_base = buffer.data_ptr()
            target_addr, ptrs = self._get_addr_ptrs(owner_rank)
        for b in bucket.ranges:
            assert offset + b.size <= bucket.size, (
                f"offset {offset} + size {b.size} > bucket_size {bucket.size}"
            )
            if owner_rank is not None:
                buf_ptrs.append(ptr_base + offset)
                remote_ptrs.append(ptrs[b.idx][0] + b.offset)
                lens.append(b.size)
            else:
                pool = self._get_memory_pool(checkpoint_name)[b.idx]
                buffer[offset : offset + b.size].data.copy_(
                    pool.buffer[b.offset : b.offset + b.size],
                    non_blocking=True,
                )
            offset += b.size
        assert offset == bucket.size, f"offset {offset} != bucket_size {bucket.size}"
        if owner_rank is not None:
            self._p2p_store.batch_transfer_sync_read(target_addr, buf_ptrs, remote_ptrs, lens)
        self.device_manager.device_module.synchronize()

    def _get_addr_ptrs(self, owner_rank: int) -> tuple[str, list[tuple[int, int]]]:
        addr = self._current_global_parameter_metas[owner_rank].p2p_store_addr
        metas_list = self._current_global_parameter_metas[owner_rank].memory_buffer_metas_list
        return addr, [(metas.ptr, metas.size) for metas in metas_list]

    def _register_parameters_to_p2p_store(self, checkpoint_name: str):
        assert self._p2p_store is not None, "p2p store is not initialized"
        pool = self._get_memory_pool(checkpoint_name)
        if len(pool) == 0:
            return
        named_tensors, tensor_ptrs = {}, []
        register_name = (
            checkpoint_name
            if checkpoint_name != self._current_shared_memory_pool_user
            else self.shared_memory_pool_name
        )
        for idx, memory_buffer in enumerate(pool):
            named_tensors[f"memory_pool_{register_name}_{idx}"] = memory_buffer.buffer
            tensor_ptrs.append((memory_buffer.buffer.data_ptr(), memory_buffer.size))
        self._p2p_store.register_named_tensors(named_tensors)

    def _unregister_parameters_from_p2p_store(self, checkpoint_name: str) -> int:
        assert self._p2p_store is not None, "p2p store is not initialized"
        pool = self._get_memory_pool(checkpoint_name)
        if len(pool) == 0:
            return 0
        unregister_name = (
            checkpoint_name
            if checkpoint_name != self._current_shared_memory_pool_user
            else self.shared_memory_pool_name
        )
        return self._p2p_store.unregister_named_tensors(
            [f"memory_pool_{unregister_name}_{idx}" for idx, _ in enumerate(pool)]
        )

    def _update_per_bucket(
        self,
        checkpoint_name: str,
        req_func: Callable[[list[tuple[str, str]]], None],
        ranks_group: dist.ProcessGroup | None,
        ranks: list[int] | None = None,
    ):
        assert len(self._current_global_parameter_metas) != 0, "parameter metas is empty"
        assert dist.is_initialized(), "process group is not initialized"

        # if both ranks is None or [], it will use fully broadcast to update to all ranks
        if not ranks:
            logger.info(f"[rank{self._rank}] update checkpoint {checkpoint_name}")
        # if ranks is set, it will use p2p to update to the ranks
        else:
            assert self._p2p_store is not None, "p2p store is not initialized"
            assert ranks, "ranks should be set"

            need_update = self._rank in ranks
            logger.info(
                f"[rank{self._rank}] update checkpoint {checkpoint_name} p2p, {need_update=} with {ranks=}, "
                f"gpu_count {self._gpu_count}, world_size {self._world_size}"
            )

            if not need_update:
                return
            # first execute a barrier to avoid subsequent device oom
            dist.barrier(group=ranks_group)

        bucket_size, disable_h2d_buffer = self._detect_bucket_size(ranks_group)
        buckets = _gen_h2d_buckets(
            self._current_global_parameter_metas,
            bucket_size,
            self._local_rdma_devices,
            self._remote_rdma_devices,
            ranks,
        )

        h2d_buffer: torch.Tensor | None = (
            None
            if disable_h2d_buffer
            else torch.empty(bucket_size, dtype=torch.uint8, device=self.device_manager.device_type)
        )
        # p2p store need to register h2d_buffer to let other ranks read
        if ranks:
            h2d_buffer_name = "__h2d_buffer__"
            if h2d_buffer is not None and self._p2p_store is not None:
                self._p2p_store.register_named_tensors({h2d_buffer_name: h2d_buffer})
        receiver_rank_buckets: list[tuple[int, H2DBucket]] = []
        for receiver_rank, owner_rank, bucket in buckets:
            if receiver_rank != self._rank:
                continue
            receiver_rank_buckets.append((owner_rank, bucket))

        buffer = torch.empty(
            bucket_size * 2, dtype=torch.uint8, device=self.device_manager.device_type
        )
        handle = reduce_tensor(buffer)

        buckets_by_receiver_rank: dict[int, list[H2DBucket]] = defaultdict(list)
        max_len = 0
        for receiver_rank, _, bucket in buckets:
            buckets_by_receiver_rank[receiver_rank].append(bucket)
            if len(buckets_by_receiver_rank[receiver_rank]) > max_len:
                max_len = len(buckets_by_receiver_rank[receiver_rank])

        socket, socket_paths = self._bind_zmq_socket()
        req_thread = threading.Thread(
            target=req_func,
            args=(socket_paths,),
        )
        req_thread.start()
        socket.send_pyobj(handle)

        gidx = 0
        ret_code = torch.zeros((), device=self.device_manager.device_type, dtype=torch.int64)
        try:
            for i in range(max_len):
                if i < len(receiver_rank_buckets) and not disable_h2d_buffer:
                    self._copy_to_buffer(
                        checkpoint_name,
                        receiver_rank_buckets[i][1],
                        h2d_buffer,
                        receiver_rank_buckets[i][0] if ranks else None,
                    )
                for receiver_rank, _buckets in buckets_by_receiver_rank.items():
                    if i >= len(_buckets):
                        continue
                    bucket = _buckets[i]
                    alloc, reserved = (
                        self.device_manager.device_module.memory_allocated() / 1024 / 1024,
                        self.device_manager.device_module.memory_reserved() / 1024 / 1024,
                    )
                    self._logger_rank0(
                        f"[rank{self._rank}] begin to update bucket {gidx + 1}/{len(buckets)} receiver_rank {receiver_rank} in checkpoint {checkpoint_name}, bucket_size: {bucket.size / 1024 / 1024:.2f}MiB, length: {len(bucket.items)}. "
                        f"Current device allocated {alloc:.2f} MB, "
                        f"reserved {reserved:.2f} MB."
                    )
                    start = gidx % 2 * bucket_size
                    buffer_b: torch.Tensor = buffer[start : start + bucket.size]
                    if receiver_rank == self._rank:
                        if disable_h2d_buffer:
                            self._copy_to_buffer(checkpoint_name, bucket, buffer_b)
                        else:
                            buffer_b.data.copy_(h2d_buffer[: bucket.size])
                    dist.broadcast(buffer_b, src=receiver_rank, group=ranks_group)
                    resp = socket.recv()
                    if resp != b"":
                        msg = resp.decode("utf-8")
                        logger.error(
                            f"[rank{self._rank}] receive error response from rank {receiver_rank} for bucket {gidx} in checkpoint {checkpoint_name}: {msg}"
                        )
                        ret_code.fill_(1)
                    dist.all_reduce(ret_code, op=dist.ReduceOp.SUM, group=ranks_group)
                    self.device_manager.device_module.synchronize()
                    if ret_code.item() != 0:
                        # quit early if any rank failed
                        socket.send_pyobj(RuntimeError("Some workers failed to update weights"))
                        raise RuntimeError("Failed to update weights due to remote errors")
                    socket.send_pyobj(_to_named_tensor(bucket.items, gidx % 2 * bucket_size))
                    gidx += 1

            socket.recv()
            socket.send_pyobj(None)
            socket.recv()
        finally:
            req_thread.join()
            dist.barrier(group=ranks_group)
            socket.close()
            if ranks and h2d_buffer is not None:
                self._p2p_store.unregister_named_tensors([h2d_buffer_name])

            self.device_manager.device_module.empty_cache()


def _init_api(ps: ParameterServer) -> Any:
    import fastapi
    from fastapi import Request
    from fastapi.responses import JSONResponse, Response

    app = fastapi.FastAPI()

    class RegisterRequest(BaseModel):
        files: list[str]

    class UpdateRequest(BaseModel):
        ranks: list[int] = []
        update_url: str | None = None
        inference_group_ranks: list[int] = []
        timeout: float = 300.0
        uds: str | None = None

    def wrap_exception(func: Callable[[], None]) -> Response:
        try:
            func()
        except Exception as e:  # noqa: BLE001
            logger.exception(f"wrap exception {func} failed")
            return JSONResponse(content=str(e), status_code=500)
        return Response(status_code=200)

    @app.post("/v1/checkpoints/{checkpoint_name}/files")
    async def register_files(checkpoint_name: str, req: RegisterRequest, raw: Request) -> Response:
        return wrap_exception(lambda: ps.register_checkpoint(checkpoint_name, files=req.files))

    @app.delete("/v1/checkpoints/{checkpoint_name}")
    async def unregister_checkpoint(checkpoint_name: str) -> Response:
        return wrap_exception(lambda: ps.unregister_checkpoint(checkpoint_name))

    @app.get("/v1/healthz")
    async def healthz() -> Response:
        return Response(status_code=200)

    @app.post("/v1/checkpoints/{checkpoint_name}/gather-metas")
    async def gather_metas(checkpoint_name: str) -> Response:
        return wrap_exception(lambda: ps.gather_metas(checkpoint_name))

    @app.post("/v1/checkpoints/{checkpoint_name}/update")
    async def update(checkpoint_name: str, req: UpdateRequest) -> Response:
        def update_func(socket_paths: list[tuple[str, str]]):
            if req.update_url is None:
                return
            if req.inference_group_ranks:
                socket_paths = [socket_paths[i] for i in req.inference_group_ranks]
            request_inference_to_update(
                req.update_url, dict(socket_paths), timeout=req.timeout, uds=req.uds
            )

        return wrap_exception(lambda: ps.update(checkpoint_name, update_func, ranks=req.ranks))

    return app


@logger.catch(reraise=True)
def run_from_cli():
    import uvicorn

    parser = argparse.ArgumentParser(description="Parameter Server")
    parser.add_argument("--uds", type=str)

    args = parser.parse_args()
    logger.info(
        f"Parameter Server {args=}, master addr: {os.getenv('MASTER_ADDR')}, master port {os.getenv('MASTER_PORT')}"
    )

    assert args.uds and len(args.uds) > 0, args.uds
    ps = ParameterServer(auto_pg=True)
    uvicorn.run(_init_api(ps), uds=args.uds, timeout_keep_alive=60)


if __name__ == "__main__":
    run_from_cli()
