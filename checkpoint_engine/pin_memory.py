import concurrent.futures
import json
import os
import pickle
from typing import TYPE_CHECKING, Any, BinaryIO

import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel
from safetensors.torch import _getdtype, safe_open

from checkpoint_engine.data_types import (
    MemoryBuffer,
    ParameterMeta,
)


if TYPE_CHECKING:
    from checkpoint_engine.data_types import FileMeta

# 256 bytes alignment when flatten torch tensors to uint8 buffer
_ALIGN_SIZE = 256


def _align_size(dtype: torch.dtype, shape: torch.Size) -> int:
    return (dtype.itemsize * shape.numel() + _ALIGN_SIZE - 1) // _ALIGN_SIZE * _ALIGN_SIZE


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
    If meta.concat_dim is -1, means this is shared tp weights, just use the first weights.
    Else we will cat weights in concat_dim.
    """
    if tp_concat_dim == -1:
        return tp_weights[0]
    assert tp_size == len(tp_weights)
    if len(tp_weights) == 1:
        return tp_weights[0]
    return torch.cat([w for w in tp_weights], dim=tp_concat_dim)


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
    device_index = torch.cuda.current_device()

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
            torch.cuda.set_device(device_index)
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
