from __future__ import annotations

import argparse
import concurrent.futures
import os
import pickle
import random
import socket
import subprocess
import threading
import time
import uuid
from collections import defaultdict
from datetime import timedelta
from functools import cached_property, lru_cache
from typing import Callable, NamedTuple

import numpy as np
import requests
import torch
import torch.distributed as dist
import zmq
from loguru import logger
from pydantic import BaseModel, ConfigDict
from safetensors.torch import safe_open
from torch.multiprocessing.reductions import reduce_tensor
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypedDict

    class FileMeta(TypedDict):
        key: str  # parameter name
        dtype: torch.dtype
        shape: torch.Size
        type: type
        tp_concat_dim: int


class ParameterMeta(BaseModel):
    # now all classes are changed to pydantic BaseModel
    # it will directly report validation errors for unknown types
    # like torch.dtype, torch.Size, so we need this configuration
    # see https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.validate_assignment
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    dtype: torch.dtype
    shape: torch.Size


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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    buffer: torch.Tensor
    size: int
    metas: list[ParameterMeta]


class MemoryBufferMetaList(BaseModel):
    p2p_store_addr: str | None
    memory_buffer_metas_list: list[MemoryBufferMetas]


class DataToGather(MemoryBufferMetaList):
    host_ip: str
    zmq_socket_path: tuple[str, str]


# 256 bytes alignment when flatten torch tensors to uint8 buffer
_ALIGN_SIZE = 256


def _align_size(dtype: torch.dtype, shape: torch.Size) -> int:
    return (dtype.itemsize * shape.numel() + _ALIGN_SIZE - 1) // _ALIGN_SIZE * _ALIGN_SIZE


def _to_named_tensor(metas: list[ParameterMeta], offset=0) -> list[dict]:
    ret = []
    for meta in metas:
        size = _align_size(meta.dtype, meta.shape)
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


def _load_checkpoint_file(file_path: str) -> tuple[int, dict[str, tuple[FileMeta, torch.Tensor]]]:
    def _safetensors_load(fn) -> dict[str, tuple[FileMeta, torch.Tensor]]:
        ret = {}
        with safe_open(fn, framework="pt") as f:
            for name in f.keys():
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
    def _fast_np_load(fn) -> dict[str, tuple[FileMeta, torch.Tensor]]:
        """load *.np file and return memmap and related tensor meta"""

        def parse_npy_header(fin):
            start = fin.tell()
            major, minor = np.lib.format.read_magic(fin)
            if major == 1 and minor == 0:
                read_header_fn = np.lib.format.read_array_header_1_0
            elif major == 2 and minor == 0:
                read_header_fn = np.lib.format.read_array_header_2_0
            else:
                raise ValueError(f"unknown version {major}.{minor} when parsing npy header from {fn}")
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


def _concat_tp_weights(tp_weights: list[torch.Tensor], tp_concat_dim: int, tp_size: int) -> torch.Tensor:
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


def _get_physical_gpu_id(rank: int) -> str:
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(result.stdout)
    lines = result.stdout.strip().split("\n")
    for line in lines:
        if f"GPU {rank}" in line:
            uuid = line.split("UUID: ")[1].strip(")")
            return uuid
    raise ValueError(f"not found gpu{rank} uuid")


@lru_cache(maxsize=1)
def _get_ip():
    try:
        # try to get ip from network interface
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except:
        # fallback to get ip from hostname
        logger.warning("fail to get ip from network interface, fallback to get ip from hostname")
        return socket.gethostbyname(socket.gethostname())


def _get_rdma_devices() -> list[str]:
    """
    use script like below to get RDMA devices, if NCCL_IB_HCA has multiple values, just return
    ```bash
    pushd /sys/class/infiniband/ > /dev/null;
    for i in mlx5_*; do cat "$i"/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo "$i" ; done;
    popd > /dev/null;
    ```
    """
    devices_str = os.getenv("PS_P2P_STORE_RDMA_DEVICES")
    if devices_str:
        return devices_str.split(",")
    # if PS_P2P_STORE_RDMA_DEVICES is not set, try to use NCCL_IB_HCA to get RDMA devices
    hca = os.getenv("NCCL_IB_HCA", None)
    if hca:
        l = hca.split(",")
        if len(l) > 1:
            # if NCCL_IB_HCA has multiple values, just return
            return l
        else:
            hca = l[0]
    basepath = "/sys/class/infiniband/"
    port_path = "ports/1/gid_attrs/types"
    devices = []
    for device in sorted(os.listdir(basepath)):
        if hca is not None and hca not in device:
            continue
        path = os.path.join(basepath, device, port_path)
        if not os.path.exists(path) or not os.path.isdir(path):
            continue
        for port in os.listdir(path):
            try:
                content = open(os.path.join(path, port)).read()
                if "v" in content:
                    print(f"found rdma device {device} in port {port}: {content.strip()}")
                    devices.append(device)
                    break
            except Exception:
                pass
    return devices


def _get_my_rdma_device(local_rank: int, gpu_count: int, devices: list[str]):
    """
    implement network card device allocation, if network card is "mlx5_0,mlx5_1", then 0-3 will share mlx5_0, 4-7 will share mlx5_1, etc.
    """
    if not devices:
        raise RuntimeError("no rdma devices found")
    assert len(devices) <= gpu_count, f"rdma devices count {len(devices)} should be less than or equal to gpu count {gpu_count}"
    assert gpu_count % len(devices) == 0, f"gpu count {gpu_count} should be divisible by rdma devices count {len(devices)}"
    return devices[local_rank // (gpu_count // len(devices))]


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
                assert isinstance(meta["dtype"], torch.dtype), f"meta {meta} dtype should be torch.dtype"
                assert isinstance(meta["shape"], torch.Size), f"meta {meta} shape should be torch.Size"
                parameter_metas[parameter_name] = ParameterMeta(
                    name=parameter_name,
                    shape=meta["shape"],
                    dtype=meta["dtype"],
                )
            tp_meta = tp_metas[parameter_name]
            if tp_meta.concat_dim != -1:
                tp_meta.size = max(tp_meta.size, tp_rank + 1)
    for name, tp_meta in tp_metas.items():
        if tp_meta.concat_dim != -1:
            shape = list(parameter_metas[name].shape)
            shape[tp_meta.concat_dim] = shape[tp_meta.concat_dim] * tp_meta.size
            parameter_metas[name] = ParameterMeta(name=name, shape=torch.Size(shape), dtype=parameter_metas[name].dtype)
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


def _register_checkpoint(
    *,
    files: list[str] = [],
    named_tensors: dict[str, torch.Tensor] = {},
    rank: int | None = None,
) -> list[MemoryBuffer]:
    logger.info(f"[rank{rank}] start to register checkpoint with {len(files)} files and {len(named_tensors)} named_tensors")
    if not files and not named_tensors:
        return []
    parameters = _load_checkpoint(files)
    if named_tensors:
        parameters.update(named_tensors)
    bucket_size = max(4 << 30, max(map(lambda x: _align_size(x.dtype, x.shape), parameters.values())))

    class MemoryBucket(BaseModel):
        size: int
        metas: list[ParameterMeta]

    buckets: list[MemoryBucket] = [MemoryBucket(size=0, metas=[])]
    for name, tensor in sorted(parameters.items()):
        size = _align_size(tensor.dtype, tensor.shape)
        if buckets[-1].size + size > bucket_size:
            assert buckets[-1], f"buckets[{len(buckets) - 1}] should not be empty"
            buckets.append(MemoryBucket(size=0, metas=[]))
        buckets[-1].metas.append(ParameterMeta(name=name, shape=tensor.shape, dtype=tensor.dtype))
        buckets[-1].size += size

    memory_buffers = [MemoryBuffer(buffer=torch.empty(0), size=bucket.size, metas=bucket.metas) for bucket in buckets]

    def register_pin_memory(idx: int, size: int) -> tuple[int, torch.Tensor]:
        buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        return idx, buffer

    def register_tensor(buffer: torch.Tensor, offset: int, tensor: torch.Tensor):
        buffer[offset : offset + tensor.nbytes] = tensor.view(-1).view(dtype=torch.uint8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(register_pin_memory, idx, bucket.size) for idx, bucket in enumerate(buckets)]
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


def request_inference_to_update(url: str, socket_paths: dict[str, str], timeout: float = 300.0):
    resp = requests.post(
        url,
        json={
            "method": "update_weights_from_ipc",
            "args": [socket_paths],
            "timeout": timeout,
        },
        timeout=timeout,
    )
    resp.raise_for_status()


def _gen_h2d_buckets(global_metas: dict[int, MemoryBufferMetaList], bucket_size: int) -> list[tuple[int, H2DBucket]]:
    buckets: list[tuple[int, H2DBucket]] = []

    for owner_rank, items in global_metas.items():
        buckets.append((owner_rank, H2DBucket(size=0, ranges=[], items=[])))
        for idx, metas in enumerate(items.memory_buffer_metas_list):
            start_offset, offset = 0, 0
            for meta in metas.metas:
                s = _align_size(meta.dtype, meta.shape)
                if buckets[-1][1].size + s > bucket_size:
                    if offset - start_offset > 0:
                        buckets[-1][1].ranges.append(BucketRange(idx, start_offset, offset - start_offset))
                    start_offset = offset
                    buckets.append((owner_rank, H2DBucket(size=0, ranges=[], items=[])))
                offset += s
                buckets[-1][1].size += s
                buckets[-1][1].items.append(meta)
            buckets[-1][1].ranges.append(BucketRange(idx, start_offset, offset - start_offset))
        assert buckets[-1][1].size > 0, f"buckets[-1][1].size {buckets[-1][1].size} should be greater than 0"
    return buckets


def _get_master_port(master_port: int | None = None) -> int:
    if master_port is None:
        # HACK: use MASTER_PORT + 1 as master_port, avoid conflict with torchrun's rendezvous port
        # TODO: check whether master_port is available or use a more elegant way
        master_port = int(os.getenv("MASTER_PORT")) + 1
    return master_port


class P2PStore:
    def __init__(self):
        from mooncake.engine import TransferEngine

        self.rank = int(os.getenv("RANK"))
        gpu_count = torch.cuda.device_count()
        local_rank = self.rank % gpu_count
        device = _get_my_rdma_device(local_rank, gpu_count, _get_rdma_devices())
        self.ip = _get_ip()

        # we will start at most 8 ps processes, so we use 8 retries to avoid port conflicts in extreme cases
        retry_count = 8
        for i in range(retry_count):
            self.engine = TransferEngine()
            ret = self.engine.initialize(self.ip, "P2PHANDSHAKE", "rdma", device)
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
        logger.info(f"[rank{self.rank}] p2p store initialized, addr is {self.addr}, rdma device is {device}")

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
            logger.info(f"[rank{self.rank}] p2p store unregister tensor {name} with addr {hex(buffer_addresses[i])}")
            num_unregistered += 1
        return num_unregistered

    def batch_transfer_sync_read(self, target_hostname: str, buf_ptrs: list[int], remote_ptrs: list[int], lens: list[int]):
        assert self.engine.batch_transfer_sync_read(target_hostname, buf_ptrs, remote_ptrs, lens) == 0


class ParameterServer:
    def __init__(self, *, auto_pg: bool = False):
        """
        Initialize the parameter server. env RANK, WORLD_SIZE and MASTER_ADDR must be set.

        Args:
            auto_pg: Whether to automatically initialize the process group.
                Notice that if auto_pg is True, will destroy the process group after update.
        """
        self._rank = int(os.environ.get("RANK", None))
        self._world_size = int(os.environ.get("WORLD_SIZE", None))
        self._master_addr = os.getenv("MASTER_ADDR")
        self._gpu_count = torch.cuda.device_count()
        self._local_rank = self._rank % self._gpu_count
        self._auto_pg = auto_pg
        self._all_hosts = []
        self._global_socket_paths: list[tuple[str, str]] = []

        assert self._rank is not None and self._rank >= 0, self._rank
        assert self._world_size and self._world_size > 0, self._world_size

        self._device_uuid = _get_physical_gpu_id(self._local_rank)
        self._zmq_ctx = zmq.Context()

        self._memory_pool: dict[str, list[MemoryBuffer]] = {}
        # dict key is owner_rank, value is a bucket metas list in owner_rank
        self._current_global_parameter_metas: dict[int, MemoryBufferMetaList] = {}
        try:
            self._p2p_store = P2PStore()
        except ImportError as e:
            logger.warning(f"[rank{self._rank}] fail to initialize p2p store due to {e}")
            self._p2p_store = None

        torch.cuda.set_device(self._local_rank)

    def _logger_rank0(self, msg):
        if self._local_rank == 0:
            logger.info(msg)

    def get_metas(self):
        return self._current_global_parameter_metas

    def load_metas(self, metas: dict[int, MemoryBufferMetaList]):
        self._current_global_parameter_metas = metas

    def register_checkpoint(self, checkpoint_name: str, *, files: list[str] = [], named_tensors: dict[str, torch.Tensor] = {}):
        """
        Register a checkpoint to the parameter server. Both files and named_tensors will be registered together.

        Args:
            checkpoint_name: The name of the checkpoint.
            files: The safetensors files to register.
            named_tensors: The named tensors to register.
        """
        try:
            assert checkpoint_name not in self._memory_pool, f"checkpoint {checkpoint_name} already registered"
            self._memory_pool[checkpoint_name] = _register_checkpoint(files=files, named_tensors=named_tensors, rank=self._rank)
            if self._p2p_store is not None:
                self._register_parameters_to_p2p_store(checkpoint_name)
        except Exception:
            logger.exception(f"[rank{self._rank}] fail to register checkpoint {checkpoint_name} with files {files}")
            if self._p2p_store is not None:
                self._unregister_parameters_from_p2p_store(checkpoint_name)
            self.unregister_checkpoint(checkpoint_name)
            raise

    def unregister_checkpoint(self, checkpoint_name: str):
        """
        Unregister a checkpoint from the parameter server. This function will also unregister the checkpoint
        from p2p store if p2p store is initialized.
        """
        if checkpoint_name not in self._memory_pool:
            return
        if self._p2p_store is not None:
            num_unregistered = self._unregister_parameters_from_p2p_store(checkpoint_name)
            logger.info(
                f"[rank{self._rank}] unregister {num_unregistered} parameters from p2p store for checkpoint {checkpoint_name}"
            )
        del self._memory_pool[checkpoint_name]
        # see https://github.com/pytorch/pytorch/blob/31d5c675394705f8a6bc767f80ae14bf4f01246b/torch/csrc/cuda/Module.cpp#L2018
        # this works by using torch>=2.5.0
        torch._C._host_emptyCache()

    @cached_property
    def _zmq_socket_path(self) -> str:
        return f"ipc://@checkpoint-engine-{uuid.uuid4()}.sock"

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
        metas = DataToGather(
            memory_buffer_metas_list=list(
                map(
                    lambda x: MemoryBufferMetas(
                        metas=x.metas,
                        ptr=x.buffer.data_ptr(),
                        size=x.size,
                    ),
                    self._memory_pool.get(checkpoint_name, []),
                ),
            ),
            p2p_store_addr=None if self._p2p_store is None else self._p2p_store.addr,
            host_ip=_get_ip(),
            zmq_socket_path=(self._device_uuid, self._zmq_socket_path),
        )

        dist.all_gather_object(metas_lst, metas)

        self._current_global_parameter_metas = {}
        num_parameters = 0
        all_hosts: list[str] = []
        global_socket_paths: list[tuple[str, str]] = []
        for i, metas_buckets in enumerate(metas_lst):
            assert metas_buckets is not None, f"metas_buckets {i} should not be None"
            if i % self._gpu_count == 0 and not self._all_hosts:
                all_hosts.append(metas_buckets.host_ip)
            if not self._global_socket_paths:
                global_socket_paths.append(metas_buckets.zmq_socket_path)
            if metas_buckets.memory_buffer_metas_list:
                self._current_global_parameter_metas[i] = metas_buckets
                num_parameters += sum(map(lambda x: len(x.metas), metas_buckets.memory_buffer_metas_list))
        if not self._all_hosts:
            self._all_hosts = all_hosts
        if not self._global_socket_paths:
            self._global_socket_paths = global_socket_paths
        logger.info(f"[rank{self._rank}] gather parameter metas finished, num_parameters: {num_parameters}")

    def init_process_group(self, *, master_port: int | None = None, timeout: timedelta = timedelta(minutes=10)):
        """
        Initialize the process group for the ranks. This global group can be easily destroyed by calling dist.destroy_process_group.

        Args:
            master_port: The specified port of the master node. If not set, will use _get_master_port to get the port.
            timeout: The timeout of the process group.
        """
        store = dist.TCPStore(
            self._master_addr, _get_master_port(master_port), self._world_size, timeout=timeout, is_master=self._rank == 0
        )
        dist.init_process_group(backend="nccl", world_size=self._world_size, rank=self._rank, timeout=timeout, store=store)
        logger.info(f"[rank{self._rank}] init process group successfully.")

    def update(
        self,
        checkpoint_name: str,
        req_func: Callable[[list[tuple[str, str]]], None],
        *,
        ranks: list[int] = [],
    ):
        """
        Update the checkpoint to inference engine. This function should be called after gather_metas.

        Args:
            checkpoint_name: The name of the checkpoint.
            req_func: The function to request the inference of inference engine.
            ranks: The ranks to update. If not set, will use fully broadcast to update to all ranks,
                which is the fastest way to update weights, especially in colocated architecture.
                If set, will use p2p to update to the ranks, this is flexible to update to a group of ranks,
                which is useful in disaggregated architecture.
        """
        try:
            if not ranks:
                if self._auto_pg and not dist.is_initialized():
                    self.init_process_group()
                self._update_per_bucket(checkpoint_name, req_func)
            else:
                if self._rank not in ranks:
                    return
                if self._auto_pg:
                    if dist.is_initialized():
                        dist.destroy_process_group()
                        # HACK: wait 2s to ensure destroy is finished
                        time.sleep(2)
                    self.init_process_group_for_ranks(ranks)
                self._update_per_bucket_p2p(checkpoint_name, req_func, ranks)
            if self._auto_pg:
                dist.destroy_process_group()

            torch.cuda.empty_cache()

            logger.info(
                f"[rank{self._rank}] update checkpoint {checkpoint_name} with ranks {ranks} done. "
                f"Current CUDA allocated {torch.cuda.memory_allocated() / 1024 / 1024} MB, "
                f"reserved {torch.cuda.memory_reserved() / 1024 / 1024} MB."
            )
        except Exception as e:
            logger.exception(f"[rank{self._rank}] update checkpoint {checkpoint_name} with ranks {ranks} error {e}")
            raise e

    def _get_bucket_size(self, *, disable_h2d_buffer: bool = False) -> tuple[int, bool]:
        GiB_bytes = 1 << 30
        # auto detect bucket size
        free_bytes_tensor = torch.tensor(
            int(float(torch.cuda.mem_get_info()[0]) * 0.9),
            dtype=torch.int64,
            device="cuda",
        )
        dist.all_reduce(free_bytes_tensor, op=dist.ReduceOp.MIN)
        free_bytes = free_bytes_tensor.item()
        max_tensor_bytes = 0
        max_bytes = int(os.getenv("PS_MAX_BUCKET_SIZE_GB", 8)) * GiB_bytes
        for items in self._current_global_parameter_metas.values():
            for metas_list in items.memory_buffer_metas_list:
                for meta in metas_list.metas:
                    max_tensor_bytes = max(max_tensor_bytes, _align_size(meta.dtype, meta.shape))
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
        bucket_size = min(max(max_bytes, max_tensor_bytes), free_bytes)
        logger.info(f"[rank{self._rank}] auto detect bucket size {bucket_size / GiB_bytes:.2f} GiB")
        return bucket_size, disable_h2d_buffer

    def _copy_to_buffer(self, checkpoint_name: str, bucket: H2DBucket, buffer: torch.Tensor, owner_rank: int | None = None):
        offset = 0
        if owner_rank is not None:
            buf_ptrs, remote_ptrs, lens = [], [], []
            ptr_base = buffer.data_ptr()
            target_addr, ptrs = self._get_addr_ptrs(owner_rank)
        for b in bucket.ranges:
            assert offset + b.size <= bucket.size, f"offset {offset} + size {b.size} > bucket_size {bucket.size}"
            if owner_rank is not None:
                buf_ptrs.append(ptr_base + offset)
                remote_ptrs.append(ptrs[b.idx][0] + b.offset)
                lens.append(b.size)
            else:
                pool = self._memory_pool[checkpoint_name][b.idx]
                buffer[offset : offset + b.size].data.copy_(
                    pool.buffer[b.offset : b.offset + b.size],
                    non_blocking=True,
                )
            offset += b.size
        assert offset == bucket.size, f"offset {offset} != bucket_size {bucket.size}"
        if owner_rank is not None:
            self._p2p_store.batch_transfer_sync_read(target_addr, buf_ptrs, remote_ptrs, lens)
        torch.cuda.synchronize()

    def init_process_group_for_ranks(
        self, ranks: list[int], *, master_port: int | None = None, timeout: timedelta = timedelta(minutes=10)
    ):
        """
        Initialize the process group for the ranks. This global group can be easily destroyed by calling dist.destroy_process_group.

        Args:
            ranks: The ranks to initialize the process group. ranks should be a subset of all ranks.
            master_port: The specified port of the master node. If not set, will use _get_master_port to get the port.
            timeout: The timeout of the process group.
        """
        assert not dist.is_initialized()
        assert ranks, "ranks should be set"
        if self._rank not in ranks:
            return
        assert self._all_hosts, "all_hosts should be set"
        assert len(self._all_hosts) == self._world_size // self._gpu_count, (
            f"world_size {self._world_size} should be equal to all_hosts {len(self._all_hosts)}"
        )
        rank = ranks.index(self._rank)
        master_addr = self._all_hosts[ranks[0] // self._gpu_count]
        master_port = _get_master_port(master_port)
        logger.info(
            f"[rank{self._rank}] start to init process group as virtual_rank {rank}, "
            f"master_addr {master_addr}, master_port {master_port}, world_size {len(ranks)}, "
        )
        # only initialize process group and store for ranks, other nodes are not initialized
        # and will not participate in this update. Since they have registered memory addresses
        # to p2p_store at the beginning, update ranks can directly get the memory addresses
        # from other nodes and put the weights into the buffer.
        store = dist.TCPStore(master_addr, master_port, len(ranks), is_master=rank == 0, timeout=timeout)
        dist.init_process_group(backend="nccl", world_size=len(ranks), rank=rank, timeout=timeout, store=store)

    def _update_per_bucket_p2p(
        self,
        checkpoint_name: str,
        req_func: Callable[[list[tuple[str, str]]], None],
        ranks: list[int],
    ):
        assert self._p2p_store is not None, "p2p store is not initialized"
        assert ranks, "ranks should be set"
        if len(self._current_global_parameter_metas) == 0:
            raise ValueError("parameter metas is empty")
        assert dist.is_initialized(), "process group is not initialized when update model per bucket p2p"

        need_update = self._rank in ranks
        logger.info(
            f"[rank{self._rank}] update checkpoint {checkpoint_name} p2p, {need_update=} with {ranks=}, "
            f"gpu_count {self._gpu_count}, world_size {self._world_size}"
        )

        if not need_update:
            return

        # first execute a barrier to avoid subsequent cuda oom
        dist.barrier()

        bucket_size, _ = self._get_bucket_size(disable_h2d_buffer=True)
        buffer = torch.empty(bucket_size * 2, dtype=torch.uint8, device="cuda")
        IPC_BUFFER_NAME = "__ipc_buffer___"
        self._p2p_store.register_named_tensors({IPC_BUFFER_NAME: buffer})
        logger.info(
            f"[rank{self._rank}] register buffer, shape={buffer.shape}, dtype={buffer.dtype}, data_ptr={buffer.data_ptr()}, nbytes={buffer.nbytes}"
        )
        handle = reduce_tensor(buffer)

        gidx = 0
        buckets = _gen_h2d_buckets(self._current_global_parameter_metas, bucket_size)
        req_thread = threading.Thread(
            target=req_func,
            args=(self._global_socket_paths,),
        )
        req_thread.start()
        socket = self._zmq_ctx.socket(zmq.REQ)
        socket.bind(self._zmq_socket_path)
        socket.send_pyobj(handle)
        for owner_rank, bucket in buckets:
            self._logger_rank0(
                f"[rank{self._rank}] begin to update bucket {gidx + 1}/{len(buckets)} owner_rank {owner_rank} in checkpoint {checkpoint_name}, bucket_size: {bucket.size / 1024 / 1024:.2f}MiB, length: {len(bucket.items)}. "
            )
            _buffer = buffer[gidx % 2 * bucket_size : gidx % 2 * bucket_size + bucket.size]
            if dist.get_rank() == 0:
                self._copy_to_buffer(checkpoint_name, bucket, _buffer, owner_rank)
            # broadcast the collected data to all ranks
            dist.broadcast(_buffer, src=0)
            socket.recv()
            dist.barrier()
            socket.send_pyobj(_to_named_tensor(bucket.items, gidx % 2 * bucket_size))
            gidx += 1

        socket.recv()
        socket.send_pyobj(None)
        socket.recv()
        req_thread.join()
        dist.barrier()
        socket.close()
        self._p2p_store.unregister_named_tensors([IPC_BUFFER_NAME])
        torch.cuda.empty_cache()

    def _get_addr_ptrs(self, owner_rank: int) -> tuple[str, list[tuple[int, int]]]:
        addr = self._current_global_parameter_metas[owner_rank].p2p_store_addr
        metas_list = self._current_global_parameter_metas[owner_rank].memory_buffer_metas_list
        return addr, [(metas.ptr, metas.size) for metas in metas_list]

    def _register_parameters_to_p2p_store(self, checkpoint_name: str):
        assert self._p2p_store is not None, "p2p store is not initialized"
        pool = self._memory_pool[checkpoint_name]
        if len(pool) == 0:
            return
        named_tensors, tensor_ptrs = {}, []
        for idx, memory_buffer in enumerate(pool):
            named_tensors[f"memory_pool_{checkpoint_name}_{idx}"] = memory_buffer.buffer
            tensor_ptrs.append((memory_buffer.buffer.data_ptr(), memory_buffer.size))
        self._p2p_store.register_named_tensors(named_tensors)

    def _unregister_parameters_from_p2p_store(self, checkpoint_name: str) -> int:
        assert self._p2p_store is not None, "p2p store is not initialized"
        pool = self._memory_pool[checkpoint_name]
        if len(pool) == 0:
            return 0
        return self._p2p_store.unregister_named_tensors([f"memory_pool_{checkpoint_name}_{idx}" for idx, _ in enumerate(pool)])

    def _update_per_bucket(
        self,
        checkpoint_name: str,
        req_func: Callable[[list[tuple[str, str]]], None],
    ):
        if len(self._current_global_parameter_metas) == 0:
            raise ValueError("parameter metas is empty")

        assert dist.is_initialized(), "process group is not initialized"

        logger.info(f"[rank{self._rank}] update checkpoint {checkpoint_name}")

        bucket_size, disable_h2d_buffer = self._get_bucket_size()
        buckets = _gen_h2d_buckets(self._current_global_parameter_metas, bucket_size)

        h2d_buffer: torch.Tensor | None = (
            None if disable_h2d_buffer else torch.empty(bucket_size, dtype=torch.uint8, device="cuda")
        )

        owner_rank_buckets: list[H2DBucket] = []
        for owner_rank, bucket in buckets:
            if owner_rank != self._rank:
                continue
            owner_rank_buckets.append(bucket)

        buffer = torch.empty(bucket_size * 2, dtype=torch.uint8, device="cuda")
        handle = reduce_tensor(buffer)

        buckets_by_owner_rank: dict[int, list[H2DBucket]] = defaultdict(list)
        max_len = 0
        for owner_rank, bucket in buckets:
            buckets_by_owner_rank[owner_rank].append(bucket)
            if len(buckets_by_owner_rank[owner_rank]) > max_len:
                max_len = len(buckets_by_owner_rank[owner_rank])

        req_thread = threading.Thread(
            target=req_func,
            args=(self._global_socket_paths,),
        )
        req_thread.start()
        socket = self._zmq_ctx.socket(zmq.REQ)
        socket.bind(self._zmq_socket_path)
        socket.send_pyobj(handle)

        gidx = 0
        for i in range(max_len):
            if i < len(owner_rank_buckets) and not disable_h2d_buffer:
                self._copy_to_buffer(checkpoint_name, owner_rank_buckets[i], h2d_buffer)

            for owner_rank, _buckets in buckets_by_owner_rank.items():
                if i >= len(_buckets):
                    continue
                bucket = _buckets[i]
                alloc, reserved = torch.cuda.memory_allocated() / 1024 / 1024, torch.cuda.memory_reserved() / 1024 / 1024
                self._logger_rank0(
                    f"[rank{self._rank}] begin to update bucket {gidx + 1}/{len(buckets)} owner_rank {owner_rank} in checkpoint {checkpoint_name}, bucket_size: {bucket.size / 1024 / 1024:.2f}MiB, length: {len(bucket.items)}. "
                    f"Current CUDA allocated {alloc:.2f} MB, "
                    f"reserved {reserved:.2f} MB."
                )
                start = gidx % 2 * bucket_size
                buffer_b: torch.Tensor = buffer[start : start + bucket.size]
                if owner_rank == self._rank:
                    if disable_h2d_buffer:
                        self._copy_to_buffer(checkpoint_name, bucket, buffer_b)
                    else:
                        buffer_b.data.copy_(h2d_buffer[: bucket.size])
                dist.broadcast(buffer_b, src=owner_rank)
                socket.recv()
                dist.barrier()
                socket.send_pyobj(_to_named_tensor(bucket.items, gidx % 2 * bucket_size))
                gidx += 1

        socket.recv()
        socket.send_pyobj(None)
        socket.recv()
        req_thread.join()
        dist.barrier()
        socket.close()
        torch.cuda.empty_cache()


def _init_api(ps: ParameterServer):
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

    def wrap_exception(func):
        try:
            func()
        except Exception as e:
            logger.exception(f"wrap exception {func} failed")
            return JSONResponse(content=str(e), status_code=500)
        return Response(status_code=200)

    @app.post("/v1/checkpoints/{checkpoint_name}/files")
    async def register_files(checkpoint_name: str, req: RegisterRequest, raw: Request):
        return wrap_exception(lambda: ps.register_checkpoint(checkpoint_name, files=req.files))

    @app.delete("/v1/checkpoints/{checkpoint_name}")
    async def unregister_checkpoint(checkpoint_name: str):
        return wrap_exception(lambda: ps.unregister_checkpoint(checkpoint_name))

    @app.get("/v1/healthz")
    async def healthz():
        return Response(status_code=200)

    @app.post("/v1/checkpoints/{checkpoint_name}/gather-metas")
    async def gather_metas(checkpoint_name: str):
        return wrap_exception(lambda: ps.gather_metas(checkpoint_name))

    @app.post("/v1/checkpoints/{checkpoint_name}/update")
    async def update(checkpoint_name: str, req: UpdateRequest):
        def update_func(socket_paths: list[tuple[str, str]]):
            if req.update_url is None:
                return
            if req.inference_group_ranks:
                socket_paths = [socket_paths[i] for i in req.inference_group_ranks]
            request_inference_to_update(req.update_url, dict(socket_paths), timeout=req.timeout)

        return wrap_exception(lambda: ps.update(checkpoint_name, update_func, ranks=req.ranks))

    return app


@logger.catch(reraise=True)
def run_from_cli():
    import uvicorn

    parser = argparse.ArgumentParser(description="Paramter Server")
    parser.add_argument("--uds", type=str)

    args = parser.parse_args()
    logger.info(f"Parameter Server {args=}, master addr: {os.getenv('MASTER_ADDR')}, master port {os.getenv('MASTER_PORT')}")

    assert args.uds and len(args.uds) > 0, args.uds
    ps = ParameterServer(auto_pg=True)
    uvicorn.run(_init_api(ps), uds=args.uds, timeout_keep_alive=60)


if __name__ == "__main__":
    run_from_cli()
