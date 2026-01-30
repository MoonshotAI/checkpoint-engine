import ctypes
import gc
import os
import threading
from collections import defaultdict
from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING

import torch
import torch.distributed
import zmq
from loguru import logger
from torch.multiprocessing.reductions import reduce_tensor

import checkpoint_engine.distributed as dist
from checkpoint_engine.data_types import (
    BucketRange,
    DataToGather,
    H2DBucket,
    MemoryBuffer,
    MemoryBufferMetaList,
    MemoryBufferMetas,
    ParameterMeta,
)
from checkpoint_engine.device_utils import DeviceManager, get_ip, npu_generate_uuid
from checkpoint_engine.p2p_store import P2PStore
from checkpoint_engine.pin_memory import _ALIGN_SIZE, _register_checkpoint


if TYPE_CHECKING:
    from checkpoint_engine.data_types import T


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


def _get_physical_gpu_id(device_manager: DeviceManager, device_index: int | None = None) -> str:
    try:
        if device_manager.device_type == "npu":
            return f"NPU-{npu_generate_uuid()}"
        else:
            return f"GPU-{device_manager.device_module.get_device_properties(device_index).uuid!s}"
    except AssertionError as e:
        raise ValueError(f"fail to get physical gpu id {device_index}") from e


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
        master_port_str = os.getenv("MASTER_PORT")
        assert master_port_str, "MASTER_PORT is required if no master_port is provided."
        master_port = int(master_port_str) + 1
    return master_port


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
        master_addr: str | None = None,
        master_port: int | None = None,
    ):
        """
        Initialize the parameter server. env RANK, WORLD_SIZE and MASTER_ADDR must be set.

        Args:
            auto_pg: Whether to automatically initialize the process group.
                Notice that if auto_pg is True, will destroy the process group after update. It is recommended to set auto_pg to True!
            mem_fraction: The proportion (as a fraction) of the current free device memory for allocation.
        """
        self._rank = rank or int(os.environ["RANK"])
        self._world_size = world_size or int(os.environ["WORLD_SIZE"])
        self.device_manager = DeviceManager()
        self._gpu_count = gpu_count or self.device_manager.device_module.device_count()
        self._local_rank = self._rank % self._gpu_count
        self._auto_pg = auto_pg
        self._all_hosts = []
        self._global_device_uuids: list[str] = []
        self._local_rdma_devices: dict[str, set[int]] = defaultdict(set)
        self._remote_rdma_devices: dict[str, set[int]] = defaultdict(set)
        self._mem_fraction = mem_fraction or float(os.getenv("PS_MEM_FRACTION", "0.9"))

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

        master_addr = master_addr or os.getenv("MASTER_ADDR")
        assert master_addr, "master_addr is required"
        self._store = torch.distributed.TCPStore(
            master_addr,
            _get_master_port(master_port),
            self._world_size,
            timeout=timedelta(minutes=10),
            is_master=self._rank == 0,
        )
        self._store_counter = 0

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
                if r != 0:
                    error_msg = cudart.cudaGetErrorString(r)
                    raise RuntimeError(
                        f"unpin memory error, error code: {r}, error message: {error_msg}"
                    )

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
        if self.device_manager.device_type == "cuda":
            torch._C._host_emptyCache()
        else:
            # torch._C._host_emptyCache() is not supported on NPU, so we call gc.collect() to empty host cache.
            import gc

            gc.collect()

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
        timeout: timedelta = timedelta(minutes=10),
    ):
        """
        Initialize the process group for the ranks. This global group can be easily destroyed by calling dist.destroy_process_group.

        Args:
            master_port: The specified port of the master node. If not set, will use _get_master_port to get the port.
            timeout: The timeout of the process group.
        """
        self._store_counter += 1
        sub_store = torch.distributed.PrefixStore(f"prefix-{self._store_counter}", self._store)
        dist.init_process_group(
            backend=self.device_manager.backend,
            world_size=self._world_size,
            rank=self._rank,
            timeout=timeout,
            store=sub_store,
        )
        logger.info(f"[rank{self._rank}] init process group successfully.")

    def store_based_barrier(self, timeout: timedelta = timedelta(minutes=5)) -> None:
        """
        Perform a store-based barrier synchronization across all ranks.

        This barrier uses a TCP store directly rather than a process group,
        allowing all ranks to synchronize regardless of which process group
        they belong to.

        Args:
            store: The TCPStore instance to use for synchronization.
        """
        torch.distributed.distributed_c10d._store_based_barrier(
            rank=self._rank,
            store=self._store,
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
            if self._auto_pg and not dist.is_initialized():
                self.init_process_group(timeout=timeout)
            # if ranks is None or [], it will use fully broadcast to update to all ranks
            ranks_group = dist.new_group(ranks) if ranks else None
            self._update_per_bucket(checkpoint_name, req_func, ranks_group, ranks)
            self.store_based_barrier()
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
        self,
        ranks_group: dist.DistributedProcessGroup | None,
        *,
        disable_h2d_buffer: bool = False,
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
        dist.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN, group=ranks_group)
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
        max_bytes = int(float(os.getenv("PS_MAX_BUCKET_SIZE_GB", "8")) * GiB)
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
        ranks_group: dist.DistributedProcessGroup | None,
        ranks: list[int] | None = None,
    ):
        assert len(self._current_global_parameter_metas) != 0, "parameter metas is empty"
        assert dist.is_initialized(), "process group is not initialized"

        p2p_update = False
        # if both ranks is None or [], it will use fully broadcast to update to all ranks
        if not ranks:
            logger.info(f"[rank{self._rank}] update checkpoint {checkpoint_name}")
        # if ranks is set, it will use p2p to update to the ranks
        else:
            assert self._p2p_store is not None, "p2p store is not initialized"
            assert ranks, "ranks should be set"

            p2p_update = True
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
        receiver_rank_buckets: list[tuple[int, H2DBucket]] = []
        for receiver_rank, owner_rank, bucket in buckets:
            if receiver_rank != self._rank:
                continue
            receiver_rank_buckets.append((owner_rank, bucket))

        buffer = torch.empty(
            bucket_size * 2, dtype=torch.uint8, device=self.device_manager.device_type
        )
        if p2p_update:
            # p2p store need to register buffer to let other ranks read
            p2p_ipc_buffer_name = "__ipc_buffer__"
            self._p2p_store.register_named_tensors(
                {p2p_ipc_buffer_name: buffer if disable_h2d_buffer else h2d_buffer}
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
                            if p2p_update:
                                assert bucket == receiver_rank_buckets[i][1]
                            self._copy_to_buffer(
                                checkpoint_name,
                                bucket,
                                buffer_b,
                                receiver_rank_buckets[i][0] if p2p_update else None,
                            )
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
                    dist.all_reduce(ret_code, op=torch.distributed.ReduceOp.SUM, group=ranks_group)
                    self.device_manager.device_module.synchronize()
                    if ret_code.item() != 0:
                        # quit early if any rank failed
                        socket.send_pyobj(RuntimeError("Some workers failed to update weights"))
                        raise RuntimeError("Failed to update weights due to remote errors")
                    socket.send_pyobj(_to_named_tensor(bucket.items, gidx % 2 * bucket_size))
                    gidx += 1

            socket.recv()
            device_mem = self.device_manager.device_module.mem_get_info()
            logger.info(
                f"[rank{self._rank}] weights broadcast done, device mem usage: {(device_mem[1] - device_mem[0]) / 1024 / 1024:.2f} MB, allocated memory: {self.device_manager.device_module.memory_allocated() / 1024 / 1024:.2f} MB, reserved memory: {self.device_manager.device_module.memory_reserved() / 1024 / 1024:.2f} MB"
            )
            # Notify worker to release handle
            socket.send_pyobj(None)
            socket.recv()
            # Set to None in correct order (views first, then base tensors)
            del buffer_b, h2d_buffer, buffer, handle
            self.device_manager.device_module.synchronize()
            gc.collect()
            self.device_manager.device_module.ipc_collect()
            self.device_manager.device_module.empty_cache()
            self.device_manager.device_module.synchronize()

            # Log actual memory usage
            device_mem = self.device_manager.device_module.mem_get_info()
            logger.info(
                f"[rank{self._rank}] post-release: device mem usage: {(device_mem[1] - device_mem[0]) / 1024 / 1024:.2f} MB, "
                f"allocated: {self.device_manager.device_module.memory_allocated() / 1024 / 1024:.2f} MB, "
                f"reserved: {self.device_manager.device_module.memory_reserved() / 1024 / 1024:.2f} MB"
            )
            # Notify worker to call post_hook
            socket.send_pyobj(None)
            socket.recv()
        finally:
            req_thread.join()
            dist.barrier(group=ranks_group)
            socket.close()
            if p2p_update:
                self._p2p_store.unregister_named_tensors([p2p_ipc_buffer_name])

            self.device_manager.device_module.empty_cache()


# we need this CLI entry point for compatibility with former versions
if __name__ == "__main__":
    from .__main__ import run_from_cli

    run_from_cli()
