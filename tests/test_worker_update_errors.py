"""A failed worker must fail the whole update, including the last bucket."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import zmq

from checkpoint_engine.data_types import (
    MemoryBuffer,
    MemoryBufferMetaList,
    MemoryBufferMetas,
    ParameterMeta,
)
from checkpoint_engine.ps import ParameterServer
from checkpoint_engine.worker import update_weights_from_ipc


class _DeviceOps:
    def __init__(self, device: str) -> None:
        self.device = device

    def synchronize(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    def empty_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def memory_allocated(self) -> int:
        return torch.cuda.memory_allocated() if self.device == "cuda" else 0

    def memory_reserved(self) -> int:
        return torch.cuda.memory_reserved() if self.device == "cuda" else 0

    def mem_get_info(self) -> tuple[int, int]:
        return torch.cuda.mem_get_info() if self.device == "cuda" else (4096, 8192)


class _LocalIPC:
    """The real socket protocol with an in-process stand-in for device IPC."""

    def export(self, buffer: torch.Tensor) -> str:
        self.buffer = buffer
        return "test handle"

    def attach(self, handle: str, device_id: int) -> torch.Tensor:
        assert handle == "test handle"
        return self.buffer

    def detach(self):
        pass


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)],
)
@pytest.mark.parametrize("bucket_count, fail_at", [(1, 0), (2, 0), (2, 1), (2, None)])
def test_worker_failure_is_reported_on_every_bucket(
    device: str, bucket_count: int, fail_at: int | None
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    ops = _DeviceOps(device)
    manager = SimpleNamespace(
        device_type=device,
        device_module=ops,
        supports_device_ipc=lambda: True,
        ipc_collect=lambda: None,
    )
    ipc = _LocalIPC()
    source = torch.arange(1, bucket_count + 1, dtype=torch.float32).view(torch.uint8)
    metas = [
        ParameterMeta(name=f"w{index}", dtype=torch.float32, shape=torch.Size([1]), aligned_size=4)
        for index in range(bucket_count)
    ]
    memory = MemoryBuffer(buffer=source, size=source.nbytes, metas=metas)
    global_meta = MemoryBufferMetaList(
        p2p_store_addr=None,
        rdma_device="",
        memory_buffer_metas_list=[
            MemoryBufferMetas(ptr=source.data_ptr(), size=source.nbytes, metas=metas)
        ],
    )
    context = zmq.Context()
    context.setsockopt(zmq.RCVTIMEO, 5000)
    context.setsockopt(zmq.SNDTIMEO, 5000)
    context.setsockopt(zmq.LINGER, 0)
    address = f"inproc://checkpoint-worker-error-{uuid.uuid4().hex}"
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    socket.setsockopt(zmq.SNDTIMEO, 5000)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(address)
    ps = ParameterServer.__new__(ParameterServer)
    ps._rank = 0
    ps.device_manager = manager
    ps._current_global_parameter_metas = {0: global_meta}
    ps._current_shared_memory_pool_user = ""
    ps._memory_pool = {"checkpoint": [memory]}
    ps._local_rdma_devices = {}
    ps._remote_rdma_devices = {}
    ps._detect_bucket_size = lambda *_args, **_kwargs: (4, True)
    ps._bind_zmq_socket = lambda: (socket, [("local", address)])
    ps._logger_rank0 = lambda *_args: None
    loaded, worker_errors, post_hooks = [], [], []

    def load_weights(weights: list[tuple[str, torch.Tensor]]) -> None:
        name, value = weights[0]
        loaded.append((name, value.item()))
        if name == f"w{fail_at}":
            raise RuntimeError("synthetic weight load failure")

    def request_worker(paths: list[tuple[str, str]]) -> None:
        try:
            update_weights_from_ipc(
                context,
                dict(paths)["local"],
                0,
                run=load_weights,
                post_hook=lambda: post_hooks.append(True),
            )
        except Exception as exc:  # noqa: BLE001
            worker_errors.append(str(exc))

    try:
        with (
            patch("checkpoint_engine.ps.dist.is_initialized", return_value=True),
            patch("checkpoint_engine.ps.dist.broadcast"),
            patch("checkpoint_engine.ps.dist.all_reduce"),
            patch("checkpoint_engine.ps.dist.barrier"),
            patch("checkpoint_engine.worker.DeviceManager", return_value=manager),
            patch("checkpoint_engine.worker._ipc_handler_for_handle", return_value=ipc),
        ):
            if fail_at is None:
                ps._update_per_bucket("checkpoint", request_worker, ipc, None)
            else:
                with pytest.raises(
                    RuntimeError, match="Failed to update weights due to remote errors"
                ):
                    ps._update_per_bucket("checkpoint", request_worker, ipc, None)
    finally:
        socket.close(linger=0)
        context.term()

    assert loaded == [
        (f"w{index}", float(index + 1))
        for index in range(bucket_count if fail_at is None else fail_at + 1)
    ]
    if fail_at is None:
        assert post_hooks == [True]
        assert not worker_errors
    else:
        assert not post_hooks
        assert worker_errors == ["Some workers failed to update weights"]


def test_p2p_failure_reaches_nonparticipants_and_barrier_can_be_reused() -> None:
    from checkpoint_engine.ps import _WorkerUpdateError

    timeout = timedelta(seconds=4)
    server = torch.distributed.TCPStore(
        "127.0.0.1", 0, 3, True, timeout=timeout, wait_for_workers=False, use_libuv=False
    )
    stores = [server] + [
        torch.distributed.TCPStore(
            "127.0.0.1", server.port, 3, False, timeout=timeout, use_libuv=False
        )
        for _ in range(2)
    ]
    failed = {"value": True}
    servers = []
    for rank, store in enumerate(stores):
        ps = ParameterServer.__new__(ParameterServer)
        ps._rank = rank
        ps._world_size = 3
        ps._auto_pg = False
        ps._store = store
        ps._store_barrier_counter = 0
        ps._store_barrier_trash = []
        ps.device_manager = SimpleNamespace(device_module=_DeviceOps("cpu"))
        barrier = ps.store_based_barrier
        ps.store_based_barrier = lambda barrier=barrier: barrier(timeout)

        def update_bucket(*_args, rank: int = rank, **_kwargs) -> None:
            if rank == 1 and failed["value"]:
                raise _WorkerUpdateError("Failed to update weights due to remote errors")

        ps._update_per_bucket = update_bucket
        servers.append(ps)

    def run_updates() -> list[BaseException | None]:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(ps.update, "checkpoint", lambda _paths: None, ranks=[1, 2])
                for ps in servers
            ]
            return [future.exception(timeout=20) for future in futures]

    with (
        patch("checkpoint_engine.ps.dist.is_initialized", return_value=True),
        patch("checkpoint_engine.ps.dist.new_group", return_value=object()),
        patch("checkpoint_engine.ps.dist.destroy_process_group"),
        patch(
            "checkpoint_engine.ps.build_ipc_handler",
            side_effect=lambda _manager: nullcontext(object()),
        ),
    ):
        errors = run_updates()
        assert all(isinstance(error, RuntimeError) for error in errors)
        assert all("Failed to update weights" in str(error) for error in errors)
        assert [ps._store_barrier_counter for ps in servers] == [1, 1, 1]
        failure_key = "parameter_server_barrier-1:update_failed"
        assert server.check([failure_key])
        failed["value"] = False
        assert run_updates() == [None, None, None]
        assert run_updates() == [None, None, None]
        assert not server.check([failure_key])


def test_p2p_uncoordinated_error_does_not_wait_at_store_barrier() -> None:
    ps = ParameterServer.__new__(ParameterServer)
    ps._rank = 0
    ps._auto_pg = False
    ps._store = Mock()
    ps._store_barrier_counter = 0
    ps.device_manager = SimpleNamespace(device_module=_DeviceOps("cpu"))
    ps._update_per_bucket = Mock(side_effect=ValueError("failed to export"))
    ps.store_based_barrier = Mock()
    with (
        patch("checkpoint_engine.ps.dist.new_group", return_value=object()),
        patch("checkpoint_engine.ps.dist.destroy_process_group"),
        patch("checkpoint_engine.ps.build_ipc_handler", return_value=nullcontext(object())),
        pytest.raises(ValueError, match="failed to export"),
    ):
        ps.update("checkpoint", lambda _paths: None, ranks=[0])
    ps._store.set.assert_not_called()
    ps.store_based_barrier.assert_not_called()
