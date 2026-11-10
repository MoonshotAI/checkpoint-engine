import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from checkpoint_engine.pin_memory import _inplace_pin_memory, _register_checkpoint


class _CudaWithFailedRegistration:
    def __init__(self, runtime: Any = None):
        self.runtime = runtime
        self.failed = threading.Event()
        self.registered = []
        self.unregistered = []

    def cudaHostRegister(self, ptr: int, size: int, flags: int) -> Any:
        if size == 16:
            self.failed.set()
            return 2
        assert self.failed.wait(timeout=10)
        result = self.runtime.cudaHostRegister(ptr, size, flags) if self.runtime else 0
        if result == 0:
            self.registered.append(ptr)
        return result

    def cudaHostUnregister(self, ptr: int) -> Any:
        result = self.runtime.cudaHostUnregister(ptr) if self.runtime else 0
        if result == 0:
            self.unregistered.append(ptr)
        return result

    def cudaGetErrorString(self, error: int) -> str:
        return "injected registration failure"


def _run_failed_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_cuda: bool,
    through_register: bool = False,
) -> tuple[Path, Path, _CudaWithFailedRegistration]:
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    save_file({"first": torch.arange(4, dtype=torch.float32)}, str(first))
    save_file({"second": torch.arange(4096, dtype=torch.float32)}, str(second))

    runtime = torch.cuda.cudart() if use_cuda else None
    cuda = _CudaWithFailedRegistration(runtime)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cuda)
    if not use_cuda:
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)

    with pytest.raises(RuntimeError, match="pin memory error"):
        files = [str(first), str(second)]
        if through_register:
            _register_checkpoint(files=files, named_tensors={}, inplace_pin=True)
        else:
            _inplace_pin_memory(files)

    return first, second, cuda


def _assert_files_preserved(first: Path, second: Path) -> None:
    assert torch.equal(load_file(str(first))["first"], torch.arange(4, dtype=torch.float32))
    assert torch.equal(load_file(str(second))["second"], torch.arange(4096, dtype=torch.float32))


def _assert_unpinned(cuda: _CudaWithFailedRegistration) -> None:
    assert len(cuda.registered) == 1
    assert cuda.unregistered == cuda.registered


def test_inplace_registration_failure_preserves_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second, _ = _run_failed_registration(tmp_path, monkeypatch, use_cuda=False)
    _assert_files_preserved(first, second)


def test_inplace_registration_failure_unpins_other_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, cuda = _run_failed_registration(tmp_path, monkeypatch, use_cuda=False)
    _assert_unpinned(cuda)


@pytest.mark.skipif(
    sys.platform != "linux" or not Path("/dev/shm").is_dir(),  # noqa: S108
    reason="requires Linux /dev/shm",
)
def test_register_checkpoint_preserves_shm_files_on_pin_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TemporaryDirectory(dir="/dev/shm") as temp_dir:
        first, second, cuda = _run_failed_registration(
            Path(temp_dir), monkeypatch, use_cuda=False, through_register=True
        )
        _assert_files_preserved(first, second)
        _assert_unpinned(cuda)


@pytest.mark.parametrize(
    "use_cuda",
    [
        False,
        pytest.param(
            True,
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
            ],
        ),
    ],
)
def test_inplace_unlink_failure_keeps_registered_buffers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_cuda: bool
) -> None:
    paths = [tmp_path / f"{i}.safetensors" for i in range(3)]
    expected = [torch.arange(4096 + i, dtype=torch.float32) for i in range(3)]
    for path, tensor in zip(paths, expected):
        save_file({"weight": tensor}, str(path))
    cuda = _CudaWithFailedRegistration(torch.cuda.cudart() if use_cuda else None)
    cuda.failed.set()
    if not use_cuda:
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cuda)
    warnings = []
    monkeypatch.setattr("checkpoint_engine.pin_memory.logger.warning", warnings.append)
    real_remove = os.remove

    def fail_second_remove(path: str) -> None:
        if Path(path) == paths[1]:
            raise PermissionError("injected unlink failure")
        real_remove(path)

    monkeypatch.setattr("checkpoint_engine.pin_memory.os.remove", fail_second_remove)
    buffers = _inplace_pin_memory([str(path) for path in paths])
    try:
        assert [path.exists() for path in paths] == [False, True, False]
        assert len(buffers) == len(cuda.registered) == 3
        assert not cuda.unregistered
        assert any("injected unlink failure" in warning for warning in warnings)
        for memory_buffer, tensor in zip(buffers, expected):
            assert memory_buffer.manually_pinned
            assert torch.equal(memory_buffer.buffer.view(dtype=torch.float32), tensor)
    finally:
        for memory_buffer in buffers:
            if memory_buffer.manually_pinned:
                assert cuda.cudaHostUnregister(memory_buffer.buffer.data_ptr()) == 0


def test_inplace_submit_failure_waits_for_queued_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [tmp_path / f"{i}.safetensors" for i in range(2)]
    for i, path in enumerate(paths):
        save_file({"weight": torch.arange(4096 + i, dtype=torch.float32)}, str(path))
    cuda = _CudaWithFailedRegistration()
    cuda.failed.set()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cuda)

    class SubmitFailure(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.submissions = 0

        def submit(self, *args, **kwargs) -> Any:
            result = super().submit(*args, **kwargs)
            self.submissions += 1
            if self.submissions == 2:
                raise RuntimeError("injected submit failure")
            return result

    monkeypatch.setattr(
        "checkpoint_engine.pin_memory.concurrent.futures.ThreadPoolExecutor", SubmitFailure
    )
    with pytest.raises(RuntimeError, match="injected submit failure"):
        _inplace_pin_memory([str(path) for path in paths])
    assert all(path.exists() for path in paths)
    assert len(cuda.registered) == 2
    assert set(cuda.unregistered) == set(cuda.registered)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_inplace_registration_failure_unpins_cuda_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second, cuda = _run_failed_registration(tmp_path, monkeypatch, use_cuda=True)
    _assert_unpinned(cuda)
    _assert_files_preserved(first, second)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_inplace_registration_on_cuda(tmp_path: Path) -> None:
    path = tmp_path / "weights.safetensors"
    expected = torch.arange(4096, dtype=torch.float32)
    save_file({"weight": expected}, str(path))
    buffers = _inplace_pin_memory([str(path)])
    try:
        assert not path.exists()
        assert len(buffers) == 1
        assert buffers[0].manually_pinned
        assert torch.equal(buffers[0].buffer.view(dtype=torch.float32), expected)
    finally:
        for memory_buffer in buffers:
            if memory_buffer.manually_pinned:
                assert torch.cuda.cudart().cudaHostUnregister(memory_buffer.buffer.data_ptr()) == 0
