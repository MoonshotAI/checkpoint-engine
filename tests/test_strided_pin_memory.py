import threading

import pytest
import torch

from checkpoint_engine.data_types import MemoryBuffer
from checkpoint_engine.pin_memory import _align_size, _normal_pin_memory
from checkpoint_engine.ps import _to_named_tensor
from checkpoint_engine.worker import _extract_weights


def _weights(device: str) -> dict[str, torch.Tensor]:
    return {
        "transposed": torch.arange(24, device=device, dtype=torch.float32)
        .reshape(4, 6)
        .t()
        .requires_grad_(),
        "strided": torch.arange(32, device=device, dtype=torch.float16)[1::3],
        "fp8": torch.arange(24, device=device, dtype=torch.float32)
        .to(torch.float8_e4m3fn)
        .reshape(4, 6)
        .t(),
        "scalar": torch.tensor(7.0, device=device),
        "empty": torch.empty((3, 0), device=device),
    }


def _check_weights(
    weights: dict[str, torch.Tensor], buffers: list[MemoryBuffer], device: str
) -> None:
    actual = {}
    for bucket in buffers:
        payload = _to_named_tensor(bucket.metas)
        actual.update(_extract_weights(payload, bucket.buffer.to(device)))
    assert actual.keys() == weights.keys()
    for name, expected in weights.items():
        assert actual[name].shape == expected.shape
        assert actual[name].dtype == expected.dtype
        actual_bytes = actual[name].cpu().contiguous().reshape(-1).view(torch.uint8)
        expected_bytes = expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        assert torch.equal(actual_bytes, expected_bytes)


def test_register_strided_weights_in_shared_memory() -> None:
    weights = _weights("cpu")
    size = sum(_align_size(tensor.dtype, tensor.shape) for tensor in weights.values())
    shared = [MemoryBuffer(buffer=torch.empty(size, dtype=torch.uint8), size=size, metas=[])]

    buffers = _normal_pin_memory(files=[], named_tensors=weights, shared_pin_memory=shared)

    _check_weights(weights, buffers, "cpu")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("source", ["cpu", "cuda"])
def test_register_strided_weights_to_pinned_memory(source: str) -> None:
    weights = _weights(source)

    buffers = _normal_pin_memory(files=[], named_tensors=weights)

    assert all(bucket.buffer.is_pinned() for bucket in buffers)
    _check_weights(weights, buffers, "cuda")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_register_strided_cuda_copies_do_not_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    weights = {f"weight{i}": torch.randn(8, 16, device="cuda").t() for i in range(4)}
    original_copy = torch.Tensor.copy_
    count_lock = threading.Lock()
    second_started = threading.Event()
    active = 0
    maximum = 0
    copies = 0

    def track_copy(target: torch.Tensor, source: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        nonlocal active, maximum, copies
        if not source.is_cuda or source.is_contiguous():
            return original_copy(target, source, *args, **kwargs)
        with count_lock:
            active += 1
            maximum = max(maximum, active)
            copies += 1
            first = copies == 1
            if active > 1:
                second_started.set()
        try:
            if first:
                second_started.wait(timeout=1)
            return original_copy(target, source, *args, **kwargs)
        finally:
            with count_lock:
                active -= 1

    monkeypatch.setattr(torch.Tensor, "copy_", track_copy)
    buffers = _normal_pin_memory(files=[], named_tensors=weights)

    assert copies == len(weights)
    assert maximum == 1
    _check_weights(weights, buffers, "cuda")
