import pytest
import torch

from checkpoint_engine.data_types import MemoryBuffer
from checkpoint_engine.p2p_store import P2PStore
from checkpoint_engine.ps import ParameterServer


def test_register_named_tensors_does_not_mutate_state_on_engine_failure() -> None:
    class FakeEngine:
        def batch_register_memory(self, addrs: list[int], caps: list[int]) -> int:
            return -1

    store = P2PStore.__new__(P2PStore)
    store.rank = 0
    store.engine = FakeEngine()
    store.named_tensors = {}

    with pytest.raises(AssertionError):
        store.register_named_tensors({"w": torch.zeros(1)})

    assert store.named_tensors == {}


def test_register_named_tensors_records_state_after_engine_success() -> None:
    class FakeEngine:
        def batch_register_memory(self, addrs: list[int], caps: list[int]) -> int:
            return 0

    store = P2PStore.__new__(P2PStore)
    store.rank = 0
    store.engine = FakeEngine()
    store.named_tensors = {}
    tensor = torch.zeros(1)

    store.register_named_tensors({"w": tensor})

    assert store.named_tensors == {"w": tensor}


def test_register_checkpoint_cleans_memory_pool_after_p2p_registration_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDeviceManager:
        device_type = "npu"

    class FakeP2PStore:
        def __init__(self) -> None:
            self.named_tensors: dict[str, torch.Tensor] = {}

        def register_named_tensors(self, named_tensors: dict[str, torch.Tensor]) -> None:
            raise AssertionError("engine register failed")

        def unregister_named_tensors(self, names: list[str]) -> int:
            raise AssertionError("should not unregister names that were never registered")

    buffer = MemoryBuffer(buffer=torch.zeros(1), size=4, metas=[])

    ps = ParameterServer.__new__(ParameterServer)
    ps.device_manager = FakeDeviceManager()
    ps._rank = 0
    ps._p2p_store = FakeP2PStore()
    ps._memory_pool = {ps.shared_memory_pool_name: []}
    ps._current_shared_memory_pool_user = ""

    monkeypatch.setattr("checkpoint_engine.ps._register_checkpoint", lambda **_: [buffer])

    with pytest.raises(AssertionError, match="engine register failed"):
        ps.register_checkpoint("ckpt", named_tensors={"w": torch.zeros(1)})

    assert "ckpt" not in ps._memory_pool
