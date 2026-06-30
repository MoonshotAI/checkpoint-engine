import pytest
import torch

from checkpoint_engine.p2p_store import P2PStore


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
