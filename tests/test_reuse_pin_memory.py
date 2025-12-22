import os

import pytest
import torch

from checkpoint_engine.ps import ParameterServer


def generate_dummy_checkpoint() -> dict[str, torch.Tensor]:
    """
    Generate dummy checkpoint data
    """
    named_tensors = {
        "layer1.weight": torch.randn(1024, 1024),
        "layer1.bias": torch.randn(1024),
        "layer2.weight": torch.randn(2048, 1024),
        "layer2.bias": torch.randn(2048),
    }
    return named_tensors


@pytest.mark.gpu
def test_register_pin_memory():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    ps = ParameterServer()
    checkpoint1 = generate_dummy_checkpoint()
    checkpoint_shared1 = generate_dummy_checkpoint()
    checkpoint2 = generate_dummy_checkpoint()
    checkpoint_shared2 = generate_dummy_checkpoint()
    checkpoint_shared3 = generate_dummy_checkpoint()
    checkpoint_shared3["layer3.weight"] = torch.randn(4096, 2048)
    checkpoint_shared3["layer3.bias"] = torch.randn(4096)
    ps.register_checkpoint("test_checkpoint1", named_tensors=checkpoint1)
    ps.unregister_checkpoint("test_checkpoint1")
    assert "test_checkpoint1" not in ps._memory_pool
    ps.register_checkpoint(
        "test_checkpoint_shared1", named_tensors=checkpoint_shared1, use_shared_memory_pool=True
    )
    ps.register_checkpoint("test_checkpoint2", named_tensors=checkpoint2)
    assert "test_checkpoint_shared1" not in ps._memory_pool
    assert "__shared_memory_pool__" in ps._memory_pool
    assert ps._current_shared_memory_pool_user == "test_checkpoint_shared1"
    assert "test_checkpoint2" in ps._memory_pool
    try:
        ps.register_checkpoint(
            "test_checkpoint_shared2", named_tensors=checkpoint_shared2, use_shared_memory_pool=True
        )  # this will fail
    except AssertionError:
        print("Caught expected AssertionError when registering second shared memory pool user")
    assert "test_checkpoint_shared2" not in ps._memory_pool
    assert ps._current_shared_memory_pool_user == "test_checkpoint_shared1"
    ps.unregister_checkpoint("test_checkpoint_shared1")
    assert ps._current_shared_memory_pool_user == ""
    assert "__shared_memory_pool__" in ps._memory_pool
    ps.register_checkpoint(
        "test_checkpoint_shared2", named_tensors=checkpoint_shared2, use_shared_memory_pool=True
    )
    assert "test_checkpoint_shared2" not in ps._memory_pool
    assert "__shared_memory_pool__" in ps._memory_pool
    assert ps._current_shared_memory_pool_user == "test_checkpoint_shared2"
    ps.unregister_checkpoint("test_checkpoint1")  # this will trigger an warning
    assert "test_checkpoint1" not in ps._memory_pool
    ps.unregister_checkpoint("test_checkpoint2")
    assert "test_checkpoint2" not in ps._memory_pool
    ps.unregister_checkpoint("test_checkpoint_shared2", force=True)
    assert ps._current_shared_memory_pool_user == ""
    assert "__shared_memory_pool__" in ps._memory_pool
    ps.register_checkpoint(
        "test_checkpoint_shared3", named_tensors=checkpoint_shared3, use_shared_memory_pool=True
    )
    assert "test_checkpoint_shared3" not in ps._memory_pool
    assert "__shared_memory_pool__" in ps._memory_pool
    assert ps._current_shared_memory_pool_user == "test_checkpoint_shared3"
    ps.unregister_checkpoint("test_checkpoint_shared3")
    assert ps._current_shared_memory_pool_user == ""
    assert "__shared_memory_pool__" in ps._memory_pool
