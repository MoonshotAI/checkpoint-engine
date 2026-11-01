from unittest.mock import MagicMock

import pytest
import torch

from checkpoint_engine.data_types import MemoryBuffer
from checkpoint_engine.ps import ParameterServer


@pytest.mark.parametrize("shared", [False, True])
def test_duplicate_registration_keeps_existing_checkpoint(
    monkeypatch: pytest.MonkeyPatch, shared: bool
) -> None:
    ps = ParameterServer.__new__(ParameterServer)
    ps._rank = 0
    ps._p2p_store = None
    ps._current_shared_memory_pool_user = ""
    ps._memory_pool = {ps.shared_memory_pool_name: []}
    ps.device_manager = MagicMock()
    pool = [MemoryBuffer(buffer=torch.arange(16, dtype=torch.uint8), size=16, metas=[])]
    register = MagicMock(return_value=pool)
    monkeypatch.setattr("checkpoint_engine.ps._register_checkpoint", register)

    ps.register_checkpoint("weights", use_shared_memory_pool=shared)
    with pytest.raises(AssertionError, match=r"already (registered|using shared memory pool)"):
        ps.register_checkpoint("weights", use_shared_memory_pool=shared)

    assert ps._get_memory_pool("weights") is pool
    assert torch.equal(pool[0].buffer, torch.arange(16, dtype=torch.uint8))
    assert ps._current_shared_memory_pool_user == ("weights" if shared else "")
    register.assert_called_once()
    ps.device_manager.host_empty_cache.assert_not_called()
