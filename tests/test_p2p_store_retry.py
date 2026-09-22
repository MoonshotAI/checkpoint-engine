"""Regression test for P2PStore's TransferEngine retry cleanup (issue #94).

CPU-only: mooncake.engine is stubbed out with a fake TransferEngine so this
exercises the retry loop's cleanup ordering without needing RDMA hardware or
the real mooncake package.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from collections.abc import Iterator


def _fake_device_manager() -> SimpleNamespace:
    return SimpleNamespace(
        device_module=SimpleNamespace(device_count=lambda: 1),
        rdma_device=lambda local_rank: "mlx5_0",
        transfer_engine_protocol="rdma",
    )


class _FakeTransferEngine:
    """Fails `n_failures` times, then succeeds.

    Tracks how many instances are simultaneously alive via a class-level
    counter incremented in __init__ and decremented in __del__, so a test
    can assert that a failed instance is released before the next one is
    constructed.
    """

    n_failures: int = 0
    active_count: int = 0
    max_concurrent: int = 0
    _attempts: int = 0

    def __init__(self) -> None:
        type(self).active_count += 1
        type(self).max_concurrent = max(type(self).max_concurrent, type(self).active_count)
        self._freed = False

    def initialize(self, ip: str, mode: str, protocol: str, device: str) -> int:
        type(self)._attempts += 1
        if type(self)._attempts <= type(self).n_failures:
            return -1
        return 0

    def get_rpc_port(self) -> int:
        return 12345

    def __del__(self) -> None:
        if not self._freed:
            self._freed = True
            type(self).active_count -= 1


@pytest.fixture
def fake_mooncake() -> Iterator[type[_FakeTransferEngine]]:
    _FakeTransferEngine.n_failures = 0
    _FakeTransferEngine.active_count = 0
    _FakeTransferEngine.max_concurrent = 0
    _FakeTransferEngine._attempts = 0
    fake_module = types.ModuleType("mooncake.engine")
    fake_module.TransferEngine = _FakeTransferEngine

    # NOT mock.patch.dict(sys.modules, ...): its __exit__ clears the *entire*
    # sys.modules dict and restores only the pre-__enter__ snapshot, which
    # silently evicts every module imported *during* the `with` block --
    # including torch (pulled in transitively by `checkpoint_engine`).  A
    # later test then re-imports torch from scratch in the same process,
    # which segfaults (loading torch._C twice per-process is not supported).
    # Setting/popping just this one key avoids touching anything else.
    had_key = "mooncake.engine" in sys.modules
    previous = sys.modules.get("mooncake.engine")
    sys.modules["mooncake.engine"] = fake_module
    try:
        yield _FakeTransferEngine
    finally:
        if had_key:
            sys.modules["mooncake.engine"] = previous
        else:
            sys.modules.pop("mooncake.engine", None)


def test_failed_transfer_engine_is_freed_before_next_retry(
    fake_mooncake: type[_FakeTransferEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only one TransferEngine should ever be alive at a time.

    Before the fix, `self.engine = TransferEngine()` constructs the new
    engine while the previous failed one is still referenced (and thus
    still holding its native resources), so max_concurrent would reach 2.
    """
    monkeypatch.setenv("RANK", "0")
    fake_mooncake.n_failures = 2  # fail twice, succeed on the 3rd attempt
    monkeypatch.setattr("time.sleep", lambda _: None)

    from checkpoint_engine.p2p_store import P2PStore

    store = P2PStore(_fake_device_manager())

    assert fake_mooncake._attempts == 3
    assert fake_mooncake.max_concurrent == 1
    assert fake_mooncake.active_count == 1  # the successful engine stays alive
    assert store.port == 12345


def test_raises_after_exhausting_retries(
    fake_mooncake: type[_FakeTransferEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "0")
    fake_mooncake.n_failures = 8  # never succeeds
    monkeypatch.setattr("time.sleep", lambda _: None)

    from checkpoint_engine.p2p_store import P2PStore

    with pytest.raises(RuntimeError, match="fail to initialize transfer engine"):
        P2PStore(_fake_device_manager())

    assert fake_mooncake.max_concurrent == 1
