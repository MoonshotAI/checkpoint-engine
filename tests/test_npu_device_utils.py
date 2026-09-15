import subprocess

import pytest
import torch

from checkpoint_engine.device_utils import npu_generate_uuid


def test_npu_generate_uuid_checks_reported_device_count(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeNpu:
        @staticmethod
        def device_count() -> int:
            return 10

    pid = 12345
    seen_npu_ids: list[int] = []

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        npu_id = int(cmd[-1])
        seen_npu_ids.append(npu_id)
        stdout = "Chip Count: 2\n"
        if npu_id == 9:
            stdout += f"PID: {pid}\nChip ID: 1\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(torch, "npu", FakeNpu(), raising=False)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("checkpoint_engine.device_utils.os.getpid", lambda: pid)
    monkeypatch.setattr("checkpoint_engine.device_utils.get_ip", lambda: "10.0.0.1")
    monkeypatch.setattr("checkpoint_engine.device_utils.subprocess.run", fake_run)

    assert npu_generate_uuid() == "10.0.0.1-19"
    assert seen_npu_ids == list(range(10))


def test_npu_generate_uuid_scans_visible_physical_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeNpu:
        @staticmethod
        def device_count() -> int:
            return 1

    pid = 12345
    seen_npu_ids: list[int] = []

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        npu_id = int(cmd[-1])
        seen_npu_ids.append(npu_id)
        stdout = "Chip Count: 2\n"
        if npu_id == 4:
            stdout += f"PID: {pid}\nChip ID: 1\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(torch, "npu", FakeNpu(), raising=False)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4")
    monkeypatch.setattr("checkpoint_engine.device_utils.os.getpid", lambda: pid)
    monkeypatch.setattr("checkpoint_engine.device_utils.get_ip", lambda: "10.0.0.1")
    monkeypatch.setattr("checkpoint_engine.device_utils.subprocess.run", fake_run)

    assert npu_generate_uuid() == "10.0.0.1-9"
    assert seen_npu_ids == [4]
