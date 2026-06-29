"""CPU-only tests for the metas endpoints in api.py."""

from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from checkpoint_engine.api import _init_api
from checkpoint_engine.data_types import (
    MemoryBufferMetaList,
    MemoryBufferMetas,
    ParameterMeta,
)


_METAS_ADAPTER = TypeAdapter(dict[int, MemoryBufferMetaList])


def _make_meta(rdma_device: str, ip: str) -> MemoryBufferMetaList:
    return MemoryBufferMetaList(
        p2p_store_addr=f"{ip}:12345",
        rdma_device=rdma_device,
        memory_buffer_metas_list=[
            MemoryBufferMetas(
                metas=[
                    ParameterMeta(
                        name="w",
                        dtype=torch.float16,
                        shape=torch.Size([2, 3]),
                        aligned_size=12,
                    )
                ],
                ptr=0x12345678,
                size=1024,
            )
        ],
    )


@pytest.fixture
def fake_metas() -> dict[int, MemoryBufferMetaList]:
    return {
        0: _make_meta("mlx5_0", "192.168.1.1"),
        1: _make_meta("mlx5_1", "192.168.1.1"),
    }


@pytest.fixture
def ps_mock(fake_metas: dict[int, MemoryBufferMetaList]) -> MagicMock:
    ps = MagicMock()
    ps.get_metas.return_value = fake_metas
    return ps


def test_get_metas_returns_json(
    ps_mock: MagicMock, fake_metas: dict[int, MemoryBufferMetaList]
) -> None:
    client = TestClient(_init_api(ps_mock))
    resp = client.get("/v1/metas")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    assert _METAS_ADAPTER.validate_json(resp.content) == fake_metas
    ps_mock.get_metas.assert_called_once_with()


def test_get_metas_propagates_ps_error(ps_mock: MagicMock) -> None:
    ps_mock.get_metas.side_effect = RuntimeError("metas not gathered yet")
    client = TestClient(_init_api(ps_mock))
    resp = client.get("/v1/metas")
    assert resp.status_code == 500
    assert "metas not gathered yet" in resp.text


def test_load_metas_decodes_and_calls_ps(
    ps_mock: MagicMock, fake_metas: dict[int, MemoryBufferMetaList]
) -> None:
    client = TestClient(_init_api(ps_mock))
    resp = client.post(
        "/v1/metas",
        content=_METAS_ADAPTER.dump_json(fake_metas),
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 200
    ps_mock.load_metas.assert_called_once_with(fake_metas)


def test_load_metas_rejects_bad_json(ps_mock: MagicMock) -> None:
    client = TestClient(_init_api(ps_mock))
    resp = client.post(
        "/v1/metas",
        content=b"not a valid json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    ps_mock.load_metas.assert_not_called()


def test_load_metas_rejects_schema_mismatch(ps_mock: MagicMock) -> None:
    """JSON that parses but doesn't match MemoryBufferMetaList shape -> 422."""
    client = TestClient(_init_api(ps_mock))
    resp = client.post(
        "/v1/metas",
        content=b'{"0": {"foo": "bar"}}',
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    ps_mock.load_metas.assert_not_called()


def test_load_metas_propagates_ps_error(
    ps_mock: MagicMock, fake_metas: dict[int, MemoryBufferMetaList]
) -> None:
    ps_mock.load_metas.side_effect = RuntimeError("rdma device mismatch")
    client = TestClient(_init_api(ps_mock))
    resp = client.post(
        "/v1/metas",
        content=_METAS_ADAPTER.dump_json(fake_metas),
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 500
    assert "rdma device mismatch" in resp.text


def test_round_trip_get_then_load(
    ps_mock: MagicMock, fake_metas: dict[int, MemoryBufferMetaList]
) -> None:
    """JSON bytes returned by GET /v1/metas must be accepted by POST /v1/metas."""
    client = TestClient(_init_api(ps_mock))
    get_resp = client.get("/v1/metas")
    assert get_resp.status_code == 200
    load_resp = client.post(
        "/v1/metas",
        content=get_resp.content,
        headers={"content-type": "application/json"},
    )
    assert load_resp.status_code == 200
    ps_mock.load_metas.assert_called_once_with(fake_metas)
