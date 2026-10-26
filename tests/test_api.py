"""CPU-only tests for the metas endpoints in api.py."""

from types import TracebackType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import TypeAdapter
from typing_extensions import Self

from checkpoint_engine.api import _init_api, request_inference_to_update
from checkpoint_engine.data_types import (
    MemoryBufferMetaList,
    MemoryBufferMetas,
    ParameterMeta,
)
from checkpoint_engine.ps import ParameterServer


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


def test_load_metas_filters_empty_owners(fake_metas: dict[int, MemoryBufferMetaList]) -> None:
    ps = ParameterServer.__new__(ParameterServer)
    empty_meta = MemoryBufferMetaList(
        p2p_store_addr="192.168.1.2:12345",
        rdma_device="mlx5_2",
        memory_buffer_metas_list=[],
    )

    ps.load_metas({**fake_metas, 2: empty_meta})

    assert ps.get_metas() == fake_metas
    assert all(2 not in ranks for ranks in ps._remote_rdma_devices.values())


def test_request_inference_to_update_closes_httpx_client() -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

    class FakeClient:
        closed = False
        payload = None

        def __init__(self, *, transport: Any):
            self.transport = transport

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            type(self).closed = True

        def post(self, url: str, *, json: dict[str, Any], timeout: float) -> FakeResponse:
            type(self).payload = (url, json, timeout)
            return FakeResponse()

    with (
        patch("checkpoint_engine.api.httpx.HTTPTransport", return_value="transport"),
        patch("checkpoint_engine.api.httpx.Client", FakeClient),
    ):
        request_inference_to_update("http://example/update", {"GPU-0": "ipc://x"}, timeout=1.5)

    assert FakeClient.closed is True
    assert FakeClient.payload == (
        "http://example/update",
        {
            "method": "update_weights_from_ipc",
            "args": [{"GPU-0": "ipc://x"}],
            "timeout": 1.5,
        },
        1.5,
    )
