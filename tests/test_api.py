"""CPU-only tests for the metas endpoints in api.py."""

import pickle
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from checkpoint_engine.api import _init_api


@pytest.fixture
def fake_metas() -> dict:
    # Mimic ParameterServer.get_metas() return shape (dict[int, ...]),
    # exact value type doesn't matter for the round-trip test.
    return {0: {"foo": [1, 2, 3]}, 1: {"bar": "baz"}}


@pytest.fixture
def ps_mock(fake_metas: dict) -> MagicMock:
    ps = MagicMock()
    ps.get_metas.return_value = fake_metas
    return ps


def test_get_metas_returns_pickle_bytes(ps_mock: MagicMock, fake_metas: dict) -> None:
    client = TestClient(_init_api(ps_mock))
    resp = client.get("/v1/checkpoints/my-ckpt/metas")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    assert pickle.loads(resp.content) == fake_metas
    ps_mock.get_metas.assert_called_once_with()


def test_get_metas_propagates_ps_error(ps_mock: MagicMock) -> None:
    ps_mock.get_metas.side_effect = RuntimeError("metas not gathered yet")
    client = TestClient(_init_api(ps_mock))
    resp = client.get("/v1/checkpoints/my-ckpt/metas")
    assert resp.status_code == 500
    assert "metas not gathered yet" in resp.text


def test_load_metas_decodes_and_calls_ps(ps_mock: MagicMock, fake_metas: dict) -> None:
    client = TestClient(_init_api(ps_mock))
    resp = client.post(
        "/v1/checkpoints/my-ckpt/load-metas",
        content=pickle.dumps(fake_metas),
        headers={"content-type": "application/octet-stream"},
    )
    assert resp.status_code == 200
    ps_mock.load_metas.assert_called_once_with(fake_metas)


def test_load_metas_rejects_bad_pickle(ps_mock: MagicMock) -> None:
    client = TestClient(_init_api(ps_mock))
    resp = client.post(
        "/v1/checkpoints/my-ckpt/load-metas",
        content=b"not a valid pickle",
    )
    assert resp.status_code == 400
    ps_mock.load_metas.assert_not_called()


def test_load_metas_propagates_ps_error(ps_mock: MagicMock, fake_metas: dict) -> None:
    ps_mock.load_metas.side_effect = RuntimeError("rdma device mismatch")
    client = TestClient(_init_api(ps_mock))
    resp = client.post(
        "/v1/checkpoints/my-ckpt/load-metas",
        content=pickle.dumps(fake_metas),
    )
    assert resp.status_code == 500
    assert "rdma device mismatch" in resp.text


def test_round_trip_get_then_load(ps_mock: MagicMock, fake_metas: dict) -> None:
    """Pickle bytes returned by GET /metas must be accepted by POST /load-metas."""
    client = TestClient(_init_api(ps_mock))
    get_resp = client.get("/v1/checkpoints/source/metas")
    assert get_resp.status_code == 200
    load_resp = client.post(
        "/v1/checkpoints/dest/load-metas",
        content=get_resp.content,
    )
    assert load_resp.status_code == 200
    ps_mock.load_metas.assert_called_once_with(fake_metas)
