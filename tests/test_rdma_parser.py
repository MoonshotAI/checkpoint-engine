import os
from unittest.mock import patch

import pytest

from checkpoint_engine.ps import (
    _get_my_rdma_device,
    _get_rdma_devices,
    _ibv_get_device_list,
    _parse_NCCL_IB_HCA,
)


@pytest.fixture
def mock_available_devices() -> list[str]:
    """Provide mock available device list"""
    return ["mlx5_0", "mlx5_1", "mlx4_0", "mlx4_1"]


def test_detect_ibv_list():
    """Test detection of _ibv_get_device_list function"""
    # Skip this test if no real infiniband devices exist
    if not os.path.exists("/sys/class/infiniband"):
        pytest.skip("No infiniband devices found on system")

    real_ibv_list = sorted(os.listdir("/sys/class/infiniband"))
    if real_ibv_list:
        devices = _ibv_get_device_list()
        assert isinstance(devices, list)


def test_parse_max_hcas_limit():
    """Test maximum HCA quantity limit"""
    # Create mock data with more than 32 devices
    many_devices = [f"device_{i}" for i in range(50)]
    result = _parse_NCCL_IB_HCA("", many_devices)
    assert len(result) == 32
    assert result == many_devices[:32]


def test_get_rdma_devices_no_env_vars(mock_available_devices: list[str]):
    """Test _get_rdma_devices with no environment variables"""
    with (
        patch.dict(os.environ, clear=True),
        patch("checkpoint_engine.ps._ibv_get_device_list", return_value=mock_available_devices),
    ):
        devices = _get_rdma_devices()
        assert sorted(devices) == sorted(mock_available_devices)


@pytest.mark.parametrize(
    "input_value,expected",
    [
        pytest.param("", ["mlx5_0", "mlx5_1", "mlx4_0", "mlx4_1"], id="empty string"),
        pytest.param("   \t\n  ", ["mlx5_0", "mlx5_1", "mlx4_0", "mlx4_1"], id="whitespace"),
        pytest.param("None", [], id="None string"),
        pytest.param("^", ["mlx5_0", "mlx5_1", "mlx4_0", "mlx4_1"], id="caret"),
        pytest.param("^=", ["mlx5_0", "mlx5_1", "mlx4_0", "mlx4_1"], id="caret-equals"),
        pytest.param("=^", [], id="equals-caret"),
        pytest.param("^^", ["mlx5_0", "mlx5_1", "mlx4_0", "mlx4_1"], id="double-caret"),
        pytest.param("=", [], id="equals"),
        pytest.param("==", [], id="double-equals"),
    ],
)
def test_parse_basic_cases(
    input_value: str, expected: list[str], mock_available_devices: list[str]
):
    """Test basic parsing cases: empty string, whitespace, None"""
    result = _parse_NCCL_IB_HCA(input_value, mock_available_devices)
    assert result == expected


@pytest.mark.parametrize(
    "input_value,expected",
    [
        # prefix
        ("mlx5_0", ["mlx5_0"]),
        ("mlx5", ["mlx5_0", "mlx5_1"]),
        # exact match
        ("=mlx5_0", ["mlx5_0"]),
        ("=mlx5_0,mlx5_1", ["mlx5_0", "mlx5_1"]),
        # ignore ports, whitespace and duplicated commas
        ("mlx5_0:1,mlx5_1:2", ["mlx5_0", "mlx5_1"]),
        ("mlx5_0:1,mlx5_1", ["mlx5_0", "mlx5_1"]),
        (" mlx5_0 , mlx5_1 ", ["mlx5_0", "mlx5_1"]),
        ("mlx5_0,,mlx5_1", ["mlx5_0", "mlx5_1"]),
        # exclusion
        ("^mlx5_0", ["mlx5_1", "mlx4_0", "mlx4_1"]),
        ("^mlx5_0,mlx5_1", ["mlx4_0", "mlx4_1"]),
        ("^mlx5", ["mlx4_0", "mlx4_1"]),
        ("^=mlx5_0,mlx5_1", ["mlx4_0", "mlx4_1"]),
        ("^=mlx4", ["mlx5_0", "mlx5_1", "mlx4_0", "mlx4_1"]),
    ],
)
def test_parse_various_patterns(
    input_value: str, expected: list[str], mock_available_devices: list[str]
):
    """Test various parsing patterns"""
    result = _parse_NCCL_IB_HCA(input_value, mock_available_devices)
    assert result == expected


@pytest.mark.parametrize(
    "input_value,expected_result,expected_warning",
    [
        ("=mlx5_100", [], "No RDMA device match device_name='mlx5_100' where is_exact_match=True."),
        ("mlx5_100", [], "No RDMA device match device_name='mlx5_100' where is_exact_match=False."),
        (
            "^mlx5_100",
            ["mlx5_0", "mlx5_1", "mlx4_0", "mlx4_1"],
            "No RDMA device match device_name='mlx5_100' where is_exact_match=False.",
        ),
        ("mlx6", [], "No RDMA device match device_name='mlx6' where is_exact_match=False."),
        ("=mlx6", [], "No RDMA device match device_name='mlx6' where is_exact_match=True."),
    ],
)
def test_parse_exact_match_with_nonexistent_device(
    input_value: str,
    expected_result: list[str],
    expected_warning: str,
    mock_available_devices: list[str],
):
    """Test exact matching with non-existent device"""
    with patch("checkpoint_engine.ps.logger") as mock_logger:
        result = _parse_NCCL_IB_HCA(input_value, mock_available_devices)
        assert result == expected_result
        mock_logger.warning.assert_called_once_with(expected_warning)


@pytest.mark.parametrize(
    "env_var_name,env_var_value,expected_devices",
    [
        ("PS_P2P_STORE_RDMA_DEVICES", "mlx5_0,mlx5_1", ["mlx5_0", "mlx5_1"]),
        ("NCCL_IB_HCA", "mlx5", ["mlx5_0", "mlx5_1"]),
        ("NCCL_IB_HCA", "mlx5_0,mlx5_1", ["mlx5_0", "mlx5_1"]),
        ("NCCL_IB_HCA", "^mlx5_0", ["mlx5_1", "mlx4_0", "mlx4_1"]),
        ("NCCL_IB_HCA", "mlx6", ["mlx5_0", "mlx5_1", "mlx4_0", "mlx4_1"]),
        ("NCCL_IB_HCA", "", ["mlx5_0", "mlx5_1", "mlx4_0", "mlx4_1"]),
    ],
)
def test_get_rdma_devices_with_env_vars(
    env_var_name: str,
    env_var_value: str,
    expected_devices: list[str],
    mock_available_devices: list[str],
):
    """Test _get_rdma_devices with various environment variables"""
    env_dict = {env_var_name: env_var_value}
    with (
        patch.dict(os.environ, env_dict),
        patch("checkpoint_engine.ps._ibv_get_device_list", return_value=mock_available_devices),
    ):
        devices = _get_rdma_devices()
        assert sorted(devices) == sorted(expected_devices)


@pytest.mark.parametrize(
    "local_rank,gpu_count,expected_device",
    [
        (0, 4, "mlx5_0"),
        (3, 4, "mlx5_3"),
        (4, 8, "mlx5_2"),
        (7, 8, "mlx5_3"),
    ],
)
def test_get_my_rdma_device_basic(local_rank: int, gpu_count: int, expected_device: str):
    """Test _get_my_rdma_device with basic allocation"""
    # Use fewer devices to match the GPU count constraint
    devices = ["mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3"]
    device = _get_my_rdma_device(local_rank, gpu_count, devices)
    assert device == expected_device


@pytest.mark.parametrize(
    "local_rank,gpu_count,devices,error",
    [
        (
            0,
            4,
            ["mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3", "mlx5_4"],
            AssertionError,
        ),  # Too many devices
        (
            0,
            8,
            ["mlx5_0", "mlx5_1", "mlx5_2"],
            AssertionError,
        ),  # GPU count not divisible by device count
        (0, 8, [], RuntimeError),  # No devices
    ],
)
def test_get_my_rdma_device_invalid_config(
    local_rank: int, gpu_count: int, devices: list[str], error: type
):
    """Test _get_my_rdma_device with invalid configuration"""
    with pytest.raises(error):
        _get_my_rdma_device(local_rank, gpu_count, devices)
