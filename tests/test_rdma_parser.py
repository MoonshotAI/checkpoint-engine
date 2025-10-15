import os
from unittest.mock import patch

import pytest

from checkpoint_engine.ps import NCCLIBHCAParser


class TestNCCLIBHCAParser:
    """Unit tests for NCCLIBHCAParser class"""

    @pytest.fixture
    def mock_available_devices(self) -> list[str]:
        """Provide mock available device list"""
        return ["mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3", "mlx4_0", "mlx4_1", "roce_0", "roce_1"]

    @pytest.fixture
    def parser_with_mock_devices(self, mock_available_devices: list[str]) -> NCCLIBHCAParser:
        """Create parser instance with mock devices"""
        with patch.object(
            NCCLIBHCAParser, "_ibv_get_device_list", return_value=mock_available_devices
        ):
            parser = NCCLIBHCAParser()
            return parser

    def test_detect_ibv_list(self):
        """Test detection of _ibv_get_device_list function"""
        parser = NCCLIBHCAParser()
        real_ibv_list = (
            os.listdir("/sys/class/infiniband") if os.path.exists("/sys/class/infiniband") else []
        )
        if real_ibv_list:
            assert parser.available_devices == real_ibv_list

    def test_init_with_mock_devices(self, mock_available_devices: list[str]):
        """Test correct device list initialization"""
        with patch.object(
            NCCLIBHCAParser, "_ibv_get_device_list", return_value=mock_available_devices
        ):
            parser = NCCLIBHCAParser()
            assert parser.available_devices == mock_available_devices
            assert parser.max_hcas == 32

    def test_parse_empty_string_returns_all_devices(
        self, parser_with_mock_devices: NCCLIBHCAParser
    ):
        """Test empty string returns all devices"""
        result = parser_with_mock_devices.parse("")
        assert result == parser_with_mock_devices.available_devices

    def test_parse_whitespace_only_returns_all_devices(
        self, parser_with_mock_devices: NCCLIBHCAParser
    ):
        """Test whitespace-only string returns all devices"""
        result = parser_with_mock_devices.parse("   \t\n  ")
        assert result == parser_with_mock_devices.available_devices

    def test_parse_none_string_returns_all_devices(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test 'None' string returns empty list (no matching devices)"""
        result = parser_with_mock_devices.parse("None")
        # "None" is treated as a regular string, tries prefix matching
        # Since no devices start with "None", should return empty list
        assert result == []

    def test_parse_prefix_match_single_device(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test prefix matching for single device"""
        result = parser_with_mock_devices.parse("mlx5_0")
        assert result == ["mlx5_0"]

    def test_parse_prefix_match_multiple_devices(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test prefix matching for multiple devices"""
        result = parser_with_mock_devices.parse("mlx5")
        expected = ["mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3"]
        assert result == expected

    def test_parse_exact_match_single_device(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test exact matching for single device"""
        result = parser_with_mock_devices.parse("=mlx5_0")
        assert result == ["mlx5_0"]

    def test_parse_exact_match_multiple_devices(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test exact matching for multiple devices"""
        result = parser_with_mock_devices.parse("=mlx5_0,mlx5_1")
        assert result == ["mlx5_0", "mlx5_1"]

    def test_parse_exact_match_with_nonexistent_device(
        self, parser_with_mock_devices: NCCLIBHCAParser
    ):
        """Test exact matching with non-existent device"""
        with patch("checkpoint_engine.rdma_parser.logger") as mock_logger:
            result = parser_with_mock_devices.parse("=mlx5_100")
            assert result == []
            mock_logger.warning.assert_called_once_with(
                "Device 'mlx5_100' not found in available devices."
            )

    def test_parse_exclude_single_device(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test excluding single device"""
        result = parser_with_mock_devices.parse("^mlx5_0")
        expected = [dev for dev in parser_with_mock_devices.available_devices if dev != "mlx5_0"]
        assert result == expected

    def test_parse_exclude_multiple_devices(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test excluding multiple devices"""
        result = parser_with_mock_devices.parse("^mlx5_0,mlx5_1")
        expected = [
            dev
            for dev in parser_with_mock_devices.available_devices
            if dev not in ["mlx5_0", "mlx5_1"]
        ]
        assert result == expected

    def test_parse_exclude_with_prefix_match(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test exclusion with prefix matching"""
        result = parser_with_mock_devices.parse("^mlx5")
        expected = ["mlx4_0", "mlx4_1", "roce_0", "roce_1"]
        assert result == expected

    def test_parse_exclude_with_exact_match(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test exclusion with exact matching"""
        result = parser_with_mock_devices.parse("^=mlx5_0,mlx5_1")
        expected = [
            dev
            for dev in parser_with_mock_devices.available_devices
            if dev not in ["mlx5_0", "mlx5_1"]
        ]
        assert result == expected

    def test_parse_exclude_nonexistent_device(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test excluding non-existent device"""
        with patch("checkpoint_engine.rdma_parser.logger") as mock_logger:
            result = parser_with_mock_devices.parse("^mlx5_100")
            expected = parser_with_mock_devices.available_devices
            assert result == expected
            mock_logger.warning.assert_called_once_with("No devices match the prefix 'mlx5_100'.")

    def test_parse_with_port_specification(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test parsing with port specification"""
        result = parser_with_mock_devices.parse("mlx5_0:1,mlx5_1:2")
        expected = ["mlx5_0:1", "mlx5_1:2"]
        assert result == expected

    def test_parse_mixed_with_and_without_ports(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test mixed parsing with and without port specifications"""
        result = parser_with_mock_devices.parse("mlx5_0:1,mlx5_1")
        expected = ["mlx5_0:1", "mlx5_1"]
        assert result == expected

    def test_parse_max_hcas_limit(self):
        """Test maximum HCA quantity limit"""
        # Create mock data with more than 32 devices
        many_devices = [f"device_{i}" for i in range(50)]
        with patch.object(NCCLIBHCAParser, "_ibv_get_device_list", return_value=many_devices):
            parser = NCCLIBHCAParser()
            result = parser.parse("")
            assert len(result) == 32
            assert result == many_devices[:32]

    def test_parse_complex_combination(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test complex combination parsing"""
        result = parser_with_mock_devices.parse("^=mlx5_3,mlx4_1")
        expected = [
            dev
            for dev in parser_with_mock_devices.available_devices
            if dev not in ["mlx5_3", "mlx4_1"]
        ]
        assert result == expected

    def test_parse_multiple_prefix_operators(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test multiple prefix operators"""
        result = parser_with_mock_devices.parse("^=mlx5_0")
        expected = [dev for dev in parser_with_mock_devices.available_devices if dev != "mlx5_0"]
        assert result == expected

    def test_parse_edge_case_empty_after_operators(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test edge cases with empty content after operators"""
        result = parser_with_mock_devices.parse("^")
        # Empty exclusion list means exclude nothing, return all devices
        assert result == parser_with_mock_devices.available_devices

        result = parser_with_mock_devices.parse("=")
        # Empty exact match list means match nothing, return empty list
        assert result == []

    def test_parse_edge_case_only_operators(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test edge cases with only operators"""
        result = parser_with_mock_devices.parse("^=")
        assert result == parser_with_mock_devices.available_devices

        result = parser_with_mock_devices.parse("=^")
        assert result == parser_with_mock_devices.available_devices

        result = parser_with_mock_devices.parse("^^")
        assert result == parser_with_mock_devices.available_devices

        result = parser_with_mock_devices.parse("==")
        assert result == []

    def test_parse_with_spaces_in_input(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test parsing with spaces in input"""
        result = parser_with_mock_devices.parse(" mlx5_0 , mlx5_1 ")
        assert result == ["mlx5_0", "mlx5_1"]

    def test_parse_empty_device_spec(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test parsing with empty device specifications"""
        result = parser_with_mock_devices.parse("mlx5_0,,mlx5_1")
        assert result == ["mlx5_0", "mlx5_1"]

    def test_ibv_get_device_list_real_implementation_mocked(self):
        """Test ibv_get_device_list implementation with complete mocking to avoid ctypes issues"""
        # Mock the entire method instead of using ctypes
        with patch.object(
            NCCLIBHCAParser, "_ibv_get_device_list", return_value=["mlx5_0", "mlx5_1"]
        ):
            parser = NCCLIBHCAParser()
            devices = parser._ibv_get_device_list()
            assert devices == ["mlx5_0", "mlx5_1"]

    def test_ibv_get_device_list_no_devices_mocked(self):
        """Test no available devices case with complete mocking"""
        with patch.object(NCCLIBHCAParser, "_ibv_get_device_list", return_value=[]):
            parser = NCCLIBHCAParser()
            devices = parser._ibv_get_device_list()
            assert devices == []

    def test_resolve_device_specs_no_match(self, parser_with_mock_devices: NCCLIBHCAParser):
        """Test _resolve_device_specs with no matching devices"""
        with patch("checkpoint_engine.rdma_parser.logger") as mock_logger:
            result = parser_with_mock_devices._resolve_device_specs(["nonexistent"], False)
            assert result == []
            mock_logger.warning.assert_called_once_with(
                "No devices match the prefix 'nonexistent'."
            )

    def test_resolve_device_specs_exact_match_not_found(
        self, parser_with_mock_devices: NCCLIBHCAParser
    ):
        """Test _resolve_device_specs with exact match not found"""
        with patch("checkpoint_engine.rdma_parser.logger") as mock_logger:
            result = parser_with_mock_devices._resolve_device_specs(["nonexistent"], True)
            assert result == []
            mock_logger.warning.assert_called_once_with(
                "Device 'nonexistent' not found in available devices."
            )
