import ctypes
import os
import re
import socket
import subprocess
from functools import lru_cache

import torch
from loguru import logger


@lru_cache(maxsize=1)
def get_ip() -> str:
    try:
        # try to get ip from network interface
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as e:  # noqa: BLE001
        # fallback to get ip from hostname
        logger.warning(
            f"fail to get ip from network interface, fallback to get ip from hostname: {e}"
        )
        return socket.gethostbyname(socket.gethostname())


def npu_generate_uuid() -> str:
    str_pid = str(os.getpid())
    npu_num = 8
    try:
        for npu_id in range(npu_num):
            cmd = ["npu-smi", "info", "-t", "proc-mem", "-i", str(npu_id)]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
            str_result = str(result.stdout)
            if str_pid in str_result:
                # In A3 server, one NPU has two chips.
                match_chip_count = re.search(r"Chip Count[^\d]*(\d+)", str_result)
                chip_count = int(match_chip_count.group(1))
                search_after_pid = str_result[str_result.find(str_pid) + len(str_pid) :]
                match_chip_id = re.search(r"Chip ID[^\d]*(\d+)", search_after_pid)
                chip_id = int(match_chip_id.group(1))
                return f"{get_ip()}-{npu_id * chip_count + chip_id}"
        raise ValueError("The current process is not running on the npu device")
    except subprocess.CalledProcessError as e:
        raise ValueError("The current process is not running on the npu device") from e


def _ibv_get_device_list() -> list[str]:
    lib = ctypes.CDLL("libibverbs.so.1")
    lib.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]  # int *num_devices
    lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)  # struct ibv_device **

    lib.ibv_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.ibv_get_device_name.argtypes = [ctypes.c_void_p]  # struct ibv_device *
    lib.ibv_get_device_name.restype = ctypes.c_char_p  # const char *

    num = ctypes.c_int()
    dev_array = lib.ibv_get_device_list(ctypes.byref(num))
    if not dev_array or num.value <= 0:
        return []

    devices = []
    for i in range(num.value):
        dev_ptr = dev_array[i]  # struct ibv_device *
        name = lib.ibv_get_device_name(dev_ptr)  # const char *
        devices.append(name.decode())
    lib.ibv_free_device_list(dev_array)
    return devices


def _get_rdma_devices() -> list[str]:
    """
    use _ibv_get_device_list to get RDMA devices, if NCCL_IB_HCA has multiple values, just return
    """
    devices_str = os.getenv("PS_P2P_STORE_RDMA_DEVICES")
    if devices_str:
        return devices_str.split(",")
    # if PS_P2P_STORE_RDMA_DEVICES is not set, try to use NCCL_IB_HCA to get RDMA devices
    hca = os.getenv("NCCL_IB_HCA", None)
    return _parse_NCCL_IB_HCA(hca or "", _ibv_get_device_list()) or _ibv_get_device_list()


def _get_my_rdma_device(local_rank: int, gpu_count: int, devices: list[str]) -> str:
    """
    implement network card device allocation, if network card is "mlx5_0,mlx5_1", then 0-3 will share mlx5_0, 4-7 will share mlx5_1, etc.
    """
    if not devices:
        raise RuntimeError("no rdma devices found")
    try:
        if len(devices) <= gpu_count:
            assert gpu_count % len(devices) == 0, (
                f"gpu count {gpu_count} should be divisible by rdma devices count {len(devices)}"
            )
            return devices[local_rank // (gpu_count // len(devices))]
        else:
            assert len(devices) % gpu_count == 0, (
                f"rdma devices count {len(devices)} should be divisible by gpu count {gpu_count}"
            )
            device_per_rank = len(devices) // gpu_count
            return ",".join(
                devices[local_rank * device_per_rank : (local_rank + 1) * device_per_rank]
            )
    except AssertionError:
        logger.error(
            "Please set 'NCCL_IB_HCA' or 'PS_P2P_STORE_RDMA_DEVICES' environment variable to choose proper number of RDMA devices."
            "The number of RDMA devices should be less than or equal to GPU count, and GPU count should be divisible by the number of RDMA devices."
            "The acceptable value by NCCL_IB_HCA is documented in 'https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#id8'."
        )
        raise


def _parse_NCCL_IB_HCA(value: str, available_devices: list[str]) -> list[str]:
    """
    The acceptable value by NCCL_IB_HCA is documented in https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#id8.
    The Python version parser is referred to the CPP parser in NCCL: https://github.com/NVIDIA/nccl/blob/v2.28.3-1/src/transport/net_ib.cc#L658-L662.

    The list is comma-separated; port numbers are NOT supported yet.
    An optional prefix '^' indicates the list is an exclude list.
    A second optional prefix '=' indicates that the tokens are exact names, otherwise by default NCCL would treat each token as a prefix.
    Please note that when '^' and '=' appear together, only '^=' is allowed, '=^' is not supported.

    Examples:
    - `NCCL_IB_HCA="mlx5"`: Use all cards starting with `mlx5`.
    - `NCCL_IB_HCA="=mlx5_0,mlx5_1"`: Use specific cards `mlx5_0` and `mlx5_1`.
    - `NCCL_IB_HCA="^mlx5"`: Use all cards except those starting with `mlx5`.
    - `NCCL_IB_HCA="^=mlx5_0,mlx5_1"`: Use all cards except `mlx5_0` and `mlx5_1`.
    """
    max_hcas = 32
    if not value or value.strip() == "":
        return available_devices[:max_hcas]

    value = value.strip()
    result = []
    is_exclude = value.startswith("^")
    if is_exclude:
        value = value.removeprefix("^")
    is_exact_match = value.startswith("=")
    if is_exact_match:
        value = value.removeprefix("=")

    device_specs = [spec.strip() for spec in value.split(",") if spec.strip()]

    result = _resolve_device_specs(device_specs, is_exact_match, available_devices)
    if is_exclude:
        result = [dev for dev in available_devices if dev not in result]
    if len(result) > max_hcas:
        result = result[:max_hcas]

    logger.info(f"RDMA Devices from 'NCCL_IB_HCA': {result}")

    return result


def _resolve_device_specs(
    device_specs: list[str], is_exact_match: bool, available_devices: list[str]
) -> list[str]:
    devices = set()
    for spec in device_specs:
        parts = spec.split(":", 1)
        device_name = parts[0].strip()
        # HACK: mooncake transfer engine does not support port specification yet, so we ignore it
        # port = parts[1].strip() if len(parts) > 1 else None
        base_devices = (
            [device_name]
            if device_name in available_devices
            else []
            if is_exact_match
            else [dev for dev in available_devices if dev.startswith(device_name)]
        )

        if not base_devices:
            logger.warning(f"No RDMA device match {device_name=} where {is_exact_match=}.")
            continue

        for base_dev in base_devices:
            devices.add(base_dev)

    return sorted(devices)


class DeviceManager:
    def __init__(self):
        self.device_type = self._detect_device_type()
        self._setup_device_module()

    def _is_torch_npu_available(self) -> bool:
        try:
            if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)):
                return torch.npu.is_available()
            else:
                return False
        except ImportError:
            return False

    def _detect_device_type(self) -> str:
        if self._is_torch_npu_available():
            return "npu"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            raise TypeError("The current device type is not supported")

    def _setup_device_module(self):
        if self.device_type == "npu":
            import torch_npu

            self.device_module = torch_npu.npu
        elif self.device_type == "cuda":
            self.device_module = torch.cuda
        else:
            raise TypeError("The current device type is not supported")

    @property
    def backend(self) -> str:
        if self.device_type == "npu":
            return "hccl"
        elif self.device_type == "cuda":
            return "nccl"
        else:
            raise TypeError("The current device type is not supported")

    @property
    def transfer_engine_protocol(self) -> str:
        if self.device_type == "npu":
            return "ascend_direct"
        elif self.device_type == "cuda":
            return "rdma"
        else:
            raise TypeError("The current device type is not supported")

    def rdma_device(self, rank: int) -> str:
        if self.transfer_engine_protocol == "ascend_direct":
            return ""
        elif self.transfer_engine_protocol == "rdma":
            return _get_my_rdma_device(rank, self.device_module.device_count(), _get_rdma_devices())
        else:
            raise TypeError("The current transfer engine protocol is not supported")
