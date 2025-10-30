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
        ValueError("The current process is not running on the npu device")
    except subprocess.CalledProcessError:
        ValueError("The current process is not running on the npu device")


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
