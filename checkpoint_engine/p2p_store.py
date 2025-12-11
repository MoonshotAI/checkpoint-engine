import ctypes
import os
import random
import time

import torch
from loguru import logger

from checkpoint_engine.device_utils import DeviceManager, get_ip


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
        assert len(devices) <= gpu_count, (
            f"rdma devices count {len(devices)} should be less than or equal to gpu count {gpu_count}"
        )
        assert gpu_count % len(devices) == 0, (
            f"gpu count {gpu_count} should be divisible by rdma devices count {len(devices)}"
        )
        return devices[local_rank // (gpu_count // len(devices))]
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


class P2PStore:
    def __init__(self, device_manager: DeviceManager):
        from mooncake.engine import TransferEngine

        self.rank = int(os.getenv("RANK"))
        gpu_count = device_manager.device_module.device_count()
        local_rank = self.rank % gpu_count
        device_type = device_manager.device_type
        if device_type == "npu" and os.getenv("PS_P2P_STORE_RDMA_DEVICES") is None:
            self.device = ""
        else:
            self.device = _get_my_rdma_device(local_rank, gpu_count, _get_rdma_devices())
        self.ip = get_ip()

        # we will start at most 8 ps processes, so we use 8 retries to avoid port conflicts in extreme cases
        retry_count = 8
        for i in range(retry_count):
            self.engine = TransferEngine()
            ret = self.engine.initialize(
                self.ip,
                "P2PHANDSHAKE",
                "ascend_direct" if device_type == "npu" else "rdma",
                self.device,
            )
            if ret == 0:
                break
            # sleep 0.5 ~ 2.0s, to avoid port conflicts when two processes retry at the same time
            sleep_ms = random.randint(500, 2000)
            logger.warning(
                f"[rank{self.rank}] fail to initialize transfer engine, ret {ret}, retry {i + 1}/{retry_count} in {sleep_ms}ms"
            )
            time.sleep(sleep_ms / 1000)
        else:
            raise RuntimeError(f"[rank{self.rank}] fail to initialize transfer engine")
        self.port = self.engine.get_rpc_port()
        self.named_tensors: dict[str, torch.Tensor] = {}
        logger.info(
            f"[rank{self.rank}] p2p store initialized, addr is {self.addr}, rdma device is {self.device}"
        )

    @property
    def addr(self) -> str:
        return f"{self.ip}:{self.port}"

    def register_named_tensors(self, named_tensors: dict[str, torch.Tensor]):
        buffer_addresses = [tensor.data_ptr() for tensor in named_tensors.values()]
        capacities = [tensor.nbytes for tensor in named_tensors.values()]
        self.named_tensors.update(named_tensors)
        for i, name in enumerate(named_tensors.keys()):
            logger.info(
                f"[rank{self.rank}] p2p store register tensor {name} with addr {hex(buffer_addresses[i])} and capacity {capacities[i]}"
            )
        assert self.engine.batch_register_memory(buffer_addresses, capacities) == 0

    def unregister_named_tensors(self, names: list[str]) -> int:
        buffer_addresses = [self.named_tensors[name].data_ptr() for name in names]
        assert self.engine.batch_unregister_memory(buffer_addresses) == 0
        num_unregistered = 0
        for i, name in enumerate(names):
            del self.named_tensors[name]
            logger.info(
                f"[rank{self.rank}] p2p store unregister tensor {name} with addr {hex(buffer_addresses[i])}"
            )
            num_unregistered += 1
        return num_unregistered

    def batch_transfer_sync_read(
        self, target_hostname: str, buf_ptrs: list[int], remote_ptrs: list[int], lens: list[int]
    ):
        assert (
            self.engine.batch_transfer_sync_read(target_hostname, buf_ptrs, remote_ptrs, lens) == 0
        )
