import os
import random
import time

import torch
from loguru import logger

from checkpoint_engine.device_utils import DeviceManager, get_ip


class P2PStore:
    def __init__(self, device_manager: DeviceManager):
        from mooncake.engine import TransferEngine

        self.rank = int(os.environ["RANK"])  # ENV RANK is required
        gpu_count = device_manager.device_module.device_count()
        local_rank = self.rank % gpu_count
        self.device = device_manager.rdma_device(local_rank)
        self.ip = get_ip()

        # we will start at most 8 ps processes, so we use 8 retries to avoid port conflicts in extreme cases
        retry_count = 8
        for i in range(retry_count):
            self.engine = TransferEngine()
            ret = self.engine.initialize(
                self.ip,
                "P2PHANDSHAKE",
                device_manager.transfer_engine_protocol,
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
