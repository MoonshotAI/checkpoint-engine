import os
import subprocess
import time

import pytest
import torch.distributed as dist
from test_update import device_manager, gen_test_tensors, get_world_size

from checkpoint_engine.ps import ParameterServer


def get_files() -> list[str]:
    rank = int(os.getenv("RANK"))
    named_tensors = dict(gen_test_tensors(rank))
    import safetensors.torch

    files = []
    dev_shm_dir = "/dev/shm/checkpoint_engine_tests"  # noqa: S108
    os.makedirs(dev_shm_dir, exist_ok=True)
    tensors_in_dev_shm = named_tensors
    time.sleep(1)
    dev_shm_files = [
        os.path.join(dev_shm_dir, f"rank{rank}_checkpoint.safetensors")
        for _ in range(get_world_size())
    ]
    safetensors.torch.save_file(tensors_in_dev_shm, dev_shm_files[rank])
    time.sleep(1)
    files.append(dev_shm_files[rank])
    return files


def run_pin_and_unpin(num_runs: int):
    ps = ParameterServer(auto_pg=True)
    checkpoint_name = "test_with_files"
    for _ in range(num_runs):
        files = get_files()
        ps.register_checkpoint(checkpoint_name, files=files)
        ps.gather_metas(checkpoint_name)
        dist.barrier()
        ps.unregister_checkpoint(checkpoint_name)
    dist.destroy_process_group()


@pytest.mark.gpu
def test_unpin_files():
    world_size = device_manager.device_module.device_count()
    assert world_size >= 2, "This test requires at least 2 GPUs."
    master_addr = "localhost"
    master_port = 25400
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(world_size),
        "--master_addr",
        master_addr,
        "--master_port",
        str(master_port),
        __file__,
    ]

    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=False,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        shell=False,
        check=False,
    )

    assert result.returncode == 0


if __name__ == "__main__":
    run_pin_and_unpin(3)
