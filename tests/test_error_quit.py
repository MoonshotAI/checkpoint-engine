import os
import random
import subprocess
import time

import pytest
import torch
import zmq
from torch.multiprocessing import Queue, get_context

from checkpoint_engine.ps import ParameterServer, _get_physical_gpu_id
from checkpoint_engine.worker import update_weights_from_ipc


def gen_test_tensors(rank: int) -> list[tuple[str, torch.Tensor]]:
    tensors = []
    for layer in range(random.randint(10, 50)):
        for num in range(random.randint(50, 100)):
            r = random.randint(0, 16)
            if r < 4:
                dtype = torch.bfloat16
            elif r < 10:
                dtype = torch.float16
            elif r < 14:
                dtype = torch.float8_e4m3fn
            else:
                dtype = torch.float
            tensors.append(
                (
                    f"rank{rank}.layer{layer}.num{num}",
                    torch.randn([random.randint(100, 500), random.randint(500, 1000)]).to(dtype),
                )
            )
    return tensors


def receiver_proc_with_error(
    rank: int, device_uuid: str, named_tensors: dict[str, torch.Tensor], queue: Queue
):
    torch.cuda.set_device(rank)
    named_tensors = {name: tensor.cuda() for name, tensor in named_tensors.items()}
    _zmq_ctx = zmq.Context()

    def trigger_error(socket_paths: list[tuple[str, str]]):
        socket_paths = dict(socket_paths)
        update_weights_from_ipc(
            _zmq_ctx,
            socket_paths[device_uuid],
            device_id=rank,
            run=error_run,
            post_hook=lambda: torch.cuda.synchronize(),
        )

    def error_run(weights: list[tuple[str, torch.Tensor]]):
        weights = weights  # Do some fake processing
        time.sleep(random.uniform(0.1, 0.5))
        if random.random() < 0.6:
            raise RuntimeError("Intentional Error for testing.")

    while True:
        socket_paths: list[tuple[str, str]] = queue.get()
        if socket_paths is None:
            break
        try:
            trigger_error(socket_paths)
        except RuntimeError:
            print(f"[rank{rank}] successfully triggered error.")
            raise


def run():
    rank = int(os.getenv("RANK"))
    ctx = get_context("spawn")
    queue = ctx.Queue()
    _device_uuid = _get_physical_gpu_id(rank)
    ps = ParameterServer(auto_pg=True)
    named_tensors = dict(gen_test_tensors(rank))
    checkpoint_name = "test"
    proc = ctx.Process(
        target=receiver_proc_with_error, args=(rank, _device_uuid, named_tensors, queue)
    )
    proc.daemon = True
    proc.start()
    try:
        ps.register_checkpoint(checkpoint_name, named_tensors=named_tensors)
        ps.gather_metas(checkpoint_name)
        ranks = []
        ps.update(checkpoint_name, queue.put, ranks=ranks)
        # sleep 3s to wait process group is destroyed
        time.sleep(3)
    except RuntimeError as e:
        print(f"[rank{rank}] Caught expected RuntimeError from worker process: {e}")
        assert "failed to update weights due to remote error(s)" in str(e)
    except Exception as e:
        print(f"[rank{rank}] Caught unexpected exception: {e}")
        raise
    finally:
        ps.unregister_checkpoint(checkpoint_name)
        queue.put(None)


@pytest.mark.gpu
def test_update():
    world_size = torch.cuda.device_count()
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
        "tests/test_error_quit.py",
    ]

    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        shell=False,
        check=False,
    )

    assert result.returncode == 0


if __name__ == "__main__":
    run()
