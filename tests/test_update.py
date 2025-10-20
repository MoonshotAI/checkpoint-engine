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


def checker_proc(rank: int, device_uuid: str, named_tensors: dict[str, torch.Tensor], queue: Queue):
    torch.cuda.set_device(rank)
    named_tensors = {name: tensor.cuda() for name, tensor in named_tensors.items()}
    _zmq_ctx = zmq.Context()

    def check(names_to_check: dict[str, bool], weights: list[tuple[str, torch.Tensor]]):
        for name, weight in weights:
            if name not in named_tensors:
                continue
            assert (weight == named_tensors[name]).all()
            names_to_check[name] = True

    def check_weights(names_to_check: dict[str, bool], socket_paths: list[tuple[str, str]]):
        socket_paths = dict(socket_paths)
        update_weights_from_ipc(
            _zmq_ctx,
            socket_paths[device_uuid],
            device_id=rank,
            run=lambda weights: check(names_to_check, weights),
            post_hook=lambda: torch.cuda.synchronize(),
        )
        assert all(names_to_check.values())

    while True:
        socket_paths: list[tuple[str, str]] = queue.get()
        if socket_paths is None:
            break
        names_to_check = dict.fromkeys(named_tensors.keys(), False)
        check_weights(names_to_check, socket_paths)


def run_with_specified_ranks(ranks: list[int]):
    rank = int(os.getenv("RANK"))
    ctx = get_context("spawn")
    queue = ctx.Queue()
    _device_uuid = _get_physical_gpu_id(rank)
    ps = ParameterServer(auto_pg=True)
    named_tensors = dict(gen_test_tensors(rank))
    checkpoint_name = "test"
    proc = ctx.Process(target=checker_proc, args=(rank, _device_uuid, named_tensors, queue))
    proc.start()
    ps.register_checkpoint(checkpoint_name, named_tensors=named_tensors)
    ps.gather_metas(checkpoint_name)
    ps.update(checkpoint_name, queue.put, ranks=ranks)
    time.sleep(5)
    ps.unregister_checkpoint(checkpoint_name)
    queue.put(None)
    proc.join()


def run():
    world_size = int(os.getenv("WORLD_SIZE"))
    random.seed(42)
    ranklist = [
        list(random.sample(range(world_size), k=num_ranks)) for num_ranks in range(world_size + 1)
    ]
    for ranks in ranklist:
        run_with_specified_ranks(ranks)


@pytest.mark.gpu
def test_update():
    world_size = torch.cuda.device_count()
    assert world_size >= 2, "This test requires at least 2 GPUs."

    master_addr = "localhost"
    master_port = random.randint(20000, 30000)

    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(world_size),
        "--master_addr",
        master_addr,
        "--master_port",
        str(master_port),
        "tests/test_update.py",
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
