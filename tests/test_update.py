import os
import random
import time

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


def run():
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
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
    ranks_list = [[], list(range(world_size // 2)), [], list(range(world_size))]
    for ranks in ranks_list:
        ps.update(checkpoint_name, queue.put, ranks=ranks)
        # sleep 3s to wait process group is destroyed
        time.sleep(3)
    ps.unregister_checkpoint(checkpoint_name)
    queue.put(None)
    proc.join()


if __name__ == "__main__":
    run()
