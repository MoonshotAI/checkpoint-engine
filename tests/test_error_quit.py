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
        weights = weights  # unused
        time.sleep(random.uniform(0.1, 0.5))
        if random.random() < 0.6:
            raise RuntimeError("Intentional Error for testing.")

    while True:
        socket_paths: list[tuple[str, str]] = queue.get()
        if socket_paths is None:
            break
        try:
            trigger_error(socket_paths)
        except:
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
        print(f"[rank{rank}] Caught exception from worker process: {e}")
        assert isinstance(e, RuntimeError)
    finally:
        ps.unregister_checkpoint(checkpoint_name)
        queue.put(None)


if __name__ == "__main__":
    run()
