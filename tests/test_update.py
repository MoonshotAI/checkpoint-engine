import json
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import zmq
from torch.multiprocessing import Queue, get_context

from checkpoint_engine.device_utils import DeviceManager
from checkpoint_engine.ps import ParameterServer, _get_physical_gpu_id
from checkpoint_engine.worker import update_weights_from_ipc


try:
    device_manager = DeviceManager()
except TypeError:
    device_manager = SimpleNamespace(device_module=SimpleNamespace(device_count=lambda: 0))


def get_world_size() -> int:
    return device_manager.device_module.device_count()


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


def checker_proc_with_error(
    rank: int, device_uuid: str, named_tensors: dict[str, torch.Tensor], queue: Queue
):
    device_manager.device_module.set_device(rank)
    named_tensors = {
        name: tensor.to(device_manager.device_type) for name, tensor in named_tensors.items()
    }
    _ = named_tensors
    _zmq_ctx = zmq.Context()

    def trigger_error(socket_paths: list[tuple[str, str]]):
        socket_paths = dict(socket_paths)
        update_weights_from_ipc(
            _zmq_ctx,
            socket_paths[device_uuid],
            device_id=rank,
            run=error_run,
            post_hook=lambda: device_manager.device_module.synchronize(),
        )

    def error_run(weights: list[tuple[str, torch.Tensor]]):
        _ = weights  # Do some fake processing
        time.sleep(random.uniform(0.1, 0.5))
        if rank == 0:
            raise RuntimeError("Intentional Error for testing.")

    while True:
        socket_paths: list[tuple[str, str]] = queue.get()
        if socket_paths is None:
            break
        try:
            trigger_error(socket_paths)
        except RuntimeError as e:
            assert str(e) == "Some workers failed to update weights"


def checker_proc(rank: int, device_uuid: str, named_tensors: dict[str, torch.Tensor], queue: Queue):
    device_manager.device_module.set_device(rank)
    named_tensors = {
        name: tensor.to(device_manager.device_type) for name, tensor in named_tensors.items()
    }
    _zmq_ctx = zmq.Context()

    def check(names_to_check: dict[str, bool], weights: list[tuple[str, torch.Tensor]]):
        for name, weight in weights:
            if name not in named_tensors:
                continue
            assert (weight == named_tensors[name]).all(), f"Tensor {name} does not match!"
            names_to_check[name] = True

    def check_weights(names_to_check: dict[str, bool], socket_paths: list[tuple[str, str]]):
        socket_paths = dict(socket_paths)
        update_weights_from_ipc(
            _zmq_ctx,
            socket_paths[device_uuid],
            device_id=rank,
            run=lambda weights: check(names_to_check, weights),
            post_hook=lambda: device_manager.device_module.synchronize(),
        )
        assert all(names_to_check.values())

    while True:
        socket_paths: list[tuple[str, str]] = queue.get()
        if socket_paths is None:
            break
        names_to_check = dict.fromkeys(named_tensors.keys(), False)
        check_weights(names_to_check, socket_paths)


def run(
    checker_func: callable,
    rank_list: list[list[int]],
    need_error: bool = False,
    expected_exception: Exception | None = None,
    exception_msg: str | None = None,
):
    if need_error:
        assert expected_exception is not None, (
            "expected_exception must be provided when need_error is True."
        )
        assert exception_msg is not None, "exception_msg must be provided when need_error is True."
    else:
        assert expected_exception is None, (
            "expected_exception must be None when need_error is False."
        )
        assert exception_msg is None, "exception_msg must be None when need_error is False."

    rank = int(os.getenv("RANK"))
    ctx = get_context("spawn")
    queue = ctx.Queue()
    _device_uuid = _get_physical_gpu_id(device_manager, rank)
    ps = ParameterServer(auto_pg=True)
    _device_uuid = _get_physical_gpu_id(ps.device_manager, rank)
    named_tensors = dict(gen_test_tensors(rank))
    checkpoint_name = "test"
    proc = ctx.Process(target=checker_func, args=(rank, _device_uuid, named_tensors, queue))
    proc.start()
    with pytest.raises(expected_exception) if need_error else nullcontext() as e:
        ps.register_checkpoint(checkpoint_name, named_tensors=named_tensors)
        ps.gather_metas(checkpoint_name)
        for ranks in rank_list:
            ps.update(checkpoint_name, queue.put, ranks=ranks)
            # sleep 3s to wait process group is destroyed
            time.sleep(3)
        if need_error:
            pytest.fail("Test failed: Expected RuntimeError was not raised. Should not reach here.")
    if need_error:
        assert exception_msg in str(e.value)
    ps.unregister_checkpoint(checkpoint_name)
    queue.put(None)
    proc.join()
    assert proc.exitcode == 0


def run_with_files(
    checker_func: callable,
):
    rank = int(os.getenv("RANK"))
    ctx = get_context("spawn")
    queue = ctx.Queue()
    _device_uuid = _get_physical_gpu_id(device_manager, rank)
    ps = ParameterServer(auto_pg=True)
    _device_uuid = _get_physical_gpu_id(ps.device_manager, rank)
    named_tensors = dict(gen_test_tensors(rank))

    # Save 1/3 tensors to /dev/shm/ as .safetensors files
    # Save 1/3 tensors to ./tmp (disk) as .safetensors files
    # Keep 1/3 tensors in memory
    import safetensors.torch

    files = []
    dev_shm_dir = "/dev/shm/checkpoint_engine_tests"  # noqa: S108
    disk_dir = "/tmp/checkpoint_engine_tests"  # noqa: S108
    os.makedirs(dev_shm_dir, exist_ok=True)
    os.makedirs(disk_dir, exist_ok=True)
    tensors_items = list(named_tensors.items())
    tensors_in_dev_shm = named_tensors
    tensors_in_dev_shm = dict(tensors_items[: len(tensors_items) // 2])
    tensors_in_disk = dict(tensors_items[len(tensors_items) // 3 : 2 * len(tensors_items) // 3])
    tensors_in_memory = dict(tensors_items[1 * len(tensors_items) // 2 :])
    disk_files = [
        os.path.join(disk_dir, f"rank{_rank}_checkpoint.safetensors")
        for _rank in range(get_world_size())
    ]
    safetensors.torch.save_file(tensors_in_disk, disk_files[rank])
    time.sleep(1)
    files.append(disk_files[rank])
    dev_shm_files = [
        os.path.join(dev_shm_dir, f"rank{rank}_checkpoint.safetensors")
        for _ in range(get_world_size())
    ]
    safetensors.torch.save_file(tensors_in_dev_shm, dev_shm_files[rank])
    time.sleep(1)
    files.append(dev_shm_files[rank])

    checkpoint_name = "test_with_files"
    proc = ctx.Process(target=checker_func, args=(rank, _device_uuid, named_tensors, queue))
    proc.start()
    ps.register_checkpoint(checkpoint_name, named_tensors=tensors_in_memory, files=files)
    ps.gather_metas(checkpoint_name)
    ps.update(checkpoint_name, queue.put, ranks=[])
    # sleep 3s to wait process group is destroyed
    time.sleep(3)
    ps.unregister_checkpoint(checkpoint_name)
    queue.put(None)
    proc.join()
    assert proc.exitcode == 0


@pytest.mark.gpu
@pytest.mark.parametrize(
    "test_name,rank_list",
    [
        (
            "test_no_error",
            [
                list(range(get_world_size() // 2)),
                list(range(get_world_size() // 2, get_world_size())),
                [],
                list(range(get_world_size())),
            ],
        ),
        ("test_with_remote_error", [[]]),
        # ("long_test_no_error", [list(random.sample(range(get_world_size()), k=num_ranks)) for num_ranks in range(get_world_size() + 1)]),
    ],
)
def test_update(test_name: str, rank_list: list[list[int]] | None):
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
        test_name,
        json.dumps(rank_list) if rank_list is not None else "[]",
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


@pytest.mark.gpu
def test_update_with_files(test_name: str = "test_with_files"):
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
        test_name,
        "[]",
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
    run_with_pytest = "PYTEST_CURRENT_TEST" in os.environ
    if not run_with_pytest:
        print("ERROR: This script is designed to run only through pytest!")
        print("Please use: pytest test_update.py")
        sys.exit(1)
    assert len(sys.argv) > 2
    test_type = sys.argv[1]
    rank_list = json.loads(sys.argv[2])
    if test_type == "test_no_error":
        run(checker_proc, rank_list, need_error=False)
    elif test_type == "test_with_remote_error":
        run(
            checker_proc_with_error,
            rank_list,
            need_error=True,
            expected_exception=RuntimeError,
            exception_msg="Failed to update weights due to remote errors",
        )
    elif test_type == "test_with_files":
        run_with_files(checker_proc)
    else:
        raise ValueError(f"Unknown TEST_TYPE: {test_type}")
