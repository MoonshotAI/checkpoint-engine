from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from unittest.mock import Mock, patch

import torch.distributed as dist

from checkpoint_engine.ps import ParameterServer


def test_store_based_barrier_uses_unique_group_name() -> None:
    ps = ParameterServer.__new__(ParameterServer)
    ps._rank = 0
    ps._world_size = 2
    ps._store = Mock()
    ps._store_barrier_counter = 0
    timeout = timedelta(seconds=5)

    target = "torch.distributed.distributed_c10d._store_based_barrier"
    with patch(target) as barrier:
        ps.store_based_barrier(timeout)
        ps.store_based_barrier(timeout)

    assert [call.kwargs["group_name"] for call in barrier.call_args_list] == [
        "parameter_server_barrier-1",
        "parameter_server_barrier-2",
    ]
    assert all(call.kwargs["store"] is ps._store for call in barrier.call_args_list)
    assert all(call.kwargs["timeout"] == timeout for call in barrier.call_args_list)


def test_store_based_barrier_is_reusable_with_shared_tcp_store() -> None:
    timeout = timedelta(seconds=5)
    server_store = dist.TCPStore("127.0.0.1", 0, 2, True, timeout=timeout, wait_for_workers=False)
    client_store = dist.TCPStore("127.0.0.1", server_store.port, 2, False, timeout=timeout)

    parameter_servers = []
    for rank, store in enumerate((server_store, client_store)):
        ps = ParameterServer.__new__(ParameterServer)
        ps._rank = rank
        ps._world_size = 2
        ps._store = store
        ps._store_barrier_counter = 0
        parameter_servers.append(ps)

    with ThreadPoolExecutor(max_workers=2) as executor:
        for _ in range(2):
            futures = [executor.submit(ps.store_based_barrier, timeout) for ps in parameter_servers]
            for future in futures:
                future.result()

    prefix = dist.distributed_c10d.STORE_BASED_BARRIER_PREFIX
    for generation in (1, 2):
        store_key = f"{prefix}:parameter_server_barrier-{generation}"
        assert server_store.add(store_key, 0) == 2
        assert server_store.get(f"{store_key}:last_worker") == b"1"
