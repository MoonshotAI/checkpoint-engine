import os
import sys
import pytest
from checkpoint_engine.ps import _assign_receiver_ranks


@pytest.mark.parametrize("buckets,local_topo,remote_topo,expected_result", [
    (
        [(i % 8, f"bucket{i}") for i in range(100)],
        {f"rdma{i}": {i} for i in range(8)},
        {f"rdma{i}": {i} for i in range(8)},
        [(i % 8, i % 8, f"bucket{i}") for i in range(100)],
    ),
    (
        [(i % 8, f"bucket{i}") for i in range(100)],
        {f"rdma{i}": {i} for i in range(8)},
        {f"rdma{i}": {i, i+1} for i in range(4)},
        [((i % 4)*2, i % 8, f"bucket{i}") for i in range(100)],
    ),
    (
        [(i % 8, f"bucket{i}") for i in range(100)],
        {f"rdma{i}": {i, i+1, i+2, i+3} for i in range(2)},
        {f"rdma{i}": {i} for i in range(8)},
        [((i % 2)*4, i % 8, f"bucket{i}") for i in range(100)],
    ),
])
def test_basic_functionality(buckets, local_topo, remote_topo, expected_results):
    assert expected_results == _assign_receiver_ranks(buckets, local_topo, remote_topo)
