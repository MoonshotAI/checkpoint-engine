import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from checkpoint_engine.ps import _assign_receiver_ranks


class TestAssignReceiverRanks:
    def test_basic_functionality(self):
        buckets = [(i % 8, f"bucket{i}") for i in range(100)]
        local_topo = {f"rdma{i}": {i} for i in range(8)}
        remote_topo = {f"rdma{i}": {i} for i in range(8)}

        result = _assign_receiver_ranks(buckets, local_topo, remote_topo)

        assert len(result) == 100
        for item in result:
            assert len(item) == 3
            assert isinstance(item[0], int)  # receiver_rank
            assert isinstance(item[1], int)  # owner_rank
            assert isinstance(item[2], str)  # bucket

        for receiver_rank, owner_rank, bucket in result:
            assert receiver_rank in range(8)
            assert owner_rank % 8 == receiver_rank
            assert bucket in {f"bucket{i}" for i in range(100)}

    def test_empty_buckets(self):
        buckets = []
        local_topo = {"rdma0": {0}}
        remote_topo = {"rdma0": {0}}

        result = _assign_receiver_ranks(buckets, local_topo, remote_topo)

        assert result == []
