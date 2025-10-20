import pytest

from checkpoint_engine.ps import _assign_receiver_ranks


@pytest.mark.parametrize(
    "buckets,local_topo,remote_topo,expected_results",
    [
        (
            [(i % 8, f"bucket{i}") for i in range(80)],
            {f"rdma{i}": {i} for i in range(8)},
            {f"rdma{i}": {i} for i in range(8)},
            [(i % 8, i % 8, f"bucket{i}") for i in range(80)],
        ),
        (
            [(i % 8, f"bucket{i}") for i in range(80)],
            {f"rdma{i}": {i} for i in range(8)},
            {f"rdma{i}": {i, i + 1} for i in range(0, 8, 2)},
            [((i // 2 % 4), i % 8, f"bucket{i}") for i in range(80)],
        ),
        (
            [(i % 8, f"bucket{i}") for i in range(80)],
            {f"rdma{i}": {i, i + 1, i + 2, i + 3} for i in range(0, 8, 4)},
            {f"rdma{i}": {i} for i in range(8)},
            [((i % 2) * 4, i % 8, f"bucket{i}") for i in range(80)],
        ),
        (
            [(i % 8, f"bucket{i}") for i in range(13)],
            {f"rdma{i}": {i} for i in range(8)},
            {f"rdma{i}": {i, i + 1} for i in range(0, 8, 2)},
            [((i // 2 % 4), i % 8, f"bucket{i}") for i in range(13)],
        ),
        (
            [(i % 8, f"bucket{i}") for i in range(13)],
            {f"rdma{i}": {i, i + 1} for i in range(0, 8, 2)},
            {f"rdma{i}": {i} for i in range(8)},
            [((i % 4) * 2, i % 8, f"bucket{i}") for i in range(13)],
        ),
        (
            [(i % 8, f"bucket{i}") for i in range(13)],
            {f"rdma{i}": {i} for i in range(3)},
            {f"rdma{i}": {i, i + 1} for i in range(0, 8, 2)},
            [
                (0, 0, "bucket0"),
                (1, 1, "bucket1"),
                (1, 2, "bucket2"),
                (2, 3, "bucket3"),
                (2, 4, "bucket4"),
                (0, 5, "bucket5"),
                (0, 6, "bucket6"),
                (1, 7, "bucket7"),
                (2, 0, "bucket8"),
                (2, 1, "bucket9"),
                (0, 2, "bucket10"),
                (0, 3, "bucket11"),
                (1, 4, "bucket12"),
            ],
        ),
    ],
)
def test_basic_functionality(
    buckets: list[tuple[int, str]],
    local_topo: dict[str, int],
    remote_topo: dict[str, int],
    expected_results: list[tuple[int, int, str]],
):
    assert len(expected_results) == len(buckets)
    assert set(expected_results) == set(_assign_receiver_ranks(buckets, local_topo, remote_topo))
