from checkpoint_engine.distributed.vllm_compat import create_stateless_process_group


class CurrentProcessGroup:
    def __init__(self, rank: int, world_size: int, store: object):
        self.arguments = rank, world_size, store


class LegacyProcessGroup:
    def __init__(self, rank: int, world_size: int, store: object, socket: object):
        self.arguments = rank, world_size, store, socket


def test_current_vllm_process_group_signature():
    store = object()

    group = create_stateless_process_group(
        CurrentProcessGroup,
        rank=2,
        world_size=8,
        store=store,
    )

    assert group.arguments == (2, 8, store)


def test_legacy_vllm_process_group_signature():
    store = object()

    group = create_stateless_process_group(
        LegacyProcessGroup,
        rank=2,
        world_size=8,
        store=store,
    )

    assert group.arguments == (2, 8, store, None)
