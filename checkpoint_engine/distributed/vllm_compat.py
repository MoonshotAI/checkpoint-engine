import inspect
from typing import Any


def create_stateless_process_group(
    group_cls: type,
    *,
    rank: int,
    world_size: int,
    store: Any,
) -> Any:
    """Construct a vLLM stateless process group across supported APIs."""
    kwargs = {"rank": rank, "world_size": world_size, "store": store}
    if "socket" in inspect.signature(group_cls).parameters:
        kwargs["socket"] = None
    return group_cls(**kwargs)
