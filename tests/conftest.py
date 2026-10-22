import pytest

from checkpoint_engine.device_utils import DeviceManager


@pytest.fixture(autouse=True)
def device_manager(request: pytest.FixtureRequest) -> DeviceManager | None:
    if request.node.get_closest_marker("gpu") is None:
        return None
    try:
        return DeviceManager()
    except TypeError as exc:
        pytest.skip(f"GPU/NPU runtime is unavailable: {exc}")
