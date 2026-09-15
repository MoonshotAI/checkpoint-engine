from checkpoint_engine.ps import _is_valid_manual_pin_memory_flag


def test_manual_pin_memory_flags_accept_default_and_mapped() -> None:
    assert _is_valid_manual_pin_memory_flag(0x00)
    assert _is_valid_manual_pin_memory_flag(0x02)
    assert not _is_valid_manual_pin_memory_flag(0x01)
