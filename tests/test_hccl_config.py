import ctypes
import importlib
import sys
import types
from typing import ClassVar

import pytest
import torch


def test_hccl_comm_config_field_names_are_assignable(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeFunction:
        def __init__(self, *args: object) -> None:
            self.args = args

    class FakeHcclLibrary:
        exported_functions: ClassVar[list[object]] = []

    class FakePyHcclCommunicator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class FakeHcclDataTypeEnum:
        @staticmethod
        def from_torch(dtype: torch.dtype) -> int:
            return 0

    class FakeNpu:
        class Stream:
            npu_stream = 0

    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.distributed": types.ModuleType("vllm.distributed"),
        "vllm.distributed.utils": types.ModuleType("vllm.distributed.utils"),
        "vllm_ascend": types.ModuleType("vllm_ascend"),
        "vllm_ascend.distributed": types.ModuleType("vllm_ascend.distributed"),
        "vllm_ascend.distributed.device_communicators": types.ModuleType(
            "vllm_ascend.distributed.device_communicators"
        ),
        "vllm_ascend.distributed.device_communicators.pyhccl": types.ModuleType(
            "vllm_ascend.distributed.device_communicators.pyhccl"
        ),
        "vllm_ascend.distributed.device_communicators.pyhccl_wrapper": types.ModuleType(
            "vllm_ascend.distributed.device_communicators.pyhccl_wrapper"
        ),
        "vllm_ascend.utils": types.ModuleType("vllm_ascend.utils"),
    }
    modules["vllm.distributed.utils"].StatelessProcessGroup = object
    modules[
        "vllm_ascend.distributed.device_communicators.pyhccl"
    ].PyHcclCommunicator = FakePyHcclCommunicator
    wrapper = modules["vllm_ascend.distributed.device_communicators.pyhccl_wrapper"]
    wrapper.Function = FakeFunction
    wrapper.HCCLLibrary = FakeHcclLibrary
    wrapper.aclrtStream_t = ctypes.c_void_p
    wrapper.buffer_type = ctypes.c_void_p
    wrapper.hcclComm_t = ctypes.c_void_p
    wrapper.hcclDataType_t = ctypes.c_int
    wrapper.hcclDataTypeEnum = FakeHcclDataTypeEnum
    wrapper.hcclResult_t = int
    modules["vllm_ascend.utils"].current_stream = lambda: FakeNpu.Stream()

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch, "npu", FakeNpu(), raising=False)
    monkeypatch.delitem(sys.modules, "checkpoint_engine.distributed.vllm_hccl", raising=False)

    module = importlib.import_module("checkpoint_engine.distributed.vllm_hccl")
    field_names = [name for name, _ in module.HcclCommConfig._fields_]

    assert "hccl_op_expansion_mode" in field_names
    assert "hccl_world_rank_id" in field_names
    assert "hcll_world_rank_id" not in field_names
    config = module.HcclCommConfig(hccl_op_expansion_mode=7, hccl_world_rank_id=3)
    assert config.hccl_op_expansion_mode == 7
    assert config.hccl_world_rank_id == 3
