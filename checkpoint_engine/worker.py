import gc
import traceback
from collections.abc import Callable
from functools import cached_property
from typing import TypedDict

import torch
import zmq

from checkpoint_engine.device_utils import DeviceManager, npu_generate_uuid
from checkpoint_engine.ipc_handler import (
    IPCHandler,
    TorchIPCHandler,
    XpuIPCHandler,
)


_WEIGHTS_TYPE = list[tuple[str, torch.Tensor]]


def _ipc_handler_for_handle(handle: object) -> IPCHandler:
    """Pick the consumer-side IPC handler based on the handle wire format.

    CUDA/NPU send a ``reduce_tensor`` tuple; XPU sends a dict tagged with its kind.
    """
    if isinstance(handle, dict) and handle.get("kind") == XpuIPCHandler.kind:
        return XpuIPCHandler()
    return TorchIPCHandler()


class FlattenedTensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    # specify the start offset of this tensor in shared ipc_buffer tensor
    offset: int


def _extract_weights(payload: list[FlattenedTensorMetadata], buffer: torch.Tensor) -> _WEIGHTS_TYPE:
    assert buffer is not None
    weights: _WEIGHTS_TYPE = []
    for item in payload:
        shape = item["shape"]
        if isinstance(shape, list | tuple):
            shape = torch.Size(shape)
        assert isinstance(shape, torch.Size)
        dtype, offset = item["dtype"], item["offset"]
        size = dtype.itemsize * shape.numel()
        tensor = buffer[offset : offset + size].view(dtype=dtype).view(shape)
        weights.append((item["name"], tensor))
    return weights


def update_weights_from_ipc(
    zmq_ctx: zmq.Context,
    zmq_handle: str,
    device_id: int,
    *,
    run: Callable[[list[tuple[str, torch.Tensor]]], None],
    post_hook: Callable[[], None] | None = None,
):
    socket = zmq_ctx.socket(zmq.REP)
    socket.connect(zmq_handle)
    buffer: torch.Tensor | None = None
    device_manager = DeviceManager()
    ipc_handler: IPCHandler | None = None
    try:
        ipc_handle = socket.recv_pyobj()
        ipc_handler = _ipc_handler_for_handle(ipc_handle)
        buffer = ipc_handler.attach(ipc_handle, device_id)
        assert buffer.dtype == torch.uint8
        socket.send(b"")
    except Exception as e:
        msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        socket.send_string(msg)
        socket.recv()  # wait for ack
        raise
    # State machine:
    # + receive tensor_metadata -> update_weights
    # + receive Exception -> raise and stop
    # + receive None first time -> release resources
    # + receive None second time -> call post_hook and stop
    try:
        released = False
        while True:
            payload: list[FlattenedTensorMetadata] | Exception | None = socket.recv_pyobj()
            if released:
                assert payload is None, "Should not receive any payload after released"
                if post_hook is not None:
                    post_hook()
                device_manager.device_module.synchronize()
                socket.send(b"")
                break
            if payload is None:  # done signal
                # TODO: wrap all messages to an object instead of None and Exception
                device_manager.device_module.synchronize()
                released = True
                buffer = None
                if ipc_handler is not None:
                    ipc_handler.detach()

                gc.collect()
                device_manager.ipc_collect()
                device_manager.device_module.empty_cache()
                device_manager.device_module.synchronize()
                socket.send(b"")
                continue
            if isinstance(payload, list):  # still updating weights
                try:
                    run(_extract_weights(payload, buffer))
                    device_manager.device_module.synchronize()
                    socket.send(b"")
                except Exception as e:  # noqa: BLE001
                    # Send exception back to Parameter Server.
                    # Don't raise here. Because all workers should quit in the same way by receiving the exception from PS
                    msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                    socket.send_string(msg)
            elif isinstance(
                payload, Exception
            ):  # error occurred, got force quit signal from Parameter Server
                raise payload
            else:
                raise TypeError(f"Unexpected payload type: {type(payload)}")

    finally:
        socket.close()
        del buffer
        if ipc_handler is not None:
            ipc_handler.detach()
        gc.collect()
        device_manager.device_module.empty_cache()


class VllmColocateWorkerExtension:
    """
    Worker extension for vLLM to update weights from checkpoint-engine.

    This class provides a worker extension mechanism that allows vLLM workers to receive
    and apply weight updates from the checkpoint-engine via IPC (Inter-Process Communication).
    The methods in this worker extension will be injected into the vLLM worker class and
    are callable from the `collective_rpc` API, enabling seamless weight updates for both
    vLLM V0 and V1 versions.

    Note:
        This class is defined in a separate module. The fully qualified name
        `checkpoint_engine.worker.VllmColocateWorkerExtension` should be passed as the
        `worker_extension_cls` argument when initializing the vLLM worker.
    """

    @cached_property
    def _device_uuid(self) -> str:
        from vllm.platforms import current_platform

        if current_platform.device_type == "cuda":
            return current_platform.get_device_uuid(self.device.index)
        elif current_platform.device_type == "npu":
            return f"NPU-{npu_generate_uuid()}"
        elif current_platform.device_type == "xpu":
            # Must match ps.py::_get_physical_gpu_id ("GPU-<uuid>") for the ZMQ key to resolve.
            return f"GPU-{torch.xpu.get_device_properties(self.device.index).uuid!s}"
        else:
            raise ValueError(f"Unsupported device type: {current_platform.device_type}")

    @cached_property
    def _zmq_ctx(self) -> zmq.Context:
        return zmq.Context()

    def update_weights_from_ipc(self, zmq_handles: dict[str, str]):
        """
        Update model weights from checkpoint-engine via IPC communication.

        This method establishes a ZMQ connection to the checkpoint-engine and receives
        weight updates through a shared memory buffer. The update process includes:
        1. Receiving IPC handles to reconstruct shared memory tensors
        2. Extracting flattened metadata describing tensor weights in the shared memory tensor
        3. Loading weights into the model
        4. Post-processing weights after loading

        Args:
            zmq_handles: A dictionary mapping device UUIDs to ZMQ socket handles.
                        The device UUID is platform-specific:
                        - For CUDA: UUID from `current_platform.get_device_uuid()`
                        - For NPU: Format "NPU-{generated_uuid}"
                        - For XPU: Format "GPU-{torch.xpu device uuid}"

        Raises:
            ValueError: If the device type is not supported (not CUDA, NPU, or XPU).
            AssertionError: If the device is not properly initialized.

        Note:
            This method is called by vLLM's collective RPC mechanism. The ZMQ context
            is lazily initialized on first call and reused for subsequent updates.
        """
        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        from vllm.platforms import current_platform

        # vllm-ascend not init device
        if current_platform.device_type == "npu" and self.device is None:
            self.device = torch.device(f"npu:{self.local_rank}")
        elif current_platform.device_type == "xpu" and self.device is None:
            self.device = torch.device(f"xpu:{self.local_rank}")
        assert self.device is not None

        def _load_weights(weights: _WEIGHTS_TYPE):
            # Load main model weights
            self.model_runner.model.load_weights(weights)
            # Load drafter model weights if MTP/speculative decoding is enabled
            if (
                getattr(self.model_runner, "drafter", None) is not None
                and getattr(self.model_runner.drafter, "model", None) is not None
            ):
                self.model_runner.drafter.model.load_weights(weights=weights)
            if getattr(self.model_runner, "get_draft_model", None) is not None:
                draft_model = self.model_runner.get_draft_model()
                if draft_model is not None:
                    draft_model.load_weights(weights=weights)

        def _post_hook():
            process_weights_after_loading(self.model_runner.model, self.model_config, self.device)
            # Also trigger drafter model's post processing if MTP is enabled
            if (
                getattr(self.model_runner, "drafter", None) is not None
                and getattr(self.model_runner.drafter, "model", None) is not None
            ):
                process_weights_after_loading(
                    self.model_runner.drafter.model, self.model_config, self.device
                )
            elif getattr(self.model_runner, "get_draft_model", None) is not None:
                draft_model = self.model_runner.get_draft_model()
                if draft_model is not None:
                    process_weights_after_loading(
                        draft_model,
                        self.vllm_config.speculative_config.draft_model_config,
                        self.device,
                    )

        update_weights_from_ipc(
            self._zmq_ctx,
            zmq_handles[self._device_uuid],
            device_id=self.device.index,
            run=_load_weights,
            post_hook=_post_hook,
        )
