import gc
from collections.abc import Callable
from typing import TypedDict

import torch
import zmq

from checkpoint_engine.device_utils import DeviceManager, npu_generate_uuid


def _rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer


class FlattenedTensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    # specify the start offset of this tensor in shared ipc_buffer tensor
    offset: int


def _extract_weights(
    payload: list[FlattenedTensorMetadata], buffer: torch.Tensor
) -> list[tuple[str, torch.Tensor]]:
    assert buffer is not None
    weights: list[tuple[str, torch.Tensor]] = []
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
    device_mananger = DeviceManager()
    while True:
        payload: tuple[Callable, tuple] | list[FlattenedTensorMetadata] | None | Exception = (
            socket.recv_pyobj()
        )
        if payload is None or isinstance(payload, Exception):
            # means the update is done or error occurred
            # only call post_hook if update is successful
            if post_hook is not None and not isinstance(payload, Exception):
                post_hook()
            device_mananger.device_module.synchronize()
            socket.send(b"")
            if isinstance(payload, Exception):
                raise payload
            break
        if isinstance(payload, tuple):
            # an ipc handle that vLLM can use `func, args = handle`
            # and `func(*args)` to rebuild GPU tensor.
            buffer = _rebuild_ipc(payload, device_id)
            assert buffer.dtype == torch.uint8
            socket.send(b"")
            continue
        assert isinstance(payload, list)
        try:
            run(_extract_weights(payload, buffer))
            device_mananger.device_module.synchronize()
            socket.send(b"")
        except Exception as e:  # noqa: BLE001
            socket.send_pyobj(e)

    socket.close()
    del buffer
    gc.collect()
    device_mananger.device_module.empty_cache()


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

        Raises:
            ValueError: If the device type is not supported (not CUDA or NPU).
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
        assert self.device is not None
        if not hasattr(self, "_zmq_ctx") or self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        if current_platform.device_type == "cuda":
            device_uuid = current_platform.get_device_uuid(self.device.index)
        elif current_platform.device_type == "npu":
            device_uuid = f"NPU-{npu_generate_uuid()}"
        else:
            raise ValueError(f"Unsupported device type: {current_platform.device_type}")
        update_weights_from_ipc(
            self._zmq_ctx,
            zmq_handles[device_uuid],
            device_id=self.device.index,
            run=self.model_runner.model.load_weights,
            post_hook=lambda: process_weights_after_loading(
                self.model_runner.model, self.model_config, self.device
            ),
        )
