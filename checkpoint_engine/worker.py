import gc
from typing import Callable, Optional, TypedDict

import torch
import zmq


def _rebuild_ipc(handle: tuple[Callable, tuple], device_id: Optional[int] = None) -> torch.Tensor:
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


def _extract_weights(payload: list[FlattenedTensorMetadata], buffer: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    assert buffer is not None
    weights: list[tuple[str, torch.Tensor]] = []
    for item in payload:
        shape = item["shape"]
        if isinstance(shape, (list, tuple)):
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
    post_hook: Callable[[], None] = None,
):
    socket = zmq_ctx.socket(zmq.REP)
    socket.connect(zmq_handle)
    buffer: Optional[torch.Tensor] = None
    while True:
        payload: tuple[Callable, tuple] | list[FlattenedTensorMetadata] | None = socket.recv_pyobj()
        if payload is None:
            # means the update is done
            if post_hook is not None:
                post_hook()
            torch.cuda.synchronize()
            socket.send(b"")
            break
        if isinstance(payload, tuple):
            # an ipc handle that vLLM can use `func, args = handle`
            # and `func(*args)` to rebuild GPU tensor.
            buffer = _rebuild_ipc(payload, device_id)
            assert buffer.dtype == torch.uint8
            socket.send(b"")
            continue
        assert isinstance(payload, list)
        run(_extract_weights(payload, buffer))
        torch.cuda.synchronize()
        socket.send(b"")

    socket.close()
    del buffer
    gc.collect()
    torch.cuda.empty_cache()


class VllmColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """

    def update_weights_from_ipc(self, zmq_handles: dict[str, str]):
        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        from vllm.platforms import current_platform

        assert self.device is not None
        if not hasattr(self, "_zmq_ctx") or self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        device_uuid = current_platform.get_device_uuid(self.device.index)
        update_weights_from_ipc(
            self._zmq_ctx,
            zmq_handles[device_uuid],
            device_id=self.device.index,
            run=self.model_runner.model.load_weights,
            post_hook=lambda: process_weights_after_loading(self.model_runner.model, self.model_config, self.device),
        )
