from collections.abc import Callable
from typing import Any

import fastapi
import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response
from loguru import logger
from pydantic import BaseModel

from checkpoint_engine.ps import ParameterServer


def request_inference_to_update(
    url: str,
    socket_paths: dict[str, str],
    timeout: float = 300.0,
    uds: str | None = None,
):
    """Send an inference update request to inference server via HTTP or Unix socket.

    Args:
        url (str): The HTTP URL or request path (e.g., "http://localhost:19730/inference") to send the request to.
        socket_paths (dict[str, str]): A dictionary containing device uuid and IPC socket paths for updating weights.
        timeout (float, optional): Request timeout in seconds. Defaults to 300.0.
        uds (str, optional): Path to a Unix domain socket. If provided, the request
            will be sent via the Unix socket instead of HTTP. Defaults to None.

    Raises:
        httpx.HTTPStatusError: If the response contains an HTTP error status.
        httpx.RequestError: If there was an issue while making the request.
    """
    resp = httpx.Client(transport=httpx.HTTPTransport(uds=uds)).post(
        url,
        json={
            "method": "update_weights_from_ipc",
            "args": [socket_paths],
            "timeout": timeout,
        },
        timeout=timeout,
    )
    resp.raise_for_status()


def _init_api(ps: ParameterServer) -> Any:
    app = fastapi.FastAPI()

    class RegisterRequest(BaseModel):
        files: list[str]

    class UpdateRequest(BaseModel):
        ranks: list[int] = []
        update_url: str | None = None
        inference_group_ranks: list[int] = []
        timeout: float = 300.0
        uds: str | None = None

    def wrap_exception(func: Callable[[], None]) -> Response:
        try:
            func()
        except Exception as e:  # noqa: BLE001
            logger.exception(f"wrap exception {func} failed")
            return JSONResponse(content=str(e), status_code=500)
        return Response(status_code=200)

    @app.post("/v1/checkpoints/{checkpoint_name}/files")
    async def register_files(checkpoint_name: str, req: RegisterRequest, raw: Request) -> Response:
        return wrap_exception(lambda: ps.register_checkpoint(checkpoint_name, files=req.files))

    @app.delete("/v1/checkpoints/{checkpoint_name}")
    async def unregister_checkpoint(checkpoint_name: str) -> Response:
        return wrap_exception(lambda: ps.unregister_checkpoint(checkpoint_name))

    @app.get("/v1/healthz")
    async def healthz() -> Response:
        return Response(status_code=200)

    @app.post("/v1/checkpoints/{checkpoint_name}/gather-metas")
    async def gather_metas(checkpoint_name: str) -> Response:
        return wrap_exception(lambda: ps.gather_metas(checkpoint_name))

    @app.post("/v1/checkpoints/{checkpoint_name}/update")
    async def update(checkpoint_name: str, req: UpdateRequest) -> Response:
        def update_func(socket_paths: list[tuple[str, str]]):
            if req.update_url is None:
                return
            if req.inference_group_ranks:
                socket_paths = [socket_paths[i] for i in req.inference_group_ranks]
            request_inference_to_update(
                req.update_url, dict(socket_paths), timeout=req.timeout, uds=req.uds
            )

        return wrap_exception(lambda: ps.update(checkpoint_name, update_func, ranks=req.ranks))

    return app
