"""checkpoint_engine.join_cli

Join an existing P2P weight world over mooncake RDMA: pull weights from a
remote ParameterServer's pinned CPU memory directly into local GPUs and
update the local inference engine (vLLM by default).

The remote side must already have done ``gather_metas`` so that its
``ps.get_metas()`` returns a usable ``dict[int, MemoryBufferMetaList]``,
and the metas JSON bytes must be reachable via either a file path or
an HTTP URL.

Usage (one process per local GPU, e.g. via torchrun):

    # From a local file (e.g. shared moonfs):
    torchrun --nproc-per-node N -m checkpoint_engine.join_cli \\
        --load-metas-file /path/to/metas.json \\
        --endpoint http://localhost:19730 \\
        --inference-parallel-size N \\
        [--checkpoint-name <name>]

    # Or directly from the source ParameterServer's HTTP endpoint:
    torchrun --nproc-per-node N -m checkpoint_engine.join_cli \\
        --metas-url http://main-ps-host:19710/v1/checkpoints/<name>/metas \\
        --endpoint http://localhost:19730 \\
        --inference-parallel-size N

Environment variables (same as ``torchrun`` sets): ``RANK``, ``WORLD_SIZE``,
``LOCAL_RANK``, ``MASTER_ADDR``, ``MASTER_PORT``.
"""

import argparse
import os
import time
from collections.abc import Callable
from contextlib import contextmanager

import httpx
from loguru import logger
from pydantic import TypeAdapter

import checkpoint_engine.distributed as dist
from checkpoint_engine import request_inference_to_update
from checkpoint_engine.data_types import MemoryBufferMetaList
from checkpoint_engine.ps import ParameterServer


_METAS_ADAPTER = TypeAdapter(dict[int, MemoryBufferMetaList])


@contextmanager
def _timer(msg: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"{msg} duration: {elapsed:.2f}s")


def _check_vllm_ready(
    rank: int, endpoint: str, inference_parallel_size: int, uds: str | None
) -> None:
    if rank != rank // inference_parallel_size * inference_parallel_size:
        return
    transport = httpx.HTTPTransport(uds=uds) if uds else None
    retry = 0
    while True:
        try:
            response = httpx.Client(transport=transport).get(f"{endpoint}/health", timeout=10)
            response.raise_for_status()
            break
        except (httpx.ConnectError, httpx.HTTPStatusError) as e:
            retry += 1
            logger.warning(f"vllm not ready, retry {retry}: {e}")
            time.sleep(5)


def _req_func_for_inference(
    rank: int, endpoint: str, inference_parallel_size: int, uds: str | None
) -> Callable[[list[tuple[str, str]]], None]:
    src = rank // inference_parallel_size * inference_parallel_size

    def req(socket_paths: list[tuple[str, str]]) -> None:
        if rank == src:
            request_inference_to_update(
                f"{endpoint}/collective_rpc",
                dict(socket_paths[src : src + inference_parallel_size]),
                uds=uds,
            )

    return req


def _load_metas(args: argparse.Namespace) -> dict[int, MemoryBufferMetaList]:
    if args.load_metas_file:
        with open(args.load_metas_file, "rb") as f:
            return _METAS_ADAPTER.validate_json(f.read())
    if args.metas_url:
        resp = httpx.get(args.metas_url, timeout=300.0)
        resp.raise_for_status()
        return _METAS_ADAPTER.validate_json(resp.content)
    raise ValueError("either --load-metas-file or --metas-url is required")


def join(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    metas = _load_metas(args)
    logger.info(f"[rank{rank}] loaded metas: {len(metas)} owner entries")

    ps = ParameterServer(auto_pg=True)
    req_func = _req_func_for_inference(rank, args.endpoint, args.inference_parallel_size, args.uds)

    ps.init_process_group()
    _check_vllm_ready(rank, args.endpoint, args.inference_parallel_size, args.uds)
    dist.barrier()
    with _timer(f"[rank{rank}] gather_metas"):
        ps.gather_metas(args.checkpoint_name)
    ps.load_metas(metas)
    with _timer(f"[rank{rank}] p2p update ranks=range(0, {args.inference_parallel_size})"):
        ps.update(
            args.checkpoint_name,
            req_func,
            ranks=list(range(args.inference_parallel_size)),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join an existing P2P weight world via mooncake RDMA"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--load-metas-file", type=str, help="Path to a metas JSON file")
    src.add_argument(
        "--metas-url",
        type=str,
        help="HTTP URL returning a metas JSON (application/json)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:19730",
        help="Local inference engine endpoint",
    )
    parser.add_argument(
        "--inference-parallel-size",
        type=int,
        required=True,
        help="Tensor parallel size of the local inference engine",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="join-checkpoint",
        help="Name to use for this checkpoint locally (any unique string)",
    )
    parser.add_argument(
        "--uds",
        type=str,
        default=None,
        help="Optional UDS path for talking to local vLLM instead of HTTP",
    )
    parser.add_argument(
        "--custom-dist",
        type=str,
        default=None,
        help="Optional custom distributed backend name",
    )
    args = parser.parse_args()
    if args.custom_dist:
        dist.use_backend(args.custom_dist)
    join(args)


if __name__ == "__main__":
    main()
