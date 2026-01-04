import argparse
import os

from loguru import logger

from checkpoint_engine.api import _init_api
from checkpoint_engine.ps import ParameterServer


@logger.catch(reraise=True)
def run_from_cli():
    import uvicorn

    parser = argparse.ArgumentParser(description="Parameter Server")
    parser.add_argument("--uds", type=str)

    args = parser.parse_args()
    logger.info(
        f"Parameter Server {args=}, master addr: {os.getenv('MASTER_ADDR')}, master port {os.getenv('MASTER_PORT')}"
    )

    assert args.uds and len(args.uds) > 0, args.uds
    ps = ParameterServer(auto_pg=True)
    uvicorn.run(_init_api(ps), uds=args.uds, timeout_keep_alive=60)


if __name__ == "__main__":
    run_from_cli()
