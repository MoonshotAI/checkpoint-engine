# Getting start in ascend

## Overview

Due to hardware differences in Ascend devices, the method for running the Checkpoint Engine on Ascend platforms requires specific adaptations.

## Environment

To support features like IPC Buffer and Transfer Engine, the following Ascend software versions are required:

| Software    | version     |
|-------------|-------------|
| Ascend HDK  | \>=25.3.rc1 |
| cann        | \>=8.3.RC1  | <!-- codespell:ignore -->
| python      | 3.11        |
| torch       | 2.7.1       |
| torch_npu   | 2.7.1       |
| vllm        | 0.11.0      |
| vllm_ascend | 0.11.0rc0   |

## Installation

```angular2html
pip install checkpoint-engine
```
Using the flexible P2P implementation requires installation of the Transfer Engine. However, ascend device cannot install transfer engine via pip, requires source compilation.

Reference document: [Ascend Direct Transport documentation](https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/ascend_direct_transport.md)


## Deploy vLLM Service

Since HCCL uses the default port 16666, when executing single-device multi-process tasks, you need to manually assign port to the processes.
Additionally, the underlying HIXL used by the Transfer Engine also defaults to port 16666 during link establishment, and currently there is no interface to modify this. Therefore, when Deploying vLLM serve, you must manually specify the port for the device via the ranktable file.

**ranktable file example:**
```
{
    "version": "1.0",
    "server_count": "2",
    "server_list": [
        {
            "server_id": "server1",
            "device": [
                {
                    "device_id": "0",
                    "device_ip": "ip1",
                    "device_port": "23333", // Choose an available port other than 16666
                    "rank_id": "0"
                },
                {
                    "device_id": "1",
                    "device_ip": "ip2",
                    "device_port": "23333",
                    "rank_id": "1"
                }...
            ]
        },
        {
            "server_id": "server2",
            "device": [
                {
                    "device_id": "0",
                    "device_ip": "ip8",
                    "device_port": "23333",
                    "rank_id": "8"
                }...
            ]
        }...
    ]
}
```

Set the `RANK_TABLE_FILE` environment variable when starting vLLM.
```shell
RANK_TABLE_FILE=ranktable.json VLLM_SERVER_DEV_MODE=1 python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port 19730 --trust-remote-code --tensor-parallel-size=8 --max-model-len 4096 \
    --load-format dummy --served-model-name checkpoint-engine-demo \
    --model /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/ \
    --worker-extension-cls checkpoint_engine.worker.VllmColocateWorkerExtension
```

The command to start the Checkpoint Engine remains the same.
```shell
torchrun --nproc-per-node 8 --log_dir=$(pwd)/logs --redirect 3 examples/update.py --update-method all --checkpoint-path /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
```

## Important Notes

1. Set the `ASCEND_RT_VISIBLE_DEVICES` environment variable according to the actual NPUs in use. Failure to do so will cause host quantity validation to fail in P2P mode.
