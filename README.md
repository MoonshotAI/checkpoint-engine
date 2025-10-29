# Checkpoint Engine
Checkpoint-engine is a simple middleware to update model weights in LLM inference engines -- a critical step in reinforcement learning.
We provide an efficient and lightweight implementation for inplace weight update:
updating our [Kimi-K2](https://github.com/MoonshotAI/Kimi-K2) model (1 Trillion parameters) across thousands of GPUs takes about 20s.


<div align="center">
  <picture>
      <img src="figures/checkpoint-engine.png" width="80%" alt="ckpt-engine">
  </picture>
</div>

## Architecture

The core weight update logic is in `ParameterServer` class, a service colocated with inference engines. It provides two implementations of weight update: Broadcast and P2P.

- **Broadcast**: Used when a large number of inference instances need to update weights in synchronous. This is the fastest implementation and should be used as the default update method. See `_update_per_bucket` with `ranks == None or []`.
- **P2P**: Used when new inference instances are dynamically added (due to restarts or dynamic availability) while the existing instances are already serving requests. Under this scenario, to avoid affecting the workloads on existing instances, we use the [`mooncake-transfer-engine`](https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#use-python-package) to P2P send weights from CPUs in existing instances to GPUs in new instances. See `_update_per_bucket` with `ranks` specified.

### Optimized Weight Broadcast
In the *Broadcast* implementation, the checkpoint-engine holds references to sharded weights in CPU memory, and need to efficiently broadcast them to a cluster of inference instances, often under a different sharding pattern.
We arrange the data transfer into 3 stages:
1. H2D: moving weights to GPU memory. These weights may come from disk or the training engine.
2. broadcast: broadcast among checkpoint engine workers; the data results in a CUDA IPC buffer shared with inference engine.
3. reload: inference engine decides what subset of weights to copy from the broadcasted data.

Checkpoint-engine orchestrates the entire transfer process. It first gathers necessary metadata to create a plan, including deciding the proper bucket size for data transfer.
It then executes the transfer, where it controls the inference engine through a ZeroMQ socket. To maximize performance, it organizes the data transfers into a pipeline with overlapped communication and copy, illustrated below. The details can be found in [Kimi-K2 Technical Report](https://arxiv.org/abs/2507.20534).


<div align="center">
  <picture>
      <img src="figures/pipeline.png" width="80%" alt="pipeline">
  </picture>
</div>

Pipelining naturally requires more GPU memory. When memory is not enough, checkpoint-engine will fallback to serial execution.

### Optimized P2P Bucket Assignment
In the *P2P* implementation, checkpoint-engine needs to send weights from existing instances to new instances.
To minimize the overall transfer time, checkpoint-engine optimizes the bucket assignment for each sender-receiver pair.
The optimization goal is to make full use of the available network bandwidth for each sender and receiver.
See [issue #25](https://github.com/MoonshotAI/checkpoint-engine/issues/25)

## Benchmark

| Model                                | Device Info  | GatherMetas | Update (Broadcast) | Update (P2P)            |
| :----------------------------------- | :----------- | :---------- |:-------------------| :---------------------- |
| GLM-4.5-Air (BF16)                   | 8xH800 TP8   | 0.12s       | 3.47s (3.02GiB)    | 4.12s (3.02GiB)         |
| Qwen3-235B-A22B-Instruct-2507 (BF16) | 8xH800 TP8   | 0.33s       | 6.22s (2.67GiB)    | 7.10s (2.68GiB)         |
| DeepSeek-V3.1 (FP8)                  | 16xH20 TP16  | 1.17s       | 10.19s (5.39GiB)   | 11.80s (5.41GiB)        |
| Kimi-K2-Instruct (FP8)               | 16xH20 TP16  | 1.33s       | 14.36s (5.89GiB)   | 17.49s (5.91GiB)        |
| DeepSeek-V3.1 (FP8)                  | 256xH20 TP16 | 0.80s       | 11.33s (8.00GiB)   | 11.81s (8.00GiB)        |
| Kimi-K2-Instruct (FP8)               | 256xH20 TP16 | 1.22s       | 16.04s (8.00GiB)   | 16.75s (8.00GiB)        |

All results above are tested by [`examples/update.py`](./examples/update.py) and use [vLLM v0.10.2rc1](https://github.com/vllm-project/vllm/tree/v0.10.2rc1) as inference engine. Some notes:

* FP8 test needs additional vLLM patches, see [FP8 quantization](#fp8-quantization).
* Device Info: we tested various combination of devices and parallelism setups. For example, a 256-GPU TP16 setup means that we deploy 16 vLLM instances, each with 16-way tensor parallelism.
* Since update duration is related to IPC bucket size, we provide the bucket size in the table.
* The P2P time were tested for updating no more than two nodes (16 GPUs) (`ParameterServer.update(ranks=range(0, 16))`) out of the entire cluster.
* We bind each GPU to its corresponding NUMA node to ensure stable H2D transfer speeds.

## Installation

Use the fastest broadcast implementation

```Bash
pip install checkpoint-engine
```

Use the flexible P2P implementation, notice this will install `mooncake-transfer-engine` to support RDMA transfer between different ranks.

```Bash
pip install 'checkpoint-engine[p2p]'
```

If set `NCCL_IB_HCA` env, checkpoint-engine will use it to auto select net devices for different ranks. Available patterns can be found from [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#id8). If not set, it will read all RDMA devices and try to divide them into each rank.

## Getting Started

Prepare an H800 or H20 machine with 8 GPUs with latest vLLM. Be sure to include [/collective_rpc API endpoint](https://github.com/vllm-project/vllm/commit/f7cf5b512ee41f36613deb2471a44de5f304f70d) commit (available in main branch) since checkpoint-engine will use this endpoint to update weights.

```Bash
cd /opt && git clone https://github.com/vllm-project/vllm && cd vllm
uv venv --python 3.12 --seed
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

Install checkpoint-engine

```Bash
uv pip install 'checkpoint-engine[p2p]'
```

We use `Qwen/Qwen3-235B-A22B-Instruct-2507` (BF16) as the test model

```Bash
hf download Qwen/Qwen3-235B-A22B-Instruct-2507 --local-dir /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
```

Start vLLM in dev mode and set `--load-format dummy`. Notice that we also set `--worker-extension-cls=checkpoint_engine.worker.VllmColocateWorkerExtension`

```Bash
VLLM_SERVER_DEV_MODE=1 python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 19730 --trust-remote-code \
    --tensor-parallel-size=8 --max-model-len 4096 --load-format dummy \
    --served-model-name checkpoint-engine-demo --model /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/ \
    --worker-extension-cls checkpoint_engine.worker.VllmColocateWorkerExtension
```

Meanwhile, use this command to update weights by checkpoint-engine. No need to wait for vLLM to get ready.

```Bash
torchrun --nproc-per-node 8 examples/update.py --update-method all --checkpoint-path /opt/models/Qwen/Qwen3-235B-A22B-Instruct-2507/
```

### Reuse weights from existing instances

New checkpoint-engine instances can join existing instances and reuse their weights. This is simple to achieve.

First, start the existing instances with `--save-metas-file global_metas.pkl` to save global metas to a file and use `--sleep-time 300` to make sure they stay alive.

```Bash
torchrun --nproc-per-node 8 examples/update.py --checkpoint-path $MODEL_PATH \
    --sleep-time 300 --save-metas-file global_metas.pkl
```

After a checkpoint is registered, new instances can obtain a copy of the checkpoint by setting `--load-metas-file global_metas.pkl`.

```Bash
torchrun --nproc-per-node 8 examples/update.py --load-metas-file global_metas.pkl
```

### FP8 quantization

FP8 quantization currently do not natively work in vLLM when updating weights.
We provide a simple patch in [`patches/vllm_fp8.patch`](./patches/vllm_fp8.patch) to handle the correct weight update.
Notice this patch is only tested in DeepSeek-V3.1 and Kimi-K2. Other models may meet some compatible issues.

A [PR](https://github.com/vllm-project/vllm/pull/24488) is opened to the vLLM project and waiting to discuss and review.

### Test

Run a simple correctness test for checkpoint_engine

```bash
torchrun --nproc-per-node 8 tests/test_update.py
```

Other unit tests can be done with pytest.
## Limitations and Future Work

- This project is currently only tested with vLLM. But it is easy to integrate with other frameworks like SGLang.
- The perfect three-stage pipeline mentioned in our paper is currently not implemented. This could be useful for architectures where H2D and broadcast do not conflict in PCIE.

## Acknowledgments

This open source project uses the same vLLM interface in https://github.com/vllm-project/vllm/pull/24295 . Thanks for the comments and insights from [youkaichao](https://github.com/youkaichao).
