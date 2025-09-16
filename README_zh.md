在今年 7 月发布的 [Kimi K2](https://github.com/MoonshotAI/Kimi-K2) 模型中，我们实现了 RL 期间 1T 参数模型仅用约 20s 完成整个参数更新流程，显著优化了的 RL E2E 耗时的关键效率瓶颈。在实现高效参数更新的路上我们也踩了很多坑，希望写一篇文章也来聊聊我们遇到的一些问题和尝试的解法。

## 我们的场景是怎样的

目前在 LLM 的强化学习训练主要分为 colocate 和 disaggregation 两种架构，colocate 即训推共享 GPU 资源，会交替占用 GPU；disaggregation 即训推分离，各自占用不同的 GPU device。在这两种架构里面，一个很重要的阶段是参数更新，即每一轮训练结束之后需要把参数同步给推理框架然后开始 rollout，这个阶段里面如果占用时间过长，会导致 GPU 一直在空转，无法提高整体的 GPU 利用率，训练端到端的性能会有比较大的损失，因此在 RL 训练中，高效的参数更新是一个值得优化的方向。

我们内部的实验中，colocate 场景会多一些，因此我们也着重针对了 colocate 做了相关优化。我们的 colocate RL 和社区大部分开源框架的不太一样，我们并没有采用 Ray 这种通用的方案，而是让训练和推理用两个不同的容器共享一台机器来部署，这样的好处是能够让训练和推理完全解耦，各自的功能迭代互相不影响，同时它们的环境和镜像也是完全解耦的，这套方案让我们内部训练和推理的迭代非常顺畅，同时这种部署方式和我们线上 Kimi 的部署方式比较类似，因此也能较好的复用一些线上服务的基础设施。当然它也带来了一个问题，即训练和推理较难感知各自的 Process Group，因此给参数更新带来了一定的难度。因此我们当时在设计的时候，期望能提供一层轻量的中间层，把训推能够连起来，尽可能的不去动训练和推理 Process Group 的逻辑，于是我们开始着手设计 **checkpoint-engine**，希望在训推之间建立一个 bridge 去实现参数更新。

![checkpoint-engine](//statics.moonshot.cn/checkpoint-engine-blog/assets/checkpoint-engine.png)

需要注意的是，我们当时选择这个方案也是经过了一些考量和权衡的。从理论上来说，性能最优的方案当然是让 RL 框架能感知训推的并行度，然后有一个统一的控制器去做 reshard 实现参数更新，这样肯定是能实现最快速的参数更新，而且也不会存在冗余的数据发送。但是从工程上来说，我们不想侵入太多逻辑到训练和推理引擎中，因此做了一个权衡，把 checkpoint-engine 设计成了每张卡都会收到全量的权重，这样在推理已经拆分 TP or EP 的情况下肯定会存在数据的冗余，但是从网卡和 NVLink 带宽上来说，H800 这种机型能提供机间 or 机内至少 100GiB/s 的性能，因此全量 1TiB 模型权重在所有机器通讯一次使用 10s 在我们看来是一个可以接受的范围，因此我们采用了这个从工程上最解耦的一个方案。

## 最初的逻辑

在最初开发 [k1.5](https://arxiv.org/abs/2501.12599) 的时候，我们就实现了 checkpoint-engine 参数更新的逻辑，思路就是通过 CUDA IPC 来传输 tensor 数据，在 inference 里提供一个接口接受 CUDA IPC 的 tensor，然后 train 的每个 rank 会对应一个 checkpoint-engine，每次 train 完会把 weights 传给对应的 checkpoint-engine，checkpoint-engine 会先通过 broadcast 把参数传给每一个 rank，然后再通过 CUDA IPC 把每一个 tensor 打包成 IPC handle 共享给 inference 实现参数更新，大致流程如下图所示。在传输的时候，因为显存无法装下所有 weights，因此我们最初采取的分组是 per layer per ep，这个组合当时基本能满足我们的需求。

![init](//statics.moonshot.cn/checkpoint-engine-blog/assets/init.png)
*最初的 checkpoint-engine 参数更新方案*

## 踩坑和填坑

随着模型尺寸的进一步增加，我们发现这种 per layer per ep 和把每一个 tensor 都打包成 IPC handle 的方式会造成很大的开销。我们在千卡 H800 上跑 Kimi K2 RL 的时候遇到了很大的性能瓶颈，参数更新时间甚至能达到 10min，因此我们不得不着手做深度优化。

### 共用 buffer 并实现两阶段流水线

在做 profile 之后，我们发现了几个问题

- per layer per ep 这种模式会导致显存占用很不稳定，有时候会导致参数更新的时候直接 CUDA OOM
- per layer per ep 是每个 tensor 各自发 async broadcast，最终等发完再去做 update weights，可能会存在很多细碎的小通讯
- 每个 tensor 一个 IPC handle 这种模式会导致 vLLM 那边做序列化和反序列化时间非常长，初版实现会把所有 rank 的 ipc handle 都 gather 到一起传给 vLLM，vLLM 自己传给各个 TP 或 EP 的时候会传全量数据
- Checkpoint-engine 里的通讯操作和 vLLM 里 update weights 的操作是串行的

因此基于上面的问题，我们的解决方案也比较清晰了，就是采用 bucket 的方式攒一下 batch，把细碎的 tensor 放入一个固定 bucket size 的 buffer 里

1. 每次只 broadcast 这个 buffer，这样就能干掉小通讯带来的 overhead，也能让显存占用非常稳定
2. 把这个 buffer 一开始就 share 给 vLLM，二者把 buffer 当做 channel 来做数据面的传输，这样能避免每次请求传输 ipc handle 带来的开销，仅第一次请求有传 ipc handle 的开销，后面传输的都是 tensor meta 信息，能极大的减轻 vLLM 序列化和反序列化开销
3. 同时我们针对 ipc handle 的 gather 逻辑做了优化
   1. 不用每次请求都 gather 一次 ipc handle，仅第一个请求之前 gather 就行，减少了通讯操作
   2. 不用所有 rank all_gather ipc handles，因为每个 vLLM 实例只会有一个 rank 上的 checkpoint-engine 会发起请求，因此把 ipc_handle 都 gather 给它就行，不用做全局 all_gather
4. 为了把 broadcast 和 vLLM 里的 update weights 操作 overlap 起来，我们需要使用双 buffer 实现两阶段流水线

总体的实现思路如下

![overlap](//statics.moonshot.cn/checkpoint-engine-blog/assets/overlap.png)
*双 buffer 实现两阶段流水线*

通过上面的 4 个优化，我们成功的把千卡 H800 上参数更新的时间从 10min 降低到了 2min，基本能满足内部 RL 训练的需求了。

### 提高 H2D 速度并优化推理引擎参数更新性能

但是从理论上，这个速度应该是有提升空间的，在 H800 或者 H20 这类 GPU 机器之间互联的网络 broadcast 带宽至少是能达到 100GiB/s，所以 Kimi K2 1TiB 的权重文件同步一次的时间理论最优应该是可以达到 10s 以内的，那这里一定是有一些优化空间的。

我们发现在上面的方案里有几个问题

- checkpoint-engine 在每次 Broadcast 之前都需要等待单个 rank 做完一次 H2D (Host To Device)，然而机器上的 H2D 速度基本上只能打到 40-50GiB/s，因此会导致带宽会被 bound 到单次的 H2D 上，无法发挥网卡的全部性能
- vLLM 在 update weights 的时候，有一些开销，具体体现在
  - [每个请求都需要算一遍 dict(self.named_parameters())](https://github.com/vllm-project/vllm/blob/v0.10.2rc1/vllm/model_executor/models/deepseek_v2.py#L939)，这里每次都需要让 Python 做一些 CPU bound 操作
  - 在 expert 权重更新的时候[使用 .item 会频繁触发 GPU -> CPU 的同步](https://github.com/vllm-project/vllm/blob/v0.10.2rc1/vllm/model_executor/layers/fused_moe/layer.py#L1151)，会导致 update weights 的速度非常不稳定，时快时慢。

因此我们又做了3个优化

1. 尝试把 H2D 和 Broadcast 做 Overlap
2. 缓存 `dict(self.named_parameters())`
3. 在 cpu 上缓存 `expert_map`，避免频繁 CUDA synchronize

优化 2 和 3 实现起来很简单，优化 1 我们想的很完美，认为能实现类似下图一样的非常完美的 overlap

![three-stage-pipeline](//statics.moonshot.cn/checkpoint-engine-blog/assets/three-stage-pipeline.png)

但是实测发现在 H800 和 H20 这种机型里 Broadcast 和 H2D 会互相抢占 PCIE 带宽，让二者都出现了降速，结果变成了这样

![three-stage-pipeline-pcie](//statics.moonshot.cn/checkpoint-engine-blog/assets/three-stage-pipeline-pcie.png)

那是否存在能不被单个 PCIE 速度 bound 的方案呢？我们发现可以让每个节点先同时做一遍 H2D，这样就能达到比较大的聚合 H2D 带宽而不是大家都在等一个 rank 做 H2D，到时候需要 broadcast 的时候直接从已经 H2D 的数据里走一次快速的 D2D 就可以把数据放到 broadcast 的 buffer 里做 broadcast 了，因为 D2D 的速度非常快，这里基本可以忽略它的开销。使用这个方案我们就能在 H2D 的时候把所有机器的 PCIE 用起来，提高整体的吞吐，最终实现的流水线如下图所示

![fixed-two-stage-pipeline](//statics.moonshot.cn/checkpoint-engine-blog/assets/fixed-two-stage-pipeline.png)

经过这些优化，我们内部测试**在千卡 H800 下仅用 20s** 即可以实现 Kimi K2 模型的参数更新，而且速度稳定，基本没有发生过某几次参数更新很慢的情况。

## 推理引擎的故障自愈

当我们实现了高效的参数更新逻辑之后，我们发现在强化学习中还存在另一个问题，就是推理引擎会时不时出现一些故障导致 RL 训练 Crash。当然这个问题能想到一个简单的解决方案，即一个推理引擎故障了，我们不要让整个任务挂掉，直接重启它即可。但是需要注意的是，在 RL 流程中，我们是直接通过不落盘的方式把 weights 从训练传给推理的，这个时候如果推理想重启，看起来只能从 train 的 checkpoint 中重新 convert 一份重启，这样会有比较长的 IO 等待，会导致重启速度比较慢。最好的方式是希望通过 checkpoint-engine 实现重启的推理实例在线更新 weights。

在当时的设计里我们其实是无法实现这个功能的，因为所有的 inference engine 的参数更新逻辑是同步的，无法实现只更新部分 inference 实例的权重。如果无脑的触发一次所有 inference engine 的参数更新流程，虽然我们可以尝试让那些运行中的实例不要 update weights 只做 broadcast，但是这个会导致它们需要额外申请 GPU 显存，这对本来对显存敏感的 RL 任务而言是不可接受的，因此我们需要一种能直接从运行中实例的 CPU RDMA 把 weights 读到故障实例 GPU 中的传输框架。刚好 [mooncake-transfer-engine](https://github.com/kvcache-ai/Mooncake) 能完美满足这个要求。

因此我们和 Mooncake 的同学一起把 `mooncake-transfer-engine` 集成到了我们的系统里，我们实现了一个简单的方案，即让故障机器的 rank0 把数据按照 bucket_size 从 remote CPU RDMA P2P 的读到 rank0 的 GPU 里，然后 broadcast 给故障实例的其他 ranks，接着再触发参数更新，这样就能优雅的实现仅更新部分实例的参数，实现高效的故障自愈。这个方案能实现 40s 更新故障实例的权重，这个速度对于单点故障恢复而言完全够用了。

![recover](//statics.moonshot.cn/checkpoint-engine-blog/assets/recover.png)

推理引擎故障自愈

## 推理启动加速

在我们内部的非 RL 场景通常也会有需求启动一批推理服务，针对这个场景我们内部已经做了一些优化，能把权重先预热到 `/dev/shm` 里让推理引擎读取，这样能比直接从分布式文件系统读快很多。但是代价是会占用较多的内存空间，同时等待预热也需要时间。

但当我们把 Kimi K2 参数同步的开销优化到 20s 这个量级之后，我们发现这个速度比推理引擎直接从磁盘甚至 `/dev/shm` 中读 weights 都快了很多。而且我们发现 checkpoint-engine 从磁盘里注册 checkpoint 的操作可以和 vLLM 启动完全 overlap 起来，vLLM 启动的时候也有一些类似 `torch.compile` 和 Capture CUDA Graph 的操作，我们没必要串行的让 vLLM 读完 weights 在做这些操作。

于是我们让 vLLM 先 dummy 的启动，同时开一个 sidecar 启动 checkpoint engine 注册 checkpoint。等待 vLLM 就绪了之后直接触发所有实例 update weights。实践下来我们可以做到在和 vLLM dummy 启动比较接近的时间内启动所有 vLLM 实例，极大的提高了启动速度。我们内部相当一部分的推理服务已经在使用这个功能，极大的提高了用户的使用体验。

## 开源

在之后的两个月里，这套高性能参数更新的方案稳定的支撑了我们的 RL 训练，我们逐渐意识到它具有较好的可扩展性和灵活性，因此萌生了把它开源给社区使用的想法。但是我们内部的 checkpoint-engine 有两层，一层耦合了一些我们的 RL 业务逻辑，承担了 Convert checkpoint、管理 vLLM 生命周期、自动故障恢复等逻辑，另一层是核心的参数更新逻辑，即 `ParameterServer`。我们希望能把 `ParameterServer` 解耦出来，提供方便灵活的接口，让大家能更好更快的用上我们的优化。另一方面我们也希望和 vLLM 社区讨论一个高性能的 update weights 接口。

于是我们将我们内部 vLLM update weights 的思路[提给了 vLLM 官方](https://github.com/vllm-project/vllm/issues/24163)，在和 [@游凯超](https://www.zhihu.com/people/176cf88046a1cae595b55e12d58c95e9) 的讨论中也凯超也给了我们一些 idea，最终让我们敲定了一个比较优雅的接口，将和 vLLM 的控制面交互从 HTTP 请求改成了 [zmq](https://zeromq.org/) 队列，最终 merge 到了 [vLLM 官方的 examples](https://github.com/vllm-project/vllm/pull/24295) 里。

最后我们把 `ParameterServer` 单独拆出来，开源了 [checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine)，在实现高效参数更新的情况下也提供了灵活好用的接口，测试下来在各大模型的参数更新速度都比较高效

| Model                                | Device Info  | GatherMetas | Update (Broadcast) | Update (P2P)      |
| ------------------------------------ | ------------ | ----------- | ------------------ | ----------------- |
| GLM-4.5-Air (BF16)                   | 8xH800 TP8   | 0.17s       | 3.94s (1.42GiB)    | 8.83s (4.77GiB)   |
| Qwen3-235B-A22B-Instruct-2507 (BF16) | 8xH800 TP8   | 0.46s       | 6.75s (2.69GiB)    | 16.47s (4.05GiB)  |
| DeepSeek-V3.1 (FP8)                  | 16xH20 TP16  | 1.44s       | 12.22s (2.38GiB)   | 25.77s (3.61GiB)  |
| Kimi-K2-Instruct (FP8)               | 16xH20 TP16  | 1.81s       | 15.45s (2.93GiB)   | 36.24s (4.46GiB)  |
| DeepSeek-V3.1 (FP8)                  | 256xH20 TP16 | 1.40s       | 13.88s (2.54GiB)   | 33.30s (3.86 GiB) |
| Kimi-K2-Instruct (FP8)               | 256xH20 TP16 | 1.88s       | 21.50s (2.99GiB)   | 34.49s (4.57 GiB) |

开源版的 checkpoint-engine 并没有和推理框架做耦合，而是可以让用户在 update weights 的时候提供一个自定义的 `req_func` 来决定要怎么和推理引擎交互，这样能非常方便的去对接各种不同的推理引擎。推理引擎也可以自己按照需求去决定一些量化的逻辑。

通过我们的接口，大家使用下面的代码就可以比较容易的实现权重更新逻辑

```python3
ps = ParameterServer(auto_pg=True)
ps.register_checkpoint(name, files=files, named_tensors=named_tensors)
ps.gather_metas()
ps.update(name, req_func)
```

上面的代码会帮你托管 NCCL Group 的创建和销毁，如果你想自己管理 NCCL，**也可以不配置 `auto_pg=True`，这样可以自己来管理 NCCL Group。**

另一方面 checkpoint-engine 也支持多种不同的用法

- 支持 **Fully Broadcast** 和 **P2P** 两种参数更新方式，前者是我们最快的实现，能够一把同时更新上千张卡的推理引擎权重，后者更灵活，适合用于**故障自愈、disaggregation** 这些场景，能够动态的从已有权重的 checkpoint-engine RDMA 的拉取权重，无需额外占用运行中推理实例的显存即可快速实现新实例的参数更新。
- **支持注册多个 checkpoint**，你可以灵活的在多个 checkpoint 中间切换，只需调用多次 register_checkpoint 把不同 name 的 checkpoint 注册到 ps 里，在 update 的时候指定 checkpoint_name 即可使用这个 checkpoint 去更新推理引擎，这样 checkpoint-engine 就成为了一个参数服务。在线和离线的推理引擎都能非常方便的去切换模型的不同版本做测试。

需要注意的是，P2P 目前的实现方案还比较 naive，每次都是 `rank0` 串行的读取权重再广播给其他 rank，这里其实可以做 overlap，例如让别的 rank 也同时读取权重，实现类似 Fixed two-stage pipeline 的效果，这块未来我们也会持续优化，也欢迎社区的朋友们感兴趣来一起实现。

最后感谢 vLLM 社区 [游凯超](https://github.com/youkaichao)  提供了一个科学的推理引擎接口。本文的一些相关技术方案也能在 [Kimi K2 Technical Report](https://arxiv.org/abs/2507.20534) 里找到。希望大家来试用一下 [checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine)，如果发现有什么问题，求提 Issue or PR，期望我们能和社区一起持续迭代优化，提高参数更新的速度和体验！
