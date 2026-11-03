# USL-Ray

`usl-ray` 是本仓库中基于 Ray 的独立 U-shaped Split Learning 实现。旧版
`usl/` 包保持不变，新包不会导入它。

## 架构

- Client/server 边界张量使用 Ray 对象存储：发送方调用 `ray.put`，将顶层
  `ObjectRef` 传给 `peer.push.remote`，接收方的 raylet 自动拉取对象。
- 服务端 GPU 使用 torch 2.9 的 `PipelineStage`，调度器为 `ScheduleGPipe`
  或 `Schedule1F1B`；stage 间张量通过 NCCL 传输。
- Attention mask、position ID 和 RoPE 张量不会跨越边界。`SequenceSpec`
  只携带标量元数据，每个 partition 根据本地模型配置重建这些张量。
- 客户端调度使用有界 `for` 循环和 Future，不使用轮询线程、`while True`
  或应用层网络 sleep。
- 真实测量与模拟运行共用 `Span`、`StepTimeline`、`StepMetrics`、
  `LinkModel` 以及同一个层划分函数。

## 安装

依赖版本与开发机器一致：Python 3.11、torch 2.9.1、transformers 5.0.0、
Ray 2.58.0。

```bash
cd /opt/tiger/CostEfficientUSL/usl-ray
pip install -e '.[dev]'
```

`pyproject.toml` 是版本的唯一来源，仓库根目录的 `requirements.txt`
同步其运行时固定版本。不保证 legacy 包可以在这些版本下运行。

网络整形还需要：

```bash
tc -V
sudo -n true
```

## 配置

每个实验目录包含：

- `common.yaml`：模型、切分、batch、全局 step、流水线、预期网络条件和
  profile 配置；
- `server.yaml`：继承公共配置，并增加服务端 world size 和 backend；
- `client.yaml`：继承公共配置，并增加数据和 offload 配置。

Client 和 server 在 attach 时逐项比较共享字段。若配置不一致，训练开始前
会直接失败，并输出字段级差异。

命令行参数支持带类型的覆盖：

```bash
usl-ray server \
  --config experiments/configs/qwen3-1.7b/server.yaml \
  --set batch.batch_size=16
```

未知字段和非法组合会在加载阶段失败。主要约束包括：

- `batch_size` 必须能被 `micro_batch_size` 整除；
- 对称切分后必须至少保留一层 server layer；
- 多 stage 服务端使用固定 sequence shape，不能对变长序列排序；
- 左 padding 要求 `position_kind: from_valid_lens`；
- NF4 通信要求 hidden size 能被 64 整除。

解析后的配置不可变，并与每份 profile 一起保存。

## 单机实验

该功能测试拓扑只使用一个 raylet。Ray 传输仍走共享内存，因此流量控制不会
对其生效：

```bash
TOPO=same-node experiments/run_single.sh \
  experiments/configs/smoke-qwen3-0.6b
```

脚本遵循原始实验生命周期：

1. 启动 Ray；
2. 在后台启动 server；
3. 在前台启动 client；
4. 等待 server；
5. 正常退出或失败时停止 Ray。

使用 `--server-set` 和 `--client-set` 覆盖角色专属配置：

```bash
experiments/run_single.sh experiments/configs/smoke-qwen3-0.6b \
  --set train.steps=2 \
  --server-set server.activation_offload_mb=2 \
  --client-set client.activation_offload_mb=2 \
  --client-set client.model_state_offload_layers=1
```

### 性能分析与甘特图检查

运行默认的随机权重、合成数据 smoke 实验，并绘制最后一个非 warmup step：

```bash
experiments/run_profile_gantt.sh
```

脚本会输出生成的 `profile.json` 和 `gantt.png` 路径。它也接受实验目录，
并支持与 `run_single.sh` 相同的 `--set`、`--server-set` 和
`--client-set`：

```bash
experiments/run_profile_gantt.sh experiments/configs/qwen3-1.7b \
  --set train.steps=3 \
  --set profile.warmup_steps=1 \
  --server-set server.world_size=2
```

甘特图始终只包含 warmup 之后最后一个 steady global step，避免跨 step
重叠或首次优化器初始化干扰时间线。默认的 `TOPO=same-node` 运行可以验证
时间线结构，但 Ray 在同一个 raylet 内使用共享内存，因此其传输 Span
不代表 WAN 性能。

## 双机 WAN 实验

Ray 对象传输必须跨越两个 raylet，`tc` 才会生效。

在 server 机器上：

```bash
ray start --head \
  --node-ip-address=<SERVER_IP> \
  --port=26379 \
  --num-gpus=4 \
  --resources='{"usl_server": 4}' \
  --disable-usage-stats

sudo experiments/tc/shape.sh apply <BOND_IFACE> <CLIENT_IP> 230 10
export RAY_ADDRESS=<SERVER_IP>:26379
experiments/run_server.sh experiments/configs/qwen3-1.7b/server.yaml
```

在 client 机器上：

```bash
ray start \
  --address=<SERVER_IP>:26379 \
  --node-ip-address=<CLIENT_IP> \
  --num-gpus=1 \
  --resources='{"usl_client": 1}' \
  --disable-usage-stats

sudo experiments/tc/shape.sh apply <BOND_IFACE> <SERVER_IP> 230 10
export RAY_ADDRESS=<SERVER_IP>:26379
experiments/run_client.sh experiments/configs/qwen3-1.7b/client.yaml
```

`shape.sh` 使用 HTB 和 netem，并按对端 IP 过滤，因此本机 NCCL 和无关流量
不会被限速。实验结束后运行 `shape.sh clear <iface>`。如果物理网卡是 bond
的 slave，应在 bond 设备上进行整形。

`experiments/bench_link.py` 会在两个方向传输新的 1/4/16/64 MiB Ray 对象，
并拟合 \(T(n)=a+n/b\)：

```bash
python experiments/bench_link.py --expected-mbps 230
```

带宽使用 tc 的 SI 单位：1 Mbit/s 等于 1,000,000 bit/s。

### 网络命名空间拓扑

`experiments/tc/netns.sh` 和 `TOPO=netns` 会创建两个通过 veth 对连接的本地
raylet。该操作需要 `CAP_SYS_ADMIN`。开发容器不具备此 capability
（`mount --make-shared /run/netns` 被拒绝），因此尚未在这里运行该拓扑。
请在具备相应 capability 的宿主机上运行，或使用上面的双机拓扑。

## 流水线模式

| 新名称 | 旧实现名称 | 客户端顺序 | 服务端 |
|---|---|---|---|
| `sequential` | `naive` | 单个微批执行完整 U 形路径 | GPipe |
| `gpipe` | `gpipe` | Head 和 Tail 均使用 fill-drain | GPipe |
| `splitmind` | `pipedream_wc` | 全部 Head forward、Tail F/B、全部 Head backward | GPipe |
| `splitmind_eager` | `pipedream_wc_eager` | 激活或梯度 Future 就绪后立即消费 | GPipe |
| `1f1b` | `pipedream_strict` | 有界 warmup，之后 one-forward/one-backward | 1F1B |

Torch 2.9.1 的原生 pipeline 邻居初始化存在 bug：它将 int64 标量发送到
float32 接收 buffer。`USLPipelineStage` 只覆盖该初始化方法，使 gloo 和
NCCL 使用一致的 float32 元数据；其余调度仍使用原生 torch 实现。

## 模型支持与扩展

Qwen3、Llama 和 Gemma2 共用一个 `DecoderPartition`，架构差异集中在
`models/arch.py`。增加新模型时：

1. 添加一条 `ArchSpec`；
2. 在 `tests/unit/test_partition_equivalence.py` 中增加一组小型随机配置；
3. 验证完整模型和 partition 链输出的 logits 一致。

Gemma2 保留原始 layer index，并使用 transformers 实际的滑窗 attention
mask。GPT-2 被有意排除，不提供支持。

## 性能分析与绘图

当 `profile.enabled` 为 true 时，client 写入：

```text
runs/<run.name>/<run_tag>/
├── config.yaml
└── profile.json
```

`profile.json` 包含解析后的配置、版本信息、client/server 显存、计算操作与
Ray 传输 Span，以及排除 warmup step 后的指标。为了保持与 legacy 相同的
测量口径，计时 Span 前后会同步 CUDA 操作。

甘特图只绘制最后一个 steady global step，在每个微批 Span 中标注
microbatch 编号，并隐藏 `head_step`、`tail_step` 和 `server_step`
优化器更新。原始 profile 数据不会被删除。绘图命令为：

```bash
usl-ray plot runs/.../profile.json
```

## 校准、模拟与搜索

使用七组实测配置拟合成本模型：

```bash
MAX_SPLIT_POINT=14 \
experiments/calibrate.sh experiments/configs/qwen3-1.7b
```

也可以直接调用拟合命令：

```bash
usl-ray calibrate <seven profile.json paths> \
  --base-split-point 2 \
  --base-batch-size 8 \
  --max-split-point 14 \
  --lora \
  --output runs/cost_model.json
```

模拟并对比：

```bash
usl-ray simulate \
  --config experiments/configs/qwen3-1.7b/client.yaml \
  --cost-model runs/cost_model.json \
  --output runs/simulation.json

usl-ray compare \
  --profile runs/.../profile.json \
  --cost-model runs/cost_model.json
```

Pareto 搜索：

```bash
usl-ray search \
  --config experiments/configs/qwen3-1.7b/client.yaml \
  --cost-model runs/cost_model.json \
  --max-batch-size 64 \
  --max-client-memory-mib 49152 \
  --output runs/pareto.json
```

Python 模拟器中的 `legacy_compat=True` 会复现旧 SplitMind 的计算方式，
包括固定的 10 ms 通信开销和 B39 的 client offload 条件。正常模拟会修复
B39，并使用 SI 链路单位。

## 验证记录

已在 8×NVIDIA A800-SXM4-80GB 上执行：

- 41 个单元测试和 2 个专用 CUDA offload 测试通过；
- 8 个 Ray/gloo 集成测试通过；
- 五种客户端调度的一步 loss 和每个参数更新均与完整模型一致
  （`atol=1e-5`）；
- Qwen3-0.6B 随机初始化使用两个 NCCL server GPU 和一个 client GPU
  运行了三个 step；
- NCCL 日志中出现 `P2P/CUMEM/read`；
- Qwen3-0.6B 还在 client/server activation offload 以及一个 client 模型层
  swap 的条件下完成了两个 profile step；
- 成功生成 profile JSON 和甘特图；
- activation 和 model-state offload 通过专用梯度/参数更新测试；
- 实验结束后所有 GPU 均恢复到 0 MiB；
- 确定性模拟器与 legacy fixture 的误差不超过 `1e-6`；
- 单机 netns 整形受开发容器 capability 限制，未能执行，详见上文。

## 与旧实现的行为差异

- padding 位置的 label 为 `-100`；`loss_on: completion` 可以限制监督范围；
- 网络带宽由 tc 控制，而不是通过 socket sleep 模拟；
- server 和 client 共用同一个学习率；
- warmup 之后的每个测量 step 都参与 profile 均值计算；
- 正确启用 Gemma2 滑窗 attention；
- 模拟抖动默认值为零；
- epoch 估算使用配置的数据集大小，而不是固定的 12,460；
- 删除 GPT-2 支持。

## 故障排查

- **Client 找不到 server**：检查 `ray status`、`RAY_ADDRESS`、`usl-ray`
  namespace，并确认两个 YAML 文件继承同一个公共配置。
- **Placement group 一直等待**：server 节点至少需要
  `server.world_size` 份 `GPU` 和 `usl_server` 资源；client 节点需要
  `GPU: 1` 和 `usl_client: 1`。
- **NCCL 卡住**：确认微批 shape 固定，检查 `NCCL_DEBUG=INFO`，并确认
  所有 stage 位于同一个 server 节点。
- **tc 不生效**：client/server 很可能共用一个 raylet。请使用两台机器或
  具备权限的 netns 拓扑。
- **超时**：异常中包含 `(step, microbatch, kind)`；检查第一个缺失的 key
  以及对端在此之前的计算 Span。
