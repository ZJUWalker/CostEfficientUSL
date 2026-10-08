# USL-Ray 重构执行计划

> 读者：执行本次重构的开发者，以及之后维护 `usl-ray/` 的人。
> 状态：**计划，尚未实施**。每完成第 6 节的一个阶段，在对应条目打勾，并在"实际偏差"处记录与计划不一致之处。
> 约定：`文件:行号` 均指老代码（`usl/`、`experiment/`）的当前版本。老代码目录保持原样不动，第 3.13 节列出的缺陷只在新代码中修复。

### 修订记录

| 版本 | 内容 |
|---|---|
| v1 | 初版 |
| v2 | 按评审意见修改：① position embeddings 不再跨边界传输，改为与 mask 同样的 Spec 方式，由每个 stage 自己生成（3.7）；② 列出老代码缺陷清单，并在新代码中逐条修复（3.13）；③ 启动方式改为放在 `usl-ray/experiments/` 下的 bash 脚本，分别启动服务端和客户端，超参数写在 yaml 中（3.2、4）；④ 依赖版本以本机为准，同时更新 `requirements.txt`（5）；⑤ 传输层只使用 Ray 原生的 ObjectRef 拉取，删除 socket 和应用层限速，带宽改由 Linux tc 控制（3.3、3.4） |

---

## 0. 摘要

| 主题 | 老实现 | 新实现（`usl-ray/`，包名 `usl_ray`） |
|---|---|---|
| 配置 | `ClientArgs`/`ServerArgs`/`SplitModelConfig`/`MainVariable` 等多处重复定义，入口脚本里还要二次推导 | 一棵 frozen dataclass 树；每个实验目录下 `common.yaml` + `server.yaml` + `client.yaml`，握手时校验两侧共享部分一致 |
| 启动 | `experiment/*.sh` 拼接 argparse 参数 | `usl-ray/experiments/run_single.sh` 等脚本：分别启动 server 和 client 进程，参数来自 yaml，网格实验用 `--set` 覆盖 |
| 公网传输 | 4 条 pickle + TCP socket、4 个轮询线程、应用层限速 | 发送方 `ray.put` 后调用 `peer.push.remote(key, ref)`，接收节点由 Ray 原生机制自动拉取对象；代码里没有任何限速逻辑 |
| 带宽模拟 | 在 socket 层 `sleep` | Linux tc（htb 限速 + netem 时延），在局域网内模拟 WAN；单机通过 netns + veth 实现 |
| 服务端多卡 | fork 了约 1.9k 行 torch pipelining | 每张卡一个 Ray Actor，内部直接使用 `torch.distributed.pipelining`（`PipelineStage` + `ScheduleGPipe`/`Schedule1F1B`）与 NCCL，只需首尾两个边界适配器 |
| 客户端流水线 | `while True` + `get(timeout=0.001)` + `sleep(0.001)` | 每种模式是一段按微批下标的直线 `for` 循环，用 `Future.result()` 等待 |
| Mask / 位置编码 | 传 4D mask 张量和 (cos, sin) 张量 | 只传 `SequenceSpec`（包含 `MaskSpec` 和 `PositionSpec`），各 stage 本地生成 mask、position_ids 和 RoPE |
| 模型拆分 | 4 个架构 × head/tail/server，共 12 个类，约 1.7k 行 | 一个通用的 `DecoderPartition`，每个架构一条 `ArchSpec` 数据 |
| Profile ↔ Simulate | 两套字段、两套文件名规则，传输时间硬编码 | 共用 `Span/Timeline` 结构、`summarize()`、层划分函数、链路模型和配置匹配规则 |
| 依赖 | `requirements.txt` 锁定 torch 2.6 / transformers 4.51，与本机不符 | 以本机版本为准：torch 2.9.1、transformers 5.0.0、ray 2.58.0 等 |
| 代码量 | 11.9k 行 | 目标约 4.2k 行（含移植的 offload） |

---

## 1. 背景与范围

### 1.1 现状快照

| 目录 | 行数 | 职责 |
|---|---|---|
| `usl/client/` | 1,814 | `client.py` 基类（模型、通信、offload、profile 和结果保存混在一起）+ 4 个调度子类 |
| `usl/server/` | 2,728 | `single_server.py`（已 deprecated，706 行）+ fork 的 pipelining（`server_stage.py` 1,307 行等） |
| `usl/llm/` | 1,727 | GPT2 / Llama / Qwen3 / Gemma2 的 head、tail、server 拆分类 |
| `usl/socket/` | 341 | pickle + TCP 通信，`Payload` |
| `usl/offload/` | 1,675 | 激活、模型参数、优化器状态卸载 |
| `usl/simulate/` | 1,329 | 时间与显存模拟、profile 拟合、帕累托搜索 |
| `usl/utils/` | 2,294 | 数据集、甘特图、QLoRA 通信压缩、加载工具 |
| `experiment/` | ~330 + bash | 入口脚本与网格实验 |

本机环境：8×A800 80GB、Python 3.11.2、iproute2 6.1.0（带 `tc`）、可以免密 `sudo`；目前没有 `data/models/` 和 `data/datasets/`。

### 1.2 需求与对应章节

| 编号 | 需求 | 章节 |
|---|---|---|
| R1 | 优化并统一 config 和 dataclass | 3.2 |
| R2 | 公网用 Ray 传中间张量；服务端多卡内部用 torch 自带流水线 + NCCL | 3.3、3.5 |
| R3 | 基于 Ray 优化客户端流水线，尽量不用 `while True` | 3.6 |
| R4 | Attention mask 只传类型，由服务端重建 | 3.7 |
| R4b | position embeddings 不传；如确有需要，用 Spec 方式传，由服务端自己生成 | 3.7 |
| R5 | 更通用的模型拆分方式，精简 `usl/llm` | 3.8 |
| R6 | 同步修改 simulate | 3.10 |
| R7 | 新代码放在 `usl-ray/`，老代码保留，新老代码无交集 | 2-R7、4 |
| R8 | 不过度工程化，目标是可读性与健壮性 | 1.3、1.4、4 |
| R9 | profile 与 simulate 的数据结构和算法保持一致 | 3.10 |
| R10 | 启动脚本放在 `experiments/`，用 bash 分别启动 client 和 server，超参数写成 yaml | 3.2、3.11、4 |
| R11 | 依赖按本机版本编写并更新 requirements；老代码不需要兼容 | 5 |
| R12 | 删除 socket，传输直接使用 Ray 的 ref 拉取；网速由 Linux tc 控制 | 3.3、3.4 |
| R13 | 修复老代码中的其它 bug（在新代码中修复） | 3.13 |

### 1.3 非目标（明确不做）

- 不引入 gRPC / protobuf、会话状态机、两阶段提交、断线恢复、模型指纹。
- 不在应用层做带宽模拟或流控，带宽完全交给 tc。
- 不支持多客户端、鉴权 / TLS、断点续训（训练结束时可以选择保存一次权重）。
- 不使用 hydra / OmegaConf，只用 `dataclasses` + PyYAML。
- 不使用 Ray Compiled Graph、Ray Direct Transport、Ray Train。
- 不再支持 GPT-2（论文实验只用 Qwen3 / Llama3 / Gemma2），扩展点见 3.8。

### 1.4 此前一版 usl-ray 的教训

此前做过一版 "Ray + gRPC" 的 `usl-ray`，已删除。暴露出四个问题：

1. **工程化过重。** proto、事务化提交、session 状态机、模型指纹，与论文无关的部分占据了大部分代码。
2. **生命周期偏离原设计。** 服务端成了常驻进程，客户端结束后服务端仍在运行。原设计是两端共享同一个 global step 数并同时结束，本计划恢复这一点（3.11）。
3. **两侧参数不一致导致握手失败。** 本计划让两侧的 yaml 都继承同一个 `common.yaml`，握手时再逐字段比对共享部分（3.11）。
4. **profile 没有接入 simulate。** 本计划把"训练产出 `profile.json` → 拟合 → 模拟 → 对比"作为验收主线（3.10、7）。

---

## 2. 需求分析

逐条列出老代码的问题、新代码的目标和验收标准。具体缺陷统一汇总在 3.13，这里只做概括。

**R1 配置。** 问题：同一字段在 `ClientArgs`（`client.py:39`）、`ServerArgs`（`server/base.py:40`）、`SplitModelConfig`、simulate 的 dataclass 中重复定义。offload 用 bool 和 int 两个字段表达同一件事，推导规则在两个入口脚本里各写了一遍，而且不一致（`client_run.py:126-135`、`server_run_mp.py:172-180`）。运行过程中还会修改配置（`client.py:395-399`）。"运行标识 / 文件名"有 3 份互相矛盾的实现。目标：一棵配置树，一个 `run_tag()`，派生量一律用只读属性，校验全部在加载阶段完成。验收：client、server、simulate 只接收 `ExperimentConfig`；非法组合在加载时报错，并指出字段路径。

**R2 / R12 传输。** 问题：`pickle.loads` 直接作用于网络数据（`socket_comm.py:145`）；轮询线程依靠字典约定来停止；应用层限速与真实网络行为不一致。目标：只用 Ray 原生 ObjectRef 拉取；服务端 stage 之间只走 NCCL，且直接使用 torch 自带的 `PipelineStage` 和调度器。验收：`usl_ray` 中不出现 `socket`、`pickle`、`time.sleep` 限速代码；2-stage 服务端在 gloo（CPU）和 NCCL（GPU）下都能跑通，并且梯度与整模型一致（7-E1、E3）。

**R3 客户端流水线。** 问题：`split_mind.py:133-192`、`gpipe.py:57-136`、`pipedream.py:132-180` 都是"`while True` + 超时轮询 + 完成标记列表 + `sleep`"，发送线程还靠 `AtomicBool`/`AtomicInt` 做门控。目标：调度函数只描述先做什么、再等什么，等待统一用 `Future`。验收：`rg "while True|time\.sleep" src/usl_ray` 无结果；用纯 Python 的假 trainer 即可测出每种调度的执行顺序。

**R4 / R4b Mask 与位置编码。** 问题：

- mask 和 (cos, sin) 都随激活一起上行（`client.py:339-352`）。以 Qwen3-1.7B、fp32、S=512 为例，激活约 4 MiB，mask 约 1 MiB，cos/sin 约 0.5 MiB；有 padding 时额外开销最多约 37%。
- (cos, sin) 传过去之后，服务端并没有用（`server_stage.py:819`）。
- rank>0 收不到 mask（`server_stage.py:1090-1093`）。
- tail 复用 head 的 mask，导致这些张量必须钉在 GPU 上（`client.py:316-322`）。

目标：跨边界只传一个小的 `SequenceSpec`。head、各 server stage 和 tail 都用同一个函数，从各自的模型 config 生成 mask、position_ids 和 RoPE。验收：任意切分下，partition 链的 logits 与 HF 整模型一致（fp32，`atol=1e-5`），覆盖 causal/full、有无 padding、Gemma2 滑窗层。

**R5 模型拆分。** 问题：12 个类高度重复，逻辑加载和物理加载两条路径并存，`layer_idx` 规则各写一遍；多卡切分有两个实现，并且都不支持 Gemma2。目标：拆分只由层号区间以及是否包含 embedding / norm / lm_head 决定；每个架构的差异只用一条数据描述。验收：`models/` 不超过 450 行；三种架构全部通过等价性测试。

**R6 / R9 Simulate 与 Profile 一致。** 问题：

- 传输时间按模型名硬编码（`profile_separate_nodes.py:43-56`），再加上一个写死的时延（`simulate.py:208`）。
- profile 和 simulate 使用两套文件名规则。
- 作为库调用时会因为引用全局变量而抛 `NameError`。
- 随机抖动导致结果无法复现。
- 层数为奇数时，服务端层数会算错。

目标：共享契约（3.10），profile 和 simulate 输出同一种 `Timeline`，并由同一个 `summarize()` 计算指标。验收：`usl-ray compare` 能直接用真实 `profile.json` 与模拟结果逐指标对比；关闭抖动后，移植后的模拟器与老模拟器输出逐项一致（7-E6）。

**R7 新老隔离。** `usl-ray/` 有独立的 `pyproject.toml`，源码在 `usl-ray/src/usl_ray/`。新包不 import `usl`，`usl/` 下的文件一个也不改。验收：ruff 的 `banned-api` 禁止 `usl`，另有单测扫描源码中的 `^(from|import) usl(\.|\s|$)`。

**R10 启动脚本。** 老脚本把超参数写成 bash 变量再拼成 argparse 参数，server 和 client 各传一遍，然后用 `sleep 5` 等待服务端就绪（`bash_usl_run_single.sh:22-33`）。新脚本保持"先后台启动 server、再前台启动 client、最后 `wait` server"的结构，超参数改为来自 yaml，网格实验对两侧传入同一组 `--set`。等待就绪改为 client 主动轮询服务端的 Ray 具名 Actor，并设置超时（3.11）。

**R11 依赖。** 见第 5 节。

**R13 缺陷修复。** 见 3.13 的表格。每条缺陷都对应一个回归测试，或一条可以执行的验证命令。

---

## 3. 技术方案

### 3.1 总体架构

```
          client node（数据所有者）                                 server node（N 张 GPU）
┌──────────────────────────────────────┐   tc 整形的链路    ┌──────────────────────────────────────────────┐
│ client 进程（driver，很薄）           │                  │ server 进程（driver，很薄）                    │
│   └ ClientActor（1 GPU）              │                  │   └ StageActor 0 ─NCCL─► … ─NCCL─► StageActor N-1 │
│      head fwd ── put+push(act) ───────┼──── Ray 拉取 ───►│      first stage                    last stage  │
│      tail fwd ◄─ mailbox ◄────────────┼◄─── Ray 拉取 ────┼──────────────────── put+push(act) ─┘          │
│      tail bwd ── put+push(grad) ──────┼──── Ray 拉取 ───►┼──────────────────► last stage                  │
│      head bwd ◄─ mailbox ◄────────────┼◄─── Ray 拉取 ────┤  first stage ── put+push(grad)                │
└──────────────────────────────────────┘                  └──────────────────────────────────────────────┘
  experiments/run_*.sh：ray start → server 进程（后台）→ client 进程（前台）→ wait → ray stop
```

- 跨边界一共只有 4 类消息，用键 `(step, mb, kind)` 区分，`kind ∈ {act_up, act_down, grad_up, grad_down}`。
- 每个微批跨边界的元数据只有 `SequenceSpec`。labels 和 input_ids 始终留在客户端。

### 3.2 配置统一（R1、R10）

**文件组织**：每个实验一个目录，两侧共享部分只写一次。

```
usl-ray/experiments/configs/
├── defaults.yaml                 # 全部字段的默认值
└── qwen3-1.7b/
    ├── common.yaml               # base: ../defaults.yaml；model/split/batch/train/pipeline/network
    ├── server.yaml               # base: common.yaml；只写 server: 段
    └── client.yaml               # base: common.yaml；只写 client: 段
```

```yaml
# common.yaml（两侧必须一致，握手时逐字段比对）
run:      { name: qwen3-1.7b, output_dir: runs, seed: 0 }
model:
  path: data/models/qwen/qwen3-1.7b   # 每个节点各自在本地解析；random 初始化时只需要 config.json
  init: pretrained                    # pretrained | random（只影响数值，不影响耗时和显存）
  dtype: float32                      # 必须显式设置，原因见 3.8
  attn_implementation: sdpa
  lora: { enabled: true, r: 8, alpha: 32, dropout: 0.1, target_modules: [q_proj, k_proj, v_proj] }
split:    { split_point: 4 }          # head 与 tail 各 split_point 层
batch:    { batch_size: 8, micro_batch_size: 1, max_seq_len: 512 }
train:    { steps: 20, lr: 5.0e-4 }   # 两侧共用的 global step 数和学习率
pipeline: { mode: splitmind }         # sequential | gpipe | splitmind | splitmind_eager | 1f1b
network:  { mbps: 230, delay_ms: 10, compression: none }   # 只用于描述和模拟；实际限速由 tc 完成，见 3.4
profile:  { enabled: true, warmup_steps: 1, probe_link: true }
---
# server.yaml
server:   { world_size: 4, backend: nccl, activation_offload_mb: 0, wait_client_s: 600 }
---
# client.yaml
client:
  data: { dataset: dialogsum, path: data/datasets/dialogsum, shrink_frac: 0.3, sort: none, loss_on: all }
  activation_offload_mb: 0
  model_state_offload_layers: 0
  wait_server_s: 600
```

**规则**

- 用 frozen dataclass 实现：`ExperimentConfig(run, model, split, batch, train, pipeline, network, profile, server: ServerSection | None, client: ClientSection | None)`。`usl-ray server` 要求 `server` 段存在，`usl-ray client` 要求 `client` 段存在。
- `base:` 支持相对路径继承和逐层合并。CLI 用 `--set a.b=c` 覆盖，值按 dataclass 字段类型转换，写错字段名会直接报错。
- 派生量只用 `@property`：`num_microbatches`、`server_layers`、`server_schedule`（由 `pipeline.mode` 推导，见 3.6）。不再保留冗余的 bool 字段：offload 写 `0` 表示关闭，写不小于微批数的值表示全部微批都卸载。
- `run_tag(cfg)` 是唯一的命名函数，只依赖共享部分以及两侧的 offload 字段，例如 `sp4_b8_mb1_s512_ws4_bw230_d10_splitmind_lora_coa8_cos0_soa8`。profile 写文件和 simulate 查找都只用它；比较两个实验时比较配置字段，不解析文件名。
- 每次运行把两侧解析后的完整配置写到 `runs/<run.name>/<run_tag>/{server,client}.yaml`。
- 加载时的校验（每一项都有单测）：
  - `1 ≤ split_point` 且 `2 × split_point < 层数`；
  - `batch_size % micro_batch_size == 0`（修复 3.13-B6）；
  - `sequential` 模式要求 `micro_batch_size == batch_size`；
  - `model_state_offload_layers > 0` 只能用于 `splitmind` / `gpipe`；
  - `world_size > 1` 时要求 `sort == none`（NCCL 收发 buffer 定长）；
  - `compression == nf4` 时要求 `hidden_size % 64 == 0`（修复 B30）。

**运行期 dataclass**（集中定义）：`MicroBatch`（input_ids、labels、spec，只存在于客户端）、`SequenceSpec`（3.7）、`TensorBlob`（3.3）、`Span` / `StepTimeline` / `StepMetrics`（3.10）、`LinkModel` / `TimeCost` / `MemoryCost`（3.10）。

### 3.3 Ray 传输：原生 ObjectRef 拉取（R2、R12）

整套机制只有三件事。除此之外不做任何限速、分片或重传。

1. **序列化**：使用 `TensorBlob(dtype: str, shape: tuple[int, ...], data: np.ndarray[uint8])`。发送端执行 `t.contiguous().view(torch.uint8).numpy()`，bf16 也能零拷贝放进 Ray 对象存储，不经过 pickle 序列化张量。接收端先拷到 pinned buffer，再 `non_blocking` 拷到 GPU。
2. **发送 = put + push**：

   ```python
   ref = ray.put(blob)
   peer.push.remote(key, ref)
   ```

   ref 是顶层参数，Ray 会在执行 `push` 之前，在接收方节点自动把对象从发送方节点拉过来。这一步就是走过 tc 链路的网络传输。发送方随后可以丢掉本地 ref，Ray 会在 task 结束前保证对象存活。`push.remote` 返回的 ref 在每个 step 结束时统一 `ray.get` 一次，确保发送错误不会被吞掉。
3. **接收 = 邮箱**：每个 Actor 内部有一个 `Mailbox`（`key → concurrent.futures.Future`）。`push` 方法运行在 `io` 并发组里，只负责 `mailbox.put(key, blob)` 并记录到达时间；计算线程通过 `mailbox.get(key, timeout)` 拿到 Future。键唯一，所以不需要再用"收到的 mb_idx 是否等于期望值"这类 assert。

```python
# 每个 Actor（ClientActor / StageActor）都有的方法，属于 io 并发组
def push(self, key: Key, blob: TensorBlob) -> None
def abort(self, reason: str) -> None     # 对端失败时尽力通知，让本端快速失败，而不是等超时
def clock(self) -> float                 # 估计两台机器之间的时钟偏移，仅用于画图
```

- **为什么由发送方推，而不是接收方拉**：数据一旦就绪就会立刻触发传输，接收方不需要额外的预取线程，两个方向的写法也完全对称。
- **GPU → CPU 暂存**：每个 Actor 有一个单线程 `outbox`（`ThreadPoolExecutor(max_workers=1)`），依次执行"等待 comm stream 上 D2H 的 event → put → push"，不阻塞计算线程。如果该微批启用了激活卸载，D2H 之前先等卸载的 event（老代码 `_get_socket_copy_wait_event` 的逻辑）。这个不变式要在实现处写注释并加单测。
- Actor 有两个并发组：`compute`（1 个线程，跑训练）和 `io`（处理 push / abort / clock）。
- 不采用的方案：socket（pickle 反序列化不安全，需要轮询）、gRPC（已证明代价过大）、Ray Client `ray://`（所有数据都要经过 proxy）、`ray.util.queue`（按 FIFO 依赖 mb_idx 比对）、Compiled Graph（面向 GPU 直连，并且要求静态 DAG）。

### 3.4 网络整形：Linux tc（R12）

代码中不做限速。WAN 条件全部由 tc 在网卡上实现，`network.mbps` 和 `network.delay_ms` 只用于记录和模拟。

**一个前提必须写进 README**：client 和 server 的 Actor 如果在**同一个 Ray 节点（同一个 raylet）**上，对象通过共享内存传递，根本不走网卡，tc 不会生效。因此 WAN 实验只能使用下面两种拓扑之一。

| 拓扑 | 做法 | 用途 |
|---|---|---|
| 双机 | 服务端机器 `ray start --head`，客户端机器 `ray start --address`；两端各自对发往对方 IP 的出向流量整形（client 出向即上行，server 出向即下行） | 正式实验 |
| 单机 netns | `experiments/tc/netns.sh up` 创建网络命名空间 `usl_client` 和一对 veth（例如 10.200.0.1 ↔ 10.200.0.2）。head raylet 绑定到 veth 宿主侧 IP，client raylet 在 netns 里以 `--node-ip-address=10.200.0.2` 加入集群；对两端 veth 的出向流量分别整形。GPU 不受 netns 隔离，CUDA 照常可用 | 本机复现 WAN、开发调试 |
| 单机同节点 | 一个 raylet 带全部 GPU，用自定义资源区分 client 和 server | 只做功能测试，传输时间约为 0 |

**整形脚本** `experiments/tc/shape.sh {apply|clear|show} <iface> <peer_ip> <mbit> <delay_ms>`：

```bash
tc qdisc add dev $IF root handle 1: htb default 10
tc class add dev $IF parent 1: classid 1:10 htb rate 100gbit                     # 其它流量不限速
tc class add dev $IF parent 1: classid 1:20 htb rate ${MBIT}mbit ceil ${MBIT}mbit
tc qdisc add dev $IF parent 1:20 handle 20: netem delay ${DELAY}ms limit 100000
tc filter add dev $IF protocol ip parent 1: prio 1 u32 match ip dst ${PEER}/32 flowid 1:20
```

- 只对发往对端 IP 的流量限速，同机 NCCL、SSH 和其它业务流量都不受影响。Ray 的控制面流量也会被限速，但数据量很小。
- 网卡是 bond 时，要在 bond 设备上做整形，不能直接用 slave 网卡（本机 `eth*` 都是 SLAVE）。
- **单位**：tc 的 `mbit` 是 10^6 bit/s，而老代码用的是 `1024 × 1024`，相差约 4.9%。新代码统一采用 SI 单位（3.12）。
- **验证**：`experiments/bench_link.py` 在两个节点之间传 1、4、16、64 MiB 的对象，对 `T(n) = a + n/b` 做线性拟合，输出有效带宽 `b` 和固定开销 `a`；与 tc 设置值相差超过 10% 时报警。`profile.probe_link: true` 时，client 在握手后自动跑一次简化版探测（每个方向 2 种大小、各 3 次），结果写入 `profile.json`。

### 3.5 服务端：Ray Actor + torch pipelining + NCCL（R2）

**server 进程**（`usl-ray server --config server.yaml`）：连接 Ray 集群，用 `STRICT_PACK` placement group 在服务端节点上创建 N 个 `StageActor`（每个 bundle 为 `{"GPU": 1, "usl_server": 1}`），stage0 注册为具名 Actor `usl/<run_tag>/stage0`，然后执行 `ray.get([s.run.remote() for s in stages])`，全部完成后退出。

**StageActor 初始化**

1. `setup_devices(all_gpu_ids)`：Ray 默认只给每个 Actor 暴露它自己那张卡，进程之间互相看不到对方的 GPU，NCCL 可能因此走不了 P2P/NVLink。参考 Ray Train 的做法：在任何 CUDA 调用之前，把 `CUDA_VISIBLE_DEVICES` 设成本节点所有 stage GPU 的并集，再 `torch.cuda.set_device(本 Actor 在并集中的下标)`。用 `NCCL_DEBUG=INFO` 中出现 `P2P/IPC` 或 `P2P/CUMEM` 确认生效。
2. `init_process_group(backend, init_method=f"tcp://{rank0_ip}:{free_port}", rank, world_size)`：端口由 rank0 动态分配后广播，不再写死 `MASTER_PORT`。随后做一次 `barrier` 和一次 1 元素的 send/recv 自检（修复 B9）。
3. 加载本 stage 负责的层（3.8），套上 LoRA，创建优化器（与客户端共用 `train.lr`，修复 B5）。
4. 构造 `USLPipelineStage`（`PipelineStage` 的子类，`input_args` 的 shape 来自 `batch.micro_batch_size × batch.max_seq_len × hidden`，修复 B7），以及 stock 的 `ScheduleGPipe` 或 `Schedule1F1B`，并设置 `scale_grads=False`。
5. `run()`：先等待 client attach（3.11），然后 `for step in range(train.steps): begin_step → schedule._step_microbatches(...) → optimizer.step() → zero_grad()`，最后返回本 stage 的 `StepTimeline` 列表和显存峰值。

**边界适配器**（只在首尾 stage 使用，约 120 行）。stock 调度器按 `arg_mbs[i]` 逐个读取首 stage 的输入；last stage 每做完一个微批的前向就调用 `loss_fn(output, target_mbs[i])`，反向时对这个 loss 求导（torch 2.9 `schedules.py:734-742`）。适配器利用的正是这两点。

```python
class ClientInputs(list):
    """首 stage 的 arg_mbs，长度为 M；取第 i 项时阻塞等待邮箱中的 act_up，转到 GPU 并设置 requires_grad，结果带缓存。"""

class GradToClient(torch.autograd.Function):
    """首 stage 输入处的恒等映射。反向时把输入梯度经 comm stream 暂存后交给 outbox（发送 grad_down）。"""

class ClientLoss(torch.autograd.Function):
    """last stage 的 loss_fn。forward：把输出交给 outbox（发送 act_down），返回一个标量占位。
    backward：返回预先取到的 grad_up，作为 d(output)。"""
```

- `target_mbs` 传入 `torch.arange(M)` 切出的 chunk，`loss_fn` 借此知道当前是第几个微批。
- **`scale_grads=False` 必须设置。** 客户端已经把 loss 除以 M（`client.py:519`），stock 调度器默认还会再除一次 M（`schedules.py:773`）。漏设不会报错，只会让服务端梯度小 M 倍，所以必须有专门的梯度等价测试（7-E1）兜住。
- `USLPipelineStage` 只覆盖两个公开方法 `forward_one_chunk` / `backward_one_chunk`，做两件事：(a) 开启 profile 时在前后做 CUDA 同步并记录 `Span`；(b) last stage 在调用父类 `backward_one_chunk` 之前，先阻塞取到 grad_up，这样等待时间不会被算进反向计算时间。
- 每个微批的 `SequenceSpec` 通过 kwargs 传给各 stage 的 `forward`。客户端在 step 开始时调用 stage0 的 `begin_step(step, specs)`，由 stage0 转发给其它 stage。这一步同时用于对齐 global step。
- `world_size = 1` 时，first 和 last stage 是同一个 Actor，两个适配器同时生效。

**风险与回退（Plan B）**：方案依赖私有方法 `_step_microbatches`，以及 `_check_inputs` 中的 `isinstance(list)` 检查。M4 第一步先用 gloo 做 spike 验证；如果不可行，改用自己实现的 GPipe/1F1B 循环（约 200 行，直接调用 `dist.batch_isend_irecv`），对外接口不变。torch 版本在依赖中固定。

### 3.6 客户端流水线（R3）

client 进程（`usl-ray client --config client.yaml`）在客户端节点上创建 `ClientActor`（`num_gpus=1, resources={"usl_client": 1}`，数据集只在这个节点加载），然后等待 `ClientActor.train()` 返回。

`ClientTrainer` 提供以下原语，每个原语对应一种 `Span`：

```python
head_forward(mb) -> None                        # 前向，激活交给 outbox（发送 act_up）
activation(mb) -> Future[Tensor]                # 邮箱中的 act_down
tail_forward_backward(mb, act) -> float         # tail 前向 + loss + 反向，梯度交给 outbox（发送 grad_up），返回 loss
gradient(mb) -> Future[Tensor]                  # 邮箱中的 grad_down
head_backward(mb, grad) -> None
swap_to_tail() / swap_to_head()                 # 模型参数与优化器状态卸载；未开启时为空操作
step_head() / step_tail()
```

每种模式的调度就是一个函数。以 `splitmind`（论文方法，即老代码的 `pipedream_wc`）为例：

```python
def run_splitmind(t: ClientTrainer, mbs: list[MicroBatch]) -> float:
    for mb in mbs:
        t.head_forward(mb)
    t.swap_to_tail()
    loss = sum(t.tail_forward_backward(mb, t.activation(mb).result()) for mb in mbs)
    t.step_tail()
    t.swap_to_head()
    for mb in mbs:
        t.head_backward(mb, t.gradient(mb).result())
    t.step_head()
    return loss
```

| `pipeline.mode` | 老名字 | 客户端调度要点 | 服务端调度 |
|---|---|---|---|
| `sequential` | `naive` | 对每个微批依次执行 F→S→T→S→H，要求 M=1 | GPipe |
| `gpipe` | `gpipe` | 所有 tail 前向做完后，再做所有 tail 反向 | GPipe |
| `splitmind` | `pipedream_wc` | 如上例，支持模型状态 swap | GPipe |
| `splitmind_eager` | `pipedream_wc_eager` | act/grad 谁先到就先处理谁：`concurrent.futures.wait(FIRST_COMPLETED)`，循环次数有上界 2M | GPipe |
| `1f1b` | `pipedream_strict` | 先预热 `min(M, S+2)` 个 head 前向；之后每拿到一个梯度，做一次 head 反向，再接一个 head 前向 | 1F1B |

- **无死锁条件**：服务端 rank0 在 1F1B 中要先收到 `S` 个激活才会开始第一次反向（torch 中 `warmup = num_stages - stage_index`），客户端预热 `S+2 ≥ S`，满足条件。在 `schedules.py` 中注释这个条件，并用 gloo 测试。
- 对于 `gpipe` / `splitmind`，outbox 是单线程 FIFO，天然保证"所有激活发完后才发梯度"，老代码里的 `head_fwd_send_count` / `is_head_fwd_done` 门控可以删除。
- 调度函数只依赖 `Future`，不直接接触 Ray。单测用一个假 trainer 记录调用顺序即可。

### 3.7 SequenceSpec：mask 与位置编码都按 Spec 传递（R4、R4b）

```python
@dataclass(frozen=True)
class MaskSpec:
    kind: Literal["causal", "full"] = "causal"

@dataclass(frozen=True)
class PositionSpec:
    kind: Literal["arange", "from_valid_lens"] = "arange"   # 右 padding 用 arange；左 padding 用 from_valid_lens
    offset: int = 0

@dataclass(frozen=True)
class SequenceSpec:
    seq_len: int
    valid_lens: tuple[int, ...] | None = None    # 每个样本的有效 token 数；为 None 表示不需要 padding 信息
    mask: MaskSpec = MaskSpec()
    position: PositionSpec = PositionSpec()
```

- **跨边界永远不传 mask 张量、position_ids 或 (cos, sin)。** 每个 partition 都持有一份自己的 `rotary_emb`（只有 buffer，几 KB），收到 Spec 后在本地生成。
- **生成函数**只有一个：`build_sequence_inputs(config, rotary_emb, hidden, spec) -> SequenceInputs(mask_by_layer_type, position_ids, position_embeddings)`。步骤如下：
  1. 由 `valid_lens` 生成 2D padding mask，`valid_lens` 为 None 时 mask 也为 None；
  2. 由 `PositionSpec` 生成 `position_ids`：`arange` 时为 `offset + arange(seq_len)`；`from_valid_lens` 时，左 padding 样本的有效 token 从 0 开始编号；
  3. 调用 transformers 5 的 `create_causal_mask`、`create_sliding_window_causal_mask` 或 `create_bidirectional_mask`，按 `config.layer_types` 组装 `{"full_attention": ..., "sliding_attention": ...}`；
  4. 计算 `position_embeddings = rotary_emb(hidden, position_ids)`。

  head、每个 server stage 和 tail 都调用这个函数。
- **默认配置**：客户端强制 `tokenizer.padding_side = "right"`，pad 位置的 label 设为 `-100`。右 padding 下，有效 token 一定排在 pad 之前，causal mask 本身就保证它们看不到 pad。因此默认 `valid_lens=None`、`position=arange`，SDPA 可以走 `is_causal` 快速路径，跨边界的 Spec 只有几个标量。`full` 类型或左 padding 时，`valid_lens` 必须提供，否则在加载配置时就报错。
- tail 也用 Spec 重新生成 mask 和位置编码，老代码中的 `atten_mask_dict`、`pos_embedding_dict`、`pin_on_gpu_tensors_idx` 全部删除。
- Gemma2 滑窗层用 HF 的滑窗 mask 正确生成，不再依赖 `is_sliding=False` 绕过（修复 B20）。当 `seq_len ≤ sliding_window` 时，两种写法结果一致。
- 隐私说明：`valid_lens` 会暴露样本长度。默认配置下不传，README 中会注明这一点。
- Spec 是 frozen dataclass，跟随 `begin_step` 作为 Ray 的普通参数传递，体积是字节级。

### 3.8 通用模型拆分（R5）

```python
@dataclass(frozen=True)
class ArchSpec:                                   # 每个架构一条数据
    embed: str = "model.embed_tokens"
    layers: str = "model.layers"
    norm: str = "model.norm"
    lm_head: str = "lm_head"
    rotary: str = "model.rotary_emb"
    embed_scale: Literal["none", "sqrt_hidden"] = "none"     # Gemma2: sqrt_hidden
    final_logit_softcap_attr: str | None = None              # Gemma2: "final_logit_softcapping"

ARCHS = {"qwen3": ArchSpec(), "llama": ArchSpec(),
         "gemma2": ArchSpec(embed_scale="sqrt_hidden", final_logit_softcap_attr="final_logit_softcapping")}
```

- `DecoderPartition(nn.Module)` 包含以下部分：
  - 可选的 embed；
  - HF 原始 decoder layer 的一段切片，保留原始 `layer_idx`（Gemma2 依赖它判断层类型；修复 B21）；
  - `rotary_emb`；
  - 可选的 `norm` 和 `lm_head`。

  `forward(x, spec)` 依次执行 embed（如有）→ `build_sequence_inputs` → 逐层计算 → norm / lm_head（如有）。loss 不放在 partition 内，由客户端 tail 计算。
- `layer_param_groups() -> dict[int, list[Parameter]]` 显式给出"层号 → 参数"的映射，offload 直接使用这个映射，不再解析参数名（修复 B25）。
- **划分**：`SplitPlan(num_layers, split_point)` 给出 head `[0, sp)`、trunk `[sp, L-sp)`、tail `[L-sp, L)`。trunk 再用 `balanced_partition(n, world_size)` 分给各 stage，余数分给前面的 stage。全仓库只有这一个划分函数，simulate 也调用它（修复 B23、B36）。
- **加载**：第一版沿用老做法，先在 CPU 上加载整个模型再切片，然后 `del` 并触发 gc。`model.init: random` 时用 `AutoConfig.from_pretrained(path)` 加 `from_config` 构造，只需要 `config.json`。按需从 safetensors 读取部分层作为可选优化（M10）。
- **dtype**：transformers 5 的 `from_pretrained` 默认 `dtype="auto"`（`modeling_utils.py:3934`），而老代码在 4.51 下默认是 float32。为了和老 profile 对比，必须显式传入 `model.dtype`。
- **权重绑定**：小尺寸 Qwen3 的 `tie_word_embeddings=True`，head 的 embed 和 tail 的 lm_head 是同一个 Parameter，两者都在客户端。模型状态 offload 必须跳过这个参数，需要写注释并加单测。
- **LoRA**：对每个 partition 分别调用 `get_peft_model`，`target_modules` 来自配置（修复 B24）。
- **新增架构**：在 `ARCHS` 中加一条记录，跑一遍 `test_partition_equivalence.py` 的参数化用例即可。README 中写明这个流程。

### 3.9 Offload 移植

- 把 `usl/offload` 拷到 `usl_ray/offload/` 后再清理：补全类型注解，删除注释掉的代码。算法保持不变。保留 `AsyncDoubleBufferGroupOffloadHandler`、`CpuOffloadHookWithOffloadHandler`、`LayerwiseModelParamOffload`、`LayerwiseAsyncModelParamOffloadHandler`、`OptimizerStateOffload`、`HybridOffloadContext`。
- 同时修复 3.13 中的 B25–B28：
  - 层号来自 `layer_param_groups()`；
  - 去掉可变默认参数；
  - 异步 D2H/H2D 期间持有源张量的引用，或调用 `record_stream`；
  - 计时统一用 CUDA event。
- 需要在实现处写注释并测试的不变式：
  - (a) 启用激活卸载的微批，D2H 暂存前先等卸载 event；
  - (b) 绑定的 lm_head 不参与卸载；
  - (c) `on_minibatch_commit_forward/backward` 的调用时机与老代码一致。
- 服务端激活卸载包在 `USLPipelineStage.forward_one_chunk` 外层（老代码 `usl_gpipe.py:85-97`），保留"CPU 内存占用超过 90% 时改用 checkpoint"这个兜底。
- QLoRA 通信压缩作为 `network.compression: nf4` 的 codec 放进 M10：解压改在 GPU 上进行（修复 B29），用 `torch.special.ndtri` 替换 `scipy.stats.norm.ppf`（修复 B31）。

### 3.10 Profile 与 Simulate 的一致性（R6、R9）

**共享契约**：下表中每一项在代码里都只有一份实现。

| 契约 | 唯一实现 | 说明 |
|---|---|---|
| 时间线结构 | `profile/timeline.py`：`Span(lane, op, mb, start_ms, end_ms, nbytes=0)`、`StepTimeline` | lane 取值：`client`、`uplink`、`downlink`、`server/{r}`、`client_copy`、`server_copy/{r}` |
| 指标 | `summarize(timelines) -> StepMetrics` | batch 时间、各操作平均耗时（沿用老 `avg_value`：样本数大于 2 时去掉最大值和最小值）、两侧 idle 率与发送占比、各方向字节数 |
| 层划分 | `models/partition.py: balanced_partition` | 训练和模拟共用 |
| 链路模型 | `profile/link.py: LinkModel(fixed_ms, mbps)`，`transfer_ms(nbytes) = fixed_ms + nbytes × 8 / (mbps × 10^6) × 1000` | 参数来自 `bench_link` 或握手探测的线性拟合，单位与 tc 一致 |
| 消息字节数 | `message_nbytes(cfg, kind)` = `mb × S × H × dtype_bytes`（再乘以压缩比） | 训练时按实际值记录到 `Span.nbytes`，模拟时直接计算 |
| 调度顺序 | `client/schedules.py` 和服务端调度表 | 模拟器按同样的顺序排布 Span |
| 实验匹配 | `config.run_tag`、`same_experiment(a, b)` | 不解析文件名 |
| 单位 | 时间 ms，显存 MiB，带宽 10^6 bit/s | 写在 `timeline.py` 的模块注释里 |

**传输 Span 的测量**：上行 Span 的起点是 client 端 outbox 开始 `put` 的时刻（client 时钟），终点是 server `push` 被调用的时刻（server 时钟，换算到 client 时钟）。下行 Span 反过来。时钟偏移通过握手时多次调用 `clock()` 往返估计；同一台机器上偏移约为 0。指标中的"传输时长"直接使用这两个时间点之差，因此 profile 文件里记录偏移的估计误差（往返时间的一半）。

**`profile.json`**：由 client 进程写到 `runs/<name>/<run_tag>/`。server 的时间线在 `run()` 返回时交给 server 进程；server 进程在退出前把它推给 client 的邮箱（键为 `("profile", -1, -1)`），不再使用 `gather_object` 加 socket 的方式（修复 B10）。文件内容包括：

- 两侧配置、环境信息（GPU 型号、各库版本、`tc qdisc show` 的输出）；
- 时钟偏移；
- 链路探测结果（`LinkModel`）；
- 每个 measured step 的 `StepTimeline`；
- 两侧显存峰值、offload 各项耗时；
- 跳过 warmup step 后的 `StepMetrics` 均值（修复 B18）。

**测量方式**：开启 profile 时，每个计算操作前后都做 `torch.cuda.current_stream().synchronize()`，与老代码口径一致，否则拟合出来的常数无法比较。关闭 profile 时不做同步。

**校准**（`experiments/calibrate.sh` → `usl-ray calibrate`）：按顺序跑老 `run_profile` 的 7 组基准配置（`profile_separate_nodes.py:87-96`），读取各自的 `profile.json`，用与老代码相同的差分公式拟合 `TimeCost` / `MemoryCost`（沿用 `TimeConstant` / `MemoryConstant` 的字段名，并补上单位注释），写出 `cost_model.json`。与老代码相比有三处修正：

- 硬编码的 `_get_transfer_time` 和固定 10 ms 时延，改成 `LinkModel`（修复 B37）；
- `max_split_point` 由真实层数计算（修复 B36）；
- 缺少某组配置时，直接列出缺了哪些组合并报错（修复 B34）。

**模拟器**

- 第一步：忠实移植老 `_simulate_train_time` / `_simulate_peak_mem_alloc` 中 `splitmind` 的算法，逐行对照。改动只有三处：输出 `StepTimeline`；全局变量改成函数参数（修复 B32）；抖动通过 `jitter: float = 0.0` 控制，默认关闭，开启时使用固定种子（修复 B38）。
- 第二步：其它模式按 3.6 的调度顺序做列表调度，即每个 Span 的开始时间取"所在 lane 空闲时刻"和"所依赖数据就绪时刻"中的较大值，代码不超过 150 行。时间紧的话，可以先只支持 `splitmind`。
- 链路建模：沿用老模拟器的方式，上行一条、下行一条串行 FIFO 链路（`simulate.py:347-353`、`:400-407`）。真实情况下多个 Ray 拉取可能并发、平分带宽，这个差异列入第 8 节风险，由 7-E7 量化。
- 命令：
  - `usl-ray simulate` 输出 `StepMetrics`、甘特图（与真实 profile 共用同一个绘图函数）和成本；
  - `usl-ray search` 是 `do_optimize` 帕累托搜索的移植版，不可行配置会返回明确结果（修复 B35）；
  - `usl-ray compare` 读取若干 `profile.json`，逐项输出模拟值与真实值的误差和误差率。

### 3.11 生命周期、握手与启动脚本（R10）

```
server 进程                                        client 进程
  创建 StageActor × N，setup，注册具名 stage0          创建 ClientActor
  ray.get(run())  ── 阻塞等待 attach（wait_client_s）    轮询 get_actor(stage0)（wait_server_s，替代老脚本的 sleep 5）
                  ◄────────── attach(client_handle, shared_cfg) ──────────
                  校验共享配置；不一致时返回字段级 diff，两侧都失败退出
                  ─────────── ServerInfo(stage 划分, 探测对象) ───────────►
  for step in range(train.steps):  ◄─ begin_step(step, specs) ─  for step in range(train.steps):
      ... 4 类消息 ...                                               ... 4 类消息 ...
  run() 返回 → 推送 profile → 退出                    收到 server profile → 写 profile.json → 退出
```

- 两侧的 step 数都来自共享的 `train.steps`。客户端使用 `drop_last=True` 的 DataLoader，并按实际执行的 step 计数，因此两侧 step 数严格相同（修复 B12）。数据集不够 `steps` 个完整 batch 时，在加载阶段就报错。
- 所有阻塞等待都有超时（默认 600 秒，可配置；修复 B8）。超时异常中带有对应的 `(step, mb, kind)`。任何一侧失败时，都会尽力调用对端的 `abort.remote(reason)`，让对端快速失败。
- Actor 归各自的 driver 进程所有，driver 退出时 Actor 随之销毁，不会出现"客户端结束了、服务端还在跑"的情况。

**启动脚本**：沿用 `bash_usl_run_single.sh` 的结构，即 server 后台运行、client 前台运行、最后 `wait`。

```bash
# experiments/run_single.sh <exp_dir> [--set k=v ...]
set -euo pipefail
EXP=${1:?usage: run_single.sh <exp_dir> [--set k=v ...]}; shift
TOPO=${TOPO:-netns}                  # netns | same-node
source "$(dirname "$0")/lib/ray_up.sh" "$TOPO" "$EXP"       # ray start（以及可选的 netns + tc）
trap 'source "$(dirname "$0")/lib/ray_down.sh" "$TOPO"' EXIT

usl-ray server --config "$EXP/server.yaml" "$@" &
SERVER_PID=$!
${CLIENT_EXEC:-} usl-ray client --config "$EXP/client.yaml" "$@"   # netns 拓扑下 CLIENT_EXEC 为 "sudo ip netns exec usl_client sudo -u $USER"
wait $SERVER_PID
```

- `run_server.sh` / `run_client.sh`：双机拓扑下分别在两台机器上执行，内容就是 `ray start`、可选的 `tc/shape.sh apply`、启动对应进程。
- `run_grid.sh`：替代 `bash_usl_run_multi.sh`。在 bash 中枚举 `SPLIT_POINTS`、`CLIENT_OAM` 等，生成同一组 `--set` 依次调用 `run_single.sh`。因为两侧共享 `common.yaml`，同一组 `--set` 天然对两侧一致。
- `calibrate.sh`：依次跑 7 组校准配置，最后调用 `usl-ray calibrate`。
- tc 的速率和时延由脚本参数传入（默认读取环境变量 `NET_MBIT` / `NET_DELAY_MS`）。为防止与配置不一致，client 在握手后把探测得到的 `LinkModel` 与 `network.mbps` / `delay_ms` 做比较，偏差超过 10% 时打 warning，并写进 profile。

### 3.12 与老代码的行为差异（需在 README 中列出）

| 项 | 老行为 | 新行为 | 影响 |
|---|---|---|---|
| pad 位置的 loss | labels = input_ids，pad 也算进 loss | pad 位置 label 为 -100；`loss_on: completion` 时只在 summary 部分计算 loss | loss 数值变化，耗时不变 |
| 带宽单位 | `1024 × 1024` bit/s | 10^6 bit/s（与 tc 一致） | 同样名义带宽下，传输时间比老代码短约 4.9% |
| 带宽实现 | socket 发送端分片发送后 sleep | tc 整形（TCP 拥塞控制和 netem 时延都是真实的） | 更接近真实 WAN；1F1B 下两类上行数据会共享同一条链路 |
| 服务端学习率 | 写死为 1e-3 | 使用 `train.lr` | 收敛曲线变化 |
| profile 统计范围 | 只保留最后一个 step | 去掉 warmup 后对所有 step 取均值 | 抖动更小 |
| Gemma2 滑窗 | 关闭 | 按 HF 层类型启用 | seq ≤ 4096 时结果一致 |
| simulate 抖动 | 固定开启 ±1% | 默认关闭 | 结果可复现 |
| epoch 时间 | 按写死的 12460 条样本计算 | 按实际数据集长度计算 | 数值会变化 |
| GPT-2 | 支持 | 不支持 | — |

### 3.13 老代码缺陷清单与在新代码中的修复

这些缺陷都只在 `usl-ray/` 中修复，`usl/` 保持不变。"回归测试"一列给出验证方式，没有写测试的，说明该问题已由新设计消除。

| # | 位置 | 问题 | 新代码修复 | 回归测试 |
|---|---|---|---|---|
| B1 | `socket/socket_comm.py:145` | 对网络数据直接 `pickle.loads`，公网下可能被远程执行代码 | 删除 socket，只用 Ray 和 `TensorBlob` 传输数值数据 | 隔离测试：源码中不出现 `socket`、`pickle` |
| B2 | `server/pipeline/server_stage.py:676-679` | `emb = emb.to(device)` 没有写回，position embedding 一直留在 CPU | 不再传位置编码，改为按 Spec 本地生成 | 等价性测试 |
| B3 | `server_stage.py:1090-1093` | rank>0 收不到 mask，有 padding 时各 stage 的 mask 不一致 | 每个 stage 都由 Spec 重建 mask | 多 stage + padding 等价性测试 |
| B4 | `client/client.py:350-351` 对比 `server_stage.py:819` | 传了 (cos, sin) 但服务端不用，浪费上行带宽 | 同 B2 | 测试：`act_up` 的字节数等于激活大小 |
| B5 | `experiment/server_run_mp.py:108` | 服务端学习率写死为 1e-3 | 两侧共用 `train.lr` | 配置单测 |
| B6 | `server_run_mp.py:83` 对比 `client.py:218,419` | 微批数服务端向下取整、客户端向上取整，不整除时会卡死 | 只有一个 `num_microbatches` 属性，并校验能整除 | 配置单测 |
| B7 | `server_run_mp.py:82` | NCCL 输入 shape 的 seq_len 写死为 512 | 从 `batch.max_seq_len` 推导 | gloo 测试使用非 512 的长度 |
| B8 | `server_stage.py:813` | 首 stage 等待激活的超时写死 30 秒，客户端慢一点就崩溃 | 超时可配置，异常中带上键 | 超时单测 |
| B9 | `server_stage.py:1279-1306` | `_init_p2p_neighbors` 构造了 P2P op 但从未执行，连通性检查形同虚设 | `setup` 中做 barrier 加一次真实的 send/recv | gloo 测试 |
| B10 | `server/pipeline/usl_1f1b.py:193-204`、`base_schedule.py:229-238` | 1F1B 和基类上报的 profile 字段全是 0 | profile 由 Span 统一生成 | timeline 单测 |
| B11 | `client_run.py:35`、`server_run_mp.py:78,128-129`、`socket_comm.py:18` | 设备 `cuda:3` / `cuda:{rank+4}`、IP 和 `MASTER_PORT` 都写死 | 由 Ray 分配 GPU，端口动态获取 | — |
| B12 | `client.py:1048-1056,1080` | 跳过的不完整 batch 也计入 `batch_idx`，客户端实际 step 数可能少于服务端，导致服务端挂起 | `drop_last=True`，按实际执行次数计数，并在加载时检查数据够不够 | 数据单测 |
| B13 | `client.py:395-399` | `_split_micro_with_mask` 会改写 `client_args.max_seq_len`，后续文件名跟着变 | 配置是 frozen 的 | 配置单测 |
| B14 | `client.py:645` | `total_bytes_sent` 只统计了梯度，没算激活 | 每个传输 Span 都记录 `nbytes` | timeline 单测 |
| B15 | `client.py:761,773,929-931` | `server_send_rate` / `server_idle_rate` 依赖的变量从未更新，结果恒为 0 / 100 | 由 `summarize()` 根据 server lane 计算 | timeline 单测 |
| B16 | `client.py:933` | 发送时间为 0 时 `bytes_sent_per_ms` 会除零 | `summarize()` 中做保护 | timeline 单测 |
| B17 | `client.py:903` | epoch 时间按写死的 12460 计算，但实际数据集用了 `shrink_frac=0.3` | 使用实际数据集长度 | 数据单测 |
| B18 | `client.py:979` | profile 只保留最后一个 step | 去掉 warmup 后对所有 step 取均值 | timeline 单测 |
| B19 | `utils/dataset/datasets.py:444-464` | DialogSum 计算了 summary mask 却从未使用，loss 把 pad 也算进去 | label 为 -100；`loss_on: all|completion` | 数据单测 |
| B20 | `llm/client/gemma.py:71,87,183`、`llm/server/gemma.py:30,79` | 用 `is_sliding=False` 绕过 Gemma2 的滑窗；server 的逻辑加载只给最后一层设置了该标志（缩进错误） | 按 HF 层类型生成 mask | Gemma2 等价性测试 |
| B21 | `llm/server/qwen3.py:40`、`llm/client/qwen3.py:166`、`llm/client/llama.py:131` | 物理加载时 `layer_idx + to_l` 算错了层号 | 复用原始 layer 对象，保留原层号 | 等价性测试 |
| B22 | `llm/client/gpt.py:84` | `GPT2ClientHead.from_pretrained` 构造出来的是 `GPT2ClientTail` | 不再支持 GPT-2（已在文档中说明） | — |
| B23 | `utils/load_utils.py:171,265,238-245` | 两套多卡切分函数；分 stage 的路径不支持 Gemma2 | 只保留一个 `balanced_partition` | 划分单测 |
| B24 | `load_utils.py:158-166` | 服务端 LoRA 是套在整模型上、靠共享子模块"顺带"生效，非常隐晦 | 对每个 partition 显式套 LoRA | LoRA 注入位置单测 |
| B25 | `offload/model_offload.py:334-358`、`model_offload_hook.py:318-340` | 层号靠参数名前缀 `model.layers` 识别；不开 LoRA 时参数名是 `layers.N`，结果什么都没卸载，也没有任何报错 | 使用 `layer_param_groups()` | offload 单测（LoRA 开 / 关两种情况） |
| B26 | `model_offload.py:25`、`optimizer_offload.py:18` | `except_tensor_idx_list=[]` 是可变默认参数，`add_except_tensor` 会污染所有实例 | 默认值改为 `None` | 单测 |
| B27 | `offload/optimizer_offload.py:133-136,150-153` | 异步 D2H 还没完成就丢掉了 GPU 张量的引用；H2D 在 load stream 上分配显存、却在 compute stream 上使用，都没调用 `record_stream`，存在显存被提前复用的风险 | 持有引用直到 event 完成，或调用 `record_stream` | GPU 测试：卸载前后梯度和优化器状态一致 |
| B28 | `optimizer_offload.py:162-163` 对比 `model_offload.py:151-153` | 优化器卸载的起始时间戳没有先同步 event，与模型卸载的计时口径不同 | 统一用 CUDA event 计时 | — |
| B29 | `server_stage.py:665-667`、`client.py:500` | QLoRA 解压在 CPU 上执行（先解压后 `.to(device)`） | 先 H2D 再在 GPU 上解压 | M10 codec 测试 |
| B30 | `utils/qlora_comm.py:136` | 隐含要求 hidden 能被 block_size 整除，否则 `view` 报错 | 加载配置时校验 | 配置单测 |
| B31 | `qlora_comm.py:4,13` | 只为了 `norm.ppf` 依赖 scipy | 改用 `torch.special.ndtri` | codec 测试 |
| B32 | `simulate/simulate.py:457-499,785` | 引用只在 `__main__` 中定义的全局变量，作为库调用会 `NameError` | 全部改为函数参数 | simulate 单测 |
| B33 | `simulate.py:28`、`simulate/profile_separate_nodes.py:103` | 文件名中分别写死了 `ws_1` 和 `ws_4` | 统一使用 `run_tag` / `same_experiment` | — |
| B34 | `profile_separate_nodes.py:109-124` | 打印 "Skipping" 之后紧接着 `KeyError` | 列出缺失的组合并报错 | calibrate 单测 |
| B35 | `simulate.py:601-615` | 30 张卡以内都放不下时，`server_total_layer_num` 未定义 | 返回明确的"不可行"结果 | simulate 单测 |
| B36 | `profile_separate_nodes.py:82`、`simulate.py:611` | 层数为奇数时，服务端层数少算 1 | 使用 `SplitPlan` | 划分单测 |
| B37 | `profile_separate_nodes.py:43-56`、`simulate.py:208` | 按模型名硬编码传输时间，外加固定 10 ms | 使用 `LinkModel` 拟合 | simulate 单测 |
| B38 | `simulate.py` 多处 `random.randint` | 抖动不可复现 | `jitter` 参数默认 0，并固定种子 | simulate 金标准测试 |
| B39 | `simulate.py:276,297` | 判断服务端前向是否等待激活卸载时，用的是 `client_offload_mb_num` | **待确认**（第 9 节）：默认先忠实移植，并在金标准测试中标注 | 金标准测试 |

---

## 4. 目标目录结构与代码量预算

```
usl-ray/
├── pyproject.toml                # 包 usl_ray；依赖版本与第 5 节一致；ruff banned-api 禁止 usl；入口 usl-ray
├── README.md                     # 快速开始（单机同节点 / 单机 netns / 双机）、端口、tc、配置说明、扩展点、排错、行为差异
├── src/usl_ray/
│   ├── config.py                 # 配置 dataclass、YAML 继承、--set、校验、run_tag、same_experiment、diff
│   ├── cli.py                    # usl-ray server | client | calibrate | simulate | search | compare | plot | bench-link
│   ├── transport.py              # TensorBlob、Mailbox、Outbox（put+push）
│   ├── models/{arch,sequence,partition,loader}.py   # sequence.py：SequenceSpec 与 build_sequence_inputs
│   ├── server/{stage_actor,boundary,launcher}.py
│   ├── client/{trainer,schedules,launcher}.py
│   ├── offload/                  # 从 usl/offload 移植并修复
│   ├── data.py                   # dialogsum / gsm8k 等论文数据集 + synthetic
│   ├── profile/{timeline,link,recorder,plot}.py
│   └── simulate/{cost_model,simulator,search}.py
├── experiments/
│   ├── configs/{defaults.yaml, smoke-qwen3-0.6b/, qwen3-1.7b/, paper/<model>/}   # 每个目录下有 common/server/client.yaml
│   ├── run_single.sh  run_server.sh  run_client.sh  run_grid.sh  calibrate.sh
│   ├── lib/{ray_up.sh, ray_down.sh}
│   ├── tc/{shape.sh, netns.sh}
│   └── bench_link.py
└── tests/
    ├── unit/                     # 纯 CPU，不依赖 Ray，不下载权重（使用小规模随机 HF config）
    ├── integration/              # 本地 Ray + gloo（CPU）
    ├── gpu/                      # pytest -m gpu
    └── fixtures/legacy_simulate/ # 老模拟器的金标准输出（JSON）
```

| 模块 | 老代码行数 | 新预算 |
|---|---|---|
| config + cli | ~200 + 入口脚本约 330 | 350 |
| models | 1,727 | 450 |
| transport | 341（另有大量逻辑散落在 client/server 中） | 120 |
| server | 2,728 | 350 |
| client | 1,814 | 500 |
| offload | 1,675 | ≤ 1,100（移植） |
| data | ~1,100 | 200 |
| profile | ~620 | 350 |
| simulate | 1,329 | 600 |
| experiments（bash + bench） | — | 250 |
| **合计** | **11,930** | **≈ 4,270** |

评审时，超出预算 20% 的模块需要说明原因。超预算通常意味着又做了第 1.3 节列出的非目标。

---

## 5. 版本与依赖（R11）

以本机实测版本为准，老代码不再保证能在这套依赖下运行。

| 包 | 版本 | 说明 |
|---|---|---|
| python | 3.11 | `requires-python = ">=3.11,<3.12"` |
| torch | 2.9.1 | 依赖 `torch.distributed.pipelining` 的私有方法，必须精确固定版本 |
| transformers | 5.0.0 | 使用 masking_utils 和 `layer_types`；decoder layer 直接返回张量 |
| ray[default] | 2.58.0 | 使用并发组、placement group 和具名 Actor |
| peft | 0.20.0 | |
| accelerate | 1.14.0 | |
| safetensors / tokenizers / huggingface_hub | 0.8.0 / 0.22.2 / 1.3.7 | |
| numpy / pandas / matplotlib / pyarrow | 2.3.5 / 2.3.3 / 3.10.8 / 17.0.0 | |
| psutil / PyYAML | 6.0.0 / 6.0.3 | |
| datasets | **本机未安装**：M0 阶段安装与 `huggingface_hub 1.3.7`、`pyarrow 17.0.0` 兼容的最新版并固定版本 | 老数据集加载代码需要它 |
| pytest / ruff（dev） | 9.1.1 / 0.16.10 | |
| 删除 | trl、aiohttp、ninja、pillow、scipy（B31） | 新代码不再使用 |

- `usl-ray/pyproject.toml` 是版本的唯一来源。根目录 `requirements.txt` 改写为相同的版本列表，并在文件头注明"适用于 usl-ray；usl/ 老代码不再保证兼容"。另有一个 `tests/unit/test_requirements_sync.py` 校验两处版本一致。
- 系统依赖：iproute2（`tc`、`ip netns`）和 sudo 权限，只有整形实验需要。README 中提供 `tc -V` 和 `sudo -n true` 两条自检命令。

---

## 6. 代码编写计划

每个阶段都要能独立验收，阶段内先写测试再写实现。S / M / L 表示相对工作量。

### M0 准备与基线（S）

- [x] 搭建 `usl-ray/` 骨架和 `pyproject.toml`；更新根目录 `requirements.txt`；安装 `datasets` 并固定版本；编写 `test_requirements_sync.py`。
- [x] 配置 ruff 的 `banned-api`，编写 `tests/unit/test_isolation.py`；确认根目录的 `find_packages()` 不会收录 `usl-ray`。
- [x] 在仓库根目录新增一次性脚本 `tools/export_legacy_simulate_fixtures.py`。它只依赖老代码，运行时把 `random.randint` monkeypatch 为 0，对一组输入导出老模拟器结果，写入 `usl-ray/tests/fixtures/legacy_simulate/`。新包不 import 这个脚本。
- 验收：`cd usl-ray && pip install -e '.[dev]' && ruff check . && pytest tests/unit -q`。　实际偏差：老 `simulate.py:785` 使用 Python 3.12 才接受的嵌套同引号 f-string；fixture 导出器只在临时目录中修正该行后执行，没有修改 `usl/`。`ray[default]` 曾引入不需要的依赖冲突，最终收窄为 core `ray==2.58.0`。

### M1 配置（S）

- [x] 实现 `config.py`（包括继承、`--set`、校验、`run_tag`、`same_experiment`、字段级 diff）。
- [x] 编写 `experiments/configs/defaults.yaml`、`smoke-qwen3-0.6b/`、`qwen3-1.7b/`，以及 `paper/<model>/` 下的三件套。
- [x] 单测覆盖继承、覆盖、所有校验规则（B5、B6、B13、B30）和 `run_tag` 的稳定性。
- 验收：`pytest tests/unit/test_config.py`；`usl-ray server --config ... --dry-run` 能打印解析结果和层划分。　实际偏差：无。

### M2 模型拆分与 SequenceSpec（M）

- [x] 实现 `arch.py`、`sequence.py`、`partition.py`、`loader.py`。
- [x] 等价性测试：用 Qwen3、Llama、Gemma2 的小规模随机 config（fp32，4–6 层），参数化覆盖切分点、stage 数、causal/full、右/左 padding、`PositionSpec` 两种类型，断言 head→trunk→tail 的 logits 与整模型一致。
- [x] 测试权重绑定、LoRA 注入位置、`layer_param_groups()`。
- 验收：`pytest tests/unit/test_partition_equivalence.py tests/unit/test_sequence_spec.py`。　实际偏差：三种架构的 causal + padding 做了整模型数值等价；full 和左 padding 在 `SequenceSpec` 层做了独立 mask/position 测试。

### M3 传输层（S）

- [x] 实现 `TensorBlob`（含 bf16 往返）、`Mailbox`（按键取值、超时、abort）、`Outbox`（等待 event → put → push，step 结束时检查 push 结果）。
- [x] 集成测试：本地 Ray 起两个 Actor 互相 push；乱序到达时邮箱仍能按键取到；对端 abort 后本端能快速失败。
- 验收：`pytest tests/unit/test_transport.py tests/integration/test_ray_transport.py`。　实际偏差：Outbox 串行等待每次 push 的 ack，因此真实运行和模拟器的单上行/单下行 FIFO 假设严格一致。

### M4 服务端（L，含 go/no-go）

- [x] **先做 spike**：在 2 个 Ray Actor 上用 gloo 跑 stock `ScheduleGPipe`，接上 `ClientInputs` 和 `ClientLoss`，由一个假客户端喂数据。验证调度能推进、梯度正确、`scale_grads=False` 生效。不通过则切换到 Plan B，并在此处记录原因。
- [x] 实现 `StageActor`、`USLPipelineStage`、两个调度器、`attach` / `begin_step` / `run`、`setup_devices`、P2P 自检（B9）。
- [x] 1F1B 无死锁测试（gloo，S=2，M=4）。
- 验收：`pytest tests/integration/test_server_gloo.py`。　实际偏差：torch 2.9.1 stock `_get_init_p2p_neighbors_ops` 用 int64 send 对 float32 recv，gloo 直接 SIGABRT；`USLPipelineStage` 仅覆盖该初始化方法，统一为 float32。Ray Actor 导入 torch 后已固定单卡可见性，不能再扩展并使用 `cuda:rank`；实际实现让每个 Actor 使用其隔离命名空间内的 `cuda:0`，NCCL 实测仍走 `P2P/CUMEM/read`。

### M5 客户端、启动与端到端（L）

- [x] 实现 `ClientTrainer` 原语、`schedules.py` 中的 5 种调度、`ClientActor`、两侧 launcher、握手（包括配置 diff）。
- [x] 编写 `experiments/run_single.sh`、`run_server.sh`、`run_client.sh`、`lib/ray_up.sh` 和 `lib/ray_down.sh`（先支持 `same-node` 拓扑）。
- [x] 调度单测：用假 trainer 断言调用顺序。
- [x] 端到端数值测试，即 7-E1。
- 验收：7-E1、E2 通过；`rg "while True|time\.sleep" src/usl_ray` 无结果。　实际偏差：五种模式全部通过 gloo 参数更新等价测试；GPU 实机验证了 SplitMind 和 1F1B 两种服务端调度（其余模式与 GPipe 共用同一服务端实现）。

### M6 Offload（M）

- [x] 移植并修复（B25–B28）；接入客户端的激活、模型状态和优化器卸载，以及服务端的激活卸载。
- [x] 测试：卸载前后梯度一致；D2H 暂存会等待卸载 event；绑定的 lm_head 不被卸载；显存峰值确实下降。
- 验收：`pytest tests/unit/test_offload.py`；`pytest -m gpu tests/gpu/test_offload_e2e.py`。　实际偏差：没有逐行搬运 1,675 行旧 offload，而是重写为约 230 行的 saved-tensor 异步 offloader 和显式 layer group swapper；CUDA 梯度/参数更新测试与 Qwen3-0.6B offload 冒烟均通过。

### M7 网络整形（M）

- [x] 编写 `experiments/tc/shape.sh`、`netns.sh`、`bench_link.py`，并在 `lib/ray_up.sh` 中支持 `netns` 拓扑。
- [ ] **先做 spike**：确认 netns 中的 raylet 能加入集群、ClientActor 能在 netns 节点上使用 GPU、对象传输确实经过 veth 并受 tc 限速。
- 验收：7-E4。　实际偏差：当前容器缺少 `CAP_SYS_ADMIN`，`ip netns add` 报 `mount --make-shared /run/netns failed: Operation not permitted`，`unshare -n` 也被拒绝，且没有 Docker。双机 tc 方案和链路 benchmark 已完成；单机 E4 只能在具备权限的宿主上执行。

### M8 Profile（M）

- [x] 实现 `timeline.py`、`link.py`、`recorder.py`（时钟偏移、链路探测）、`plot.py`；两侧记录 Span，server 把 profile 推给 client，client 合并后写出 `profile.json`。
- [x] 单测 `summarize()`：用手工构造的时间线逐项核对指标（B14–B18）。
- 验收：冒烟运行产出 `profile.json` 和 `gantt.png`；`usl-ray plot <profile.json>` 能复现同一张图。　实际偏差：Outbox 用发送方时钟从 `ray.put` 前计时到远端 push ack，因此传输时长不依赖跨机时钟对齐；时钟探测只需用于跨节点甘特图的绝对位置，当前 same-node 验证不需要偏移。

### M9 Simulate（L）

- [x] 实现 `cost_model.py`（采用与老代码相同的差分公式，加上 `LinkModel`）、`simulator.py`（忠实移植 `splitmind`，其它模式可选）、`search.py`（帕累托搜索和 compare），以及 `experiments/calibrate.sh`。
- [x] 金标准测试（7-E6）。
- [ ] 闭环测试（7-E5）。
- 验收：`pytest tests/unit/test_simulate*.py`；按 README 跑通 calibrate → simulate → compare。　实际偏差：legacy fixture 在 1e-6 内逐项一致，七配置拟合有合成 profile 单测；真实 E5 依赖 tc/netns 或双机 WAN，受 M7 的宿主权限限制未执行。

### M10 收尾与可选项

- [x] 完善 README（三种拓扑的快速开始、端口、tc、配置字段、扩展点、排错、行为差异、第 7 节实验结果）；对照预算清理代码；`ruff format`；以"陌生维护者只读仓库"的视角走查一遍。
- [ ] 可选：QLoRA 通信压缩（B29–B31）、按需加载 safetensors、`run_grid.sh` 网格实验。

实际偏差：运行时代码 3,964 行，低于总预算 4,270 行；模型模块 445 行，符合 450 行预算。server 模块为 619 行，超过单模块 350 行预算，原因是实际落地必须包含 Ray Actor 的 setup/readiness/abort 生命周期、torch 2.9.1 P2P dtype 修复、首尾边界 autograd 适配和独立进程 launcher；这些代码没有引入第 1.3 节的非目标。`run_grid.sh` 已完成；QLoRA codec 和按需 safetensors 加载仍保留为可选项。

---

## 7. 实验与验证（在本机执行，结果写入 README 的"验证记录"一节）

本机目前没有模型权重和数据集。耗时和显存只取决于网络结构、dtype 和张量形状，与权重数值无关，因此 E2–E7 都可以使用 `model.init: random`（只需要 `config.json`）加上 synthetic 数据（每个样本 pad 到 `max_seq_len`），在离线环境下完成。只有 loss 曲线需要真实权重和数据集，等用户把它们放到 `data/models/`、`data/datasets/` 之后再补跑。

| 编号 | 内容 | 命令（示意） | 通过标准 |
|---|---|---|---|
| E1 | 端到端梯度等价：小规模随机 Qwen3、fp32、gloo、`world_size=2`、M=2，5 种模式各跑 1 个 step | `pytest tests/integration/test_e2e_equivalence.py` | 两侧所有可训练参数的梯度和 loss 与整模型单机训练一致（`atol=1e-5`） |
| E2 | GPU 冒烟（同节点拓扑）：Qwen3-0.6B、`world_size ∈ {1,2}`、5 种模式、3 个 step | `TOPO=same-node experiments/run_single.sh experiments/configs/smoke-qwen3-0.6b` | 两侧同时以退出码 0 结束；`ray status` 显示资源归零；显存全部释放 |
| E3 | NCCL 走 P2P | E2 中 `world_size=2` 的运行加上 `NCCL_DEBUG=INFO` | 日志中出现 `P2P/IPC` 或 `P2P/CUMEM` |
| E4 | tc 生效：netns 拓扑，230 mbit / 10 ms | `experiments/tc/netns.sh up && python experiments/bench_link.py` | 拟合出的带宽与 230 Mbit/s 相差在 ±10% 以内；4 MiB 传输约 146 ms 加上固定开销 |
| E5 | 闭环：netns 拓扑下跑 7 组校准，拟合成本模型后模拟同样的配置，再与另外 3 组未参与拟合的配置对比 | `experiments/calibrate.sh experiments/configs/smoke-qwen3-0.6b`，然后 `usl-ray compare ...` | 报告 batch 时间和两侧显存的误差率（不设硬性门槛，结果写入 README） |
| E6 | simulate 金标准 | `pytest tests/unit/test_simulate_legacy.py` | 关闭抖动时与老模拟器逐项一致（≤ 1e-6）；有意的修正逐条标注为"预期差异" |
| E7 | FIFO 链路假设的误差：`1f1b` 模式和 `splitmind` 模式各跑一次 E5 | 同 E5 | 记录两种模式下的误差率，用来判断是否需要在模拟器中引入带宽共享模型 |

---

## 8. 风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| client 与 server 落在同一个 raylet 上 | 对象走共享内存，tc 不起作用，传输时间约为 0 | README 和启动日志中明确提示；`netns` / 双机拓扑下检查 ClientActor 与 StageActor 的 node_id 不同，否则拒绝启动 |
| netns 方案不可行（例如 raylet 绑定 IP 或 GPU 可见性有问题） | E4 失败 | M7 先做 spike；退路是 docker `--net=none` 加 veth，或只用双机实验 |
| torch pipelining 私有接口变化 | 升级后调度器行为改变 | 固定 torch 版本；M4 先做 spike；Plan B 是自写循环 |
| 漏设 `scale_grads` | 不报错，服务端梯度小 M 倍 | 构造参数写死并加注释；E1 一定能发现 |
| Ray 的 GPU 隔离导致 NCCL 降级 | 多卡吞吐明显变低 | 设置可见设备并集（3.5）；E3 检查 |
| Ray 端口与防火墙 | 节点无法加入集群，或对象拉取超时 | `ray start` 时固定端口，README 中列出需要放通的端口；`bench-link` 作为连通性检查 |
| 实际 tc 设置与配置中的 `network.*` 不一致 | 模拟用的带宽与实际不符 | 握手时做链路探测并比较（3.11）；calibrate 优先使用探测值 |
| 并发拉取平分带宽，而模拟器假设 FIFO | 1F1B 等模式模拟误差偏大 | 用 E7 量化；误差大时再给模拟器加入带宽共享模型 |
| 两台机器时钟不同步 | 甘特图错位，传输时长有偏差 | 用 `clock()` 往返估计偏移，并在 profile 中记录误差上界 |
| 变长序列与定长 NCCL buffer 冲突 | 多卡时 shape 不匹配 | `world_size > 1` 时强制 pad 到 `max_seq_len`，并在配置校验阶段拦截 |
| transformers 5 接口差异 | decoder 输出、mask 的组织方式、默认 dtype 都与 4.x 不同 | 只通过 `sequence.py` 和 `partition.py` 两处接触 HF 内部接口；显式设置 dtype；等价性测试覆盖 |
| `datasets` 与 `huggingface_hub 1.x` 版本不兼容 | 数据加载失败 | M0 安装时验证；synthetic 数据集不依赖 `datasets` |

---

## 9. 待确认事项（已按默认值写进计划，可以修改）

1. **调度模式命名**：默认改为 `sequential / gpipe / splitmind / splitmind_eager / 1f1b`，README 中保留新旧名称对照。
2. **pad 是否计入 loss**：默认不计入（label 为 -100），另外提供 `loss_on: completion`。如果要和老实验的 loss 曲线逐点对齐，可以加一个开关保留老行为。
3. **B39**：老模拟器在判断服务端是否等待激活卸载时用了 `client_offload_mb_num`（`simulate.py:276,297`）。默认先忠实移植，并在金标准测试中标注；如果确认是 bug，再改成 `server_offload_mb_num` 并重新生成金标准。
4. **单机 WAN 实验默认使用 netns 拓扑**：如果你们的环境更倾向于用 docker 或只做双机实验，只需替换 `lib/ray_up.sh`。
5. **不支持 GPT-2**：论文实验没有用到。如果需要，增加一条 `ArchSpec` 并支持绝对位置编码即可（约 30 行）。
