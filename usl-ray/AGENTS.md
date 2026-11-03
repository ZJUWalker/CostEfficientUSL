# CostEfficientUSL

CostEfficientUSL 是 SplitMind 论文的实验仓库，研究在数据留在客户端的前提下，以 U-shaped Split Learning 微调大语言模型。客户端持有 Head、Tail、输入和标签，服务器持有 Trunk；系统通过混合流水线、激活/模型状态卸载及 profile-driven simulation，降低客户端显存、训练时间和服务器租金。

## 规则维护

- 只维护根目录这一份 `AGENTS.md`，仅记录长期目标、架构边界、关键不变量和工作入口；实验结论与操作细节写入 `usl-ray/README.md` 或专项文档。
- **未经人类明确许可，不得编辑 `AGENTS.md`。** 单次修复不追加全局规则。
- 论文描述研究意图，`USL_RAY_REFACTOR_PLAN.md` 记录设计与实施偏差，当前行为以 `usl-ray/` 的代码、测试和 README 为准；发现三者冲突时应明确指出并同步文档。

## 项目边界

| 路径 | 职责与修改边界 |
| --- | --- |
| `usl-ray/` | 当前维护的独立 Ray 实现；新功能、修复和测试默认只在这里进行 |
| `usl/` | 论文对应的 legacy 基线；用于行为和模拟结果对照，未经明确要求不修改，也不得被 `usl_ray` import |
| `experiment/`、`test/` | legacy 启动与实验脚本；不作为新实现的公共入口 |
| `tools/export_legacy_simulate_fixtures.py` | 一次性读取 legacy 模拟器并导出金标准；运行时新包不依赖它 |
| `usl-ray/experiments/` | YAML 配置、Ray 生命周期、tc/netns、校准和网格实验入口 |
| `SplitMind__Towards_Cost_Effective_U_Shaped_Split_Learning_for_Private_LLM_Training.pdf` | 研究方法与实验目标的原始依据 |
| `USL_RAY_REFACTOR_PLAN.md` | 重构需求、B1–B39 缺陷、M0–M10 实施记录和 E1–E7 验收计划 |

- `usl-ray/pyproject.toml` 是 Python 与依赖版本的唯一来源；根 `requirements.txt` 必须与其运行时依赖保持一致。
- 模型、数据集、运行输出和 profile 属于本地实验资产；不要提交权重、数据、凭据或大型 `runs/` 产物。
- `usl-ray` 固定依赖 torch 2.9.1、transformers 5.0.0 和 Ray 2.58.0，因实现使用 torch pipeline 与 HF mask 的版本相关接口；升级必须单独验证。

## 核心架构与不变量

1. **U 形边界。** Head 为 `[0, SP)`，Trunk 为 `[SP, L-SP)`，Tail 为 `[L-SP, L)`，必须满足 `2 * SP < L`；训练与模拟共用 `SplitPlan` / `balanced_partition`。
2. **数据归属。** `input_ids`、labels、logits 和 loss 留在客户端。跨 client/server 边界只允许 `act_up`、`act_down`、`grad_up`、`grad_down` 及小型 `SequenceSpec`；这不等价于对中间表示提供形式化隐私保证。
3. **消息身份。** 每个张量由 `(step, microbatch, kind)` 唯一标识；等待必须有超时，失败需传播到对端，不能依赖 FIFO 猜测微批身份或吞掉 Ray future 异常。
4. **通信分层。** 跨节点张量走 Ray ObjectRef；服务器 stage 间走 torch distributed/NCCL。应用层不得重新引入 socket、pickle 或 sleep 限速，WAN 条件由 `tc` 控制。
5. **序列语义。** Attention mask、position IDs 和 RoPE 不跨边界；各 partition 必须由同一个 `SequenceSpec` 和模型配置本地重建。
6. **梯度语义。** 一个 global batch 内各 stage 使用同一版本权重，所有微批完成后再更新；loss 只能缩放一次，server stock schedule 保持 `scale_grads=False`；tied embedding/lm_head 只能由一个 optimizer 更新。
7. **调度语义。** 客户端五种模式集中在 `client/schedules.py`，使用有界循环和 Future；不得恢复轮询式 `while True`。修改顺序时同步检查 server schedule、无死锁预热条件和模拟器。
8. **异步资源所有权。** CUDA D2H/H2D 完成前保留源张量或正确记录 stream；compute stream 使用恢复张量前等待 event。Actor、process group、executor、placement group 和 GPU 资源在成功与失败路径都要释放。
9. **Profile/Simulate 一致。** 时间单位为 ms、显存为 MiB、带宽为 SI Mbit/s；真实执行与模拟必须共享时间线、指标、层划分和消息大小语义。same-node Ray 走共享内存，不能用于验证 WAN 或 `tc`。

## 按任务查阅

| 任务 | 入口 |
| --- | --- |
| 配置、握手、CLI | `usl-ray/src/usl_ray/config.py`、`cli.py`、`experiments/configs/` |
| 客户端调度与训练 | `client/schedules.py`、`client/trainer.py`、`client/launcher.py` |
| Ray 传输与失败传播 | `transport.py`、`tests/unit/test_transport.py`、`tests/integration/test_ray_transport.py` |
| 服务端流水线 | `server/stage_actor.py`、`server/boundary.py`、`server/launcher.py` |
| 模型拆分与序列输入 | `models/partition.py`、`models/sequence.py`、`models/loader.py` |
| Offload | `offload/activation.py`、`offload/state.py`、`tests/gpu/test_offload_e2e.py` |
| Profile、模拟与搜索 | `profile/`、`simulate/`、`experiments/calibrate.sh` |
| 端到端正确性 | `tests/integration/test_e2e_equivalence.py`、`test_server_gloo.py` |

## 变更与验证

- 修改前检查工作区差异并保留用户改动；不顺带修改 legacy、依赖版本或无关模块。
- 在 `usl-ray/` 中安装：`python -m pip install -e '.[dev]'`。Python 改动运行 `ruff format <files>`、`ruff check <files>`，再执行最小相关 pytest。
- 配置/模型/传输/调度等核心契约变化至少运行对应 unit tests；跨边界或梯度语义变化运行 `tests/integration/test_e2e_equivalence.py` 和相关 gloo 集成测试。
- CUDA、NCCL、tc/netns 和双机 WAN 结果必须注明实际硬件与拓扑；CPU/gloo 或 same-node 测试通过不能表述为已验证这些路径。
- 当前未闭环能力包括跨机 profile 时钟对齐、运行时 link probe 接线、非 SplitMind 模拟、NF4 codec 及真实 E4/E5/E7 WAN 实验；修改相关代码时不要把文档中的计划误当成已实现行为。
- 交付时说明行为变化、执行过的命令与结果、未执行的硬件/网络验证；测试通过不等于复现了论文的绝对性能数据。