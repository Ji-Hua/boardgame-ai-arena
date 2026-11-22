# AlphaZero-lite：冻结 D2 CNN 的搜索与自我博弈基准

这是现有系统内的小规模实验。训练在 `training` 上运行，沿用 Rust 规则、CNN observation / action mask、反射变换、逐 checkpoint 诊断、Arena、Decision Lab 和 MLflow。方法参考 [AlphaZero 论文](https://arxiv.org/abs/1712.01815)，没有扩大网络或引入额外搜索启发式。

## 实验边界

冻结起点是 `cnn_policy_d2_2k_update012000`（`quoridor-cnn-policy v6`），SHA256 为 `09ce6ba9b46ad1cde01407f8293da41fffe7c44492f41f8bab4850c1a1a0c5a4`。原 checkpoint 和生产 champion 不参与修改。

| 对照 | Actor | Value | 选步 |
|---|---|---|---|
| A | 原 D2 CNN | 无 | 合法 logit argmax |
| B | 与 A 参数逐字节相同 | 真实对局校准后冻结 | 50 simulations PUCT |
| C / policy | 从 B 初始化，2,000 局联合训练 | 联合训练 | 合法 logit argmax |
| C / MCTS | 同一个 C checkpoint | 同一个 Value Head | 50 simulations PUCT |

保留 7×9×9 编码、32/64/64 CNN、256 维共享隐藏层和 209 个策略 logits；增加 `Linear(256, 1) + tanh`，新增 257 个参数。checkpoint 内保存 actor、value 和模型配置；旧 checkpoint 仍按旧方式加载。无搜索推理仍只使用原 209 个 logits。

## 初始 Value Head

新增 head 权重和偏置先置零，不把随机输出当成有效叶评估。先生成 1,024 局：一半为冻结策略的随机采样自我博弈，一半对 Greedy，后者均衡 Greedy 的执手。采样温度 0.8，随机前缀 0–24 步。所有状态用实际终局结果标注当前执手的 `z ∈ {-1, 0, 1}`。

校准完成后，正式自我博弈的随机前缀排除直接到达目标行的走子。早期收集器在第 1,043 个前缀遇到终局后停止；原尝试的 1,024 局、2 个 checkpoint 和日志保留，完整重跑使用 `03-selfplay.json`。该修复和固定最终模型 ID 已在独立测试前写入 `protocol-amendment.json`。

按完整游戏划分：`episode % 5 == 0` 的 204 局作校准验证，其余 820 局训练。仅 Value Head 更新，actor 哈希必须不变。使用共享 X/Y 反射，训练 16 epochs，以验证 MSE 选择 head；如果不优于恒定 0 或训练集均值，回退到较好的常数预测。这个验证集用于 head 校准，不是最终对战测试集。

本次校准验证 MSE：恒定 0 为 **0.662156**，拟合 head 为 **0.637254**；非和局样本的符号正确率 **59.16%**，选中 epoch 11。无需常数回退，但预测能力有限。Actor 参数哈希仍为 `7987e2456e8e279248d713e3b5f8c36d2ae1dd163f7b521169337b1adeb491c9`。

## 搜索与训练

`agent_system/runtime/search/puct.py` 是训练、Arena 和 Decision Lab 共用的实现。

- 每步新建树。合法动作 prior 来自 masked softmax；选择 `Q + c_puct × P × sqrt(1 + ΣN) / (1 + N)`，`c_puct=1.5`，未访问边 Q=0。
- 每步 50 次完整模拟，不把根网络调用算成模拟。根和新叶的网络行数单独计为 `model_evaluations`，实际 batch 调用计为 `forward_batches`。
- 每条边的 Q 都是其父节点执手的视角；回传时逐边检查执手并换号。引擎终局使用真实胜者，优先于任何截断；不依赖“终局 current_player 一定是谁”的假设。
- 搜索路径遇到相同完整 observation 时按 0 回传；128 层深度上限也按 0 截断。这是实验搜索约定，不是 Rust 规则宣告和棋。没有跨根历史、tree reuse、transposition table、virtual loss 或战术选步覆盖。
- 32 局并行收集，独立树每轮各推进一次模拟，再批量计算新叶。每个棋局拥有独立可复现 RNG；不同树之间不共享访问统计。
- 自我博弈根噪声为 25% Dirichlet(0.15)，前 24 步按访问比例采样，此后取最多访问动作。访问次数同分时取最大 prior，再取最小动作索引。评估禁用噪声和采样。
- 一局真实获胜时赋 ±1；三次完整状态重复或达到 160 步时显式判实验和局并赋 0。分别记录原因。前缀不进入 policy/value 训练样本。
- 保存每步 `(obs, legal, pi, outcome, episode)`，`pi=N/ΣN`，最终结果按各步当前执手换号。只训练这些 MCTS 访问分布和最终回报，没有教师交叉熵或距离奖励。
- 64,000 状态 replay，batch 256，每批游戏 64 次 AdamW 更新，lr=2e-5，weight decay=1e-4，梯度范数上限 1，loss 为 masked soft cross-entropy + value MSE。反射时同步置换 pi，当前执手价值不变。
- 在 512 / 1,024 / 2,000 局保存并做 Greedy 诊断，最终比较固定使用 2,000 局模型，不按测试胜率挑 checkpoint。

策略学习、价值学习和搜索预算都很小。本实验只检验这套配置，不代表 AlphaZero 方法的能力上限。

## 评估与工具接入

继续使用 `scripts/evaluate_cnn_repair.py` 锁定无搜索 checkpoint、生成独立开局并运行既有诊断；再把同一 `holdout/selection.json` 传给 `scripts/run_benchmark_matrix.py --book`。后者支持 roster 中可选的 `matchups`，仅安排本实验需要的组合。

每个组合固定 100 个独立开局、200 局，逐开局交换执手，300 步上限，CPU 单 Torch 线程。固定对手为 Greedy、Minimax D1/D2、原 D2 CNN 和 `cnn_rl_ppo_imitation_ep04096`。标准初始局面两局作为单独 control，不混入排名。最终开局排除历史已记录训练状态、此次校准/自我博弈状态及历史评估开局的反射同类；同一新开局集合也按反射去重。

Arena 原有的确定性循环检测、W/D/L、pair-level Hoeffding score CI 保持使用；增补 W/D/L 比例各自的 pair-level CI、p50/p95/平均决策延迟、搜索成本、一步制胜漏判和重复率。置信区间的抽样单位是开局对，不能把两局当作独立样本；训练只跑一个 seed，区间不描述训练随机性。

`diagnose_checkpoint(..., search={...})` 复用相同的 50 局开发诊断及 18 个立即获胜探针。搜索模式回放检查实际搜索动作；策略梯度仍对应网络 logits，不是不可微搜索的梯度。开发诊断不参与最终统计结论。

Decision Lab 的神经网络页对带 Value Head 的模型显示 `PUCT / Value`：可设置 1–200 simulations 和 c_puct，查看当前执手 Value、每个合法动作的 prior、N、N/ΣN、Q 及计算量。主棋盘保持无搜索策略分析；搜索表高亮 MCTS 选择，支持导出 `neural_search`。API 为 `POST /api/neural/search`，与 Arena 调用同一个搜索函数。未经校准/训练的 head 会被拒绝。

模型目录继续由 `publish_cnn_repair.py` 发布，普通 `dqn` agent 表示 actor-only；同一 SHA 的 `_mcts50` agent 表示搜索变体。MLflow 继续注册 `quoridor-cnn-policy`，附加 value 状态、父 checkpoint SHA 和搜索配置；不更改 champion alias。

## 复现与产物

配置位于 `agent_system/training/configs/cnn_alphazero_lite/`；`protocol.json` 在对战评估之前锁定。训练目录是 `/home/jihua/quoridor-data/experiments/cnn-alphazero-lite-20261021`，所有权重、数据、审计与日志都在 Git 之外。

```bash
# 在 training:/home/jihua/workspace/boardgame-ai-arena 中运行。
.venv/bin/python scripts/train_cnn_repair.py --config agent_system/training/configs/cnn_alphazero_lite/01-calibration.json
.venv/bin/python scripts/train_cnn_repair.py --config agent_system/training/configs/cnn_alphazero_lite/03-selfplay.json
```

现有输出目录不会被覆盖。复跑需选择新的外部 output / checkpoint_dir，第二阶段 parent 路径与 SHA 必须改成此次新校准 checkpoint。即使模型参数相同，checkpoint 序列化时间/ID 也会使文件 SHA 改变。

每个训练目录包含：完整配置和输入 SHA、源码 tar/manifest、训练逐局记录、按批保存的训练样本、训练状态 geometry keys、实际计算量、checkpoint 与诊断。训练状态快照用于审计，当前 runner 不提供中断续训；可从固定 seed 完整重跑。CUDA events 是所标注区间的经过时间，可能包含调度间隙，不能当作独占 GPU kernel 时间。

聚合记录为 `assessment.json`、`holdout/comparison.json`、`holdout/arena/report.json`，逐局动作保存在现有 Arena blocks 中。MLflow 同时保留成功与失败 checkpoint 以及训练数据、校准报告和对局证据。

## 验证

```bash
PYTHONPATH=.:backend-server .venv/bin/python -m pytest \
  agent_system/training/tests/integration/test_alphazero_lite.py \
  agent_system/analysis/tests/units/test_mcts.py \
  agent_system/evaluation/arena/tests/integration/test_matrix.py -q
cd decision-lab
node --test tests/units/*.test.js
```

测试覆盖合法性、访问归一化、批量/单树一致性、双方终局、每层回传视角、对手立即获胜、路径重复、soft-target 非法动作梯度屏蔽、反射置换、实际训练数据/回报、校准不改变 actor、checkpoint 往返、Decision Lab 与 Arena 选择一致性及费用计数。

## 2026-10-21 实测结果

已完成正式 2,000 局 MCTS self-play、3,800 局独立 Arena 比赛和 38 局标准开局 control，没有非法动作或 forfeits。正式训练生成 41,909 个状态，执行 2,095,450 次模拟、1,889,433 个神经状态评估、4,032 次梯度更新；训练 wall time 699.82 秒。校准和失败尝试的成本另外保留，不计入正式 2,000 局。

| 方法 | Greedy 胜 / 和 / 负 | Greedy 得分及 95% CI | 同局面 CPU 平均延迟 |
|---|---:|---:|---:|
| A 原 D2 CNN | 115 / 0 / 85 | 57.5% [43.9%, 71.1%] | 1.01 ms |
| B 冻结策略 + MCTS | 120 / 0 / 80 | 60.0% [46.4%, 73.6%] | 38.45 ms |
| C 训练后，无搜索 | 75 / 0 / 125 | 37.5% [23.9%, 51.1%] | 1.00 ms |
| C 训练后 + MCTS | 105 / 0 / 95 | 52.5% [38.9%, 66.1%] | 37.79 ms |

延迟是比赛完成后，对相同 100 个开局串行、预热并轮换方法顺序测量，详细记录在 `holdout/latency.json`；Arena 内实际中后盘延迟另外保存。`assessment.json` 含所有固定对手的 W/D/L 和区间、配对差值、计算量、一步制胜漏判和重复率。对 Greedy 的 B−A 为 +2.5 个百分点，保守配对 CI 跨零；不能认为已经证明搜索变强。

C 无搜索对 Greedy 下降 20 个百分点，对原模型为 7 胜 / 106 和 / 87 负；搜索能恢复部分棋力，但本轮没有得到优于原模型的候选。C 搜索对 D2 为 2 胜 / 99 和 / 99 负，得分上升主要来自和局，不能解释成超过老师。原模型和 champion 保持不变。

校准与正式训练的 3,024 局、88,941 个样本经过 Rust 重放核对，胜者、当前执手、最终回报均无不一致。18 个既有立即制胜探针中，B 的 Value 有 5 个负值，C 有 4 个；actor 和 MCTS 都能 18/18 取胜。共同对手上的 C 放墙比例由原模型的 22.5% 降至 10.9%，提示需要检查策略保留与状态覆盖。它们是后续消融的依据，不是已证明的唯一退步原因。

下一轮先保持 actor 冻结，改善 Value 的中盘/战术覆盖；再独立比较 50/100/200 次搜索；确认搜索目标有效后，分别尝试更小更新幅度或教师 imitation 约束，最后用多 seed 验证。不要把这些改动一次混合。本轮详细结论和复现记录位于外部实验目录的 `benchmark-report.md`。最终回归验证为 140 个 Python 测试和 8 个 Decision Lab JavaScript 测试通过。

[MLflow 实验 8](http://192.168.68.88:8742/#/experiments/8) 保留 9 个 FINISHED run 和 1 个 FAILED run，六个 checkpoint 均 READY。B 为 `quoridor-cnn-policy v24`；正式 C 为 `v29`（`cnn_az_selfplay_fixed_ep02000`，SHA `c808a38d91d313a6b7819da5f4718e2a2d01f223241d1115a6d454218228da3e`）。v25–26 是中断尝试的已诊断 checkpoint，训练状态标记 failed；v27–28 是完整重跑的中间 checkpoint。现有 74 个模型注册索引和全部旧别名已核对不变。当前 [Decision Lab](http://192.168.68.88:8744/) 模型目录为 80 项。
