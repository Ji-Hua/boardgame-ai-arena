# D2 CNN: DAgger、PPO、自我博弈与 Double DQN 对照

本轮研究沿用现有训练入口、Arena、checkpoint 诊断和 Decision Lab。没有另建评估框架。
实验产物放在 training 的 `/home/jihua/quoridor-data/experiments/cnn-policy-rl-20261020/`，不进入 Git。

## 冻结起点

- 模型：`cnn_policy_d2_2k_update012000`，`quoridor-cnn-policy` v6。
- SHA256：`09ce6ba9b46ad1cde01407f8293da41fffe7c44492f41f8bab4850c1a1a0c5a4`。
- 原实验：同一批 2,000 局示范、12,000 次合法动作交叉熵更新。上轮独立评估对 Greedy 为 473/800；这不是本轮结果。
- Actor 保持 7×9×9 输入、32/64/64 卷积、256 维隐藏层、209 个输出和 LeakyReLU，共 1,438,545 个参数。
- A–D 从此模型的全部 Actor 权重开始；E 只复制卷积层，整个 dense head 重新初始化。所有新实验使用新的优化器。
- PPO 增加训练用的 256→1 Value Head（257 个参数，零初始化）。对战和 Decision Lab 仍使用原尺寸 Actor，不调用 Value Head、老师或搜索。

## 独立方案及预算

| 方案 | 初始化 | 新训练对局 | 更新方式 | 训练对手 |
|---|---|---:|---|---|
| A DAgger | 冻结 D2 Actor | 2,000 | 4 轮 × 1,500 次监督更新 | 运行时 Greedy |
| B PPO | 冻结 D2 Actor | 4,096 | 32 局一批，最多 4 epochs，KL 提前停止 | 运行时 Greedy |
| C PPO + Imitation | 冻结 D2 Actor | 4,096 | 与 B 相同，增加 0.3 × 示范 CE | 运行时 Greedy |
| D Self-play PPO | 冻结 D2 Actor | 4,096 | 与 B 相同，不增加示范 CE | 冻结历史策略池 |
| E Transfer DDQN | 仅 D2 卷积层 | 4,096 | 每批约 0.25 × learner transitions 次 DDQN 更新 | 运行时 Greedy |
| PPO random 对照 | 全部随机初始化 | 4,096 | 与 B 相同 | 运行时 Greedy |
| DDQN random 对照 | 全部随机初始化 | 4,096 | 与 E 相同，初始 dense head 与 E 完全相同 | 运行时 Greedy |

训练种子统一为 42，起始局面随机种子为 2026102501，各局独立派生 RNG；minibatch 种子为 1874。
这是一组单种子先导实验。开局置信区间不能代替重复训练种子；也不能把不同算法的相同对局数理解为相同算力。
所有方法的主结果固定使用最后一个 checkpoint。中途较好的开发集成绩不改变主结果选择。

A 的 checkpoint 为第 500、1,000、2,000 局；其余为 1,024、2,048、4,096 局。
每个保存的 Actor 都立即走原 `diagnose_checkpoint`：25 个开发开局交换先后手，共 50 局对 Greedy，并做 18 个一步制胜检查及逐决策激活/漏判诊断。

## 训练实现

配置在 `agent_system/training/configs/cnn_policy_rl/`，入口仍为：

```bash
.venv/bin/python scripts/train_cnn_repair.py --config agent_system/training/configs/cnn_policy_rl/01-dagger.json
```

新增训练模块 `agent_system/training/dqn/policy_research.py` 使用现有规则引擎、CNN、教师搜索、Double DQN 更新和 checkpoint 格式。

### 局面与经验

每局先从标准棋盘执行 0–24 步随机合法前缀（随机动作中走子概率 0.7），之后双方交替，由当前方案控制固定一方，先后手按局交替。前缀动作和训练动作分别计数、完整保存。

一条 learner transition 包含本方动作及对手回复。非终局的下一观测一定仍是本方视角，避免把对方价值当成本方价值。RL 奖励只有终局胜 +1、负 −1；没有距离奖励或老师动作奖励。折扣 γ=0.99。

对战部分上限为 300 plies，达到上限仍关闭对手回复，之后截断；截断不伪造终局输赢。PPO 用末局面的 critic bootstrap，DQN 保留合法下一动作和 Bellman bootstrap。

DAgger 的行为和 Greedy 都确定，因此重复同一局面可以判定循环并结束收集。PPO 的 learner 和自我博弈对手会采样动作，重复局面仍可能退出，不能采用确定性循环判定。最终 Arena 仍使用既有的确定性 argmax 和循环规则。

### A：DAgger

每轮 500 局使用当前学生的确定性合法 argmax；D2 只标注学生实际遇到的决策局面，不接管执行。各轮标注持续聚合，按四种反射的局面 hash 去重；batch 的一半来自原 D2 数据，一半来自截至当前轮的聚合数据。原数据的 25% 抽样来自一步制胜或 D1/D2 分歧局面，聚合数据均匀抽样。

监督目标仍为合法动作 CE。沿用已验证的棋盘反射增强，正确处理 9×9 走子坐标和 8×8 墙锚点；搜索剪枝边界值不作为数值标签。学习率 5e-5，batch 256，每轮 1,500 次更新。

### B/C/D：PPO

合法动作先 mask，再建立 Categorical 分布并采样；保存动作、旧 log probability 和旧 value。逐局计算 GAE，λ=0.95；每批统一标准化 advantage。PPO clipping=0.2，value coefficient=0.5（value loss 内含 0.5），entropy coefficient=0.01，target KL=0.03。学习率 2e-5，batch 256，梯度范数上限 0.5，AdamW 的 weight decay 为 0。

每批最多 4 epochs；若当前 minibatch 与采样策略的近似 KL 超过上限，在再次更新之前结束该批优化。记录实际梯度更新数、KL、clip fraction、entropy、value loss、解释方差等。PPO 不对 on-policy 数据做反射增强，避免新局面与旧 log probability 不对应。

B 不加载示范数据。C 仅额外加入原 D2 示范的合法 CE（系数 0.3），不增加新的老师搜索，也不混入 A 的标注。示范抽样使用单独 RNG，避免改变 PPO minibatch 的随机顺序。D 不加 CE，仅改变对手来源。

D 的初始对手为冻结的 D2 parent 和独立训练种子 43 的 D2 peer；每两个相邻对局使用同一对手并交换 learner 席位。在第 1,024、2,048 局诊断后将该 Actor 冻结加入历史池。对手权重永不随 learner 优化更新，动作使用其合法策略分布采样。最终固定测试池含 Greedy、Minimax D1/D2、旧 Greedy-demo CNN 和原 CNN 2k，均不在此训练池中；parent 另作起点对照，其与训练池的重合明确保留在元数据中。

### E：卷积迁移到 Double DQN

复制 parent 的 `conv.*`，重新初始化 `head.*`；不会把 policy logits 当作 Q 值。random DDQN 对照使用相同随机种子和完全相同的初始 dense head，仅卷积层初始化不同。

复用原 `trainer.train_step(..., algorithm="double_dqn")`：online 网络选取下一合法动作，target 网络估值；Huber TD loss，无示范、margin 或 MC loss。学习率 5e-5，batch 256，replay 60,000 条，每批 `ceil(learner_steps × 0.25)` 次更新，target 每 500 次更新同步。ε 从 0.3 线性降到 0.05，随机探索的走子概率为 0.7。卷积层与 head 均可训练。

## 资源与可复现记录

- `experiment.json`、输入 SHA256、source archive/manifest、初始化各层权重 hash。
- `research-metrics.jsonl`：真实游戏数、对战 plies、前缀 plies、learner decisions、梯度更新数、Minimax 调用/节点/CPU 时间、分阶段 CUDA event 时间、排除诊断后的训练 wall time。
- CUDA event 是设备队列上的经过时间，包含调度/竞争，不能解释为独占 kernel 时间或 GPU 账单；不同方案同时运行时尤其如此。
- 原 2,000 局预训练和旧示范搜索属于共享历史成本；本轮的新搜索调用单独记录。C 的新搜索调用为零，不代表原示范没有成本。
- `training-games.jsonl.gz`：完整前缀、动作、席位、对手、终止原因；`training-state-keys.json` 用于后续评估排除。
- `training-state/`：Actor/Critic、optimizer、RNG，以及 DDQN target 的快照。它不是完整自动恢复接口（尚未包含每一 checkpoint 的 replay/DAgger aggregate）。
- `dagger-aggregate.pt`：A 最终聚合标注；`opponent-pool.json`：D 的冻结对手及 SHA。
- 诊断使用独立 RNG 上下文，不改变后续训练采样。

## 复用评估框架

1. 原 `evaluate_cnn_repair.py` 锁定每个最终 checkpoint。`--comparison-prefix` 仅扩展原脚本的多方案对比，区别于重复种子；不是新评估实现。
2. 生成 400 个新开局（seed 2026102502），排除历史评估/开发开局及已记录训练反射组；每个最终模型和 parent 各对 Greedy 800 局，交换先后手。所有模型复用同一本开局。
3. 原 `run_benchmark_matrix.py --book holdout/selection.json --pairs 100` 复用其中前 100 对开局，执行统一 round robin，含 7 个新模型、parent、Greedy、Minimax D1/D2、旧 Greedy-demo CNN、原 CNN 2k。矩阵中保留额外标准开局控制局，Elo 只用于同协议内比较。
4. 原 `diagnose_cnn_teachers.py --fidelity-only` 复用已完成的 Arena 比赛，计算共同局面库上的 D2 最优动作率、强制败着、一步制胜等；不重复生成另一套比赛。共同局面来自新评估开局及轨迹，排除配置中的全部已记录训练反射组。选中动作使用完整搜索窗口判断，允许与老师同分的动作。
5. 继续报告既有的 opening-pair Hoeffding 区间；7 个新方案对 parent 的同时区间采用 Bonferroni。先后手不是独立样本，单轮结果不证明算法的普遍优劣。

## 模型管理与分析

所有 21 个 checkpoint（含失败或退化模型）沿用 `publish_cnn_repair.py` 和 `log_cnn_repair.py` 注册；原模型只读保留。A/B/C/D 和 random PPO 使用 `quoridor-cnn-policy`，E/random DDQN 使用 `quoridor-cnn-dqn`。

PPO 的 `output_semantics=policy_logits` 在 inference metadata、Decision Lab 和 MLflow 中保持一致，UI 显示 logit；DDQN 显示 Q。Value Head/optimizer 单独作为训练产物保存，不混入 209 动作输出。没有自动调整在线 champion 或对战服务 roster。

## 算法来源

- [DAgger 原论文](https://proceedings.mlr.press/v15/ross11a.html)：迭代收集当前策略实际访问的状态并聚合专家标签。
- [PPO 原论文](https://arxiv.org/abs/1707.06347)：clipped surrogate objective。
- [GAE 原论文](https://arxiv.org/abs/1506.02438)：优势估计中的 bias/variance 折中。
- [Double DQN 原论文](https://arxiv.org/abs/1509.06461)：分离下一动作的选择与估值。

## 结果

七组训练全部完成，共保存并诊断 21 个 checkpoint。以下使用预先指定的最终模型；没有根据独立测试重新挑选中间版本。

主比较为每模型 800 局对 Greedy，使用 400 个新开局并交换先后手。多对手矩阵为 13 个 agent、每一对 200 局，另有标准开局控制局；对冻结 D1 CNN 的稳定性复测也是每模型 200 局。新增执行的比赛共 24,822 局，包含开发诊断和重复开局控制，不能当成 24,822 个独立样本。

本轮排除了 1,098,534 个已记录训练反射组。D2 动作诊断使用共同的 8,335 个未命中这些训练组的局面；它不同于上轮的 2,577 局面库，只比较本轮各模型之间的差异，不直接把百分比与上轮 59.72% 相减。

| 模型 | 对 Greedy 胜率（800 局） | D2 最优动作率 | 对 D1 CNN 循环率（200 局） | 一步制胜检查 |
|---|---:|---:|---:|---:|
| 冻结 D2 CNN | 479/800 = 59.875% | 56.62% | 16.0% | 18/18 |
| A DAgger | 443/800 = 55.375% | 61.36% | 16.0% | 18/18 |
| B PPO | 469/800 = 58.625% | 53.94% | 9.0% | 18/18 |
| C PPO + Imitation | 488/800 = 61.000% | 57.36% | 12.0% | 18/18 |
| D Self-play PPO | 196/800 = 24.500% | 53.58% | 29.5% | 18/18 |
| E Transfer DDQN | 0/800 = 0.000% | 24.09% | 18.0% | 5/18 |
| PPO random | 0/800 = 0.000% | 15.84% | 24.0% | 0/18 |
| DDQN random | 0/800 = 0.000% | 25.95% | 49.0% | 8/18 |

对运行时 Minimax D2 的直接对局如下；**胜、和、负必须分开看**，循环和棋不能称为战胜老师。教师在相同的 100 对开局中对 Greedy 为 173/200（86.5%）。

| 模型 | 胜 / 和 / 负（200 局） | 胜率 | 得分率（和棋计 0.5） |
|---|---:|---:|---:|
| 冻结 D2 CNN | 0 / 64 / 136 | 0.0% | 16.00% |
| A DAgger | 3 / 20 / 177 | 1.5% | 6.50% |
| B PPO | 0 / 65 / 135 | 0.0% | 16.25% |
| C PPO + Imitation | 0 / 44 / 156 | 0.0% | 11.00% |
| D Self-play PPO | 4 / 8 / 188 | 2.0% | 4.00% |
| E Transfer DDQN | 0 / 18 / 182 | 0.0% | 4.50% |
| PPO random | 0 / 0 / 200 | 0.0% | 0.00% |
| DDQN random | 0 / 63 / 137 | 0.0% | 15.75% |

### 主要结论

1. **目前没有证据支持全面替换冻结的 D2 CNN。** C 的 Greedy 胜率最高（61%），只比 parent 的 59.875% 高 1.125 个百分点。7 组预先指定比较的 Bonferroni-Hoeffding 同时区间为 −15.66 至 +17.91 个百分点；这套区间保守，但不能据此宣称提升。C 的矩阵 Elo 为 1500.53，parent 固定为 1500，差异也很小。
2. **A 更像老师，但整局胜率没有随之提高。** D2 最优动作率从 56.62% 提高到 61.36%，共同局面中的可避免 D2 深度败着从 12 个降至 4 个；Greedy 胜率反而降到 55.375%。新增 38,749 次教师查询产生 35,895 个去重标注。A 的训练只对 Greedy，2,000 局全部正常结束，没有采到 CNN 互相循环的局面；因此没有排除针对这类局面的 DAgger 的价值。
3. **B/C 保留了大部分原有棋力。** B 对冻结 D1 CNN 的循环率从 16% 降至 9%，C 降至 12%；但 B 的完整结果为 164 胜、18 和、18 负，parent 为 160 胜、32 和、8 负，减少的和棋也转成了一些败局，不能单凭循环减少宣布棋力提高。C 为 164 胜、24 和、12 负。
4. **D 学到了针对训练对手的打法，泛化却退化了。** 对冻结 parent 为 50 胜、145 和、5 负；对 Greedy 只有 24.5%，对 Minimax D2 为 4 胜、8 和、188 负。与 parent 的直接得分率为 61.25%，保守配对区间仍包含 50%，所以这里只将它视为针对对手适应的证据。对冻结 D1 CNN 的循环率上升到 29.5%。自我博弈训练还出现了 464 次时间截断，最终对 Greedy 的放墙占比为 41.1%，parent 为 23.5%。冻结历史池本身没有防止退化。
5. **E 与两个冷启动对照都暴露了胜局经验不足。** E 在 4,096 个训练游戏中只有 2 胜，random PPO 与 random DDQN 都是 0 胜。三者对 Greedy 的独立测试均为 0/800。相同 PPO 配置下，预训练相对随机初始化的配对胜率差为 +58.625 个百分点，同时区间为 [+43.82,+73.43]；conv-only DDQN 相对 random DDQN 的差为 0，同时区间为 [−14.80,+14.80]。这说明本预算内 PPO 的已有棋力主要依赖预训练，没有显示卷积迁移能独自解决 Q 学习的冷启动。

这是一轮单训练种子实验，不能把上述结果推广为算法的普遍结论。各算法的实际交互和更新量不同，尤其 D 的 learner steps 是 B 的约 3.55 倍；这些结果也不是严格等算力排名。

### 资源记录

对战 plies 不包含单列的随机前缀；一次 learner step 包含本方动作和对手回复。CUDA 列是实际 CUDA event 经过时间的累计值，包含调度与竞争，不是独占 GPU kernel 时间。

| 方案 | 新局数 | 对战 plies | 前缀 plies | Learner steps | 新 D2 查询 | 梯度更新 | CUDA event 秒 | 训练 wall 秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A DAgger | 2,000 | 77,720 | 24,423 | 38,749 | 38,749 | 6,000 | 15.58 | 513.16 |
| B PPO | 4,096 | 128,608 | 49,298 | 64,040 | 0 | 1,236 | 7.75 | 96.26 |
| C PPO + Imitation | 4,096 | 136,474 | 49,298 | 67,943 | 0 | 1,352 | 10.85 | 103.81 |
| D Self-play PPO | 4,096 | 454,965 | 49,298 | 227,511 | 0 | 3,803 | 96.07 | 202.81 |
| E Transfer DDQN | 4,096 | 101,538 | 49,298 | 49,727 | 0 | 12,476 | 48.90 | 131.81 |
| PPO random | 4,096 | 100,438 | 49,298 | 49,176 | 0 | 1,028 | 6.79 | 82.54 |
| DDQN random | 4,096 | 113,588 | 49,298 | 55,751 | 0 | 13,988 | 52.73 | 142.25 |

总计 26,576 局新训练、1,113,331 对战 plies、320,211 前缀 plies、552,897 learner steps、39,883 次梯度更新。A 的教师搜索为 19,511,533 个节点、444.88 CPU 秒。各运行 wall time 相加为 1,272.63 秒；实验有并发，这不是从启动到全部完成的经过时间。

### Decision Lab 复现与模型管理

全部 74 个历史及新 checkpoint 已纳入同一模型目录，其中本轮新增 21 个。选中以下最终模型可直接在 [Decision Lab](http://192.168.68.88:8744/) 检查；原权重和历史模型仍保留。

| 方案 | 最终模型 ID |
|---|---|
| 冻结 D2 CNN | `cnn_policy_d2_2k_update012000` |
| A DAgger | `cnn_rl_dagger_ep02000` |
| B PPO | `cnn_rl_ppo_ep04096` |
| C PPO + Imitation | `cnn_rl_ppo_imitation_ep04096` |
| D Self-play PPO | `cnn_rl_selfplay_ppo_ep04096` |
| E Transfer DDQN | `cnn_rl_transfer_ddqn_ep04096` |
| PPO random | `cnn_rl_random_ppo_ep04096` |
| DDQN random | `cnn_rl_random_ddqn_ep04096` |

已经通过真实 HTTP API 对 parent 和全部新最终模型执行分析、分支检查、卷积激活、权重切片和输入梯度检查。示例：P1 在 a8、P2 在 e4 时，E 选择 b8 而非直接制胜的 a9；P1 在 e3、P2 在 e2 时，E 选择 b6v，D2 完整窗口分支值为 −∞，而 parent/A/B/C/D 选择防守横墙。所有这些模型都有非零激活和梯度，E 的问题并非原始 CNN 的全零隐藏层故障。

详细结果保存在实验目录的 `assessment.json`、`holdout/comparison.json`、`holdout/arena/report.json`、`holdout/teacher-audit/fidelity.json`、`holdout/d1-policy-control/report.json` 与 `decision-probes/`。训练状态、源代码、模型 SHA、逐 checkpoint 诊断和完整评估证据通过现有脚本写入 [MLflow](http://192.168.68.88:8742/)。

### 下一轮应回答的问题

- 对 B/C 做多训练种子重复，先确认微小变化能否复现；继续使用冻结 parent 作基准。
- 对 A 从新的 learner-vs-冻结 CNN / D2 对局收集循环和对抗局面，增加“相同更新数、仅继续旧示范”的对照，以区分数据覆盖与额外训练的作用；不把本轮评估轨迹拿来训练。
- 为冷启动 RL 单独测试能产生胜局的课程训练；为自我博弈保留固定棋力锚点，并用独立开发诊断决定历史模型是否进入池。若继续 E，先比较有真实回报标定的 Q head 初始化，仍不能将 logits 直接解释为 Q。
