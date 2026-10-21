# CNN Minimax 示范实验

## 目的与计数

分别从随机权重训练两个平面 CNN，老师为现有 Minimax D1、D2。每个模型训练 **2,000 个游戏 episode**；不把梯度更新数当作局数，不继承旧 CNN 或 Greedy 示范模型的权重。

旧基准 `cnn_observation_2k` 的 2k 也是游戏 episode：标准棋盘开局、学习者交替执先、达到步数上限也计为一局。旧 checkpoint 记录 `episode_count=2000`、`training_step=111135`，文件名中的 `step232954` 是环境行动数。这三个计数必须分开。

本轮每局仍从标准棋盘开始，学习者交替执先，最大 3,000 plies。对手固定为 Greedy；**Minimax 是示范老师，不是对手**。两个实验唯一的配置差别是老师深度、模型 ID 和输出路径。老师选择改变对局长度和数据分布，因此等局数不代表等样本数、等更新次数或等计算量。

## 训练方法

- CNN：7×9×9 输入，32/64/64 卷积通道，256 维隐藏层，209 个动作；保留前轮修复的 LeakyReLU。没有两级动作网络。
- 两组随机初始化 seed=42；另用独立 GPU 随机数流抽取 replay batch，seed=874。记录初始参数的 SHA-256，检查两组完全一致。
- 老师直接调用服务端 Minimax 的 `score_position`，沿用第一最大值的 tie break。搜索返回的上下界和无穷分值只用于选择动作，不作为数值 Q 目标。
- 每个学习者决策点都记录老师动作。随机探索概率按 episode 从 0.20 线性降到 0.05；非随机决策中，老师代走概率从 1.0 降到 0.25，其余由 CNN 选择。随机探索仍是原平面动作空间，只用 70% 走子 / 30% 放墙的采样比例。
- 学习者行动后必须等对手回复，才形成一个非终局转移；即使抵达 ply 上限，也关闭这一回复。下一观测保持学习者视角。
- 损失为同视角 Double DQN TD + 完整对局 Monte Carlo 回报 + 合法动作 expert margin。截断局只用 TD 和 expert margin，没有伪造的终局或 MC 回报。
- Replay 容量 100,000，warmup 512，batch 256，AdamW lr=1e-4，gamma=0.95，gradient clip=1，target 每 1,000 次更新同步。沿用距离变化 shaping：weight=0.04、clip=1。
- 每局收集结束后，每个新增学习者转移对应一次更新（warmup 前除外）。完整局面动作轨迹、实际计数、示范一致率和 loss 都保存在外部实验目录。

checkpoint 的 `episode_count` 是实际训练局数，`training_step` 是实际更新次数。`eval_summary` 记录老师深度、随机初始化摘要、代走率和累计计数。推理仍然只有 **CNN 的合法动作 masked argmax**，不调用老师或立即获胜补丁。

## 评估约束

1. 每 500 局保存一个 checkpoint，立即执行同一开发开局表上的 50 局 Greedy 对战（双方各执先 25 局）、18 个一步获胜探针以及隐藏层/Q 值诊断。
2. 主比较预先固定为两个 `ep02000`，即使早期 checkpoint 成绩更好也不替换。
3. 另生成 400 个唯一开局，交换先后手，每个模型对 Greedy **800 局**。两组新模型、原 CNN 2k、此前 fresh Greedy demo 和直接修复旧 2k 的模型使用同一开局表。
4. 排除此前评估表、可取得的 Greedy 示范 train/validation 局面，以及本轮两组所有访问过的局面及其镜像。包含截断局和从 replay 淘汰的状态；不是只排除最终 replay 中的状态。旧 CNN 2k 没有完整可用的训练局面集合，不能声称已经排除它的全部训练局面。
5. 置信区间以交换先后手的开局对为采样单位。对局结果相关，不能把 800 局当成 800 个独立样本；单个训练 seed 也不能证明普遍的模型排序。
6. 两组新模型与 fresh Greedy demo 另在前 100 个冻结开局对上交叉对战。Minimax D1/D2 老师本身也在完整开局表上对 Greedy 比赛，区分“老师不强”和“CNN 没学会”。
7. 只读 fidelity 审计使用共同的评估局面集，记录严格动作一致率、允许等价最优动作的比例、漏掉强制胜利/走入可避免失败的次数。评价 CNN 所选分支时使用完整窗口搜索，避免将 alpha-beta 的界当作精确值。

`D2 > D1 > Greedy demo` 是待检验假设。D1 的非终局动作排序与 Greedy 的距离变化评分只差局面常数；前轮 Greedy demo 已对立即获胜动作优先标注，因此更换为 D1 不会自然产生更强的示范。

## 实现与运行

- 训练器：`agent_system/training/dqn/teacher_training.py`。
- 配置：`agent_system/training/configs/cnn_minimax_demo/01_minimax_d1.json`、`02_minimax_d2.json`。
- 外部产物：`/home/jihua/quoridor-data/experiments/cnn-minimax-demo-20261020/`。
- `plan.json` 在训练启动时保存比较方案；每组保存配置、源码快照、随机初始化摘要、checkpoint、逐局轨迹、replay 和诊断。
- `training-state-keys.json` 是访问过的所有观测几何的镜像分组哈希，用于独立开局排除；`teacher-replay.pt` 是可继续分析的最后 replay。

在 training 服务器项目根目录运行：

```sh
.venv/bin/python scripts/train_cnn_repair.py --config agent_system/training/configs/cnn_minimax_demo/01_minimax_d1.json
.venv/bin/python scripts/train_cnn_repair.py --config agent_system/training/configs/cnn_minimax_demo/02_minimax_d2.json
```

输出目录已存在时训练器拒绝覆盖。重跑须使用新的实验目录和模型 ID。独立评估使用 `scripts/evaluate_cnn_repair.py --fixed-selection`，分别指定完整 `ep02000` 的 prefix，并传入 `--exclude-state-keys`；实际完整命令保存在实验根目录的 `evaluation-command.json`。评估完成后：

```sh
.venv/bin/python scripts/diagnose_cnn_teachers.py /home/jihua/quoridor-data/experiments/cnn-minimax-demo-20261020
```

模型登记沿用 `publish_cnn_repair.py` / `log_cnn_repair.py`，保留每一个 checkpoint（包括差的结果），不更新线上 champion。MLflow 实验使用 `Quoridor / CNN Minimax demonstrations`。Decision Lab 通过模型 ID 读取普通 CNN checkpoint，使用现有 Q 值、梯度、激活与候选动作视图分析。

## 验证

训练回归测试覆盖实际 Minimax 与完整窗口搜索的最大动作一致性、先后手和奇偶上限下的转移视角、真实 MC 回报、replay 环形写入、实际 checkpoint 的局数/更新数及两组相同随机初始化。独立评估回归覆盖冻结选择、参考模型共享开局、示范数据与在线训练几何排除。

## 2026-10-20 实际结果

两组各完成 2,000 局，没有截断局。初始参数 SHA-256 均为
`94abe8fd38ddb6486bd604b10f850c0529966e549c9993e216d7b91d95f2e048`，没有父 checkpoint。

| 训练计数 | D1 | D2 |
|---|---:|---:|
| 完整游戏 episode | 2,000 | 2,000 |
| 环境 plies | 31,659 | 47,471 |
| 学习者转移 / 老师查询 | 15,593 | 24,030 |
| 参数更新 | 15,085 | 23,527 |
| 访问过的镜像分组 | 5,761 | 17,430 |

老师参与代走时，训练胜率分别为 26.35%、79.45%。**这不是 CNN 的独立棋力**，不能拿来代替以下纯 CNN 推理结果。

逐 checkpoint 开发诊断（每个 50 局对 Greedy）：

| Episode | D1 胜局 | D1 一步获胜探针 | D2 胜局 | D2 一步获胜探针 |
|---|---:|---:|---:|---:|
| 500 | 5/50 | 3/18 | 4/50 | 10/18 |
| 1,000 | 3/50 | 5/18 | 10/50 | 11/18 |
| 1,500 | 4/50 | 5/18 | 7/50 | 10/18 |
| 2,000 | 4/50 | 6/18 | 11/50 | 10/18 |

独立评估使用 400 个唯一开局对，排除记录中的 90,341 个训练镜像分组。以下模型共享同一批 800 局，均由到达终点结束，无和局或非法动作：

| 模型 | 胜 / 负 | 胜率 | 执先胜局 / 400 | 执后胜局 / 400 |
|---|---:|---:|---:|---:|
| 原 CNN 2k | 18 / 782 | 2.25% | 11 | 7 |
| CNN D1，ep02000 | 75 / 725 | 9.38% | 50 | 25 |
| CNN D2，ep02000 | 98 / 702 | 12.25% | 59 | 39 |
| 旧 fresh Greedy demo，update002000 | 252 / 548 | 31.50% | 185 | 67 |
| 旧 2k 权重修复，seed42 update004000 | 255 / 545 | 31.88% | 185 | 70 |

D2 相对 D1 的胜率差为 **+2.875 个百分点**；预定的 paired Hoeffding 95% 区间是 **[-10.706, +16.456] 个百分点**。本轮点估计支持 D2 在前，但不能确认稳定优势。两组新模型都没有超过旧 Greedy 示范模型。旧 Greedy demo 的生成方式和优化流程不同，不能将跨流程的成绩差异直接归因于老师。

三种 CNN 直接交手，每组 100 个开局对 / 200 局，得分按胜=1、和=0.5：

| 左方 vs 右方 | 左胜 | 和 | 右胜 | 左方得分 |
|---|---:|---:|---:|---:|
| CNN D1 vs CNN D2 | 3 | 184 | 13 | 47.50% |
| CNN D1 vs 旧 Greedy demo | 28 | 54 | 118 | 27.50% |
| CNN D2 vs 旧 Greedy demo | 42 | 85 | 73 | 42.25% |

323 个和局全部为确定性状态循环。D1/D2 两者之间 92% 的比赛循环，因此不能用少量胜负或一个 Elo 数字宣称可靠的强弱顺序。三模型矩阵的临时 batch Elo（D1 锚定 1500）为 D1=1500、D2=1547、旧 Greedy demo=1633，只适用于这个协议。

老师本身在相同 800 局对 Greedy 中，Minimax D1 为 **400/800（50.00%）**，Minimax D2 为 **686/800（85.75%）**。D2 老师有明显更好的实际表现，学生只达到 12.25%，说明老师的能力没有充分转移。

## 诊断与下一步

本轮两个 CNN 在独立对战中都没有全零隐藏层或纯 bias 输出。实际 Decision Lab HTTP 的 Q、输入梯度、卷积激活和权重端点均可分析它们；不能继续用旧 ReLU 全零故障解释这次弱棋力。

**数据覆盖是目前最直接的证据。** D1 replay 中，12,434 / 15,593（79.74%）次决策的己方棋子处于 e 列，12,675 / 15,593（81.29%）个局面没有放置的墙。旧 Greedy demo 有 69,923 条训练转移、62,269 个训练镜像分组；本轮标准开局、固定 Greedy 对手、只有学习者随机探索的流程，产生的数据更少，也更集中。这个流程差异必须与老师深度分开看。

最终模型在各自 replay 中固定抽样 2,048 条记录，与老师动作的一致率分别是 D1 **100.00%**、D2 **99.95%**。训练记录可能重复，不将它当作验证成绩。在由全部独立开局及前 20 个开局对的模型对局轨迹组成、按镜像分组去重的 **2,725 个共同诊断局面**上：

| CNN | 与自身老师严格同动作 | 允许同分最优动作 |
|---|---:|---:|
| D1 | 882 / 2,725（32.37%） | 1,047 / 2,725（38.42%） |
| D2 | 855 / 2,725（31.38%） | 1,017 / 2,725（37.32%） |

旧 Greedy demo CNN 在同一集合上与 D1 同动作 1,489 / 2,725（54.64%），允许同分动作后为 60.33%。D1 与前轮“立即获胜优先”的 Greedy 示范标签，在全部 2,725 个局面上完全相同；D1/D2 老师之间有 575 个局面的标签不同。这支持“更强老师的标签存在，但本轮学生对常见训练局面的拟合没有泛化”的判断。轨迹中的后续局面可能在训练中出现过，因此该集合只是共同诊断集，不是额外的纯独立样本，也不用于挑 checkpoint。

可复现的局面保存在外部实验目录 `decision-probes/` 和 `holdout/teacher-audit/cycle-examples.json`，可将其中的 `position` 导入 Decision Lab：

- **边线一步获胜**：P1 a8、P2 e4，无墙，P1 行动。D1 选 b8（Q=-0.80558），漏掉 a9（Q=-0.81134）；D2 选 a9。D1 有非零输入梯度，错误发生在已激活的网络中。
- **对手立即获胜威胁**：P1 e3、P2 e2，无墙，P1 行动。D1 选 e4；D2 选横墙 d1，阻挡对手直接到 e1。这是 D2 学到了额外局部战术的具体例子。
- **四步循环**：矩阵开局 0 中，D1 的 P1 在 c4 / b4 往返，D2 的 P2 在 e6 / e7 往返。完整墙布局、循环前局面和四个动作均已保存。循环判断来自规则引擎状态重复，未将其记作胜局。

下一轮应先建立覆盖更多墙局、偏离中线和临近终点位置的共同训练局面集，让 D1/D2 在**同一批局面**上分别标注，再固定样本和更新预算比较老师。保留随机初始化、纯 CNN 推理、每 checkpoint 诊断，以及新的独立开局表。当前证据不足以认定“换更强老师即可得到更强 CNN”，也不足以归咎于 CNN 容量。

## 已登记模型与分析入口

- 最终 D1：`cnn_minimax_d1_2k_ep02000`，`quoridor-cnn-dqn` **v40**，SHA-256 `8159cdd1da861eb294463d8753b7266deaf2e6d6f59edd6bbaa6b50bb2e370d2`。
  [MLflow run](http://192.168.68.88:8742/#/experiments/5/runs/08fea8116824487bbbe985091e2fe8d8)。
- 最终 D2：`cnn_minimax_d2_2k_ep02000`，`quoridor-cnn-dqn` **v44**，SHA-256 `cad8e6da7faad6278ff96b91ac58c43d680da8653632e8ec357f7b6a61657648`。
  [MLflow run](http://192.168.68.88:8742/#/experiments/5/runs/56196e6449f742bb976fc4b8a8f7a391)。
- [MLflow 实验 5](http://192.168.68.88:8742/#/experiments/5) 保留两个训练 parent run、八个 checkpoint run 和独立评估 run。所有 checkpoint 版本为 **37–44**，包括较差结果。
- [Decision Lab](http://192.168.68.88:8744/) 已加载新 catalog，当前共有 45 个模型。上述两个模型及其 500/1,000/1,500 局 checkpoint 均可选择，模型 SHA 与注册表匹配。

结论及后续实验建议保存在 `assessment.json`；完整对局、矩阵、老师一致性审计与 HTTP 可视化诊断均作为 MLflow artifacts 留存。最终模型没有晋升为生产 champion。本次相关训练/评估回归共 29 项通过，真实 GPU 训练、全部 checkpoint 对局和 HTTP 分析校验均完成。
