# Quoridor 环境、部署与系统现状

Author: Codex（为 Ji Hua 整理）
Created Date: 2026-10-20
Last Modified: 2026-10-20
Current Version: 2
Document Type: Other
Document Subtype: Environment Inventory and Operations
Document Status: In Development
Document Authority Scope: dev / training 环境现状，不改变系统设计
Document Purpose: 记录已核查的运行环境、部署入口、验证结果、恢复方式及后续工作依赖。

## 1. 当前入口与代码

| 用途 | 地址 | 服务器目录 |
|---|---|---|
| 在线对战 | http://192.168.68.102:8765 | dev: `/home/jihua/workspace/quoridor` |
| 对战后端 | http://192.168.68.102:8764/health | 同上 |
| Agent Service | http://192.168.68.102:8090/health | 同上 |
| 训练仪表盘 | http://192.168.68.88:8741 | training: `/home/jihua/workspace/boardgame-ai-arena` |
| 仪表盘 API | http://192.168.68.88:8740/health | 同上 |

这些地址需要能够访问所在局域网。

- dev 的基准提交是 `c27e1d8`，分支 `feature/dqn`，有原先存在的 `pyproject.toml` 修改。
- training 的基准提交是 `63ef2bb`，包含更新的训练代码及仪表盘；本次部署位于新建分支 `codex/environment-deployment`，部署修改已提交为 `04bd419`（未推送）。
- 两台机器都连接到 `git@github.com:Ji-Hua/boardgame-ai-arena.git`。版本尚未统一；这次保留 dev 对战服务，部署 training 仪表盘。
- training 中原有训练脚本修改、配置删除、未跟踪实验目录和历史产物均保留，没有自动续训。
- 本地 `/Users/jihua/Documents/ChatGPT/quoridor` 是环境记录工作区，不是完整源码镜像。
- `quoridor_v0`、`quoridor_legacy` 等目录属于历史实现，不能与当前工作副本混用。

## 2. 已完成的部署和验证

训练仪表盘以 Docker Compose 常驻，容器为 `training-dashboard-training-dashboard-1`，使用 `unless-stopped` 重启策略。前端在镜像构建时生成静态产物，通过 Vite preview 提供服务；修复了 preview 模式下 `/api`、`/health` 到后端的代理。增加双端健康检查、init 和 15 秒停止宽限期，并修正入口脚本的子进程退出处理。

训练产物只读挂载：

```text
/home/jihua/workspace/boardgame-ai-arena/agent_system/training/artifacts
  -> /data/training_artifacts  (RW=false)
```

已验证：

- 仪表盘容器健康；前端页面、前端代理 API、实验列表和单局回放 JSON 均可读取。
- 仪表盘发现 50 个实验目录。它的汇总数量按现有发现规则生成，不代表 50 次训练均完成。
- 训练单元测试及 checkpoint、奖励语义集成测试：**534 passed**。重启前有一项 CUDA 初始化警告。
- 实际 CNN checkpoint 在 CPU 上成功加载，输出有限值，合法动作掩码选择动作成功，Rust 引擎接受该动作并切换玩家。
- dev 后端与 Agent Service 健康检查通过；通过 REST 创建临时房间，Greedy Agent 正常走出第一步、引擎切换到 P2，测试房间已结束并关闭。
- 重启后 CUDA 检测、实际 CNN 的 GPU 推理、CPU/GPU 输出一致性及有限梯度反向计算均通过。验证未改变 checkpoint 文件。
- 仪表盘、dcgm-exporter、node-exporter 自动恢复；Docker、Netdata、OpenClaw Gateway 均处于 active/running 状态。

用于验证的 checkpoint：

```text
agent_system/training/artifacts/dqn/cnn_curriculum_observation_2k_001/
  checkpoints/ep02000_step232954.pt
SHA256: d823125353ffb5e902ddbfba8d6086b88c87354375839682783fd38c86e0abcf
```

元数据：Double DQN、1,438,545 参数、输入 `[7,9,9]`、209 个动作。episode=2000；文件名记录环境步数 232954，checkpoint 内 `training_step=111135` 是优化器步数，两者不能混用。初始局面 131 个合法动作，CPU 推理选择 action 37。

## 3. training 依赖和 GPU

| 项目 | 重启前核查值 |
|---|---|
| Python | `.venv`，3.12.12 |
| PyTorch | 2.11.0+cu126，CUDA runtime 12.6 |
| NumPy | 2.5.4，本次补装 |
| FastAPI / Uvicorn | 0.136.1 / 0.46.0（宿主训练环境） |
| Pydantic / PyYAML / pytest | 2.13.3 / 6.0.3 / 9.0.3 |
| Rust Python 扩展 | `quoridor_engine`，本仓库 engine 的 editable 安装，导入及状态推进正常 |
| GPU | RTX 3080 Ti，12 GB |
| 运行中内核 | 5.15.0-177-generic |
| 已加载 NVIDIA 驱动 | 580.126.09 |
| 磁盘驱动和用户态库 | 580.178.04 |
| 重启前 CUDA | 不可用；NVML 版本不匹配，PyTorch Error 804 |

重启前 GPU 更新尚未生效，系统存在 reboot-required。用户在部署和记录完成后执行了重启，2026-10-20 11:34 EDT 完成验证：

- 内核：`5.15.0-198-generic`。
- NVIDIA 已加载驱动、磁盘模块和用户态库一致：`580.178.04`。
- `nvidia-smi` 正常识别 RTX 3080 Ti / 12288 MiB。
- `torch.cuda.is_available()` 为 `True`。
- 实际 CNN 的 batch=8 GPU 前向和反向计算通过；与 CPU 的推理输出在设定容差内一致。
- 新 boot ID：`c4561983-f324-4d9e-be48-164851f1222d`，确认服务器确实完成重启。

**训练环境现已通过 CPU 与 GPU 基础验证。未启动新的长时间训练。**

根目录 `pyproject.toml` / `uv.lock` 尚未完整管理训练所需的 torch、numpy。此次保留现有可工作的 `.venv`，没有用 `uv sync` 重建或清理它。依赖 freeze 是安装状态快照，其中包含本地 editable 路径；它不是可直接跨机器安装的完整锁文件。

仪表盘使用独立容器依赖，不使用宿主机的 torch/CUDA。镜像 ID：

```text
sha256:8a34ebf12555e61d8b04093b84776314a1b23fcc8e43bcc5a1320e35d09cc960
```

## 4. 运维命令与记录位置

在 training 上：

```bash
cd /home/jihua/workspace/boardgame-ai-arena
docker compose -f training-dashboard/docker-compose.yml ps
docker compose -f training-dashboard/docker-compose.yml logs --tail=80
docker compose -f training-dashboard/docker-compose.yml up -d --build
curl -fsS http://127.0.0.1:8741/health
nvidia-smi
```

本次备份与证据位于 `/home/jihua/quoridor-env-backups/20261020/`：

- `preexisting.patch`、`preexisting-status.txt`：工作前的已跟踪修改及状态。
- `deployment-files.tar.gz`：工作前的仪表盘部署文件、pyproject、uv.lock。
- `dashboard-deployment.patch`：本次仪表盘修改。
- `dashboard-compose.resolved.yml`：展开后的 Compose 配置。
- `training-environment.freeze.txt`、`dashboard-environment.freeze.txt`：两套环境依赖版本。
- `verification-before-reboot.json`：真实 checkpoint 推理、服务健康和 GPU 状态。
- `verification-after-reboot.json`：重启后驱动、CUDA、真实 CNN 的 GPU 前向与反向计算及服务检查。
- `dev-live-smoke.json`：dev 在线 Greedy 走子验证。
- `services-before-reboot.txt`：重启前 Docker 服务清单。
- `ops/verify_training.py`：可复用的 CPU/GPU 验证脚本。

这不是完整磁盘或实验产物备份；未跟踪产物保留在原目录中。恢复部署时，只从 tar 中取出对应部署文件；不要整仓 reset，也不要覆盖原先的训练修改。

本地 `ops/verify_training.py` 可通过 SSH 在 training 仓库根目录用 `.venv/bin/python -` 运行。加 `--cuda` 时，检查实际 checkpoint 的 CPU/GPU 输出一致性以及 GPU 反向计算；所有梯度只在内存中产生，不保存或改变 checkpoint。

## 5. 对系统现状的理解

设计依据仍是 `documents/system/design/quoridor-RL-vision.md`。目标生命周期为 Define → Materialize → Train → Evaluate → Decide → Deploy → Run。

当前有三条已实现但尚未完全贯通的路径：

1. 在线对战：React → FastAPI REST/WebSocket → GameManager → Rust 规则引擎；Agent Service 提供 Greedy、Minimax、随机及回放等实例。
2. 训练评测：Rust 引擎 → learner 视角的数据与奖励 → CNN/MLP Double DQN → checkpoint → 固定对手评测。
3. 观察分析：历史配置、metrics、评测和回放文件 → 只读训练仪表盘。

dev 当前可选 Agent 列表中没有训练后的 CNN。仪表盘能展示模型产物，不等于模型已进入在线对战。晋升门槛、版本绑定和运行时适配尚需打通。

训练代码已有 Double DQN、CNN、距离奖励、对手课程及并行实验，并不是仅有早期 MLP。现有 CNN 能明显胜过部分随机对手，但对固定 Minimax 的结果和长训稳定性仍不足。`cnn_curriculum_observation_2k_001` 原始 ep2000 评测对 random_legal 为 149/200，对 minimax d1/d2 为 0/200；5k_002 的后期 Q 值和 loss 显著发散。旧目录还有复用后追加不同实验记录的情况，不能把 JSONL 最后一条直接当成最初那次训练的最终结果。

仪表盘目前还有口径限制：列表上的 Episodes 可能来自目标配置，Best WR 可能跨不同对手取最大值，DQN 标签未细分 Double DQN，部分空目录也带数据徽标。应先回到原始记录，再得出棋力结论。

项目文档中的 AlphaZero 目标要求真实自我对弈、策略/价值网络、MCTS、候选评测和部署闭环。历史目录里名为 Full/AlphaZero 的运行不能仅靠名称与日志计数认定已经实现这些能力；此前核查过的旧 full_train 路径使用随机张量及自身网络标签，未形成真实 MCTS 自我对弈闭环。

## 6. 下一步工作顺序

1. 固定不可变的 run ID / checkpoint hash / 配置 / 代码版本，清理“计划数、完成数、不同对手胜率”的展示口径；整理多 seed、交换先后手的基准评测。
2. 接通一个经过验证的 CNN checkpoint 到在线 Agent Service，让训练结果可以直接对战和回放。
3. 核查长训发散与数据语义，再建立 policy/value + MCTS + 自我对弈的小规模闭环。
4. 有稳定基准和真实自我对弈后，再评估模型容量、并行采样与搜索预算。

这里的优先级是基于当前代码和实验产物的工程判断。模型容量本身尚未被证明是主要瓶颈。

## Changelog

Version 2 (2026-10-20)
- 完成用户执行重启后的 GPU、实际模型计算、服务自动恢复验证；补充在线走子证据和部署提交。

Version 1 (2026-10-20)
- 记录双服务器环境、仪表盘部署、CPU 验证、待完成的 GPU 验证与系统衔接方向。
