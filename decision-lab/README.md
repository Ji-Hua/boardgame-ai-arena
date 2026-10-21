# Quoridor Decision Lab

Author: Ji Hua / Codex
Created Date: 2026-10-20
Last Modified: 2026-10-20
Current Version: 3

Document Type: Other
Document Subtype: Developer Component Guide
Document Status: In Development
Document Authority Scope: Agent System / Decision Analysis
Document Purpose:
说明独立分析组件的使用、启动、验证和部署方式。

---

## 1. 组件定位

Decision Lab 属于 Quoridor 的 Agent System，与 Training、Evaluation/Arena
并列。它拥有自己的界面、API 和启动进程，分析逻辑位于 `agent_system/analysis/`。
这是面向开发者的诊断工具：输入 Agent、参数和局面，输出候选动作、评分、搜索记录
及可检查的原始数据。无需启动对战、训练、Arena 或 MLflow。

支持 Greedy、Minimax（深度 1–3）、DQN 的 CNN/MLP checkpoint，以及 CNN 策略蒸馏模型，均为确定性 Top1。
它直接调用实际 Agent 使用的评分与搜索函数，保留选步和同分顺序。
神经网络复用 Runtime/Arena 的 `DQNInference` 加载器与编码器。

## 2. 输入与分析流程

1. 选择 Greedy、Minimax 或「神经网络」；设置 Minimax 深度（ply），或选择 checkpoint。
   Policy 固定为 Top1；DQN 同分取最小动作编码，其余算法保持引擎顺序。
2. 导入 Position JSON、编辑棋盘或选择预设局面。算法与局面切换会自动分析；
   编辑完成后或点击「运行分析」重新计算。
3. 查看候选动作表、动作评分图和最短通路。表中序号是从 0 开始的引擎原始顺序，
   可用于核对同分选步。选择候选动作后显示评分分解和分支结果。
4. Minimax 展示根搜索返回值及其上界/下界性质、全窗口重搜值、主变化 PV、
   MAX/MIN 节点、α/β 窗口、剪枝数量和运行耗时。
5. 点击 PV 步骤或搜索树的「局面」预览状态；「返回根局面」退出预览。
   「应用动作」更新输入局面。通过「对比 Greedy / Minimax D1–3」检查同一局面下的不同配置。
6. 打开「数据 / JSON」，选择完整记录、Position、Agent Spec、候选列表、分析结果、
   当前分支、网络诊断、张量切片、对比结果或源码/引擎指纹，直接检查或保存对应 JSON。

选择神经网络后，下方「网络诊断」提供：

| 视图 | 数据与用途 |
|---|---|
| 输入编码 | CNN 的 7 个平面；MLP 的棋子、墙锚点和库存分量。显示真实输入，坐标不翻转 |
| 全部动作输出 | 全部 209 个动作、合法性、原始和屏蔽后评分；DQN 显示 Q，策略蒸馏显示 logit；合法候选可选中检查 |
| 层激活 | 各层实际输出 shape、数值统计和本局面零值占比；按层、通道读取热图 |
| 权重 | 权重分布、Conv 输出/输入通道的卷积核，或 Linear 输出单元的分页权重向量 |
| 输入敏感度 | 所选动作评分的输入梯度，或它与其余最高分合法动作之差的输入梯度；沿用该模型的 Q/logit 语义 |

梯度是当前输入附近的局部导数，不是因果贡献或注意力。棋子 one-hot、墙合法性、
库存广播平面有约束，不能把逐元素微扰当作真实合法局面；修改棋盘后重新分析可检验合法变化。
激活和权重也不直接表示某一格的战略重要性。每张热图标明自身色阶，零激活比例只针对当前局面。
空间图按 `[x,y]` 显示；MLP 中间向量、Linear 权重和卷积核使用张量索引，不能当作棋盘。
向量每次显示至多 256 项，可通过起始索引分页；完整通道/输出单元统计包含未展示项。

旧 MLP 编码 `dqn_obs_v1` 没有目标方向特征；CNN `dqn_obs_cnn_v1` 的第 6 平面包含己方目标行。
界面保留这一实际差异，不替模型补充输入。

数据面板使用打开时的快照，避免异步请求更新正在检查的数据。数据尚未产生时显示 null。
可下载当前数据对象，也可全选后复制保存；完整记录和 Position JSON 可重新导入。

棋盘 `a1` 在左下；P1 向第 9 行、P2 向第 1 行前进。编辑模式可放棋子、
增删横竖墙和切换行动方。墙工具通过交叉点放墙，点击已有墙删除。
放墙优先扣当前行动方库存，用尽时扣另一方；删除则优先返还当前方，
库存已满时返还另一方。它用于构造几何合法的输入局面，不记录墙的历史归属。
每次编辑都由服务端验证；错误编辑不会覆盖上一局面。支持撤销和 JSON 导入。

## 3. 正确理解页面数值

- **Greedy**：`对手距离变化 − 自己距离变化`，没有额外终局奖励。
- **Minimax**：叶子 `对手距离 − 自己距离`，终局用 `+∞ / −∞`。
  深度 2 指自己走一步、对手回应一步。
- **DQN**：Q 是网络的回报估计，不是胜率。展示合法 Top1/Top2 差值、未屏蔽 argmax、
  比合法 Top1 更高的非法输出数量。Q 与距离评分、不同 reward 配置的 checkpoint 不能直接横比。
  `bound=exact` 对 DQN 仅表示网络实际数值，UI 标为 Q，不表示已求得游戏精确价值。
- **CNN 策略蒸馏**：`algorithm=policy_distillation`，显示未归一化的 logit。
  它表达动作偏好，不是回报估计或胜率；动作候选、梯度与模型元数据使用这一语义。
  模型注册于 `quoridor-cnn-policy`，与历史 `quoridor-cnn-dqn` 分开。
- 通路忽略对方棋子，用于解释距离启发式；真正跳跃、走子与落墙合法性由 Rust 判定。
- 候选表是实际根搜索返回值。`≤` 表示上界，不能把它当成完整的精确排序。
  点击候选动作时，用完整窗口重新搜索，显示该分支的精确深度值。
- 「精确」仅指当前搜索深度和评价函数，不意味着已经解出整个游戏。
- 默认保留前 1,200 个访问节点用于显示。节点省略与搜索剪枝分别标注；
  记录省略不会改变结果和主变化 PV。
- 超过 75,000 次递归访问或六秒搜索预算会明确报错，不返回不完整推荐。
  服务最多同时计算两个请求。搜索深度越高，复杂局面越容易触及预算。

重新导入导出的分析文件时，只导入局面并用当前代码重新计算。
旧证据保留在原 JSON 中，不会伪装成新代码的结果。

## 4. 启动

使用仓库已有 `.venv`，需要安装本项目 Rust Python 扩展、FastAPI、Uvicorn、
Pydantic v2、PyYAML。DQN 诊断还需现有 PyTorch 推理依赖（按需导入）；无需 GPU、Node 构建或外部 CDN。
神经网络固定 CPU、单个 Torch 线程，与训练 GPU 分离。

```bash
cd /home/jihua/workspace/boardgame-ai-arena
bash decision-lab/start.sh
```

默认绑定 `127.0.0.1:8744`，UI 和 API 同源。

```bash
LAB_HOST=0.0.0.0 LAB_PORT=8744 bash decision-lab/start.sh
```

通过 `QUORIDOR_PYTHON=/absolute/path/to/python` 可指定 Python。
默认读取 `$HOME/quoridor-data/catalog.json` 中的 `models`；通过 `LAB_MODEL_CATALOG`
指定其他模型管理 catalog。只能按目录中的 model ID 选择，不接受浏览器上传权重或任意文件路径。
同目录 `registry.json` 中匹配 SHA256 的条目用于显示 MLflow 名称和版本；无需运行 MLflow 服务。
缺少目录时 Greedy/Minimax 仍可使用，DQN 面板显示配置提示。
首次加载校验 checkpoint SHA256，使用实际 checkpoint 元数据；缓存至多两个模型，文件身份变化后重新校验。
上限为单文件 512 MB、20M 参数。权重/激活请求与其他分析共享两个计算槽，同一模型诊断串行执行，
防止 hooks 和梯度互相干扰。诊断不修改参数，不累积参数梯度，不加载优化器到 GPU。

新增 checkpoint 请遵循 [模型管理流程](../model-management/README.md) 写入新的不可变 catalog，
再配置本组件；不自动扫描训练中持续变化的 checkpoint 文件。
服务用于可信开发网络，没有用户认证，不应直接作为公网服务暴露。

training 上的独立部署地址：`http://192.168.68.88:8744/`。
可以使用仓库内的 user-systemd 模板：

```bash
mkdir -p ~/.config/systemd/user
cp decision-lab/quoridor-decision-lab.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now quoridor-decision-lab
systemctl --user status quoridor-decision-lab
```

模板默认目录为 `%h/workspace/boardgame-ai-arena`，其他机器需调整。
更新 Python 代码后重启该组件：

```bash
systemctl --user restart quoridor-decision-lab
journalctl --user -u quoridor-decision-lab -n 80 --no-pager
curl -fsS http://127.0.0.1:8744/health
```

## 5. 验证

```bash
.venv/bin/python -m pytest -q agent_system/analysis/tests \
  agent_system/runtime/tests agent_system/evaluation/arena/tests \
  agent_system/definition/tests
node --test decision-lab/tests/units/*.test.js
```

测试覆盖引擎局面往返、墙几何与通路校验、实际 Arena/Runtime 的评分和选步一致性、
搜索上界的有效性、完整分支和 PV、资源上限、严格 JSON 终局表示，以及独立 HTTP 流程。
神经网络测试覆盖 CNN/MLP 双方视角与 Runtime Q/选步一致性、非法输出屏蔽、同分顺序、
梯度的数值差分验证、输入坐标、激活/权重切片、参数不变、hooks 清理和 SHA256 拒绝。

前端是原生 ES modules：`index.html`、`app.js`、`board.js`、`neural.js`、`style.css`。
`package.json` 仅为 Node 单元测试声明模块类型，无运行时 npm 依赖。
浏览器验收应包含：配置对比、候选展开、树节点/PV 预览、编辑/撤销、非法导入、
终局、数据对象切换与 JSON 导出、CNN/MLP 切换、五个网络诊断视图，以及窄屏布局。

## 6. 文档与边界

- [整体架构](../documents/system/design/system-architecture.md)
- [Agent System 架构](../documents/system/design/agent-system-design.md)
- [分析组件设计](../documents/agent/analysis/decision-analysis.md)
- [API / trace 契约](../documents/agent/analysis/decision-analysis-interface.md)
- [Greedy](../documents/agent/agent-type/greedy-agent.md)
- [Minimax](../documents/agent/agent-type/minimax-agent.md)

界面和服务不保存训练产物、不修改 checkpoint、不写 Arena 结果。
分析数据通过浏览器下载到用户指定位置；不应把导出的中间结果提交到仓库。

## Changelog

Version 3 (2026-10-20)
- 接入已有 CNN/MLP catalog，增加输入、Q、激活、权重和梯度诊断及 CPU 部署说明。

Version 2 (2026-10-20)
- 将界面收敛为开发工作台：Agent 配置、局面输入、分析结果；移除叙事引导与教程卡片。
- 增加候选动作引擎序号及按对象查看、复制和导出原始 JSON 的入口。

Version 1 (2026-10-20)
- 增加独立的 Greedy/Minimax 决策分析组件与启动、验证、部署说明。


## PUCT / Value 分析

带有已校准 Value Head 的 CNN 可以在神经网络页使用 **PUCT / Value**。输入当前局面、模型、simulations（1–200）和 c_puct，即可查看 prior、访问次数、访问比例、父节点执手视角 Q 和根 Value。使用与 Arena 相同的确定性搜索，禁用根噪声；当前主棋盘仍显示 actor-only 决策。网络层视图包含 Value Head 的权重与激活。完整搜索结果可导出 `neural_search`。

`POST /api/neural/search` 接受 `position`、`model_id`、`simulations`、`c_puct`。没有经过结果校准的 Value Head 会返回 422。终局无动作，显示实际胜负值。参见 [AlphaZero-lite 实验协议](../documents/agent/training/alphazero-lite.md)。
