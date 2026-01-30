# Spades Terminal PvE 对弈环境规划

目标：在终端中实现与可视化 PvE **一致的功能**，包括叫牌、出牌、合法动作提示、回合推进、结算信息，以及可选的自动对战/回放输出。

---

## 1. 设计原则

1. **同一后端逻辑**：复用现有 PvE 后端 [pve_server/spades_env.py](../pve_server/spades_env.py) 与 [pve_server/run_spades.py](../pve_server/run_spades.py) 的 API，保证行为一致。
2. **协议对齐**：终端端使用 `reset/step/state` API，按照文档约定的 JSON 结构渲染和交互。
3. **功能对齐**：实现叫牌阶段 + 出牌阶段 + 结算，支持合法动作限制、手牌显示/隐藏、对局结果展示。

---

## 2. 实现方式（推荐方案）

### 2.1 终端客户端（Python CLI）

新增一个终端客户端脚本（建议位置）：
- [pve_server/cli_spades.py](../pve_server/cli_spades.py)

主要功能模块：

- **API 客户端**：
  - `reset()`：创建新对局
  - `step(action_id)`：提交玩家动作
  - `state()`：刷新

- **渲染层（CLI View）**：
  - 手牌显示（当前玩家）
  - 出牌区显示（`current_trick`）
  - 叫牌状态显示（`bids`）
  - 赢墩显示（`tricks_won`）
  - Spades 破禁提示（`spades_broken`）
  - 轮到谁行动（`current_player`）

- **输入交互**：
  - 叫牌阶段：根据 `legal_actions` 生成选项（Pass / Blind Nil / Nil / Bid 1-13）
  - 出牌阶段：显示可出牌卡牌（从 `legal_actions` 过滤）

---

## 3. CLI 功能对齐清单（与可视化一致）

| 功能 | 可视化 | 终端 |
| --- | --- | --- |
| 叫牌阶段 | ✅ | ✅ |
| 出牌阶段 | ✅ | ✅ |
| 合法动作约束 | ✅ | ✅ |
| 手牌显示 | ✅ | ✅ |
| 对手手牌隐藏 | ✅ | ✅（用数量提示） |
| 当前墩出牌显示 | ✅ | ✅ |
| 结算分数 | ✅ | ✅ |
| Restart | ✅ | ✅ |

---

## 4. 终端交互流程

1. 启动 CLI：
   ```bash
   python3 pve_server/cli_spades.py
   ```

2. CLI 调用 `reset()` 初始化局面
3. 进入循环：
   - 如果 `phase == bidding`：
     - 打印合法叫牌选项
     - 用户输入选项序号或动作名
     - 调用 `step()`
   - 如果 `phase == play`：
     - 显示当前手牌与可行动作
     - 用户输入卡牌（如 `SA`）
     - 转换为 action_id 调用 `step()`
4. `terminal == true` 时显示结算
5. 提示是否 Restart

---

## 5. 输出格式建议（终端样式）

```
=== Spades PvE ===
Phase: bidding
Current Player: 0 (You)
Bids: [null, null, null, null]
Tricks: [0,0,0,0]
Spades Broken: No
Your Hand: (hidden in blind phase)
Legal Actions:
  [1] Pass
  [2] Blind Nil
> 输入: 1
```

出牌阶段：
```
Phase: play
Current Player: 0 (You)
Current Trick: [S3, _, _, _]
Your Hand: S2 S4 HJ D9 ...
Legal Actions: S2 S4 HJ
> 输入: HJ
```

---

## 6. 依赖与文件清单

- 新增文件：
  - [pve_server/cli_spades.py](../pve_server/cli_spades.py)

- 复用文件：
  - [pve_server/spades_env.py](../pve_server/spades_env.py)
  - [pve_server/run_spades.py](../pve_server/run_spades.py)
  - [src/utils/index.js](../src/utils/index.js)（动作映射逻辑可复用到 Python 端）

---

## 7. 可选增强

- **自动对弈模式**：非人类玩家自动出牌，CLI 仅观察
- **回放导出**：将 `states[]` 保存为 JSON，和可视化 Replay 格式一致
- **颜色高亮**：用 `colorama` 高亮牌面或当前回合
- **命令模式**：支持 `:help`, `:restart`, `:quit`

---

如需我继续实现 CLI 端，请确认是否采用上述方案（CLI 通过 API 调用 PvE 服务），我将开始编写 [pve_server/cli_spades.py](../pve_server/cli_spades.py)。
