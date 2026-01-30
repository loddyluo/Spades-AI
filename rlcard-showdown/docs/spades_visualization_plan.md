# Spades 可视化接入规划（RLCard-Showdown）

本文档给出 **Spades** 接入 RLCard-Showdown 的「具体数据格式」与「需要改动的文件清单」。目标是先实现**回放可视化（Replay）**，再扩展 **PvE**。

## 1. 统一回放数据格式（建议）

### 1.1 顶层结构

```json
{
  "game": "spades",
  "version": 1,
  "metadata": {
    "num_players": 4,
    "team_map": [0,1,0,1],
    "seed": 42,
    "timestamp": "2026-01-30T10:00:00Z",
    "source": "rlcard"
  },
  "initial_state": {
    "dealer": 0,
    "bidding_order": [0,1,2,3]
  },
  "states": [
    { "t": 0, "phase": "bidding", "current_player": 0, "obs": {...}, "legal_actions": [52,53], "action": 52 },
    { "t": 1, "phase": "bidding", "current_player": 0, "obs": {...}, "legal_actions": [54,55,56,57,58,59,60,61,62,63,64,65,66,67], "action": 58 },
    { "t": 2, "phase": "bidding", "current_player": 1, "obs": {...}, "legal_actions": [52,53], "action": 52 },
    { "t": 3, "phase": "bidding", "current_player": 1, "obs": {...}, "legal_actions": [54,55,56,57,58,59,60,61,62,63,64,65,66,67], "action": 55 },
    { "t": 4, "phase": "play", "current_player": 0, "obs": {...}, "legal_actions": [0,1,2], "action": 12 }
  ],
  "tricks": [
    { "trick_id": 0, "lead": 0, "cards": [12, 25, 38, 51], "winner": 3 }
  ],
  "result": {
    "team_scores": [120, -50],
    "player_scores": [120, -50, 120, -50],
    "bids": [5, 1, 3, 0],
    "tricks_won": [4, 2, 3, 4]
  }
}
```

### 1.2 关键字段说明

- `states[]`: 每一步决策/出牌的快照。
  - `t`: step 序号
  - `phase`: `bidding | play`
  - `current_player`: 0-3
  - `obs`: 观测（可压缩、可解码）
  - `legal_actions`: 动作 ID 列表
  - `action`: 实际采取动作（ID）
- `tricks[]`: 便于前端快速渲染每一墩的结果。
- `result`: 总结数据（得分/叫牌/赢墩）。

### 1.3 Spades 动作编码（对齐 rlcard）

- **0-51**：打牌（52 张）
- **52**：Pass（放弃盲叫）
- **53**：Blind Nil
- **54**：Nil
- **55-67**：Bid 1-13

### 1.4 观测 `obs` 的前端解码建议

为了可视化，建议后端同时输出 **结构化字段**（而非仅 200 维向量）：

```json
"obs": {
  "hand": [12, 25, 38],
  "bids": [5, 1, 3, 0],
  "tricks_won": [2, 1, 1, 0],
  "spades_broken": 1,
  "current_trick": [null, 25, null, null]
}
```

- `hand`: 当前玩家手牌（卡牌 ID）
- `current_trick`: 长度 4，表示当前墩的出牌（按玩家顺序，null 表示未出）

> 如需沿用 200 维向量，可在后端提供同时输出 `obs_raw`（向量）与 `obs`（结构化）以兼容两端。

### 1.5 PvE API 数据格式（建议）

PvE 建议沿用现有 `pve_server` 的风格，提供 `reset/step` 两个核心接口（可选 `state` 查询接口）。

#### 1) `POST /reset`

请求：

```json
{
  "game": "spades",
  "seed": 42,
  "human_player": 0,
  "ai_models": ["random", "random", "random"],
  "config": {
    "allow_blind_nil": true
  }
}
```

响应：

```json
{
  "game_id": "spades-0001",
  "current_player": 0,
  "phase": "bidding",
  "obs": {
    "hand": [],
    "bids": [null, null, null, null],
    "tricks_won": [0,0,0,0],
    "spades_broken": 0,
    "current_trick": [null, null, null, null]
  },
  "legal_actions": [52,53]
}
```

#### 2) `POST /step`

请求：

```json
{
  "game_id": "spades-0001",
  "action": 52
}
```

响应：

```json
{
  "current_player": 1,
  "phase": "bidding",
  "obs": { ... },
  "legal_actions": [52,53],
  "last_action": { "player": 0, "action": 52 },
  "trick": { "trick_id": 0, "lead": 0, "cards": [12,25,38,51], "winner": 3 },
  "terminal": false,
  "reward": 0,
  "result": null
}
```

#### 3) `GET /state`（可选）

用于断线重连或刷新 UI：

```json
{
  "game_id": "spades-0001"
}
```

响应：

```json
{
  "current_player": 0,
  "phase": "play",
  "obs": { ... },
  "legal_actions": [0, 5, 12],
  "history": { "bids": [5,1,3,0], "tricks_won": [2,1,1,0] }
}
```

> 注意：在“盲叫阶段”玩家未 `Pass` 前应隐藏手牌 (`hand: []`)。

---

## 2. 需要改动的文件清单（按模块）

### 2.1 后端（Django Leaderboard / Replay）

> 目标：能生成 Spades 对局、并以统一回放格式输出。

- [rlcard-showdown/server/tournament/rlcard_wrap](rlcard-showdown/server/tournament/rlcard_wrap)
  - 新增 `spades_*` 环境封装（参考 doudizhu / leduc 的 wrapper）。
  - 生成 `states / tricks / result` 结构。
- [rlcard-showdown/server/tournament/tournament.py](rlcard-showdown/server/tournament/tournament.py)
  - 增加 Spades 游戏分支。
- [rlcard-showdown/server/tournament/urls.py](rlcard-showdown/server/tournament/urls.py)
  - 新增 Spades replay/tournament 接口（或复用现有接口并扩展参数）。
- [rlcard-showdown/server/tournament/views.py](rlcard-showdown/server/tournament/views.py)
  - 处理 Spades 回放请求并返回 JSON。
- [rlcard-showdown/server/tournament/models.py](rlcard-showdown/server/tournament/models.py)
  - 如需要持久化回放或模型信息，增加 Spades 相关字段/表结构。

### 2.2 PvE 后端（Flask）

> 目标：人机对战（重点）

- [rlcard-showdown/pve_server](rlcard-showdown/pve_server)
  - 增加 `run_spades.py`（仿 run_dmc.py）并注册路由
  - 增加 `spades_env.py`（或等效封装），负责：
    - 初始化 rlcard `spades` 环境
    - 输出结构化观测（见 1.4）
    - 维护 `game_id -> env` 的会话映射
  - 实现 `reset/step/state` API（见 1.5）
- [rlcard-showdown/pve_server/utils](rlcard-showdown/pve_server/utils)
  - 增加 Spades 动作编码/解析工具（ID ↔ 牌面）
  - 补充合法动作过滤（例如黑桃破禁规则提示）

### 2.3 PvE 前端（React）

> 目标：可交互对战

- [rlcard-showdown/src/view/PvEView](rlcard-showdown/src/view/PvEView)
  - 新增 `PvESpadesView.js`
  - 在 `index.js` 注册路由
- [rlcard-showdown/src/components/GameBoard](rlcard-showdown/src/components/GameBoard)
  - 复用或扩展 `SpadesGameBoard.js` 支持交互点击出牌/叫牌
- [rlcard-showdown/src/utils](rlcard-showdown/src/utils)
  - 增加 PvE API client（`reset/step/state`）
  - 增加动作编码解码、牌面排序、可行动作高亮等 UI 辅助

### 2.4 前端（React）

> 目标：能在 Replay 页面显示 Spades 回放。

- [rlcard-showdown/src/view/ReplayView](rlcard-showdown/src/view/ReplayView)
  - 新增 `SpadesReplayView.js`
  - 在 `index.js` 注册路由/入口
- [rlcard-showdown/src/components/GameBoard](rlcard-showdown/src/components/GameBoard)
  - 新增 `SpadesGameBoard.js` 渲染牌桌、手牌、出牌、回合与得分。
- [rlcard-showdown/src/utils](rlcard-showdown/src/utils)
  - 增加 Spades 回放解析器（把 `states[]` 渲染为 UI 状态）。
- [rlcard-showdown/src/assets](rlcard-showdown/src/assets)
  - 复用现有牌面素材（cards.css / images），无需新增。

### 2.5 文档

- [rlcard-showdown/docs](rlcard-showdown/docs)
  - 新增 Spades 可视化使用说明（可选）。

---

## 3. 回放数据到 UI 的映射建议

| 数据字段 | UI 元素 | 说明 |
| --- | --- | --- |
| `hand` | 玩家手牌区域 | 仅显示当前玩家 / 或用于回放逐步渲染 |
| `current_trick` | 桌面出牌 | 每一墩逐步显示出牌 |
| `bids` | 叫牌显示 | 叫牌阶段显示 |
| `tricks_won` | 计分区域 | 实时更新 |
| `spades_broken` | 状态角标 | 可选显示“Spades 已破” |
| `result` | 终局面板 | 总分/队伍/叫牌等 |

---

## 4. 交付顺序建议

1. **后端回放 JSON 输出**（先跑通数据）
2. **前端静态回放渲染**（用样例 JSON）
3. **前后端打通**
4. **PvE（核心）**

---

## 5. PvE 交互要点（前端行为）

1. **回合控制**：只有 `current_player === human_player` 时允许出牌/叫牌。
2. **阶段 UI**：
  - `bidding` 阶段显示叫牌按钮（Pass / Blind Nil / Nil / Bid 1-13）。
  - `play` 阶段显示手牌与可出牌高亮。
3. **动作合法性**：依据 `legal_actions` 过滤 UI 按钮/手牌点击。
4. **局终显示**：`terminal === true` 时展示结算面板与 Replay 入口。

---

如需我继续实现这些改动，请直接确认优先级（只做 Replay 还是 Replay + PvE），我会开始改造代码。
