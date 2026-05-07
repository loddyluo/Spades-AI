# 黑桃王双明手求解器使用说明

## 求解器位置

项目包含**两个**双明手求解器：

| 求解器 | 文件 | 算法 | 特点 |
|--------|------|------|------|
| MCTS 求解器 | `trick_taking/solvers/double_dummy.py` | 蒙特卡洛树搜索 | 近似解，速度快，可控制搜索量 |
| 精确求解器 | `trick_taking/solvers/exact_double_dummy.py` | 极小极大 + Alpha-Beta 剪枝 | 精确解，牌少时可用 |

辅助工具函数在 `trick_taking/utils/state_tools.py` 中。

---

## 用法

### 1. 创建游戏状态

双明手求解器需要一个叫牌已结束、处于出牌阶段的 `GameState`。

**方式一：使用工具函数创建随机状态**

```python
from trick_taking.utils.state_tools import create_random_state
state = create_random_state()
```

**方式二：手动构建状态**

```python
from trick_taking.card import Card, Suit
from trick_taking.game_state import GameState, Bid

# 定义4名玩家的手牌（每人13张）
hands = [
    [Card(Suit.SPADES, "A"), Card(Suit.HEARTS, "K"), ...],  # 玩家0
    [Card(Suit.SPADES, "K"), ...],                           # 玩家1
    ...,                                                      # 玩家2
    ...,                                                      # 玩家3
]

# 创建状态
state = GameState()
state.init_for_deal(4, hands, [], all_cards)

# 设置叫牌
state.bids = [
    Bid(player_id=0, value="bid_3"),
    Bid(player_id=1, value="bid_2"),
    Bid(player_id=2, value="nil"),
    Bid(player_id=3, value="bid_3"),
]
state.max_bid = ["bid_3", "bid_2", "nil", "bid_3"]

# 设置队伍 (0&2 vs 1&3)
state.teams = [0, 1, 0, 1]

# 进入出牌阶段
state.phase = state.phase.PLAYING
state.turn = 0
state.trick_leader = 0
```

**方式三：使用工具函数从指定手牌创建**

```python
from trick_taking.utils.state_tools import create_state_from_hands
hands = [[...], [...], [...], [...]]  # 4个手牌列表，各13张
bids = ["bid_3", "bid_2", "nil", "bid_3"]
state = create_state_from_hands(hands, bids)
```

---

### 2. 使用 MCTS 求解器（近似解）

```python
from trick_taking.solvers.double_dummy import DoubleDummySolver

# 创建求解器
solver = DoubleDummySolver(max_iterations=1000)

# 求解当前局面
result = solver.solve(state, current_player=state.turn)

# 获取结果
best_action = result["best_action"]           # 最优出牌（Card对象）
action_values = result["action_values"]        # 所有动作的评估列表
state_eval = result["state_evaluation"]        # 局面总体评估
search_stats = result["search_statistics"]     # 搜索统计信息

print(f"最优出牌: {best_action}")
print(f"预期得分差: {state_eval['expected_score_diff']}")
```

`solve()` 返回的字典字段：

| 字段 | 说明 |
|------|------|
| `best_action` | 最优出牌（`Card` 对象） |
| `action_values` | 所有合法动作的列表，每项含 `action`/`value`/`visits`/`confidence` |
| `state_evaluation.expected_score_diff` | 预期得分差（己方-对方） |
| `state_evaluation.team_win_probability` | 团队胜率估计 |
| `search_statistics.iterations` | 搜索迭代次数 |
| `search_statistics.time_elapsed` | 搜索耗时（秒） |

---

### 3. 使用精确求解器（精确解，仅剩少量牌时可用）

```python
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver

solver = ExactDoubleDummySolver()
score_diff = solver.solve(state)
print(f"最优得分差（队伍0 - 队伍1）: {score_diff}")
```

返回一个 `float`，即双方最优玩法下的最终得分差（队伍0 - 队伍1）。

---

## 4. GameState 关键字段速查

手动构造状态时需设置的字段（按初始化顺序）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `phase` | `Phase.PLAYING` | 设为出牌阶段。 |
| `teams` | `list[int]` | 长度4，如 `[0,1,0,1]` 表示玩家 0&2 同队、1&3 同队。 |
| `bids` | `list[Bid]` | 叫牌记录列表，每一项用 `Bid(player_id=i, value="bid_2")` 构造。 |
| `max_bid` | `list[Any]` | 长度4，每名玩家的最终叫牌，如 `["bid_3","nil","bid_2","bid_1"]`。 |
| `table_cards` | `list[tuple[int, Card]]` | 当前墩已出的牌，每项是 `(玩家ID, Card对象)`。 |
| `turn` | `int` | 当前需要出牌的玩家 ID（0-3）。 |
| `trick_leader` | `int` | 当前墩的首攻玩家 ID。 |
| `tricks_played` | `int` | 已完成的墩数。 |
| `tricks_won` | `list[int]` | 长度4，每名玩家已赢得的墩数，总和应等于 `tricks_played`。 |
| `spades_broken` | `bool` | 黑桃是否已被打出（破禁）。 |
| `played_bitset` | `int` | 已打出所有牌的位图，可通过 `state.played_bitset \|= card.bit` 逐张标记。 |

参考 `test_15_mcts.py` 或 `test_both_solvers_17.py` 的 `create_state()` 函数获得完整示例。

## 5. 运行示例

```bash
# 运行 MCTS 求解器测试
python run_solver_test.py

# 运行精确求解器测试（最后一墩，4张牌）
python test_exact_simple.py

# 运行 MCTS 测试（最后一墩，4张牌）
python test_simple_double_dummy.py
```
