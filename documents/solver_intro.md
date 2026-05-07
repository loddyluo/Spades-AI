# 双明手求解器使用入门

## 环境与队伍约定

- 4名玩家，队伍分配：玩家0和2为**队伍0（己方）**，玩家1和3为**队伍1（对方）**
- 优化目标：最大化 **(队伍0得分 - 队伍1得分)**
- 使用前需确保：**叫牌已结束，处于出牌阶段**

---

## 1. 精确求解器（Alpha-Beta 剪枝搜索）

求解双方最优玩法下的**精确得分差**，速度慢但结果准确。适用于剩余牌数较少的局面（建议 ≤6张）。

### 核心 API

| 类 / 函数 | 文件 |
|---|------|
| `ExactDoubleDummySolver.solve(state)` | `trick_taking/solvers/exact_double_dummy.py` |

### 输入输出

- **输入**: `GameState`（必须包含所有玩家完整手牌信息，队伍为 `[0,1,0,1]`，处于 PLAYING 阶段）
- **输出**: `float` — 双方最优玩法下 **队伍0得分 - 队伍1得分**

### 代码示例

```python
from trick_taking.card import Card, Suit, Rank
from trick_taking.game_state import GameState, Bid
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver

# ---- 创建状态 ---- #
# 定义4名玩家的手牌（每人1张，已打完12墩）
hands = [
    [Card(Suit.SPADES, Rank.ACE)],   # 玩家0: ♠A
    [Card(Suit.SPADES, Rank.KING)],  # 玩家1: ♠K
    [Card(Suit.SPADES, Rank.QUEEN)], # 玩家2: ♠Q
    [Card(Suit.SPADES, Rank.JACK)],  # 玩家3: ♠J
]

state = GameState()
deck_cards = [Card(s, r) for s in Suit for r in Rank]  # 52张标准牌
state.init_for_deal(4, hands, [], deck_cards)

state.bids = [Bid(0, 1), Bid(1, 4), Bid(2, 'nil'), Bid(3, 3)]
state.max_bid = [1, 4, 'nil', 3]
state.teams = [0, 1, 0, 1]
state.phase = state.phase.PLAYING
state.turn = 0
state.trick_leader = 0
state.spades_broken = True
state.trump_broken = True
state.tricks_played = 12

# ---- 求解 ---- #
solver = ExactDoubleDummySolver()
score_diff = solver.solve(state)  # 返回 float
print(f"精确得分差: {score_diff}")  # 输出: 130.0
```

---

## 2. MCTS 求解器（蒙特卡洛树搜索）

近似求解，速度可控（通过迭代次数调节）。适用于**任意牌数**的局面，结果是估计值而非精确值。

### 核心 API

| 类 / 函数 | 文件 |
|---|------|
| `DoubleDummySolver.solve(state, current_player)` | `trick_taking/solvers/double_dummy.py` |

### 输入输出

- **输入**:
  - `state`: `GameState`（完整信息）
  - `current_player`: `int` — 当前需要出牌的玩家 ID
- **输出**: `Dict[str, Any]`，包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `best_action` | `Card` | 最优出牌 |
| `action_values` | `List[Dict]` | 每个合法动作的评估（价值、访问次数、置信度） |
| `state_evaluation` | `Dict` | 局面评估（预期得分差、团队胜率、确定性） |
| `search_statistics` | `Dict` | 搜索统计（迭代次数、耗时、扩展节点数） |

### 代码示例

```python
from trick_taking.card import Card, Suit, Rank
from trick_taking.game_state import GameState, Bid
from trick_taking.solvers.double_dummy import DoubleDummySolver

# ---- 创建状态（同上） ---- #
# （示例略，与精确求解器相同）

# ---- 创建求解器 ---- #
solver = DoubleDummySolver(
    max_iterations=5000,       # 迭代次数，越大结果越精确
    exploration_weight=1.4,    # UCT 探索权重
    rollout_epsilon=0.0        # rollout 随机探索概率
)

# ---- 求解 ---- #
result = solver.solve(state, current_player=state.turn)

# ---- 读取结果 ---- #
print(f"最优出牌: {result['best_action']}")  # Card 对象

# 动作列表（按价值降序）
for info in result['action_values'][:3]:
    print(f"  {info['action']}: 价值={info['value']:.2f}, "
          f"访问={info['visits']}, 置信度={info['confidence']:.2f}")

# 局面评估
eval_info = result['state_evaluation']
print(f"预期得分差: {eval_info['expected_score_diff']:.2f}")
print(f"团队胜率: {eval_info['team_win_probability']:.2f}")

# 搜索统计
stats = result['search_statistics']
print(f"耗时: {stats['time_elapsed']:.2f}秒")
print(f"扩展节点: {stats['nodes_expanded']}")
```

### 常用参数说明

```python
# 快速评估（~100次迭代，用于快速筛选）
solver = DoubleDummySolver(max_iterations=100)

# 标准求解（~5000次迭代，平衡速度与精度）
solver = DoubleDummySolver(max_iterations=5000)

# 高精度求解（~50000次迭代，用于关键决策）
solver = DoubleDummySolver(max_iterations=50000, exploration_weight=1.4)
```

---

## 3. 辅助工具函数

位于 `trick_taking/utils/state_tools.py`：

```python
from trick_taking.utils.state_tools import (
    create_random_state,
    create_state_from_hands,
    analyze_result,
    compare_actions,
    save_state_to_file,
    load_state_from_file,
)

# 创建随机局面（叫牌已完成，出牌开始）
state = create_random_state()

# 从指定手牌创建局面（每人必须13张）
hands = [
    [Card(Suit.SPADES, Rank.ACE), ...],  # 玩家0的13张
    [Card(Suit.HEARTS, Rank.KING), ...],  # 玩家1的13张
    [Card(Suit.CLUBS, Rank.QUEEN), ...],  # 玩家2的13张
    [Card(Suit.DIAMONDS, Rank.JACK), ...], # 玩家3的13张
]
state = create_state_from_hands(hands, ["bid_3", "bid_2", "nil", "bid_1"])

# 生成易读结果报告
print(analyze_result(result))

# 快速比较所有合法动作
action_comparison = compare_actions(state, current_player=0, solver_iterations=100)

# 状态序列化
save_state_to_file(state, "state.json")
loaded_state = load_state_from_file("state.json")
```

---

## 4. 完整测试示例参考

仓库中的测试文件可直接运行：

| 文件 | 说明 |
|------|------|
| `test_exact_simple.py` | 精确求解器最后一墩4张牌的测试 |
| `test_10_mcts.py` | MCTS求解器最后三墩12张牌的测试 |
| `test_15_mcts.py` | MCTS求解器剩余35张牌的测试（含当前墩已出一张牌） |
| `test_double_dummy.py` | 综合单元测试和性能测试 |
