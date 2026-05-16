# MCTS 阶段采样分析与改进建议

## 1. 当前 MCTS 阶段的实际行为

### 1.1 搜索结构

当 remaining > 24 时，`choose_action_with_info` 执行以下流程：

```
for det_state in det_states (K=4 个采样世界):
    for action in legal_actions (约 8-12 张合法牌):
        child_state = 模拟出 action
        child_node = 建立搜索子树根
        for _ in range(simulations_per_action):   # 默认 5
            _run_simulation(child_node)
        记录 child_node.average_value

每根动作的 value = 各世界 average_value 之和 / K
```

根动作是**外层 for 循环遍历的**，每个根动作固定分配相同数量的 simulation。

### 1.2 单次 simulation 内部做了什么

`_run_simulation` 从 child_node 开始，循环执行：

1. 检查 `remaining <= leaf_threshold (24)` → 触发 `_leaf_value` (MLP 估值)，结束
2. 检查 `_is_terminal` → 触发 `_terminal_value`，结束
3. 如果当前节点有未展开的合法动作 → **pop 第一个未展开动作**，创建新子节点，下降
4. 如果所有动作都展开过 → PUCT 选择最佳子节点，下降

**关键点**：搜索树里的每一层对应的是**当前 state.turn 的玩家的决策**——包括己方和对手。4 个玩家轮流出牌，都在同一棵搜索树的不同层里。不存在额外的"模拟策略"或"rollout 策略"——所有人的出牌完全由搜索树的展开/选择逻辑决定。

### 1.3 从 remaining=52 到 remaining=24 需要多少步

52 - 24 = 28 张牌 = 7 墩。一次 simulation 需要经过 **27-28 层**（根动作已执行一张）才能到达叶子。

每一层的分支因子（合法动作数）约：
- 首攻：8-12
- 跟牌（有花色）：2-5
- 跟牌（缺门）：8-13
- 平均约 **5-8**

### 1.4 实际搜索行为

由于每次 simulation 在每层都优先展开未访问的动作（步骤 3），且每层通常有 5-8 个合法动作，而每根动作只有 5 次 simulation：

- **每次 simulation 从根到叶子走一条 27 步的路径，每步都是"首次展开"**
- 5 次 simulation = 5 条各自独立的路径，在第 1 层就分叉
- **PUCT 选择（步骤 4）几乎不会触发**，因为没有哪一层的所有动作都被展开过
- 搜索树呈现极细长的"面条"结构，不是"扇形"结构

```
实际搜索树（5 次 simulation，每条路径 27 步）:

child_node (root action 已固定)
  ├── path1: actionA → actionB → actionC → ... (27 步) → MLP
  ├── path2: actionD → actionE → actionF → ... (27 步) → MLP
  ├── path3: actionG → actionH → actionI → ... (27 步) → MLP
  ├── path4: actionJ → actionK → actionL → ... (27 步) → MLP
  └── path5: actionM → actionN → actionO → ... (27 步) → MLP

总计 5 × 27 = 135 个节点，每个只被访问 1 次
```

---

## 2. 问题分析

### 2.1 simulation 数量严重不足（最核心问题）

5 次 simulation 面对 27 步深度、每步 5-8 的分支因子，搜索空间约 5^27 ≈ 10^18。5 条路径只覆盖了其中极小的一部分。

要让 PUCT 在某一层发挥作用，**该层所有合法动作至少各被访问 1 次**（才能进入步骤 4 的 PUCT 选择）。第 1 层约 5-8 个动作，所以至少需要 8 次 simulation 才能"铺满"第 1 层。要让前 2 层都有意义的统计量，需要约 8 × 8 = 64 次。

当前 5 次 simulation 的实质是：**5 条随机路径的 MLP 叶子估值平均**，不是搜索。

### 2.2 根动作外层遍历，预算均匀分配

标准 MCTS 做法是把根动作也交给搜索树选择（UCB/PUCT 自动分配预算），对有希望的分支投入更多。当前做法对每个根动作均匀分配 5 次 simulation，明显浪费在无希望的动作上。

### 2.3 均匀 prior 没有引导作用

```python
def _policy_priors(self, state, legal_actions):
    prob = 1.0 / len(legal_actions)
    return {action.card_id: prob for action in legal_actions}
```

均匀 prior 下，展开顺序取决于 card_id 排序，和牌面质量无关。在 simulation 极少时，先展开什么动作直接决定了搜索到什么——等于随机。

### 2.4 世界数 vs 每世界搜索深度的分配不合理

总预算 = 4 世界 × 10 根动作 × 5 simulation = 200 次 simulation。
每个世界每根动作只有 5 次。

如果改为 2 世界 × 10 根动作 × 10 simulation = 200 次（总预算不变），每个世界的搜索质量可以翻倍，而世界数从 4 减到 2 的信息损失很小。

---

## 3. 改进建议

### 3.1 大幅增加 simulations_per_action（效果最大）

从 5 提高到 50-200。每次 simulation 约 1-2ms（deepcopy + 27 步 + 1 次 MLP forward），200 次 × 10 根动作 × 4 世界 = 8000 次 simulation ≈ 8-16 秒/步。对于一局 Spades（MCTS 阶段约 28 步），总耗时约 4-8 分钟/局。

这样至少前 3-4 层能积累足够的统计量让 PUCT 发挥作用。

### 3.2 把根动作放进搜索树（结构性改进）

当前做法：
```python
for action in legal_actions:
    for _ in range(sims_per_action):
        _run_simulation(child_node)
```

改为标准 MCTS：
```python
root = SearchNode(state=state)
for _ in range(total_budget):
    _run_simulation(root)
```

让搜索自然地在有希望的根动作上投入更多预算。

### 3.3 用 policy head 或启发式提供非均匀 prior

两个选择：

**(a)** 直接用 MLP policy head（虽然在残局数据上训练，但"大牌比小牌好"这类知识是通用的）。

**(b)** 用简单启发式：黑桃给高 prior，大牌给高 prior，跟牌时考虑赢墩概率。

### 3.4 减少世界数，增加每世界搜索深度

建议 K=2 或 K=1（配合更大的 simulation 预算），让每个世界的搜索树更深更准。

### 3.5 叶子节点用精确求解器替代 MLP（可选）

当 exact_threshold == leaf_threshold == 24 时，MCTS 的叶子恰好在精确求解范围内。可以对高访问量的叶子直接调精确求解器获得完美估值。代价是速度，但可以设阈值：visits > N 时才精确求解。

---

## 4. 量化总结

| 参数 | 当前值 | 问题 | 建议值 |
|------|--------|------|--------|
| simulations_per_action | 5 | 只产生 5 条随机路径，不是搜索 | 50-200 |
| mcts_determinization_count | 4 | 分摊后每世界搜索极浅 | 1-2 |
| policy prior | 均匀 | 展开顺序和牌面质量无关 | MLP policy head 或启发式 |
| 根动作选择 | 外层遍历 | 预算均匀分配 | 放进搜索树由 PUCT 决定 |
| 叶子估值 | MLP | 可选精确求解 | MLP（默认）/ 精确求解（高访问量时） |

**核心结论**：当前 simulations_per_action=5 时，MCTS 阶段实质上不是搜索，而是 5 条随机出牌路径的 MLP 叶子估值平均。出牌质量几乎完全取决于 MLP value head 的精度，搜索本身没有贡献。
