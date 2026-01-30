# Spades 游戏环境使用文档

本文档详细介绍了如何在 RLCard 中使用新实现的 **Spades (黑桃王)** 环境。

## 1. 快速开始

您可以使用 `rlcard.make` 接口直接创建 Spades 环境。

```python
import rlcard
from rlcard.agents import RandomAgent

# 创建环境
env = rlcard.make('spades', config={'seed': 42})

# 打印环境信息
print(f"玩家人数: {env.num_players}")
print(f"动作空间大小: {env.num_actions}")
print(f"状态空间形状: {env.state_shape}")

# 设置随机智能体
agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]
env.set_agents(agents)

# 运行一局游戏
trajectories, payoffs = env.run(is_training=False)

# 打印结果
# payoff 格式为: [队0得分, 队1得分, 队0得分, 队1得分]
# (玩家0和2是队友，玩家1和3是队友)
print(f"游戏结束，得分: {payoffs}")
```

## 2. 游戏机制说明

### 2.1 阶段 (Phases)
游戏分为两个主要阶段，通过环境内部自动流转：

1.  **叫牌阶段 (Bidding Phase)**：
    *   **盲叫 (Blind)**：玩家在未看牌时，可以选择 `Blind Nil` (盲0) 或 `Pass` (看牌)。
    *   **正常叫牌**：若选择了 `Pass`，玩家将看到手牌，并需选择 `Nil` (明0) 或 `Bid 1-13`。
2.  **出牌阶段 (Play Phase)**：
    *   标准的吃墩游戏。
    *   **黑桃破禁规则**：若首攻不是黑桃且场上未出过黑桃，玩家不可首攻黑桃（除非手牌全为黑桃）。

### 2.2 动作空间 (Action Space)
动作空间共有 **68** 维，ID 映射如下：

| 动作ID | 描述 | 备注 |
| :--- | :--- | :--- |
| **0 - 51** | **打牌 (Play Card)** | 对应 52 张单牌 (S, H, D, C) |
| **52** | **Pass** | 放弃盲叫，进入看牌阶段 |
| **53** | **Blind Nil** | 叫盲 0 |
| **54** | **Nil** | 叫明 0 |
| **55 - 67** | **Bid 1 - 13** | 叫 1 到 13 墩 |

### 2.3 状态表示 (State Representation)
状态是一个字典，其中 `obs` 是一个 200 维的向量，包含以下信息：

1.  **手牌 (Current Hand)**：52 维 One-hot 编码。
    *   *注意*：在盲叫阶段，若玩家尚未选择 Pass，手牌信息将被全 0 屏蔽。
2.  **叫牌信息 (Bids)**：4 维，记录每个玩家的叫牌数（0-13）。
3.  **赢墩信息 (Tricks)**：4 维，记录每个玩家当前赢得的墩数。
4.  **黑桃破禁状态**：1 维，1 表示黑桃已被打破，0 表示未破。
5.  **当前回合出牌**：52 维 One-hot，记录本轮 Trick 中已打出的牌。

## 3. 计分规则详解

环境严格按照以下公式计算 Reward：

*   **Nil / Blind Nil**：独立结算。
    *   Blind Nil 成功 +100，失败 -100。
    *   Nil 成功 +50，失败 -50。
    *   *Covering 规则*：即使 Nil 失败，该玩家赢得的墩数也会计入队伍总墩数，帮助队友完成合约。
*   **普通合约 (Team Contract)**：
    *   设队伍叫牌总和为 $B$，赢得总墩数为 $T$。
    *   若 $T < B$：得分 $-10 \times B$ (倒罚)。
    *   若 $T \ge B$：得分 $(10 \times B) - (9 \times (T - B))$。即每多赢一墩 (Overtrick)，扣除 9 分。

## 4. 自定义开发

如果您需要修改规则（例如取消 -9 分对应的惩罚，改为 +1 分），请修改 `rlcard/games/spades/judger.py` 中的 `calculate_team_score` 方法。

如果您需要修改动作空间或状态特征，请参考 `rlcard/envs/spades.py`。
