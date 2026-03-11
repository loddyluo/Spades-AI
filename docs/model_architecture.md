# Spades AI 模型架构与训练机制

> 配置文件路径: `model_config.yaml`
>
> 训练入口脚本: `rlcard/train_spades_selfplay_drqn.py`
>
> 训练器实现: `rlcard/rlcard/agents/drqn_trainer.py`
>
> 网络与 Agent: `rlcard/rlcard/agents/drqn_agent.py`
>
> 环境: `rlcard/rlcard/envs/spades.py`

## 一、总体架构

系统采用**多 Actor + 单 Learner 的分布式 DRQN (Deep Recurrent Q-Network)** 架构：

```text
Actor 0 ─┐                          ┌─ shared_qnet (CPU, 共享内存)
Actor 1 ─┤→ episode_queue (数据) →  Learner (GPU)
  ...    ─┤                          │  训练后同步权重 →  shared_qnet
Actor N ─┘                          └─ opponent_qnet (CPU, 共享内存)
```

AI 在牌局中**只使用一套参数**：`DRQNNetwork` 的神经网络权重。叫牌和出牌共用同一个网络，通过 200 维 state vector 中的 `phase` 标志（维度 185）区分当前是叫牌阶段还是出牌阶段。

**角色分工**：

| 组件             | 设备          | 职责                                           |
| ---------------- | ------------- | ---------------------------------------------- |
| Actor 进程 (xN)  | CPU           | 使用 `shared_qnet` 自我对弈收集数据            |
| Learner 线程     | GPU           | 从 replay buffer 采样训练，更新 `learner_qnet` |
| shared_qnet      | CPU 共享内存  | Learner 定期同步过来，Actor 直接读取推理        |
| opponent_qnet    | CPU 共享内存  | 对手使用的网络，从历史快照池中随机选取          |
| target_qnet      | GPU           | Double DQN 的目标网络，定期从 learner 整体复制  |

---

## 二、每个操作用到的参数

### 2.1 网络结构 (`DRQNNetwork`)

```text
输入: (batch, seq_len, 200)
  |
BatchNorm1d(200)             — 参数: gamma(200), beta(200)   =       400 参数
  |
Linear(200 -> embed_dim) + ReLU                              =  51,456 参数
  |
LSTM(embed_dim -> lstm_hidden_size, lstm_num_layers层)        = 1,050,624 参数
  | 取最后一个时间步
Linear(lstm_hidden_size -> mlp_layers[0]) + ReLU             =  65,792 参数
  |
Linear(mlp_layers[-1] -> 68)                                 =  17,476 参数
  |
输出: 68 个 Q 值
```

网络结构由以下参数控制（均来自 `model_config.yaml` -> `drqn` 节）：

| yaml key           | 当前值  | 代码默认值 | 作用                 |
| ------------------ | ------- | ---------- | -------------------- |
| `embed_dim`        | 256     | 256        | 嵌入层输出维度       |
| `lstm_hidden_size` | 256     | 256        | LSTM 隐藏层维度      |
| `lstm_num_layers`  | **2**   | 1          | LSTM 层数            |
| `mlp_layers`       | [256]   | [256]      | MLP Head 隐藏层列表  |

> **注意**: yaml 中 `lstm_num_layers: 2`，覆盖了代码默认值 1。

68 个动作的映射：

| 动作 ID | 含义                                                  |
| ------- | ----------------------------------------------------- |
| 0-51    | 出牌（52 张牌，按 S2-SA, H2-HA, D2-DA, C2-CA 排列）  |
| 52      | pass（blind nil 阶段跳过）                            |
| 53      | blind_nil                                             |
| 54      | nil                                                   |
| 55-67   | bid_1 ~ bid_13                                        |

### 2.2 每次决策的完整流程

当轮到 AI 行动时（无论叫牌还是出牌），执行以下步骤：

#### Step 1 — 环境构造 200 维状态向量

`SpadesEnv._extract_state()` 将游戏状态编码为 200 维 int8 向量（无可学习参数）：

| 维度    | 长度 | 内容                                 |
| ------- | ---- | ------------------------------------ |
| 0-51    | 52   | 手牌 one-hot（blind nil 决策时全 0） |
| 52-55   | 4    | 四家叫品（-1 = 未叫，0-13）         |
| 56-59   | 4    | 四家已赢墩数                         |
| 60      | 1    | 黑桃是否已破                         |
| 61-112  | 52   | 当前墩的牌 one-hot                   |
| 113-164 | 52   | 历史已出牌 one-hot                   |
| 165-168 | 4    | 四家 is_nil 标志                     |
| 169-172 | 4    | 四家 is_blind_nil 标志               |
| 173-176 | 4    | 当前玩家 one-hot                     |
| 177-180 | 4    | 当前墩出牌者标志                     |
| 181-184 | 4    | 四家手牌数量                         |
| 185     | 1    | 阶段（0=叫牌, 1=出牌）              |
| 186     | 1    | 已完成墩数                           |
| 187-190 | 4    | 领出花色 one-hot (S/H/D/C)          |
| 191     | 1    | 是否自己领出                         |
| 192     | 1    | 己方队伍叫品总和                     |
| 193     | 1    | 己方队伍赢墩总和                     |
| 194     | 1    | 对方队伍叫品总和                     |
| 195     | 1    | 对方队伍赢墩总和                     |
| 196     | 1    | 己方还差多少墩完成叫品（可负）       |
| 197     | 1    | 手中黑桃数                           |
| 198     | 1    | 已出黑桃总数                         |
| 199     | 1    | 当前墩中的出牌位置 (0-3)             |

#### Step 2 — 网络前向传播

```python
# DRQNActorAgent._predict()
obs = state['obs']                          # (200,)
obs = obs.unsqueeze(0).unsqueeze(0)         # (1, 1, 200) — batch=1, seq_len=1
hidden = self.hidden_states[player_id]      # LSTM 隐藏状态 (h, c)，每人独立维护
q_values, new_hidden = self.qnet(obs, hidden)  # 前向传播
self.hidden_states[player_id] = new_hidden  # 保存新隐藏状态
```

用到的**全部可学习参数**：

| 参数                       | 所属层            | 说明                                   |
| -------------------------- | ----------------- | -------------------------------------- |
| gamma, beta                | BatchNorm1d       | 缩放和偏移；推理时还使用 running stats |
| W_embed, b_embed           | Linear(200->256)  | 特征嵌入                               |
| W_ih, W_hh, b_ih, b_hh    | LSTM 第 1 层      | 输入门/遗忘门/单元门/输出门各一组      |
| W_ih, W_hh, b_ih, b_hh    | LSTM 第 2 层      | 同上                                   |
| W_mlp, b_mlp               | Linear(256->256)  | MLP Head 第一层                        |
| W_out, b_out               | Linear(256->68)   | 输出层，产生 68 个 Q 值                |

用到的**非参数状态**：

| 状态                      | 说明                                                            |
| ------------------------- | --------------------------------------------------------------- |
| LSTM hidden state (h, c)  | 每个玩家独立维护，跨时间步传递，每局开始时重置为 None（零初始化）|
| BatchNorm running stats   | running_mean 和 running_var，推理时使用                         |

#### Step 3 — 动作选择（epsilon-贪心）

```python
# 训练时 (DRQNActorAgent.step)
if random() < epsilon:
    action = random.choice(legal_actions)     # 探索
else:
    action = argmax(q_values[legal_actions])  # 利用

# 评估时 (DRQNActorAgent.eval_step, epsilon=0)
action = argmax(q_values[legal_actions])      # 纯贪心
```

非法动作的 Q 值被设为 `-inf`，因此 argmax 只会选到合法动作。

epsilon 的调度由以下参数控制：

| yaml key (`drqn`节)    | 当前值 | 代码默认值 | 作用                     |
| ---------------------- | ------ | ---------- | ------------------------ |
| `epsilon_decay_games`  | 20000  | 5000       | 探索率线性衰减的总局数   |
| `exp_epsilon`          | 0.05   | 0.05       | 探索率最终值             |
| (无 yaml key)          | -      | 0.5        | `epsilon_start` 代码默认 |

> **注意**: yaml 中 `epsilon_decay_games: 20000`，覆盖了代码默认值 5000。yaml 中无 `epsilon_start` 键，使用代码默认值 0.5。

---

## 三、Reward 的构成

每个 transition 的格式为 `[state, action, reward, next_state, done]`。reward 由以下四层叠加：

### 3.1 Terminal Payoff（终局奖励）

来源：`reorganize()` 函数 + `SpadesEnv.get_payoffs()`

仅在每一局（round）最后一个 transition 赋予：

```text
reward = team_own_score - beta * team_opp_score
```

相关参数：

| yaml key (`training`节)     | 当前值 | 代码默认值 | 作用                                    |
| --------------------------- | ------ | ---------- | --------------------------------------- |
| `reward_beta`               | 0.5    | 1.0        | terminal payoff 中对手分数权重          |
| `game_enable_blind_nil`     | false  | true       | 是否启用 blind nil                      |

> **注意**: yaml 中 `reward_beta: 0.5`，覆盖了代码默认值 1.0；`game_enable_blind_nil: false`，覆盖了代码默认值 true。

### 3.2 Per-Trick Shaping（逐墩塑形奖励）

来源：`drqn_act()` 中的逐步 reward shaping（硬编码常量，不由 yaml 控制）

对每个非终止步，根据当前墩的赢墩变化计算：

**普通叫牌者（非 Nil）**：

| 情况                               | 奖励                            |
| ---------------------------------- | ------------------------------- |
| 队伍尚未完成叫品，自己/队友赢墩    | +1.0 x 赢墩数                  |
| 队伍已超墩，自己/队友再赢墩        | -0.5 x 赢墩数（避免超墩）      |

**Nil 叫牌者**：

| 情况              | 奖励                |
| ----------------- | ------------------- |
| 自己吃到墩        | -3.0 x 吃墩数      |
| 非 Nil 队友赢墩   | +1.0 x 赢墩数      |

**破对手 Nil**：

| 情况                      | 奖励  |
| ------------------------- | ----- |
| 对手从 0 墩变为 >0 墩     | +2.0  |

### 3.3 Bid-Quality Shaping（叫牌质量塑形奖励）

来源：`_hand_strength()` 估墩函数 (v4.2, 48 维线性模型 + Nil 规则)，硬编码在 `drqn_trainer.py` 中

仅在叫牌阶段的 transition 生效。

**数值叫牌 (`bid_N`)**：

估墩函数给出推荐值 `recommended`，根据推荐值的大小选择不同容忍范围：

| 条件               | 容忍范围 (tol)   |
| ------------------ | ---------------- |
| recommended <= 4   | diff <= 1 无惩罚 |
| recommended >= 5   | diff <= 2 无惩罚 |

奖惩规则：

| 情况                                | 奖励                                       |
| ----------------------------------- | ------------------------------------------ |
| diff == 0（恰好相等）               | +0.3 x scale                               |
| 0 < diff <= tol（容忍范围内）       | 0                                          |
| diff > tol（超出容忍范围）          | -2^(diff-tol) x scale（指数增长惩罚）      |

**Nil 叫牌 (`nil`)**：

| 情况                                   | 奖励                            |
| -------------------------------------- | ------------------------------- |
| 估墩函数也推荐 Nil (recommended=0)     | +0.3 x scale                   |
| 估墩函数推荐 <= 2                      | 0                               |
| 估墩函数推荐 > 2                       | -2^(recommended-2) x scale     |

**scale 余弦退火衰减**：

```text
scale = 0.5 * (1 + cos(pi * games_played / bid_shaping_decay_games)) / 2
```

| 时间点                  | scale 值         |
| ----------------------- | ---------------- |
| games = 0               | 0.5（满额）      |
| games = T/4             | ~0.43            |
| games = T/2             | 0.25             |
| games = 3T/4            | ~0.07            |
| games = T               | 0（完全关闭）    |

相关参数：

| yaml key (`drqn`节)           | 当前值 | 代码默认值 | 作用                           |
| ----------------------------- | ------ | ---------- | ------------------------------ |
| `bid_shaping_decay_games`     | (无)   | 20000      | 估墩 loss 余弦退火衰减总局数  |

> **注意**: yaml 中目前无 `bid_shaping_decay_games` 键，使用代码默认值 20000。如需修改请在 `drqn` 节下添加该键。

设计意图：训练早期用估墩函数引导叫牌方向，后期让模型完全依靠自身博弈经验。

### 3.4 End-of-Round Accuracy（终局叫牌准确度奖励）

来源：每局结束时，在终止 transition 上叠加（硬编码常量，不由 yaml 控制）

仅对非 Nil 玩家，比较队伍叫品与实际赢墩：

| 情况                      | 奖励            |
| ------------------------- | --------------- |
| 恰好完成叫品 (diff=0)     | +4.0            |
| 超 1 墩 (diff=1)          | +1.0            |
| 未完成 (diff<0, set)      | -2.0            |
| 超多墩 (diff>=2)          | -0.6 x diff     |

---

## 四、参数更新机制

### 4.1 数据收集流程

```text
Actor 进程 (CPU)
  |
  +-- 1. env.run()            — 用 shared_qnet 自我对弈一局
  +-- 2. reorganize()         — 重组 trajectory，添加 terminal payoff
  +-- 3. reward shaping       — 叠加 per-trick + bid-quality + end-of-round 奖励
  +-- 4. episode_queue.put()  — 打包 (obs, action, reward, next_obs, legal_actions, done) 发送
  |
  v
Learner 线程 (GPU)
  |
  +-- 5. _drain_queue()       — 从 queue 取出所有 episode
  +-- 6. memory.save()        — 存入 EpisodeMemory
  +-- 7. memory.sample()      — 采样 batch_size 个长度为 seq_len 的序列片段
  +-- 8. _train_step()        — Double DQN 梯度更新
  +-- 9. 权重同步             — 定期同步到 shared_qnet / target_qnet / opponent_pool
```

### 4.2 序列采样方式 (`EpisodeMemory.sample`)

```python
for _ in range(batch_size):
    ep = random.choice(episodes)            # 随机选一个 episode
    end_idx = random.randint(0, len(ep)-1)  # 随机选一个结束位置
    start_idx = max(0, end_idx - seq_len + 1)
    # 取 [start_idx, end_idx] 共 seq_len 步序列
    # 不足 seq_len 则左侧填零
```

### 4.3 训练步骤（Double DQN）

每次训练步：

```python
# === 目标计算（不计算梯度） ===
# Q 网络选动作
q_next = learner_qnet(next_state_seqs)    # (batch, 68)
q_next[illegal_actions] = -inf
best_next = argmax(q_next, dim=1)         # (batch,)

# Target 网络估值
q_target = target_qnet(next_state_seqs)   # (batch, 68)
targets = reward + (1-done) * gamma * q_target[best_next]

# === 梯度更新 ===
q_current = learner_qnet(state_seqs)      # (batch, 68)
q = q_current.gather(action)              # 取实际执行动作的 Q 值
loss = MSE(q, targets)
loss.backward()
clip_grad_norm_(parameters, max_norm=40)
optimizer.step()                          # Adam, lr=3e-5
```

### 4.4 权重同步链路

```text
learner_qnet (GPU)
  |
  |  每 sync_every 个训练步 (yaml drqn.sync_every = 100)
  v
shared_qnet (CPU 共享内存)  <-- Actor 进程直接读取此网络推理
  |
  |  每 update_target_every 个训练步 (yaml drqn.update_target_every = 1000)
  v
target_qnet (GPU)  <-- 整体复制 learner_qnet 的全部参数

opponent_qnet (CPU 共享内存)
  ^
  |  每 opponent_update_every 个 episode (yaml drqn.opponent_update_every = 2000)
  |  将当前 learner 快照加入 opponent_pool
  |  从 pool 中随机选一个加载到 opponent_qnet
  |  pool 最多保存 opponent_pool_size 个历史快照 (yaml drqn.opponent_pool_size = 10)
```

---

## 五、全部超参数汇总

下表列出所有超参数、yaml 中的实际配置值、代码默认值，以及 yaml 是否覆盖了默认值。

### 5.1 来自 `model_config.yaml` -> `training` 节

| yaml key                  | yaml 值  | 代码默认值 | 是否覆盖 | 作用                              | 代码位置                             |
| ------------------------- | -------- | ---------- | -------- | --------------------------------- | ------------------------------------ |
| `seed`                    | 42       | 42         | 否       | 随机种子                          | `train_spades_selfplay_drqn.py:38`   |
| `reward_beta`             | **0.5**  | 1.0        | **是**   | terminal payoff 中对手分数权重    | `spades.py:12`, `spades.py:232`      |
| `game_enable_blind_nil`   | **false**| true       | **是**   | 是否启用 blind nil                | `spades.py:10`, `spades.py:81`       |

### 5.2 来自 `model_config.yaml` -> `drqn` 节

#### 网络结构

| yaml key           | yaml 值  | 代码默认值 | 是否覆盖 | 作用               | 代码位置                    |
| ------------------ | -------- | ---------- | -------- | ------------------ | --------------------------- |
| `embed_dim`        | 256      | 256        | 否       | 嵌入层输出维度     | `drqn_agent.py:69`          |
| `lstm_hidden_size` | 256      | 256        | 否       | LSTM 隐藏层维度    | `drqn_agent.py:72-76`       |
| `lstm_num_layers`  | **2**    | 1          | **是**   | LSTM 层数          | `drqn_agent.py:72-76`       |
| `mlp_layers`       | [256]    | [256]      | 否       | MLP Head 隐藏层    | `drqn_agent.py:80-86`       |

#### 训练

| yaml key               | yaml 值       | 代码默认值   | 是否覆盖 | 作用                      | 代码位置                     |
| ---------------------- | ------------- | ------------ | -------- | ------------------------- | ---------------------------- |
| `learning_rate`        | 0.00003       | 0.00003      | 否       | Adam 学习率               | `drqn_trainer.py:707`        |
| `discount_factor`      | 0.99          | 0.99         | 否       | Q-learning 折扣因子       | `drqn_trainer.py:814`        |
| `batch_size`           | 256           | "auto"       | 是       | 每次训练的 batch 大小     | `drqn_trainer.py:710`        |
| `seq_len`              | **32**        | 16           | **是**   | LSTM 序列采样长度         | `drqn_agent.py:266,302-358`  |
| `max_episodes`         | 8000          | 8000         | 否       | Replay buffer 最大 ep 数  | `drqn_agent.py:263-267`      |
| `max_grad_norm`        | 40            | 40           | 否       | 梯度裁剪阈值              | `drqn_trainer.py:827`        |
| `total_frames`         | 5000000       | 5000000      | 否       | 总训练帧数                | `drqn_trainer.py:843`        |

#### 同步与更新

| yaml key               | yaml 值  | 代码默认值 | 是否覆盖 | 作用                                  | 代码位置                    |
| ---------------------- | -------- | ---------- | -------- | ------------------------------------- | --------------------------- |
| `sync_every`           | 100      | 100        | 否       | learner -> shared 同步频率（训练步）  | `drqn_trainer.py:859`       |
| `train_steps_per_sync` | (无)     | 16         | 否       | 每轮同步间连续训练步数                | `drqn_trainer.py:852`       |
| `update_target_every`  | 1000     | 1000       | 否       | target 网络更新频率（训练步）         | `drqn_trainer.py:836`       |

#### 探索与对手

| yaml key               | yaml 值   | 代码默认值 | 是否覆盖 | 作用                          | 代码位置                    |
| ---------------------- | --------- | ---------- | -------- | ----------------------------- | --------------------------- |
| `exp_epsilon`          | 0.05      | 0.05       | 否       | 探索率最终值                  | `drqn_trainer.py:99-101`    |
| `epsilon_start`        | (无)      | 0.5        | 否       | 探索率初始值                  | `drqn_trainer.py:99`        |
| `epsilon_decay_games`  | **20000** | 5000       | **是**   | 探索率线性衰减局数            | `drqn_trainer.py:98`        |
| `opponent_pool_size`   | 10        | 10         | 否       | 对手池最大快照数              | `drqn_trainer.py:869-876`   |
| `opponent_update_every`| **2000**  | 500        | **是**   | 对手池更新频率（episode 数）  | `drqn_trainer.py:863-876`   |

#### Reward shaping

| yaml key                    | yaml 值 | 代码默认值 | 是否覆盖 | 作用                                | 代码位置                    |
| --------------------------- | ------- | ---------- | -------- | ----------------------------------- | --------------------------- |
| `bid_shaping_decay_games`   | (无)    | 20000      | 否       | 估墩 loss 余弦退火衰减局数         | `drqn_trainer.py:177-180`   |

#### 其他

| yaml key            | yaml 值   | 代码默认值                    | 是否覆盖 | 作用                                | 代码位置                    |
| ------------------- | --------- | ----------------------------- | -------- | ----------------------------------- | --------------------------- |
| `num_actors`        | "auto"    | "auto"（CPU 核数 - 2）       | 否       | Actor 进程数                        | `drqn_trainer.py:558-561`   |
| `cuda`              | "auto"    | "auto"                        | 否       | 训练设备                            | `drqn_trainer.py:533-539`   |
| `gpu_fraction`      | 0.8       | 0.8                           | 否       | GPU 内存使用比例（auto batch 时）   | `drqn_trainer.py:617`       |
| `save_interval`     | 30        | 30                            | 否       | checkpoint 保存间隔（分钟）         | `drqn_trainer.py:893`       |
| `save_path`         | (见下)    | "experiments/spades_selfplay_drqn" | 否  | checkpoint 保存目录                 | `drqn_trainer.py:674`       |
| `eval_every_frames` | **500000**| 100000                        | **是**   | 评估频率（帧数）                    | `drqn_trainer.py:898`       |
| `eval_num_games`    | (无)      | 50                            | 否       | 每次评估的对局数                    | `drqn_trainer.py:657`       |

> `save_path` yaml 值为 `"experiments/spades_selfplay_drqn"`，与代码默认值相同。

### 5.3 yaml 覆盖代码默认值的关键差异汇总

以下参数在 yaml 中的值与代码默认值**不同**，需要特别注意：

| 参数                    | yaml 值    | 代码默认值 | 影响                                       |
| ----------------------- | ---------- | ---------- | ------------------------------------------ |
| `training.reward_beta`  | 0.5        | 1.0        | 降低了对手分数在 reward 中的权重            |
| `training.game_enable_blind_nil` | false | true  | 禁用了 blind nil 功能                      |
| `drqn.lstm_num_layers`  | 2          | 1          | 网络更深，有更强的时序建模能力             |
| `drqn.seq_len`          | 32         | 16         | 训练时 LSTM 看到更长的历史                 |
| `drqn.epsilon_decay_games` | 20000   | 5000       | 探索期更长，衰减更慢                       |
| `drqn.opponent_update_every` | 2000  | 500        | 对手池更新更慢，对手多样性建立更慢         |
| `drqn.eval_every_frames` | 500000    | 100000     | 评估频率降低，减少评估开销                 |

### 5.4 yaml 中未配置但代码有默认值的参数

以下参数在 yaml 的 `drqn` 节中不存在，使用代码默认值：

| 参数                       | 代码默认值 | 说明                                            |
| -------------------------- | ---------- | ----------------------------------------------- |
| `epsilon_start`            | 0.5        | 探索率初始值                                    |
| `train_steps_per_sync`     | 16         | 每轮同步间连续训练步数                          |
| `eval_num_games`           | 50         | 每次评估的对局数                                |
| `bid_shaping_decay_games`  | 20000      | 估墩 loss 余弦退火衰减局数（如需修改请加到 yaml）|
