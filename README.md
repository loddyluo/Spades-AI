# Spades-AI 项目总览（中文）

本文档面向第一次接触本仓库的开发者，目标是让你快速理解并跑通项目。

python evaluate/evaluate_our_mcts_vs_rule_v2.py --seed 41 --num-games 10 --num-workers 20 --torch-num-threads 1 --torch-num-interop-threads 1 --trace-log-dir logs --symmetric-seat-swap 1 --p0 our_mcts --p1 go_rule_2 --p2 our_mcts --p3 go_rule_2 --our-checkpoint mlp_test_3.pth --our-exploration-constant 25

## Trouble Shooting


git clone -b "MCTS+MLP" --recurse-submodules https://github.com/loddyluo/Spades-AI.git



submodule没拉取：

git submodule sync --recursive && git submodule update --init --recursive

git submodule add https://github.com/loddyluo/Spades_AI_GO-MCTS Spades_AI_GO-MCTS

git submodule update --init --recursive


## 1. 项目目标

本项目围绕黑桃王（Spades）构建可运行的 AI 决策系统，核心目标是：

1. 在完全信息局面下给出高质量出牌决策。
2. 提供精确求解器（Alpha-Beta）作为高置信基准。
3. 提供近似搜索（MCTS）与价值网络（MLP）结合的实战策略。
4. 建立从数据生成 -> 训练 -> 评估 -> 对局运行的完整闭环。

当前主线已经完成：

- 特征升级到 1229 维。
- 精确求解器支持返回动作 Q 值与最优动作。
- MLP 升级为 value/policy 双头结构。
- 支持“截断 MCTS + MLP 叶子评估”的出牌策略。
- 提供可直接跑完整牌局的程序与随机对局评测程序。

## 2. 如何上手运行

### 2.1 环境配置

推荐环境（与当前仓库实际运行一致）：

- Linux
- Python 3.12（conda 环境）
- PyTorch

如果你已经在本仓库环境中，通常不需要额外步骤，直接运行命令即可。

### 2.2 最快可跑通命令

先跑一局完整对局（从 52 张牌开始）+ 随机对局评估：

```bash
cd /Spades-AI
python strategy/spades_match_runner.py --seed 0 --checkpoint ./result/mlp_test_3.pth --exact_threshold 24 --leaf_threshold 24 --simulations_per_action 5 --num_eval_games 10
```

这个命令会做两件事：

1. 跑一局完整对局并打印每步出牌（含 1229 维输入维度、合法动作数、输出动作、每墩赢家）。
2. 再随机生成多局（默认 10 局），统计“玩家1=MCTS策略，其余随机玩家”时的平均得分。

### 2.3 训练与评估命令（常用）

生成数据（示例：x=24）：

```bash
python data/generate_dataset.py --xs 24 --num_samples 1000 --output_dir data
```

训练双头 MLP（示例）：

```bash
python mlp/train.py --xs 24 --data_dir data --epochs 1 --batch_size 4096 --save /tmp/mlp_test.pth
```

评估 value + policy：

```bash
python mlp/evaluate.py --checkpoint /tmp/mlp_test.pth --data_dir data --xs 24
```

评估 x=25 动作一致率（value-head 一步前瞻）：

```bash
python mlp/evaluate_action_25.py --checkpoint ./result/mlp_test_3.pth --dataset data/spades_dd_x25_n1000.pt --max_samples 1000
```

跨仓库模型评估（你这边的截断 MCTS + 合作者仓库的随机 / 启发式 / 训练模型）：

```bash
python evaluate/evaluate_model_matchups.py \
    --seed 0 \
    --num-games 10 \
    --symmetric-seat-swap 1 \
    --p0 our_mcts \
    --p1 go_rule \
    --p2 go_gomcts \
    --p3 go_random \
    --go-pv-checkpoint /path/to/collaborator_pv.pt \
    --our-checkpoint ./result/mlp_test_3.pth \
    --our-exact-threshold 24 \
    --our-number-of-exact-solvers 50
```

协作者仓库的模型和状态桥接代码放在 `evaluate/GO-MCTS/`，评估脚本会先把你这边的 `GameState` 转成对方仓库的状态格式，再统一回到本地对局引擎里执行。

如果你只想跑当前最稳定的评估组合，直接用这一条：

```bash
python evaluate/evaluate_our_mcts_vs_rule_v2.py --seed 990 --num-games 1 --num-workers 15 --torch-num-threads 1 --torch-num-interop-threads 1 --trace-log-dir logs --symmetric-seat-swap 1 --p0 our_mcts --p1 go_rule_2 --p2 our_mcts --p3 go_rule_2 --our-checkpoint mlp_test_3.pth
```

这条命令用于评估本地截断 MCTS 与协作者规则玩家的对抗表现。当前评估路径会按语义关闭 `blind_nil`，避免把“看牌后”的叫牌与“盲叫”混在一起；本地 `OurHandStrengthMCTSPlayer` 也只会在 `nil` 或普通数值叫牌之间做选择。

如果你想更接近“真实打牌能力”而不是“看牌运”，`evaluate/evaluate_model_matchups.py` 现在默认会对每个 base seed 跑两局对称对照：第一局使用原始座位顺序，第二局交换奇偶座位组（`0,2` 与 `1,3` 对调）。可以通过 `--symmetric-seat-swap 0` 关闭这一对称双跑。

## 3. 最核心接口（出牌程序）

你如果只关心“怎么出牌”，看下面 3 个位置即可。

### 3.1 策略核心

文件：`strategy/truncated_mcts_strategy.py`

核心接口：

- `TruncatedMCTSStrategy.choose_action(state: GameState) -> Card | None`

输入：

- `state`：完整牌局状态（包含手牌、桌面牌、叫牌、turn、队伍等）。

输出：

- 返回当前应出的 `Card`；若没有合法动作返回 `None`。

### 3.2 玩家封装（可直接放进对局）

文件：`strategy/spades_player_programs.py`

核心类：

- `TruncatedMCTSPlayer`：在 `play_card()` 中调用上面的 `choose_action()`。
- `RandomSpadesPlayer`：在合法动作中随机出牌（基线对手）。

`TruncatedMCTSPlayer.play_card` 输入/输出：

- 输入：`legal_cards: list[Card]` + `state_view: dict`
- 其中 `state_view` 里会带：
    - `feature`：1229 维特征（`np.ndarray`）
    - `state`：当前 `GameState`（完整状态快照）
- 输出：一张 `Card`

### 3.3 对局驱动

文件：`strategy/spades_match_runner.py`

职责：

- 随机发牌构造牌局。
- 在每个玩家回合构造 1229 维输入并调用对应玩家程序。
- 校验出牌合法性，不合法会直接抛错。
- 更新状态直到终局，并打印最终得分。

## 4. 实现方式（详细）

### 4.1 特征工程（1229 维）

文件：`trick_taking/utils/feature_encoder.py`

关键点：

- 1229 维由 7 个分块组成（手牌、叫牌、当前墩、历史、花色分析、队伍局势、全局标记）。
- 历史分块含“每张牌由谁打出 + 第几轮打出”等显式轨迹信息。

### 4.2 精确求解器（基准）

文件：`trick_taking/solvers/exact_double_dummy.py`

关键点：

- `solve(state)`：返回状态值（队伍0视角）。
- `solve_with_q(state)`：返回 `value + best_action + action_q_values`。
- 默认走 C++ 原生后端（性能高），Python 版本用于对照与回归。

### 4.3 训练数据管线

文件：`data/training_data.py`、`data/generate_dataset.py`

关键点：

- 可按剩余牌数分桶生成（当前常用 x=24/25/28/32）。
- 每条样本含：
    - `feature`
    - `value_team0`
    - `value_view`
    - `action_ids`
    - `action_q_values`
    - `best_action_id`

### 4.4 MLP 双头模型

文件：`mlp/mlp_model.py`、`mlp/train.py`、`mlp/training_utils.py`

关键点：

- 共享 backbone + value_head + policy_head。
- value head 训练目标是 `value_view`。
- policy head 监督来自 `action_q_values`（软目标 + 合法动作掩码）。

### 4.5 截断 MCTS 策略

文件：`strategy/truncated_mcts_strategy.py`

策略逻辑：

1. 若剩余牌数 `<= exact_threshold`（默认 30），直接调用精确求解器选最优动作。
2. 若剩余牌数 `> exact_threshold`，对每个根动作运行 PUCT/MCTS。
3. 搜索到 `<= leaf_threshold`（默认 24）时，用 MLP value-head 估值叶子。
4. policy-head 提供先验概率（PUCT 的 `prior`）。
5. 所有价值统一换算到“队伍0视角”；当前行动方是队伍0则取 argmax，队伍1则取 argmin。

### 4.6 为什么出牌顺序不是固定 0-1-2-3

黑桃是“按墩”推进：

- 一墩内按首攻玩家顺时针走 4 人。
- 下一墩首攻是上一墩赢家。

所以跨墩后顺序会变化，这是正确规则行为，不是程序 bug。

## 5. 每个文件夹包含什么

以下是当前仓库主要目录作用：

- `trick_taking/`
    - 通用牌类框架（卡牌、牌堆、状态、规则、驱动、玩家接口）。
    - `games/`：具体游戏规则（含 Spades）。
    - `solvers/`：求解器实现（精确、MCTS、C++后端封装）。
    - `utils/`：特征编码与状态工具。

- `strategy/`
    - 可直接用于实战对局的策略层。
    - `truncated_mcts_strategy.py`：截断 MCTS 核心。
    - `spades_player_programs.py`：随机玩家、MCTS 玩家封装。
    - `spades_match_runner.py`：完整牌局构造、运行与评估。

- `mlp/`
    - 双头模型、训练脚本、评估脚本。

- `data/`
    - 训练数据构造、保存、加载和生成脚本。

- `evaluate/`
    - 测试与评估打牌能力

- `logs`
    - 调试结果

- `tests/`
    - 回归测试、求解器正确性测试、MCTS测试、数据质量检查脚本。

- `result/`
    - 训练产生的模型权重（checkpoint）。

- `documents/`
    - 其他文档资料（如有）。

## 6. 给完全新手的建议路径

如果你第一次接触这个项目，建议按这个顺序：

1. 先跑 `strategy/spades_match_runner.py`，确认“能打一整局”。
2. 再看 `strategy/truncated_mcts_strategy.py`，理解决策逻辑。
3. 再看 `data/training_data.py` 和 `mlp/train.py`，理解数据与训练。
4. 最后看 `trick_taking/solvers/exact_double_dummy.py`，理解基准求解器。

## 7. 常见问题

### 7.1 为什么叫牌里有时同一玩家可能连续出现？

如果启用了 `blind_nil/pass` 机制，玩家会先走盲叫分支，再进入常规叫牌。当前评估入口默认关闭 `blind_nil`，改成更直观的一人一次叫牌；这也是上面那条主评估命令推荐的原因。

### 7.2 出牌程序到底吃什么输入？

策略核心吃的是完整 `GameState`。对局驱动额外会提供 1229 维 `feature`，方便你替换成只基于特征的玩家实现。

### 7.3 如何改 MCTS 搜索力度？

优先调整这几个参数：

- `simulations_per_action`
- `exact_threshold`
- `leaf_threshold`
- `exploration_constant`

通常搜索更深更稳，但会更慢。
