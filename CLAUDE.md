1 请问现在前端的“远程对战”模式，两个AI分别是什么strategy player？默认调用的是哪个config文件？
2 在evaluate文件夹中有没有评估两个具有不同的超参数的同种类玩家的脚本？评测的是哪种strategy player？超参数是什么？

---

## 超参数解释（HyperparamConfig）

定义在 [`strategy/hyperparam_config.py`](strategy/hyperparam_config.py)。

本次新增 2 个超参数：`bad_action_penalty_factor`（队友坏动作惩罚系数）、`gamma`（bid_prod 指数）。

### 1. IS 采样池（后 36 张的 determinization 阶段）
- **`num_proposals`** — 每步生成多少份对手手牌的提案（batch size）
- **`num_proposals_limit`** — 生成提案的最大尝试次数
- **`min_pool_size`** — 要求至少保留多少个有效提案（权重 > 0）

### 2. 预算表（控制精确求解的计算量）
按剩余牌数分档，每档两个上限：
- **`budget.remaining_in`** — 手牌剩余 ≤ 该值时进入对应档位
- **`budget.top_k`** — 重要性采样配额（只从 top-K 权重的提案选）
- **`budget.max_samples`** — 最终总采样数上限（含 IS + 多样性填充）

剩余牌越少，预算翻倍（因为求解更快、每步更关键）。

### 3. 重要性权重计算
- **`bad_action_weight`** — 提案中有坏动作时的惩罚系数：`"x"`=按坏动作比例加权（原版），`"0.5"`=常数折半，`"0.0"`=有坏动作直接淘汰
- **`bad_action_penalty_factor`** — 队友出牌与规则玩家不一致时的权重衰减系数（默认 0.81，对应 `_compute_batch_replay_weights` 中的 `play_weights *= 0.81`）
- **`gamma`** — `bid_prod ** gamma` 指数。>1 放大似然差异，<1 压平，1=不变（默认值）
- **`trick_num_threshold`** — 第几墩起用求解器 Q 值加权（0-index，默认 8 = 第 9 墩起）

### 4. 选择顺序
- **`swap_is_fill`** — `False`（默认）：先按 IS 权重选，不足再多样性填充；`True`：先多样性填充，再 IS

### 5. Q 值裁剪
- **`multiplier_clip`** — Q 值乘子的裁剪阈值（默认 40.0）
- **`multiplier_clip_factor`** — 超过阈值后的倍率（默认 1.0 = 不缩放，2.0 = 放大）

### 6. 并行
- **`num_workers`** — 求解器并行进程数（0 = 自动检测 CPU 核数 - 1）

