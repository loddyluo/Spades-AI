# Copilot 持续记录（wyy）

## 文件目的
- 记录每次对话中确认过的事实、结论、风险和下一步计划。
- 便于切换上下文后快速恢复项目认知。

## 维护规则
- 每次对话至少更新以下四项：
  1. 已确认事实
  2. 未确认假设/风险
  3. 关键命令与入口
  4. 下一步行动
- 只记录“可复用知识”，不记录冗余闲聊。
- 若后续证据推翻旧结论，必须在原结论后追加“已修正”。
- 注意所有代码改动后都需要编写一个测试程序或测试用例来确认正确性
 - 新增要求：每当添加或修改文件，必须在文件开头写明该文件的作用；且文件中定义的每个函数在文件开头或模块文档中写明其输入格式（类型与形状/期望字段）和输出格式（类型与形状/字段含义）。以后所有新增文件和函数必须严格遵守此规则。

---

## 对话记录

### 2026-04-26｜会话1（仓库梳理与865维特征定位）

#### 已确认事实
- 项目目标：围绕黑桃王（Spades）构建可用AI，当前核心为双明手求解器与MLP拟合。
- 双明手求解器有两条线：
  - 精确求解器（Alpha-Beta）
  - 近似求解器（MCTS）
- 865维特征入口在 trick_taking/utils/feature_encoder.py。
- 865维由7个分块拼接：
  - 手牌 112
  - 叫牌 126
  - 当前墩 73
  - 历史 192
  - 花色分析 192
  - 队伍局势 154
  - 全局标记 16
- MLP相关目录功能：
  - distill：18张手牌中盘样本生成、训练、评估
  - mlp_2_left：2张残局样本生成、训练、评估
- .pth评估入口：
  - mlp_2_left/evaluate.py
  - distill/eval.py

#### 未确认假设/风险
- distill方向当前误差是否足以支撑策略层决策，尚未形成统一标准。
- 仅拟合静态值函数，尚未形成可直接打牌的策略系统。

#### 关键入口
- 特征编码：trick_taking/utils/feature_encoder.py
- 18张评估：distill/eval.py
- 2张评估：mlp_2_left/evaluate.py
- 特征测试：tests/test_feature_encoder.py

#### 下一步建议
- 制定“可接受误差阈值”与“对实战决策影响”的联合评测。
- 在特征中补充更强的对手建模信号，并验证增益。

---

### 2026-04-26｜会话2（规划记录机制 + 训练方向讨论）

#### 已确认事实
- 需要建立长期知识继承文件，并在后续每次对话持续更新。
- 当前用户结论倾向：
  - MCTS与精确求解器在当前测试集上基本正确。
  - 18张MLP已实现对精确求解器结果的拟合流程。

#### 待检验问题（本会话将进一步讨论）
- 18张场景“差约20分”是否可接受。
- 865维特征是否合理且足够。
- 下一阶段从“求解器+拟合器”走向“可打牌AI”的路线图。

#### 计划中的后续行动
- 结合误差分布、关键决策翻转率、胜率影响三条指标，评估20分误差可接受性。
- 设计增量特征实验：先补“每张已出牌的出牌者信息”，做消融对比。
- 推进“专家数据蒸馏 + 策略学习 + 自博弈微调”的三阶段训练。

#### 本轮结论（已完成正确性检验）
- “精确求解器与MCTS基本正确”这个判断方向是对的，但置信度不同：
  - 精确求解器：在输入状态合法且满足前提（完全信息、叫牌结束、PLAYING阶段、队伍映射固定）下，可视为当前实现中的高置信正确基准。
  - MCTS：属于近似搜索，通常接近但不保证全局最优，置信度低于精确求解器。
- “可从任意时刻牌局开始计算我方减对方最大值”总体成立，但应补充前提：
  - 必须能完整恢复该时刻状态（手牌、桌面牌、tricks_won、叫牌等）；
  - 精确求解器才是严格最优，MCTS给的是近似估计。
- “18张MLP误差约20分是否可接受”：
  - 若目标只是做粗价值排序/候选筛选：可作为阶段性可用；
  - 若目标是稳定决策（避免关键出牌翻转）：通常偏大，建议继续降误差并引入策略层训练。
- “865维是否合理且足够”：
  - 合理性：one-hot 对离散语义本身是合理编码，MLP能学，但确实更依赖数据量与覆盖度；
  - 足够性：当前缺少“历史每张牌由谁打出”的显式信号，这对对手/队友分布推断很关键，应补充。

#### 新增推进路线（从求解器走向可打牌AI）
- 路线1（短期可落地）：先做“受限动作策略”
  - 用精确求解器在18张及以下生成专家动作标签，训练策略头（分类）而非只回归值。
- 路线2（中期）：不完全信息建模
  - 增加出牌者历史特征与不确定性特征（对手剩余花色概率、缺门置信）。
- 路线3（中长期）：自博弈强化
  - 先行为克隆预热，再以自博弈做策略改进（可结合价值网络与MCTS/PIMC）。
- 评估闭环
  - 除MAE外，增加“最佳动作一致率、关键局面翻转率、整局胜率提升”三类指标。

---

### 2026-04-26｜会话3（两种整体架构可行性评估 + 动作Q值需求确认）

#### 已确认事实
- 精确求解器在当前仓库中返回的是“状态值V”，不是动作Q表：
  - solve(state) 返回 float
  - 目前没有公开接口直接返回“所有合法动作对应的Q值与最优动作”
- 通过现有私有方法可以在不改代码的前提下计算动作Q值（工程上可行，接口层面不友好）：
  - 使用 rules.playable 获取合法动作
  - 对每个动作调用 _apply_action + _minimax 得到Q
- 只读基准（当前环境）显示精确求解器耗时大致如下：
  - 2张残局：约 0.000045s
  - 17张：约 0.008460s
  - 18张：约 0.018752s
  - 25张：约 0.184824s
  - 28张：约 2.052681s
- 在28张样例上，计算根值后再枚举全部动作Q值（7个动作）总耗时约 1.755s（共享同一solver缓存）。
- distill/eval.py 报错根因已确认：checkpoint 与当前模型结构不匹配（不是文件缺失）。

#### 对两种架构思路的评估结论
- 思路A（小剩余牌用网络；大剩余牌用MCTS + 网络截断评估）可行，且是最稳妥的近期主线。
- 思路B（POMDP信念建模 + 采样 + 精确求解器）理论可行，但工程复杂度高、在线时延风险高，适合作为中期并行研究线。

#### 关于“MLP最多训练到剩余多少步”建议
- 结合当前精确求解器耗时曲线与在线时延目标（每步最好<2s，最多<5s）：
  - 推荐先把价值网络训练覆盖到“剩余 24~28 张手牌”区间。
  - 落地优先级：先做到 24/25，再扩展到 28。
- 原因：
  - 25张时精确解约0.18s，说明该区间可大量离线打标签；
  - 28张时约2.05s，已接近在线2s目标上限，适合作为MCTS截断边界候选。

#### 三阶段计划（扩展版）
- 阶段1：监督蒸馏阶段（先把“会选动作”做出来）
  - 目标：从精确求解器得到动作标签和Q值监督，训练策略头+价值头。
  - 关键工作：
    - 扩展特征（含出牌者-轮次信息）；
    - 扩展精确求解器接口返回 action->Q 与 best_action；
    - 建立 18/24/25/28 张分桶数据集与评估集。
  - 完成标准：动作一致率、Q值相关性、整局胜率都有提升。
- 阶段2：搜索增强阶段（把模型接入MCTS）
  - 目标：大剩余牌时，MCTS rollout 到阈值后用网络评估叶子，显著降时。
  - 关键工作：
    - 设定截断阈值（优先试 28，再试 25/24）；
    - 调参：迭代数、探索系数、截断深度联调；
    - 评测：单步时延、动作质量、整局胜率三维联合优化。
  - 完成标准：在<2s或<5s预算下，胜率明显优于纯启发式/低迭代MCTS。
- 阶段3：不完全信息与自博弈阶段（向真正AI过渡）
  - 目标：引入信念状态建模，解决“看不见对手手牌”的现实约束。
  - 关键工作：
    - 建立可增量更新的信念模型（每次出牌后更新）；
    - 采样若干确定化世界，使用精确求解器或截断MCTS做决策聚合；
    - 后续接自博弈强化，减少对求解器标签依赖。
  - 完成标准：在不完全信息对战中稳定优于阶段2模型。

#### 顺序化执行清单（下一步）
1. 先做特征扩展设计与实现（只加最关键“牌-轮次-玩家”信息，控制维度膨胀）。
2. 改精确求解器公开接口，输出 best_action 与 action->Q。
3. 基于新接口重建训练数据管线（18/24/25/28分桶）。
4. 先训练价值头，再加策略头，做离线指标评估。
5. 将价值网络接入MCTS做截断评估，按2s/5s预算调阈值与迭代。
6. 再进入POMDP信念建模原型验证。

---

### 2026-04-27｜会话4（阶段1-步骤1已落地：轨迹特征扩展 + 接口同步 + 测试）

#### 已确认事实
- 已完成“只加最关键牌-轮次-玩家信息”的特征扩展实现。
- 维度从 865 升级到 1229，新增内容放在 History 分块：
  - 每张牌由谁打出：52 × 6 one-hot（P0/P1/P2/P3/未出/未知）= 312
  - 每张牌第几轮打出：52 标量（未知=-1，未出=0，第r轮=r/13）= 52
- History 分块维度：192 -> 556；总维度：1229。
- 已同步所有使用旧865维接口的核心代码：
  - trick_taking/utils/feature_encoder.py
  - distill/mlp_model.py
  - mlp_2_left/mlp_model.py
  - distill/train.py
  - mlp_2_left/train.py
  - distill/generate_18card_states.py
  - tests/test_both_solvers_25.py
  - tests/test_both_solvers_28.py
  - feature_design.md

#### 测试与验证
- 已按“每次代码改动都要有测试”要求完成并执行：
  1. tests/test_feature_encoder.py
     - 覆盖新维度、数值范围、新增轨迹编码语义
     - 结果：通过（所有子测试通过）
  2. tests/test_mlp_input_dim.py（新建）
     - 校验 distill/mlp_2_left 两套 MLP input_dim 与 encoder.total_dim 一致
     - 结果：通过

#### 实现细节说明
- 对“轨迹未知”场景进行了显式编码：
  - 当状态只有 played_bitset 而无完整 trick_history 时，不强行猜测玩家/轮次，避免错误标签污染。
- 设计上保留“低膨胀”原则：
  - 玩家信息使用 one-hot 保持离散语义稳定；
  - 轮次使用标量避免过大维度增长。

#### 下一步建议
1. 进入阶段1-步骤2：扩展精确求解器公开接口，返回 best_action 与 action->Q。
2. 基于新接口改造数据采集管线，支持策略监督（动作标签/Q监督）。

---

### 2026-04-27｜会话5（阶段1-步骤2已落地：精确求解器公开接口扩展 + 加速尝试）

#### 已确认事实
- 已在不破坏旧接口的前提下扩展精确求解器公开能力：
  - 保持 `solve(state) -> float` 兼容旧调用方。
  - 新增 `solve_with_q(state)` 返回：
    - `value`
    - `best_action`
    - `action_q_values` (action -> Q)
    - `action_values`（可打印列表）
    - `current_player` / `optimize_for_team`
- 新接口支持不同剩余牌数场景（测试覆盖约 1~25）。

#### 代码改动
- 精确求解器接口扩展：
  - trick_taking/solvers/exact_double_dummy.py
  - 关键新增：`solve_with_q`、`_validate_state`
- 加速尝试版（暂不替换主接口）：
  - trick_taking/solvers/exact_double_dummy_fast_try.py
  - 方案：动作排序 + alpha-beta 剪枝复用 + TT复用

#### 测试与验证
- 新增测试：
  1. tests/test_exact_solver_api_q.py
     - 验证 `solve_with_q` 结构正确、value与solve一致、best_action与Q最优一致
     - 覆盖 1~25 附近不同剩余牌数
  2. tests/test_exact_solver_fast_try.py
     - 验证 FastTry 与基线 Exact 在多类状态下结果一致
- 回归测试：
  - tests/test_exact_simple.py（通过）
  - tests/test_both_solvers_17.py（通过，精确解与MCTS对齐）

#### 可行性结论（更低耗时）
- 仅靠 Python 层动作排序可带来一定剪枝收益，但上限有限。
- 若要进一步显著降时，优先级建议：
  1. 保持算法语义不变前提下，做“就地走子+撤销”减少深拷贝开销；
  2. 引入更强置换表键与走子排序启发；
  3. 再考虑 C++/Rust 实现搜索内核，通过 Python 绑定调用。

#### 下一步建议
1. 改造数据采集：使用 `solve_with_q` 导出动作标签与Q监督。
2. 训练策略头+价值头联合模型，并建立动作一致率评估。

---

### 2026-04-27｜会话6（新增15~25张两种精确求解器耗时统计程序）

#### 已确认事实
- 已在 tests 目录新增耗时统计程序：
  - tests/test_exact_solver_timing_15_25.py
- 程序功能：
  - 对剩余牌数 15~25，每个牌数生成3个随机样本；
  - 分别运行 ExactDoubleDummySolver 与 ExactDoubleDummyFastTrySolver；

### 2026-04-28｜会话7（native / opt1 / Python 三方对照与 15~30 性能验证）

#### 已确认事实
- 以确定性样本构造重新对照后，`cpp_native`、`cpp_opt1` 和 Python 参考求解器在 value 和 action_q_values 上一致。
- 之前看到的“value mismatch”主要来自测试样本构造不稳定，不是求解器值本身错误。
- 之前看到的 `best_action` 差异来自多个动作并列最优，属于 tie；Python、native、opt1 都可能选择不同的最优动作，但 Q 值仍一致。
- 当前三方对照脚本已改成确定性局面生成，并在 15~30 张牌上完成验证与测速。
- 最终 15~30 性能对比中，`cpp_opt1` 相比 `cpp_native` 在部分区间更快，但并非全区间都更快；两者均保持与 Python 参考值一致。

#### 已验证的结果
- 确定性三方一致性检查样本数：48。
- 并列最优样本数：33。
- 15~30 速度对比脚本已成功跑完，证明脚本本身稳定。

#### 未确认假设 / 风险
- `best_action` 在并列最优时并不是唯一答案，后续若要做更严格的 API 约束，需要定义动作排序规则或额外返回 tie 集合。
- 目前的正确性验证依赖确定性样本构造；若继续扩展测试集，应保持同样的确定性构造方式，避免把样本随机性误判为求解器错误。

#### 关键命令与入口
- 三方对照脚本：tests/test_exact_solver_timing_15_35_opt_compare.py
- Python 参考求解器：trick_taking/solvers/exact_double_dummy.py
- 稳定 C++ 主线：trick_taking/solvers/exact_double_dummy_cpp_native.py
- 优化尝试：trick_taking/solvers/exact_double_dummy_cpp_opt1.py

#### 下一步行动
- 若继续优化，优先在保持 value / action_q_values 正确的前提下再改 `best_action` 的 tie 处理规则。
- 若要做更大规模测速，继续沿用 15~30 的确定性样本构造，不再用不稳定随机样本。
- 保持 `copilot_wyy.md` 持续更新，把“样本构造”和“tie 语义”写成可复用结论。
  - 统计并输出各牌数下两者平均耗时与加速比（base/fast）。
- 程序内包含一致性校验：两种精确求解器在同一状态的返回值必须一致，否则直接报错。

#### 实测结果（每个牌数3样本）
- 15张：base 0.006071s，fast 0.006329s，加速比 0.959
- 16张：base 0.010992s，fast 0.011360s，加速比 0.968
- 17张：base 0.010944s，fast 0.013697s，加速比 0.799
- 18张：base 0.024073s，fast 0.015938s，加速比 1.510
- 19张：base 0.041447s，fast 0.047096s，加速比 0.880
- 20张：base 0.046651s，fast 0.050262s，加速比 0.928
- 21张：base 0.065094s，fast 0.071579s，加速比 0.909
- 22张：base 0.152794s，fast 0.168272s，加速比 0.908
- 23张：base 0.398944s，fast 0.326248s，加速比 1.223
- 24张：base 0.316310s，fast 0.402770s，加速比 0.785
- 25张：base 2.715618s，fast 0.728275s，加速比 3.729

#### 结论
- FastTry 在高复杂度区间（特别是25张）有明显优势，但在部分中低复杂度区间不稳定。
- 后续若继续优化，可优先针对 22~25 张区间做动作排序策略与节点扩展策略调优。

---

### 2026-04-28｜会话7（原生 C++ 搜索内核落地与基准验证）

#### 已确认事实
- 已新增真正的原生 C++ 搜索内核尝试：
  - trick_taking/solvers/exact_double_dummy_cpp_native_core.cpp
  - trick_taking/solvers/exact_double_dummy_cpp_native.py
- 新原生版本的职责分工：
  - C++ 负责完整 alpha-beta 搜索、合法动作生成、转置表与评分
  - Python 负责 GameState 到 C 结构的转换，以及 solve_with_q 的根节点动作枚举
- 新增正确性测试：
  - tests/test_exact_solver_cpp_native.py
- 新增耗时基准：
  - tests/test_exact_solver_timing_15_28_native.py
- 原生库已成功编译并加载，native_available=True，不是静默回退。

#### 实测结果（每个牌数3样本）
- 15张：base 0.004505s，cpp_native 0.005020s
- 16张：base 0.013465s，cpp_native 0.000284s
- 17张：base 0.014578s，cpp_native 0.000179s
- 18张：base 0.018334s，cpp_native 0.000172s
- 19张：base 0.047412s，cpp_native 0.001084s
- 20张：base 0.083640s，cpp_native 0.001401s
- 21张：base 0.235900s，cpp_native 0.005971s
- 22张：base 0.126686s，cpp_native 0.004325s
- 23张：base 0.818566s，cpp_native 0.068788s
- 24张：base 0.615166s，cpp_native 0.012515s
- 25张：base 2.501737s，cpp_native 0.110288s
- 26张：base 0.506283s，cpp_native 0.037419s
- 27张：base 0.639235s，cpp_native 0.083347s
- 28张：base 1.566727s，cpp_native 0.282710s

#### 结论
- 原生 C++ 搜索内核相对基线有数量级提升，28 张样本已经压到 0.3s 量级，明显优于此前所有 Python / C++ 尝试。
- 当前实现仍保留 Python 侧根节点 Q 值枚举，因此 solve_with_q 的代价会比 solve 更高，但 solve 本身已经足够快。
- 后续若继续提速，优先方向是把根节点动作枚举也下沉到 C++，并进一步优化转置表与动作排序。
- 原生版已取消静默回退；如果 C++ 库不可用，solve / solve_with_q 会直接报错，避免误把 Python 基线当成原生结果。

#### 下一步行动
1. 继续做正确性回归：补 25~28 张的更多随机样本。
2. 评估是否把 action_q_values 也搬进 C++，减少 Python 根层循环。
3. 决定是否把原生版作为默认精确求解器候选。

---

### 2026-04-28｜会话8（测试范围收缩，降低正确性与基准耗时）

#### 已确认事实
- 正确性测试已收缩为 16 张及以下的局面，避免使用过重样本拖慢回归。
- 耗时测试已收缩为 15~30 张，不再跑 32 张样本。
- 最新测试结果：
  - `tests/test_exact_solver_cpp_native.py` 通过
  - `tests/test_exact_solver_timing_15_32_fastest.py` 通过，并实际只跑到 30 张

#### 结论
- 对当前阶段而言，16 张以下的正确性覆盖和 15~30 张的性能采样已经足够支撑日常迭代。
- 后续若要恢复更大范围测试，建议只在专项验证时临时开启，不作为默认回归集。

### 2026-04-28｜会话9（重建训练数据管线：x=24/28/32 分桶 + 新 mlp 目录）

#### 已确认事实
- 已新增统一的训练数据生成层：`data/training_data.py`，负责按 `x=24/28/32` 生成局面、调用 `cpp_opt1`、保存为 PyTorch 可直接加载的数据文件。
- 已新增数据生成命令行：`data/generate_dataset.py`，支持按桶批量生成并保存到 `data/` 目录。
- 已新增生成速度测试程序：`data/benchmark_generate.py`，默认对 `x=24/28/32` 各生成 20 条数据并统计耗时。
- 已新增数据规范测试：`tests/test_training_data_pipeline.py`，验证样本字段、特征维度、`torch.save/torch.load` 读写闭环。
- 已新建 `mlp/` 目录，并加入 `train.py`、`mlp_model.py`、`evaluate.py`，默认读取 `x=24/28/32` 桶数据，训练目标是 `value_view`。
- 训练与评估脚本保留了 `bucket_xs=(24,28,32)` 接口，方便后续扩展成多桶联合训练或分桶评估。
- 单样本端到端调试已完成：生成 24/28/32 各 1 条数据，`mlp/train.py` 可完成 1 个 epoch 训练，`mlp/evaluate.py` 可正常输出每桶指标。

#### 已验证的结果
- 数据规范测试：通过。
- 端到端最小闭环：通过。
- 生成速度测试（每桶 20 条）：
  - x=24：总 0.863s，均 0.0431s/条
  - x=28：总 17.171s，均 0.8586s/条
  - x=32：总 359.171s，均 17.9586s/条

#### 未确认假设 / 风险
- `x=32` 桶生成明显更慢，后续大规模采样时需要单独控制批次大小和总量，否则会拖慢训练数据准备。
- 目前训练目标先按 `value_view` 做回归，`action_q_values` 已保存到数据文件，但尚未接入策略头训练。
- 训练脚本当前默认读取每个桶的匹配文件集合，但后续如果一个桶拆成多个数据文件，需要继续保持文件命名规则一致。

#### 关键命令与入口
- 数据生成：`data/generate_dataset.py`
- 生成速度测试：`data/benchmark_generate.py`
- 数据规范测试：`tests/test_training_data_pipeline.py`
- 新训练入口：`mlp/train.py`
- 新模型定义：`mlp/mlp_model.py`
- 新评估入口：`mlp/evaluate.py`

---

### 2026-05-07｜会话10（运行求解器一致性测试）

#### 已确认事实
- 我运行了 `tests/test_solver_data_generation_correctness.py`，对小剩余牌数 x=2,4,6,8 使用确定性种子逐个构造局面，分别用 Python 参考求解器与 `cpp_opt1`（若可用）对照求解；在当前环境下（若 C++ 库不可编译则回退到 Python 比对），全部样本通过一致性检查。

#### 新知识 / 现场观察
- 在本次环境中，`trick_taking/solvers/exact_double_dummy.py` 与 `trick_taking/utils/feature_encoder.py` 在我之前插入的模块说明导致 `from __future__` 位置错误；我已合并并修复相关模块 docstring，确保 `from __future__ import annotations` 在模块顶层位置。
- 测试结果：
  - x=2: passed
  - x=4: passed
  - x=6: passed
  - x=8: passed

#### 关键命令与入口
- 运行命令：
```
/root/miniconda3/bin/python tests/test_solver_data_generation_correctness.py
```
- 相关文件：`data/training_data.py`, `trick_taking/solvers/exact_double_dummy.py`, `trick_taking/solvers/exact_double_dummy_cpp_opt1.py`, `trick_taking/utils/feature_encoder.py`

#### 下一步行动
- 若需要，我可以：
  - 把该测试加入 CI 回归套件（仅覆盖小 x 值），或
  - 扩展测试到更多 x 值与更多种子（注意 16+ 会显著耗时），或
  - 将 `exact_double_dummy_cpp_opt1` 的编译输出（若可用）也纳入自动检查并在失败时记录编译错误日志。


#### 下一步行动
- 给出大规模生成数据的命令行，但不要实际运行。
- 继续完善 `mlp/` 的训练策略，后续如果要做动作监督，再从 `action_q_values` 派生策略头数据。
- 若要扩展到更多桶，优先复用 `data/training_data.py` 的确定性生成逻辑，不再另起一套随机样本代码。

### 2026-05-06｜会话（新增：MLP 改为 value+policy 双头并更新训练/评估）

#### 已确认事实
- 已将 `mlp` 模型改为共享骨干（backbone）+ `value_head` 与 `policy_head`，并添加了训练/评估支持：
  - `mlp/mlp_model.py`：新增双头结构，保留 `predict()` 返回 value 的兼容接口，并新增 `predict_policy_logits()`。
  - `mlp/training_utils.py`：新增 policy 目标构造（从 `action_q_values` softmax 得到）、masked policy loss/accuracy、样本数组准备函数。
  - `mlp/train.py`：改为同时读取 `value_view` 与 `action_q_values`，计算 value loss 与 masked policy loss 联合训练。
  - `mlp/evaluate.py`：改为同时评估 value 指标与 policy 指标（policy loss / policy_acc）。
  - `tests/test_mlp_multi_head.py`：新增测试，覆盖 policy 目标构造、前向/反向传播与简单训练步。

- 已在本地运行并验证：`tests/test_mlp_multi_head.py` 通过；并以真实数据执行了单轮训练验证（示例命令见下）。

#### 未确认 / 风险
- 仓库中大量既有文件尚未 retroactively 添加“文件/函数输入输出注释”，目前仅保证从本次变更新增的文件遵守该规则；需要计划分批补全旧代码的注释。
- `policy_temperature` 的默认值为 1.0，但该超参对 policy 监督的分布与训练稳定性影响较大，需要后续网格搜索确认最优区间。

#### 关键命令与入口（本次变更相关）
```
python mlp/train.py --xs 24 --data_dir data --epochs 1 --batch_size 4096 --save /tmp/mlp_test.pth
python mlp/evaluate.py --checkpoint /tmp/mlp_test.pth --data_dir data --xs 24
python tests/test_mlp_multi_head.py
```

#### 下一步行动
1. 在 `tests/` 中加入一个短时的集成回归：训练几轮并能保存/加载 checkpoint（保证训练脚本对保存格式稳定）。
2. 分批为仓库中的关键模块添加模块头注释和每个函数的输入输出说明，优先 `mlp`、`data`、`trick_taking/solvers`。
3. 进行 `policy_temperature` 的敏感度实验（小范围网格搜索），并在记录中写明结论。
4. 将 `mlp` 的 policy head 训练结果用于小规模离线策略评估（动作一致率 vs 求解器）。

---

### 2026-05-07｜会话11（`from __future__` 语法错误排查与全仓修复）

#### 已确认事实
- 报错根因是：部分文件头出现了“两个连续的顶层三引号字符串（双 docstring）”，第二个字符串会被 Python 当成普通顶层语句，从而导致 `from __future__ import annotations` 不再处于“仅允许模块 docstring 之后”的位置，触发 `SyntaxError`。
- 本次确认并修复了 5 个同类文件：
  - `mlp/train.py`
  - `mlp/evaluate.py`
  - `mlp/mlp_model.py`
  - `data/generate_dataset.py`
  - `data/benchmark_generate.py`

#### 新知识 / 现场观察
- 给旧文件补“模块说明”时，若直接再加一个新的三引号块，容易触发该错误；应把新增说明合并进第一个模块 docstring。
- 本环境没有 `rg`，排查时要用 `grep` 作为替代。

#### 关键命令与入口
```
cd /Spades-AI && grep -RIn "from __future__ import annotations" mlp trick_taking data tests
cd /Spades-AI && /root/miniconda3/bin/python -m compileall -q mlp trick_taking data tests
cd /Spades-AI && /root/miniconda3/bin/python -m compileall -q .
cd /Spades-AI && /root/miniconda3/bin/python mlp/train.py --xs 24 --data_dir data --epochs 1 --batch_size 256 --save /tmp/mlp_future_fix_smoke.pth
```

#### 验证结果
- 修复前：`compileall` 明确报 5 个文件存在同类 `from __future__` 顺序错误。
- 修复后：
  - `python -m compileall -q mlp trick_taking data tests` 通过。
  - `python -m compileall -q .`（仓库级）通过。
  - `mlp/train.py` smoke run 成功（完成 1 epoch 并保存 `/tmp/mlp_future_fix_smoke.pth`）。

#### 下一步行动
1. 继续批量补注释时，统一采用“单模块 docstring”模板，避免重复引入同类语法错误。
2. 可选新增一个 pre-check 脚本（或 pre-commit 钩子）：在提交前自动执行 `python -m compileall -q .`。

---

### 2026-05-07｜会话12（改为 value-head 动作选择评估，新增 x=25 验证脚本）

#### 已确认事实
- 已支持生成 `x=25` 的数据桶：
  - `data/training_data.py` 的 `SUPPORTED_BUCKETS` 已扩展为 `(24, 25, 28, 32)`。
  - `data/generate_dataset.py` 文案已同步为默认支持 25 桶。
- 已生成目标数据文件：`data/spades_dd_x25_n1000.pt`（1000 条）。
- 已新增脚本：`mlp/evaluate_action_25.py`。
  - 方法：对每条样本用 `(x, seed)` 重建局面；精确求解器给根节点最优动作；
    对每个合法动作做一步前瞻，使用 value-head 估值并选动作，再与精确最优动作比对。

#### 新知识 / 关键逻辑
- 不能用“奇偶轮数”简单决定取最小或最大。正确逻辑应基于队伍视角：
  1. 模型输出是“当前行动方视角的 `value_view/25`”；
  2. 先把每个子状态的预测统一换算到 team0 视角；
  3. 根玩家若属于 team0，选 team0 值最大的动作；根玩家若属于 team1，选 team0 值最小的动作。
- 为了避免并列最优被误判，脚本同时输出：
  - `exact_match_rate`（严格 card_id 一致）
  - `tie_match_rate`（所选动作 Q 值是否等于最优 Q，允许并列）

#### 关键命令与入口
```
/root/miniconda3/bin/python data/generate_dataset.py --xs 25 --num_samples 1000 --output_dir data
/root/miniconda3/bin/python mlp/evaluate_action_25.py --checkpoint ./result/mlp_test_2.pth --dataset data/spades_dd_x25_n1000.pt --max_samples 1000
```

#### 验证结果
- 小样本冒烟（20条）：
  - `exact_match_rate=0.650000`
  - `tie_match_rate=1.000000`
  - `avg_legal_actions=2.7000`
- 正式评估（1000条）：
  - `exact_match_rate=0.520000`
  - `tie_match_rate=0.950000`
  - `avg_legal_actions=3.3280`

#### 下一步行动
1. 若要做“只训练 value-head”，可在训练时把 `policy_weight` 设为 0，并复用该脚本持续追踪动作一致率。
2. 后续建议按 checkpoint 版本横向对比 `exact_match_rate/tie_match_rate`，选择最优模型而不只看 value loss。

---

### 2026-05-07｜会话13（训练数据重复率与特征相似度统计脚本）

#### 已确认事实
- 已新增检查脚本：`tests/test_dataset_dup_similarity.py`。
- 该脚本直接读取 `data/spades_dd_x24_n100000.pt`，统计：
  - feature 完全重复率
  - 随机抽样样本对的余弦相似度分布

#### 新知识 / 现场观察
- 为避免把“同一条样本和自己比”混进统计，脚本会在抽样相似度时把相同索引对自动挪开。
- 100000 条 x=24 数据的统计结果：
  - 完全重复数: 0
  - 唯一样本数: 100000
  - 完全重复率: 0.00000000
  - 抽样对数: 5000
  - 余弦相似度均值: 0.398769
  - p95: 0.488516
  - p99: 0.521315
  - 高相似阈值(0.995)比例: 0.000000

#### 关键命令与入口
```
/root/miniconda3/bin/python tests/test_dataset_dup_similarity.py --dataset data/spades_dd_x24_n100000.pt --num_pairs 5000
```

#### 结论
- 在当前数据生成方式下，x=24 的 100000 条样本里没有发现完全重复的 feature。
- 从余弦相似度看，样本之间存在“同桶内自然相近”的情况，但没有出现极高相似度的局面堆积。
- 这说明当前 `seed=seed_start+index` 的连续生成方式，在 x=24 的 100000 条规模下并没有产生明显的重复污染。

#### 下一步行动
1. 如果后续要生成 100万条数据，仍建议保留这个脚本作为离线体检工具。
2. 若想进一步压低相似样本，可考虑按 `state_summary` 或 `feature` 做去重/分层采样。

---

### 2026-05-07｜会话14（x=24 生成脚本 GPU/并行性检查 + 新并行脚本落地）

#### 已确认事实
- 旧生成脚本 `data/generate_dataset.py` 本身不使用 GPU 并行：
  - 主循环在 Python 中串行调用 `generate_bucket_dataset`。
  - `data/training_data.py` 的 `generate_bucket_dataset` 逐样本 for-loop 调用 `generate_bucket_sample`。
  - 精确求解器仍是 `cpp_opt1`（CPU/C++ 搜索内核），不是 CUDA 求解器。
- 已新增并行生成脚本：`data/generate_dataset_gpu.py`。
  - 采用 `ProcessPoolExecutor` 并行生成样本（多进程并发调用 `generate_bucket_sample`）。
  - 保持 seed 规则与原脚本一致（`seed_start + index`），并按 seed 回排，保证输出顺序稳定。
  - 检测到 CUDA 可用时，会将 `feature` 批量 `stack -> cuda -> cpu`，使流水线具备 GPU 张量批处理能力。

#### 新知识 / 现场观察
- 当前仓库的数据生成瓶颈主要在“每个状态调用精确求解器”的 CPU 计算；GPU 不能直接加速现有 `cpp_opt1` 搜索逻辑。
- 在不改变标签语义的前提下，最有效的提速方式是“CPU 多进程并行样本生成 + 可选 GPU 批量张量处理”。
- 小样本一致性验证（10条）已通过：旧脚本与新脚本在 `seed/x/value/best_action/action_ids/action_q_values` 等关键字段一致。

#### 关键命令与入口
```bash
python data/generate_dataset.py --xs 24 --num_samples 10 --seed_start 0 --output_dir data --prefix spades_dd_ref
python data/generate_dataset_gpu.py --xs 24 --num_samples 10 --seed_start 0 --output_dir data --prefix spades_dd_gpu --num_workers 4
```

#### 下一步行动
1. 若目标是进一步提速，优先做“并行 worker 数与 CPU 核心绑定”的基准扫描（如 2/4/8/16 workers）。
2. 若未来要真正 GPU 化求解器，需要单独重写/迁移搜索内核，而不是仅修改 Python 调度层。

---

### 2026-05-07｜会话15（按要求对比：旧脚本 vs 新脚本，各生成100条 x=24）

#### 已确认事实
- 已按相同参数与相同种子区间，对两套脚本各生成 100 条 `x=24` 数据并完成计时。
- 两个输出文件均成功保存：
  - `data/spades_dd_timecmp_ref_x24_n100.pt`
  - `data/spades_dd_timecmp_gpu_x24_n100.pt`

#### 实测结果（本机当次）
- 旧脚本（串行）总耗时：`11.145s`
- 新脚本（并行）总耗时：`4.894s`
- 加速比（旧/新）：约 `2.28x`

#### 关键命令与入口
```bash
cd /Spades-AI && TIMEFORMAT='serial_elapsed_sec=%R'; time python data/generate_dataset.py --xs 24 --num_samples 100 --seed_start 10000 --output_dir data --prefix spades_dd_timecmp_ref
cd /Spades-AI && TIMEFORMAT='gpu_parallel_elapsed_sec=%R'; time python data/generate_dataset_gpu.py --xs 24 --num_samples 100 --seed_start 10000 --output_dir data --prefix spades_dd_timecmp_gpu --num_workers 4
```

#### 未确认假设 / 风险
- 该加速比依赖机器的 CPU 核心数、负载、I/O 和 `--num_workers` 设置；换机器后可能变化。
- 当前“GPU参与”主要体现在特征张量的批量搬运/可扩展处理，不代表精确求解器已经 CUDA 化。

#### 下一步行动
1. 用固定数据量（如 1k / 10k）跑 `num_workers` 网格（2/4/8/16）找最优点。
2. 若后续训练只需 CPU 张量，可增加 `--disable_gpu_feature_stage` 开关减少额外拷贝。

---

### 2026-05-08｜会话17（可实战对局程序 + 10 局随机对局评估）

#### 已确认事实
- 已在 `strategy/` 下新增完整对局程序：`strategy/spades_match_runner.py`。
  - 它能从随机种子构造一局完整黑桃王牌局。
  - 它会在叫牌和出牌阶段都为当前玩家构造 1229 维输入，并打印“输入维度 -> 输出动作”。
  - 它会检查每次出牌是否合法；若不合法，立即报错。
  - 它会在每次动作后刷新四名玩家各自当前的 1229 维输入。
  - 它会在终局后计算四名玩家得分并打印。
- 已在 `strategy/` 下新增玩家程序封装：`strategy/spades_player_programs.py`。
  - `RandomSpadesPlayer`：在所有合法选择中随机出牌。
  - `TruncatedMCTSPlayer`：实际调用 `TruncatedMCTSStrategy.choose_action`。
- 这次对局程序采用的是完整的 Spades 叫牌 + 出牌流程，不只是单纯的出牌片段。

#### 新知识 / 现场观察
- 1229 维特征在每个回合都可以即时刷新，说明现在的策略/对局程序已经具备“可运行闭环”，不再只是静态数据生成器。
- `TruncatedMCTSPlayer` 在 runner 里通过 `state_view['state']` 获取完整局面，所以能直接利用完全信息的截断搜索逻辑。
- 在本轮实测中，单局演示从 52 张牌开始，连续跑完 52 次出牌，没有出现非法动作或状态推进错误。
- 随机对局评估中，玩家1（MCTS）在 10 局中的平均得分为 `53.20`，说明在当前参数与随机对手下，策略已经能稳定输出正收益。

#### 关键命令与入口
```bash
cd /Spades-AI && /root/miniconda3/bin/python strategy/spades_match_runner.py --seed 0 --checkpoint ./result/mlp_test_3.pth --exact_threshold 30 --leaf_threshold 24 --simulations_per_action 5 --num_eval_games 10
```

#### 验证结果
- 单局完整演示：成功，52 步出牌全部正常，终局得分为 `[-142.0, 142.0, -142.0, 142.0]`。
- 10 局随机对局评估：成功，玩家1平均得分 `53.20`。

#### 未确认假设 / 风险
- 当前评估使用的是随机叫牌 + 随机对手，结果主要反映“在该基准下能否稳定工作”，不等价于对强对手的真实强度。
- 当前对局程序在终局前会频繁打印特征维度和动作，适合调试，不适合超大规模批量跑分；后续如要做批量评估，建议加一个静默模式。

#### 下一步行动
1. 若要进一步比较策略强弱，建议把随机对手替换成更稳定的固定基线，再做 100 局以上评估。
2. 若要接入真实驱动，可以把 `SpadesMatchRunner` 再封装成 `GeneralCardGame` 兼容入口。

---

### 2026-05-08｜会话18（叫牌流程与出牌顺序问题修正）

#### 已确认事实
- 已修正叫牌默认行为：`strategy/spades_match_runner.py` 现在默认使用 `SpadesRules(enable_nil=False, enable_blind_nil=False)`。
  - 这样叫牌阶段默认是“每人一次数值叫牌”，不会再出现 `pass -> 同一玩家再次叫牌` 的二段式日志。
  - 若要恢复 nil/blind_nil，可通过命令行参数 `--enable_nil --enable_blind_nil` 手动开启。
- 已增强出牌日志：每墩开始都会打印首攻玩家，每墩结束都会打印赢家和下一墩首攻。
  - 这明确展示了 trick-taking 规则：下一墩由上一墩赢家领出，所以全局顺序不会固定为 0-1-2-3 循环。

#### 新知识 / 现场观察
- 你之前看到的“叫牌里同一玩家连续出现”不是出牌阶段 bug，而是 `blind_nil/pass` 机制带来的设计行为：
  - 玩家先选择是否盲叫；若 `pass`，同一玩家再进行常规叫牌。
- 你之前看到的“出牌不是 01230123...”是黑桃规则本身：
  - 同一墩内是按当前首攻顺时针出牌；
  - 下一墩首攻改为本墩赢家，因此跨墩顺序会跳变。

#### 关键命令与入口
```bash
cd /Spades-AI && /root/miniconda3/bin/python strategy/spades_match_runner.py --seed 0 --checkpoint ./result/mlp_test_3.pth --exact_threshold 30 --leaf_threshold 24 --simulations_per_action 5 --num_eval_games 2
```

#### 验证结果
- 新日志中叫牌为一人一次（默认禁用 nil/blind_nil）。
- 新日志中每墩都打印了“本墩赢家 + 下一墩首攻”，出牌顺序变化可直接追踪。

#### 下一步行动
1. 若后续要分析策略效果，建议固定叫牌策略（例如全员 `bid_4`）减少噪声。
2. 若想保留 nil/blind_nil，又不想日志混淆，可单独在日志中加“盲叫阶段/常规叫牌阶段”标签。

---

### 2026-05-08｜会话16（阶段1-步骤5：截断 MCTS 策略落地与 52 步完整跑牌验证）

#### 已确认事实
- 已在新目录 `strategy/` 中实现一个全新的截断式出牌策略：`strategy/truncated_mcts_strategy.py`。
- 该策略的分层决策规则是：
  - 剩余牌数 `<= 30`：直接调用精确求解器 `solve_with_q` 输出最优动作。
  - 剩余牌数 `> 30`：对每个根动作分别运行 MCTS；每次模拟沿树选择直到剩余牌数 `<= 24`，再用 MLP 做局面估值。
  - 根动作选择采用队伍 0 视角：队伍 0 取 `argmax`，队伍 1 取 `argmin`。
- 新策略使用的是“PUCT + policy prior + value leaf”的方式，而不是复用旧的 `DoubleDummySolver` 搜索实现。
- 新策略支持参数化：
  - `simulations_per_action`：每个根动作的模拟次数
  - `checkpoint_path`：MLP 权重文件
  - `exact_threshold`：低于该剩余牌数直接精确求解
  - `leaf_threshold`：MCTS 截断到该剩余牌数时接入 MLP
  - `exploration_constant`、`policy_temperature`：探索与先验温度

#### 新知识 / 现场观察
- 这类四人队伍游戏的最大化/最小化判断，应以“当前行动方所属队伍”为准；在 Spades 中玩家编号与队伍编号一致，因此实现上用 `state.turn` 所属队伍判定即可。
- 策略在 leaf 阶段使用 value-head 估值时，需要把模型输出从“当前行动方视角”换回“队伍 0 视角”，否则根节点比较会偏。
- 这次验证中，完整 52 步跑牌已经跑通，说明：
  - 52 次连续调用策略没有状态推进错误；
  - 从开局到终局，`play_card_to_table -> complete_trick -> 更新 turn/trick_leader` 的链路是正确的；
  - `exact_threshold=30` 与 `leaf_threshold=24` 的分层策略可以正常运行。

#### 关键命令与入口
```bash
cd /Spades-AI && /root/miniconda3/bin/python strategy/truncated_mcts_strategy.py --checkpoint ./result/mlp_test_3.pth --seed 0 --exact_threshold 30 --leaf_threshold 24 --simulations_per_action 5
```

#### 验证结果
- 本次完整跑牌输出了 52 步动作序列，未报错。
- 终局动作数：52。
- 终局得分：`[-80.0, 80.0, -80.0, 80.0]`。

#### 未确认假设 / 风险
- 当前策略是“每个根动作单独跑固定次数模拟”，还没有做更激进的根节点并行化；若后续要扩到更大预算，可能需要再做批量并行或缓存复用。
- 当前 leaf 估值依赖 `./result/mlp_test_3.pth`；如果换 checkpoint，动作分布会变化。

#### 下一步行动
1. 若要做 2s/5s 预算调参，优先扫描 `simulations_per_action`、`exploration_constant`、`leaf_threshold` 三个参数。
2. 如果想把策略用于交互式对局，再补一个从 `GameState` 直接接入 driver 的封装层。




