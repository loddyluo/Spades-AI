## 核心概念

PUCT
exact求解
mcts_determinization_count: int = 4 
千万要注意，不能if self.model is None:

vs_2mcts.py是目前主要的人机交互。

手牌格式：SA S5 S4 DT D6 D3 D2 H5 H4 H2 C8 C7 C3
         SK SQ SJ S3 DA HT H9 H8 H7 CJ C5 C4 C2

#### 读log

logs/matchup_trace_seed1_games200_20260518_101402.txt 第57、58、59、60把

## 笔记
 
 命令执行流程                                                                              
                                                                                            
  这条命令做的事情：运行 2 局 Spades 牌戏，每局用不同的座位排列跑两次（对称交换），共 4     
  局对局，使用 2 个 worker 进程并行。                                                       
                  
  座位分配                                                                                  
                  
  ┌────────┬──────────┬────────────────────────────────────┐                                
  │  座位  │ 玩家类型 │               实现类               │
  ├────────┼──────────┼────────────────────────────────────┤                                
  │ Seat 0 │ our_mcts │ OurHandStrengthMCTSPlayer          │
  ├────────┼──────────┼────────────────────────────────────┤                                
  │ Seat 1 │ go_rule  │ GoPlayerAdapter(RuleBasedPlayer()) │                                
  ├────────┼──────────┼────────────────────────────────────┤                                
  │ Seat 2 │ our_mcts │ OurHandStrengthMCTSPlayer          │                                
  ├────────┼──────────┼────────────────────────────────────┤                                
  │ Seat 3 │ go_rule  │ GoPlayerAdapter(RuleBasedPlayer()) │
  └────────┴──────────┴────────────────────────────────────┘                                
                  
  Symmetric-seat-swap 会把座位 (0,1,2,3) 翻转为 (1,0,3,2)，所以每个 base seed 跑 2 次对局。2
   个 base seeds × 2 = 4 实际对局。
                                                                                            
  调用了仓库中的以下 Python/C++ 程序（按调用链）：                                          
   
  evaluate/evaluate_model_matchups.py          ← 入口                                       
  ├── strategy/spades_match_runner.py          ← 对局引擎 (SpadesMatchRunner)               
  │   ├── trick_taking/games/spades.py         ← 规则 (叫牌/出牌合法性/计分)                
  │   ├── trick_taking/card.py                 ← 牌面表示                                   
  │   ├── trick_taking/deck.py                 ← 发牌                                       
  │   ├── trick_taking/game_state.py           ← 游戏状态                                   
  │   └── trick_taking/utils/feature_encoder.py ← 1229维特征编码                            
  ├── evaluate/GO-MCTS/adapters.py              ← 玩家适配器                                
  │   ├── strategy/hand_strength.py             ← 手牌强度启发式（叫牌）                    
  │   ├── strategy/truncated_mcts_strategy.py   ← MCTS 出牌策略 ← ★ 最耗时                  
  │   │   ├── mlp/mlp_model.py                  ← MLP 价值/策略网络                         
  │   │   └── trick_taking/solvers/                                                         
  │   │       ├── exact_double_dummy_cpp_opt1.py        ← C++ solver 包装                   
  │   │       └── _exact_double_dummy_cpp_opt1_core.so  ← C++ alpha-beta 搜索               
  │   └── evaluate/GO-MCTS/bridge.py            ← 状态格式转换                              
  └── evaluate/GO-MCTS/models.py                ← go_rule 玩家（纯规则，很快）              
                                                                                            
  ---                                                                                       
  为什么这么慢？四个主要原因                                                                
                                                                                            
  1. MCTS 模拟量巨大（最主要原因）
                                                                                            
  默认 --our-simulations-per-action 5000没有被覆盖，所以每个 our_mcts                       
  玩家每次出牌决策都运行：                                                                  
                                                                                            
  每个合法动作 × 5000 次模拟 = 约 30,000 次模拟/每次出牌
                                                                                            
  truncated_mcts_strategy.py:220-224 显示了对每个合法动作跑 5000 次模拟的循环：             
                                                                                            
  for action in legal_actions:                 # 约 3-8 个合法动作                          
      child_node = _build_root_child(...)                                                   
      for _ in range(self.config.simulations_per_action):  # 5000 次                        
          self._run_simulation(child_node, ...)                                             
                                                                                            
  每次模拟都做：                                                                            
  - copy.deepcopy(state) — 深拷贝整个游戏状态                                               
  - _determinize_state() — 随机重发所有对手手牌（洗牌 + 分配）                              
  - PUCT 向下遍历 3-7 层                                      
  - 到达 leaf_threshold=24 时调用 MLP 推理（如果加载了 checkpoint）                         
                                                                                            
  总计： 全场约 52 次出牌决策 × 2 个 MCTS 玩家 × 30,000 模拟 = ~300 万次模拟（4局）         
                                                                                            
  2. 精确求解阶段每步跑 50 次 Determinization                                               
                                                                                            
  当剩余手牌 ≤ 24 张时（后约 6 墩），进入精确求解模式。在                                   
  truncated_mcts_strategy.py:401-410：
                                                                                            
  def _solve_with_determinization(self, state):
      for _ in range(self.config.determinization_count):  # 50 次！                         
          sim_state = copy.deepcopy(state)                                                  
          self._determinize_state(sim_state, observer, rng)  # 重发对手手牌                 
          res = self.exact_solver.solve_with_q(sim_state)     # C++ 全树搜索                
                                                                                            
  每次 solve_with_q 都调用 C++ exact_double_dummy_cpp_opt1_core.cpp 中的 完整 alpha-beta    
  极小极大搜索（带转置表）。对于 24 张牌的残局，搜索空间仍可能达到数百万节点。而且注意 C++  
  代码的 solve_with_q（第 405 行）对每个合法动作都清空转置表再重新搜索，浪费了共享计算。    
                  
  3. 每步大量的 copy.deepcopy 放大开销                                                      
   
  关键路径上到处都是 copy.deepcopy：                                                        
  - 每次 MCTS 模拟开头（truncated_mcts_strategy.py:285）
  - 每次 determinization 采样的开头（truncated_mcts_strategy.py:411）                       
  - C++ 层内的 NativeState 拷贝（cpp:338）                           
  - 每次 apply_action 做 _deep_copy_state（exact_double_dummy.py:296-337）                  
                                                                                            
  一个 GameState 含 hands[4][hand], bitsets[4], table_cards, trick_history, bids,           
  tricks_won[4] 等多个 Python 对象，深拷贝开销可观。                                        
                                                                                            
  4. 默认参数组合未经调优                                                                   
                  
  命令行指定的参数中：                                                                      
  - --our-simulations-per-action 未指定 → 默认 5000（对 MCTS 来说异常高，通常 50-500 即可）
  - --our-number-of-exact-solvers 50 → 每步精确决策跑 50 次完整搜索                         
  - --our-exact-threshold 24 和 --our-leaf-threshold 24 相同 → MCTS 正好在 24 张处截断用 MLP
                                                                                            
  粗略估算每步耗时                                                                          
                                                                                            
  MCTS 阶段（52→24张，约28步每局）：                                                        
    每个 our_mcts 决策: 6动作 × 5000模拟 × (3μs深拷贝 + 5μs确定化 + ...) ≈ 几百毫秒到几秒   
                                                                                            
  精确求解阶段（24→0张，约24步每局）：                                                      
    每个决策: 50个样本 × 每个样本5个动作的C++全树搜索 ≈ 可能秒级                            
                                                                                            
  两个 our_mcts 玩家各做约 52 次决策，加上 50-determinization × 50 exact-solve 的精确阶段，4
   局对局的总时间可以轻松达到 数十分钟到数小时。 




   分界点确认                                                                                
   
  在 truncated_mcts_strategy.py:198：                                                       
                  
  if remaining_cards <= self.config.exact_threshold:  # ≤ 24                                
      exact_result = self._solve_with_determinization(state)                                
      # 走精确求解（C++ alpha-beta 极小极大）                                               
  else:  # ≥ 25                                                                             
      # 走 MCTS + PUCT 搜索                                                                 
                                                                                            
  一整局牌 52 张从头出到尾，同一次运行里的同一个模型会经历：                                
  - 52→25 张：每个决策 ≈ actions × 5000 次 MCTS 模拟，每次模拟用 PUCT 选子节点，到          
  leaf_threshold=24 时用 MLP 估值                                                           
  - 24→0 张：每个决策 ≈ 50 次 determinization，每次调 C++ alpha-beta 精确求解器 算到终局
                                                                                            
  PUCT 全称与原理                                                                           
                                                                                            
  全称：Polynomial Upper Confidence Trees（多项式上置信区间树）                             
                                                                                            
  来自：AlphaGo Zero 论文（Silver et al., 2017），是对传统 UCT（Upper Confidence bounds     
  applied to Trees）的改进。
                                                                                            
  核心公式（你在 truncated_mcts_strategy.py:445-466 看到的 _select_child_puct）：           
   
  score = sign × Q(s, a) + C × P(s, a) × √N_parent / (1 + N_child)                          
                                                                                            
  其中：                                                                                    
  - Q(s, a) — 当前动作的平均价值（团队0视角，所以团队1取负号）                              
  - sign — 团队0取 +1，团队1取 -1（因为双方目标相反）                                       
  - C — 探索常数（PUCT 系数），你设的 --our-exploration-constant 1.5
  - P(s, a) — 先验概率，当前代码使用均匀分布（_policy_priors 里 1/len(legal_actions)）      
  - N_parent — 父节点访问次数                                                               
  - N_child — 子节点访问次数                                                                
                                                                                            
  直觉理解：PUCT 在选子节点时，既要选"目前看来好的"（Q 值大），也要选"还没怎么探索过的"（C ×
   P × √N_parent/(1+N_child) 大）。前者是 exploitation，后者是 exploration。                
                                                                                            
  和你这个代码的关系：你们的 MCTS 是"根节点下每个动作各自独立建一棵搜索树"，PUCT            
  负责在每棵树的 descent 过程中选路径。注意你们的 policy prior 是均匀的（代码注释说因为
  policy head 只在残局训练过，全局阶段怕用错），所以 PUCT 退化为 Q 值 + 频率驱动的探索，没有
   AlphaZero 那种"神经网络指导先验"的效果。

### 5.17 笔记

● exploration_constant 的含义                                                                 
                                         
  PUCT 公式中，exploration_constant（记作 $c_{\text{PUCT}}$）控制 探索 与 利用 之间的权衡：   
   
  $$\text{score} = \text{sgn} \cdot Q + c_{\text{PUCT}} \cdot P \cdot                         
  \frac{\sqrt{N_{\text{parent}}}}{1 + N_{\text{child}}}$$   
                                                                                              
  - $Q$ 项（利用）：选择已知胜率高的动作                                                      
  - $c_{\text{PUCT}} \cdot P \cdot \sqrt{N} / (1 + N_{\text{child}})$ 
  项（探索）：选择访问次数少的动作，让搜索去尝试未知区域                                      
                                                            
  $c_{\text{PUCT}}$ 越大 → 探索项权重越大 → 算法更愿意尝试低频动作                            
  $c_{\text{PUCT}}$ 越小 → 利用项主导 → 算法更倾向于选当前 Q 值最高的动作
                                                                                              
  ---                                                                                         
  举个例子                                                                                    
                                                                                              
  假设在某个黑桃局面下，合法的动作有 ♠A 和 ♥2，两个子节点的统计如下：
                                                                                              
  ┌──────┬──────┬────────────────────┬─────────────┐                                          
  │ 动作 │ $Q$  │ $N_{\text{child}}$ │ $P$（先验） │                                          
  ├──────┼──────┼────────────────────┼─────────────┤                                          
  │ ♠A   │ +0.8 │ 100                │ 0.5         │        
  ├──────┼──────┼────────────────────┼─────────────┤                                          
  │ ♥2   │ +0.1 │ 1                  │ 0.5         │                                          
  └──────┴──────┴────────────────────┴─────────────┘                                          
                                                                                              
  父节点访问次数 $N_{\text{parent}} = 101$，当前是 team 0（$\text{sgn}=+1$）。                
                                                            
  情况 A：$c_{\text{PUCT}} = 0.1$（几乎不探索）                                               
                                                            
  - ♠A：$0.8 + 0.1 \times 0.5 \times \sqrt{101} / (1+100) = 0.8 + 0.05 \times 10.05 / 101 =   
  0.8 + 0.005 = 0.805$                                      
  - ♥2：$0.1 + 0.1 \times 0.5 \times \sqrt{101} / (1+1) = 0.1 + 0.05 \times 10.05 / 2 = 0.1 + 
  0.251 = 0.351$                                                                              
  
  → 选 ♠A，几乎纯利用。                                                                       
                                                            
  情况 B：$c_{\text{PUCT}} = 1.5$（默认值）                                                   
                                                            
  - ♠A：$0.8 + 1.5 \times 0.5 \times 10.05 / 101 = 0.8 + 0.075 = 0.875$                       
  - ♥2：$0.1 + 1.5 \times 0.5 \times 10.05 / 2 = 0.1 + 3.769 = 3.869$
                                                                                              
  → 选 ♥2，探索项占压倒性优势，因为 ♥2 只被访问过 1 次，分母 $(1+N_{\text{child}})$ 极小。    
                                                                                              
  情况 C：$c_{\text{PUCT}} = 0.3$（go_mcts 的默认值）                                         
                                                            
  - ♠A：$0.8 + 0.3 \times 0.5 \times 10.05 / 101 = 0.8 + 0.015 = 0.815$                       
  - ♥2：$0.1 + 0.3 \times 0.5 \times 10.05 / 2 = 0.1 + 0.754 = 0.854$
                                                                                              
  → 仍然选 ♥2，但差距缩小。随着 ♥2 被多访问几次，探索项衰减，$Q$ 值也会更新，最终收敛。       
                                                                                              
  ---                                                                                         
  关键直觉                                                  
                                                                                              
  $c_{\text{PUCT}}$ 的本质是：在看到一个动作的可靠 $Q$ 估计之前，愿意多探索多少次？
                                                                                              
  - 默认值 1.5 偏探索——适合搜索空间大、需要广泛尝试黑桃对战的场景                             
  - go_mcts 的 0.3 更保守——适合与先验模型配合时，信任先验更多                                 
  - 调大 $c_{\text{PUCT}}$ → 搜索更广但单条路径变浅（固定的 simulation budget 被摊薄）        
  - 调小 $c_{\text{PUCT}}$ → 搜索更深但更窄，可能错过 $Q$ 暂时不高但实际更好的动作   