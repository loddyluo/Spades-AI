rule_based_v2被写在/home/oier/wyy/Spades-AI/Spades_AI_GO-MCTS/spades_ai/players/rule_based_v2里，被用作MCTS搜索时动作的先验概率生成。你需要注意这个模型的输入的数据格式（如何表示局面状态）. 现在问题出在这个oracle没有被正确加载，truncated_mcts_strategy.py代码出现问题了。

您需要一直修改问题（只修改truncated_mcts_strategy.py），直到print("!!!!!!!! Chosen_prior_is_None !!!!!!!")不被输出为止。我的测试的bash命令是python evaluate/evaluate_our_mcts_vs_rule_v2.py --seed 990 --num-games 1 --num-workers 15 --torch-num-threads 1 --torch-num-interop-threads 1 --trace-log-dir logs --symmetric-seat-swap 1 --p0 our_mcts --p1 go_rule_2 --p2 our_mcts --p3 go_rule_2 --our-checkpoint mlp_test_3.pth

现在有一个仓库，其中的our_mcts是我们的黑桃王人工智能。黑桃王是一种牌类游戏。

目前在还剩＞24张牌时，使用MCTS与PUCT混合的办法。在还剩≤24张牌时，使用精确求解器。这都是使用了所有人的手牌信息。但是，现在除了自己以外的三个玩家的剩余牌都是随机采样的，这不够好。

您需要做的修改是：对于另外三个玩家的剩余牌的情况，进行重要性采样。假设现在全部玩家还有 $r$ 张牌没有出。具体地，对于一种初始的牌的分布（每个玩家13张牌，我自己的牌是固定的，另三个人的已出过的牌也一定是在初始牌中的），我们按照已经进行过的游戏的过程（之前打的 $52-r$ 张牌每一张都算一个步骤）来计算打到现在这个局面的概率 $p=p_1p_2...p_{52-r}$，每个 $p_i$ 如果是当时的一种合法操作，那么其等于 $\frac{1}{D}$，$D$ 是当时的合法操作的总数；如果不是当时的合法操作，$p_i=0, p=0$。（这里概率计算是先随机确定初始的牌的分布（每个玩家13张牌），然后依次计算 $p_1,p_2,...,$. 当计算 $p_k$ 时，前 $k$ 张牌按照“轨迹”已经出出来了。无论初始的牌的分布是什么，轨迹都是那一条）

你需要做的是【每一次想要采样除了自己以外的三个玩家的剩余牌前】，都采样1234种【初始的牌的分布】，然后对它们开展重要性采样，【加权随机】选出 $x$ 种（$x$视具体情况而定）初始的牌的分布，把这作为另三个玩家的剩余牌。

上面的已经写完了，使用了our_mcts和go_rule相互对打，其中go_rule是Spades_AI_GO-MCTS/spades_ai/players/rule_based中的。现在，我们有了Spades_AI_GO-MCTS/spades_ai/players/rule_based_v2，请模仿/home/oier/wyy/Spades-AI/evaluate/evaluate_model_matchups.py写一个评测our_mcts和go_rule_2相互对打的脚本。复制evaluate_model_matchups.py，别直接在上面修改。