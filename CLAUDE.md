这是一个黑桃王AI的代码仓库。现在我们使用了前若干个回合MCTS的方法，得到了非常差的结果。以前运行的指令是：python evaluate/evaluate_cheat_mcts_vs_dds.py \
        --num-games 128 --seed 42 \
        --our-simulations-per-action 64 \
        --our-exact-threshold 36 --num-workers 32 --our-simulations-per-action 2048 --our-mcts-determinization-count 16 
这意味着前52-36=16张牌时，our_mcts是使用MCTS出牌的。后面36张牌使用精确求解器。

### 背景知识：队式赛

对于同一副牌，进行两局游戏。第一局游戏rl_exact坐在0、2位置，dds坐在1、3位置，设这时0、2位置减去1、3位置的分差是 $A$. 第2局游戏rl_exact坐在1、3位置，dds坐在0、2位置，设这时0、2位置减去1、3位置的分差是 $B$. 那么训练的学习内容是这样的两条轨迹：

轨迹1是第1局rl_exact在0、2位置做出的轨迹，奖励为 $A-B$.

轨迹2是第2局rl_exact在1、3位置做出的轨迹，奖励也为 $A-B$.

（仔细检查实现是否符合了这里的要求）

## 已完成任务
必须把前16张牌的出牌方式改成使用policy gradient方法进行强化学习的方式。训练大约10000局，让模型初步学会一个出牌方法。前16张牌的出牌不需要依赖任何采样，玩家只能观测到自己手里的牌和已经出过的牌的信息（被转化成一个高维向量）。（这将是一个新的玩家类型，不再是TruncatedMCTS，这个玩家类型叫做rl_exact）
【训练流程】
训练的时候，采取队式赛的方法，每次产生2局，一队是rl_exact，一队是dds（2局的牌是一样的，但是作为互换），以我方2人（RL+exact）两局得分之和减去对方两局得分之和作为reward（对方指的是spec == "dds"的玩家）。我方的RL+exact玩家前4墩（前16张牌）使用policy（一个MLP）出牌；后36张牌直接使用精确求解（可以看到对面的牌，出最优动作），和我们的policy无关。
【模型架构】
我方的RL所得的模型应该是一个MLP，输入状态向量，输出每个动作的logits。您必须注意的是，把牌局转化为状态向量的方式在代码仓库中已经存在（使用这个MLP时，玩家自己不允许偷看别人的牌！只能看到自己手里的牌和已经出过的牌）
【训练完之后可以得到什么】
一个checkpoint。之后当使用rl_exact的时候，前4墩（前16张牌）使用的是这个checkpoint来产生动作，剩下的墩和our_mcts的出牌方式是一样的。（必须注意，只有在测试时需要采样+精确求解，测试时不允许看到对方手里剩下的牌）
rl_exact的叫牌方式模仿/root/Spades-AI/evaluate/evaluate_cheat_mcts_vs_dds.py，使用这里面的这个checkpoint：    parser.add_argument("--bid-checkpoint", type=str,
                        default="./Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt")
【注意事项】
1. 不要只打2副牌就更新一次，多打几副再更新，减小随机波动。
2. 训练时是两个rl_exact（受训模型）打两个dds，两个rl_exact是同一个模型，要更新.
请写出相应的代码。
3. 回答我时，使用中文

## 开展训练
下面的指令必须能运行
```bash
python rl/train_rl_multicpu.py --num-games 100000 --seed 114514 --lr 0.000001  --num-workers 30 --update-interval 300 --num-epochs 1 --load-checkpoint ./rl_checkpoints/pretrain/pretrain_best.pt

python rl/pretrain_rl_multicpu.py --num-games 20000 --seed 42 --lr 0.001  --num-workers 32 --update-interval 120 --num-epochs 1 --entropy-coef 0.05
tensorboard --logdir runs/rl_train --port 6011
tensorboard --logdir runs/pretrain_rl --port 6011
```
现在的问题是rl了5000局以后，模型表现没有任何改进。您需要帮我找出问题出在哪？找出问题之后，运行python rl/train_rl.py --num-games 5000 --seed 123 --lr 0.001。（该指令运行不会超过10分钟）
【修改目标】RL初期应该有较明显的表现提升，因此，您必须使得第1~1000局的平均reward比第4001~5000局的平均reward大至少12分，而且这个目标必须对多个随机数种子都验证成立. 
请一直解决问题直到【修改目标】达成，如果【修改目标】没有被达成，接着修改，做实验，直到修改目标被达成。

python rl/pretrain_rl_multicpu.py --num-games 500000 --seed 42 --lr 0.0001 --num-workers 32 --update-interval 300 --num-epochs 1 --entropy-coef 0.05 --load-checkpoint ./rl_checkpoints/pretrain/pretrain_policy_final.pt


python rl/eval_rl_multicpu.py --num-games 1500 --seed 61 --num-workers 30 --load-checkpoint ./rl_checkpoints/best.pt

--load-checkpoint ./rl_checkpoints/pretrain/pretrain_best.pt



-1.09→-1.05