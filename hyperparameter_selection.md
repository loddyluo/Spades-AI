【第一届Scaling Law杯超参数选拔赛规程】

允许用户在一个.yaml文件中设置自己想要的超参数。让具有不同的超参数的两个rule_exact_first4_player对打。

### 赛制 

整体上，共有 $8$ 个超参数参与，采用单败淘汰赛，共 $3$ 轮 $7$ 场比赛。其中，有 $2$ 个超参数是种子选手。
每两个超参数之间的比赛采用队式赛，对打 $50$ 把，每把 $2$ 副牌（交换位置）。记分为100副牌的平均分，胜者进入下一轮。
种子段 $8880000-8880049$。

python evaluate/evaluate_hyperparam_matchup.py \
        --config-a configs/hyperparams_default.yaml \
        --config-b configs/1.yaml \
        --num-games 100 --seed 8880000 --num-workers 20 --trace-log-dir SCL

可以选择的超参数包括（后面是默认值）：
· num_proposals: int = 1234,
· num_proposals_limit: int = 5678,
· min_pool_size: int = 100,
· 以下计算budget:
if remaining_in <= 16:
            top_k, max_samples = 128, 256      # 最后3墩
        elif remaining_in <= 24:
            top_k, max_samples = 64, 128       # 倒数第5、4墩
        elif remaining_in <= 28:
            top_k, max_samples = 32, 64        # 倒数第7和第6墩
        elif remaining_in <= 32:
            top_k, max_samples = 16, 32
        else:
            top_k, max_samples = 12, 24
· 这一行的具体内容: weight *= (bad_count / total) if total > 0 else 1.0  # 坏动作 也是一个超参数，具体地，您可以假设bad_count/total=x，weight应该乘上的是x的一个函数
· 第830行if trick_num >= 8 and sim_state is not None:中的这个8也是一个超参数
· 是否要交换重要性采样和“关键牌不重复补全”。（如果交换，那么就先进行从高到低的遍历，按照关键牌不重复的原则选择，然后在剩余的牌中进行重要性采样，权重计算方法不变）Yes/No
· 第353行附近if multiplier < -40.0:
                                multiplier *= 1.0
                            agg_q[aid] = agg_q.get(aid, 0.0) + norm_w * multiplier
                        else:
                            multiplier = float(q) - min_q
                            if multiplier > 40.0:
                                multiplier *= 1.0
                            agg_q[aid] = agg_q.get(aid, 0.0) + norm_w * multiplier
    这里面的40.0和1.0也作为超参数，应该是可以选择的。

请写一个支持从两个.yaml文件中读入上述超参数，并支持两个具有不同超参数的rule_exact_first4_player对打的程序。