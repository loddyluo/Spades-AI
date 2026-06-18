【修改一】一个proposal的概率是叫牌阶段概率(bid_prod)和打牌阶段每一个动作的weight的乘积。
现在使用了简化的实现weight *= 1.0  # (uniform for simplicity)，每一步都是 $1.0$.
我们希望修改对于发生在第9、10、11、12、13墩的每个人的动作的weight的贡献。对于一个实际采取的动作 $a$，首先把当前的局面送入精确求解器（当前局面有不超过20张牌），精确求解器会得出每个动作的Q值，称Q值最大的动作（允许并列）为“好动作”，其余合法动作为坏动作，设有 $A$ 个好动作与 $B$ 个坏动作。
如果 $a$ 是好动作，那么这一步weight=$1$；
如果 $a$ 是坏动作，那么这一步weight×=$\frac{B}{B+A}$.
如果 $a$ 不合法，那么这一步weight×=$0$（直接return）。

帮我做这一修改。修改时注意两点：是否正确地把局面传给了精确求解器。以及精确求解器返回的等大牌的Q值怎么处理。

【修改二】
initial_hands = self._generate_proposal(
                state.all_cards, observer_id, state.hands[observer_id],
                played_by_player, rng,
            )
有些时候，如果一个人领出了一门花色，但是跟牌人没有出这门花色，意味着跟牌人剩下的牌里面不可能有这门花色。请修改上面的函数，把这一点考虑进去。必须考虑所有的“一个人没有一门花色”的情况，再进入bid_prod计算。

【修改三】
本来是取top-k再按照关键张不重复原则补全到max_samples。现在改成重要性采样top-k副牌（而不是按概率选前top-k个），再按照概率从大到小补全到max_samples副。