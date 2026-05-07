"""
黑桃王特征编码器

将GameState编码为~1229维特征向量，
用于MLP网络输入。

设计文档参考: feature_design.md
详细使用说明参考: usage.md (GameState字段说明)

模块函数说明（输入/输出）:

- SpadesFeatureEncoder.encode(state: GameState, player_id: int) -> np.ndarray
    输入: 完整的 `GameState` 实例 和 当前行动玩家 `player_id`
    输出: 一维 Numpy 数组, shape=(1229,), dtype=float32, 表示局面特征

- SpadesFeatureEncoder.encode_sections(state: GameState, player_id: int) -> dict
    输入: 同上
    输出: 字典, 每个键为分节名称, 值为对应分节的 numpy 数组（便于调试/消融）

其余私有方法返回内部中间结果（numpy 数组或标量），仅供编码器内部使用。
"""

from __future__ import annotations

import numpy as np

from trick_taking.card import Card, Suit, Rank
from trick_taking.game_state import GameState, Phase

# ---------------------------------------------------------------------------
# 常量：标准52张牌表，用于bitset→牌面转换
# ---------------------------------------------------------------------------
_STANDARD_CARDS: list[Card] = [Card(s, r) for s in Suit for r in Rank]


class SpadesFeatureEncoder:
    """
    黑桃王局面特征编码器

    将GameState转换为1229维特征向量，
    分为7个大类，包含原始信息和推导特征。

    说明：
    - 绝大多数维度仍为 one-hot / bit 特征
    - 仅新增“每张牌第几轮打出”使用了标量编码（-1~1），用于控制维度膨胀

    使用方式:
        encoder = SpadesFeatureEncoder()
        features = encoder.encode(state, player_id)  # ndarray shape=(1229,)

        # 或者获取分节编码（用于消融实验）
        sections = encoder.encode_sections(state, player_id)
    """

    # 各分类维度 (与 feature_design.md 严格对齐)
    DIM_HAND = 112
    DIM_BIDDING = 126
    DIM_CURRENT_TRICK = 73
    # 历史信息扩展：
    # - 原始历史 192 维
    # - 每张牌由谁打出（52 * 6 one-hot，类别: P0/P1/P2/P3/未出/未知）= 312
    # - 每张牌第几轮打出（52维标量，范围: -1~1）= 52
    DIM_HISTORY = 556
    DIM_SUIT_ANALYSIS = 192
    DIM_TEAM_SITUATION = 154
    DIM_GLOBAL_FLAGS = 16

    @property
    def total_dim(self) -> int:
        """总特征维度"""
        return (self.DIM_HAND + self.DIM_BIDDING + self.DIM_CURRENT_TRICK +
                self.DIM_HISTORY + self.DIM_SUIT_ANALYSIS +
                self.DIM_TEAM_SITUATION + self.DIM_GLOBAL_FLAGS)

    # ── 公开接口 ─────────────────────────────────────────────────────────

    def encode(self, state: GameState, player_id: int) -> np.ndarray:
        """编码为完整的1229维特征向量"""
        return np.concatenate([
            self._hand(state, player_id),
            self._bidding(state, player_id),
            self._current_trick(state, player_id),
            self._history(state, player_id),
            self._suit_analysis(state, player_id),
            self._team_situation(state, player_id),
            self._global_flags(state, player_id),
        ]).astype(np.float32)

    def encode_sections(self, state: GameState, player_id: int) -> dict[str, np.ndarray]:
        """返回各分类的编码结果，用于调试/消融实验"""
        return {
            "hand": self._hand(state, player_id),
            "bidding": self._bidding(state, player_id),
            "current_trick": self._current_trick(state, player_id),
            "history": self._history(state, player_id),
            "suit_analysis": self._suit_analysis(state, player_id),
            "team_situation": self._team_situation(state, player_id),
            "global_flags": self._global_flags(state, player_id),
        }

    # ── 第1类：手牌信息 Hand (112维) ─────────────────────────────────────

    def _hand(self, state: GameState, player_id: int) -> np.ndarray:
        hand = state.hands[player_id]

        # (1a) 我的手牌: 52-bit one-hot
        hand_52 = np.zeros(52, dtype=np.float32)
        for card in hand:
            hand_52[card.card_id] = 1.0

        # (1b) 每花色手牌数量: 4 × 14 one-hot (0~13)
        suit_counts = np.zeros(4 * 14, dtype=np.float32)
        for suit in Suit:
            cnt = sum(1 for c in hand if c.suit == suit)
            suit_counts[suit.value * 14 + cnt] = 1.0

        # (1c) 每花色是否缺门: 4-bit
        void = np.zeros(4, dtype=np.float32)
        for suit in Suit:
            if not any(c.suit == suit for c in hand):
                void[suit.value] = 1.0

        return np.concatenate([hand_52, suit_counts, void])

    # ── 第2类：叫牌信息 Bidding (126维) ──────────────────────────────────

    def _bidding(self, state: GameState, player_id: int) -> np.ndarray:
        # (2a) 每玩家叫品: 4 × 16 one-hot
        # 16类: [未叫(0), nil(1), blind_nil(2), bid_1(3) .. bid_13(15)]
        bids_onehot = np.zeros(4 * 16, dtype=np.float32)
        for pid in range(4):
            bid_val = state.max_bid[pid] if pid < len(state.max_bid) else None
            idx = self._bid_to_index(bid_val)
            bids_onehot[pid * 16 + idx] = 1.0

        # (2b) 每玩家 nil 标记
        nil_flags = np.zeros(4, dtype=np.float32)
        for pid in range(4):
            bid_val = state.max_bid[pid] if pid < len(state.max_bid) else None
            if bid_val == "nil":
                nil_flags[pid] = 1.0

        # (2c) 每玩家 blind_nil 标记
        blind_nil_flags = np.zeros(4, dtype=np.float32)
        for pid in range(4):
            bid_val = state.max_bid[pid] if pid < len(state.max_bid) else None
            if bid_val == "blind_nil":
                blind_nil_flags[pid] = 1.0

        # (2d)(2e) 我方/对方队伍总叫品: 27 one-hot (0~26)
        my_team = state.teams[player_id]
        my_team_bid = 0
        opp_team_bid = 0
        for pid in range(4):
            team = state.teams[pid]
            bid_val = state.max_bid[pid] if pid < len(state.max_bid) else None
            numeric = self._get_numeric_bid(bid_val)
            if team == my_team:
                my_team_bid += numeric
            else:
                opp_team_bid += numeric

        # 截断到 [0, 26]
        my_team_bid = min(max(my_team_bid, 0), 26)
        opp_team_bid = min(max(opp_team_bid, 0), 26)

        team_bid_self = np.zeros(27, dtype=np.float32)
        team_bid_self[my_team_bid] = 1.0
        team_bid_opp = np.zeros(27, dtype=np.float32)
        team_bid_opp[opp_team_bid] = 1.0

        return np.concatenate([
            bids_onehot, nil_flags, blind_nil_flags,
            team_bid_self, team_bid_opp,
        ])

    # ── 第3类：当前墩信息 Current Trick (73维) ───────────────────────────

    def _current_trick(self, state: GameState, player_id: int) -> np.ndarray:
        table = state.table_cards

        # (3a) 桌面牌: 52-bit one-hot
        table_52 = np.zeros(52, dtype=np.float32)
        for _, card in table:
            table_52[card.card_id] = 1.0

        # (3b) 哪些位置已出牌: 4-bit (player 0/1/2/3)
        played_positions = np.zeros(4, dtype=np.float32)
        for pid, _ in table:
            played_positions[pid] = 1.0

        # (3c) 引牌花色: 4 one-hot (S/H/D/C)
        lead_suit = np.zeros(4, dtype=np.float32)
        if table:
            lead_suit[table[0][1].suit.value] = 1.0

        # (3d) 我是第几个出牌: 4 one-hot (1st/2nd/3rd/4th)
        # 找到玩家在 table 中的位置, 尚未出牌则 index = len(table)
        played_idx = None
        for i, (pid, _) in enumerate(table):
            if pid == player_id:
                played_idx = i
                break
        if played_idx is not None:
            order = played_idx + 1  # 已经出过了, 位置固定
        else:
            order = len(table) + 1  # 还没出, 下一个出牌位置
        order_onehot = np.zeros(4, dtype=np.float32)
        if 1 <= order <= 4:
            order_onehot[order - 1] = 1.0

        # (3e) 我是否领牌: 1-bit (领牌=当前墩首攻者是我)
        is_leader = np.float32(1.0 if state.trick_leader == player_id else 0.0)

        # (3f) 当前墩谁赢: 5 one-hot (P0/P1/P2/P3/无人)
        winner_onehot = np.zeros(5, dtype=np.float32)
        if table:
            winner = self._estimate_trick_winner(state)
            winner_onehot[winner] = 1.0
        else:
            winner_onehot[4] = 1.0  # 无人出牌

        # (3g) 赢家是否我方队友: 3 one-hot (我方/对方/未定)
        teammate_flag = np.zeros(3, dtype=np.float32)
        if not table:
            teammate_flag[2] = 1.0  # 未定
        else:
            winner = self._estimate_trick_winner(state)
            my_team = state.teams[player_id]
            if state.teams[winner] == my_team:
                teammate_flag[0] = 1.0  # 我方
            else:
                teammate_flag[1] = 1.0  # 对方

        return np.concatenate([
            table_52, played_positions, lead_suit,
            order_onehot, [is_leader], winner_onehot, teammate_flag,
        ])

    # ── 第4类：历史信息 History (556维) ──────────────────────────────────

    def _history(self, state: GameState, player_id: int) -> np.ndarray:
        # (4a) 所有已出牌: 52-bit one-hot (played_bitset)
        played_52 = np.zeros(52, dtype=np.float32)
        bitset = state.played_bitset
        for i in range(52):
            if bitset & (1 << i):
                played_52[i] = 1.0

        # (4b) 每玩家已赢墩数: 4 × 14 one-hot (0~13)
        tricks_won = np.zeros(4 * 14, dtype=np.float32)
        for pid in range(4):
            tw = state.tricks_won[pid] if pid < len(state.tricks_won) else 0
            tw = min(max(tw, 0), 13)
            tricks_won[pid * 14 + tw] = 1.0

        # (4c) 每玩家手牌数量: 4 × 14 one-hot (0~13)
        hand_sizes = np.zeros(4 * 14, dtype=np.float32)
        for pid in range(4):
            sz = len(state.hands[pid]) if pid < len(state.hands) else 0
            sz = min(max(sz, 0), 13)
            hand_sizes[pid * 14 + sz] = 1.0

        # (4d) 已打墩数: 14 one-hot (0~13)
        tricks_played = min(max(state.tricks_played, 0), 13)
        tp_onehot = np.zeros(14, dtype=np.float32)
        tp_onehot[tricks_played] = 1.0

        # (4e) 剩余墩数: 14 one-hot (0~13)
        remaining = min(max(13 - state.tricks_played, 0), 13)
        rem_onehot = np.zeros(14, dtype=np.float32)
        rem_onehot[remaining] = 1.0

        # (4f) 每张牌由谁打出: 52 × 6 one-hot
        # 类别约定：
        # 0~3 = 玩家ID
        # 4 = 未出牌
        # 5 = 已出牌但无法从状态中恢复出牌玩家（例如仅有played_bitset的构造状态）
        #
        # (4g) 每张牌第几轮打出: 52维标量
        # 归一化规则：
        # 1~13轮 -> round/13 in (0,1]
        # 未出牌 -> 0
        # 已出但轮次未知 -> -1
        card_player_onehot, card_round_scalar = self._build_card_trace_history(state)

        return np.concatenate([
            played_52, tricks_won, hand_sizes,
            tp_onehot, rem_onehot,
            card_player_onehot, card_round_scalar,
        ])

    def _build_card_trace_history(self, state: GameState) -> tuple[np.ndarray, np.ndarray]:
        """
        构建“牌 -> 出牌玩家/出牌轮次”轨迹特征。

        该函数优先使用 trick_history 与 table_cards 恢复精确信息。
        若状态仅提供 played_bitset（常见于随机构造测试状态），
        则将对应牌标为“玩家未知、轮次未知”，避免引入错误伪标签。
        """
        # 初始：全部视作“未出牌”
        player_class = np.full(52, 4, dtype=np.int32)
        round_scalar = np.zeros(52, dtype=np.float32)
        known_mask = np.zeros(52, dtype=np.bool_)

        # 先编码已完成的墩历史：第1墩到第13墩
        for round_id, record in enumerate(state.trick_history, start=1):
            clamped_round = min(max(round_id, 1), 13)
            for pid, card in record.cards:
                cid = card.card_id
                player_class[cid] = pid
                round_scalar[cid] = clamped_round / 13.0
                known_mask[cid] = True

        # 再编码当前墩（若有）：其轮次 = 已完成墩数 + 1
        if state.table_cards:
            current_round = min(max(state.tricks_played + 1, 1), 13)
            for pid, card in state.table_cards:
                cid = card.card_id
                player_class[cid] = pid
                round_scalar[cid] = current_round / 13.0
                known_mask[cid] = True

        # 对“已出但缺少历史轨迹”的牌打上未知标签
        bitset = state.played_bitset
        for cid in range(52):
            if (bitset & (1 << cid)) and (not known_mask[cid]):
                player_class[cid] = 5
                round_scalar[cid] = -1.0

        # 类别转 one-hot
        player_onehot = np.zeros(52 * 6, dtype=np.float32)
        for cid in range(52):
            cls = int(player_class[cid])
            player_onehot[cid * 6 + cls] = 1.0

        return player_onehot, round_scalar

    # ── 第5类：花色分析 Suit Analysis (192维) ────────────────────────────

    def _suit_analysis(self, state: GameState, player_id: int) -> np.ndarray:
        hand = state.hands[player_id]
        played_bitset = state.played_bitset

        # (5a) 每花色未出张数: 4 × 14 one-hot (0~13)
        # 公式: 13 - 我手中的 - 已出的
        suit_remaining = np.zeros(4 * 14, dtype=np.float32)
        for suit in Suit:
            my_cnt = sum(1 for c in hand if c.suit == suit)
            played_cnt = self._count_suit_played(played_bitset, suit)
            remaining = max(13 - my_cnt - played_cnt, 0)
            remaining = min(remaining, 13)
            suit_remaining[suit.value * 14 + remaining] = 1.0

        # (5b) 我持每花色最大牌: 4 × 13 one-hot (2~A)
        my_highest = np.zeros(4 * 13, dtype=np.float32)
        for suit in Suit:
            cards_in_suit = [c for c in hand if c.suit == suit]
            if cards_in_suit:
                best = max(cards_in_suit, key=lambda c: c.rank.value)
                # rank.value 范围 2~14, 映射到 0~12
                my_highest[suit.value * 13 + (best.rank.value - 2)] = 1.0
            # 无该花色 → 全0

        # (5c) 每花色最大未出牌: 4 × 13 one-hot (2~A)
        highest_unplayed = np.zeros(4 * 13, dtype=np.float32)
        for suit in Suit:
            best = self._highest_unplayed(played_bitset, suit)
            if best is not None:
                highest_unplayed[suit.value * 13 + (best.value - 2)] = 1.0

        # (5d) 我是否持有该花色最高牌: 4-bit
        has_top = np.zeros(4, dtype=np.float32)
        for suit in Suit:
            highest = self._highest_unplayed(played_bitset, suit)
            if highest is not None and any(c.suit == suit and c.rank == highest for c in hand):
                has_top[suit.value] = 1.0

        # (5e) ♠ (trump) 剩余张数: 14 one-hot (0~13)
        trump_remaining = 0
        for suit in [Suit.SPADES]:
            my_cnt = sum(1 for c in hand if c.suit == suit)
            played_cnt = self._count_suit_played(played_bitset, suit)
            trump_remaining = max(13 - my_cnt - played_cnt, 0)
            trump_remaining = min(trump_remaining, 13)
        trump_rem_onehot = np.zeros(14, dtype=np.float32)
        trump_rem_onehot[trump_remaining] = 1.0

        # (5f) 我的♠数量: 14 one-hot (0~13)
        my_spades = sum(1 for c in hand if c.suit == Suit.SPADES)
        my_spades_onehot = np.zeros(14, dtype=np.float32)
        my_spades_onehot[min(my_spades, 13)] = 1.0

        return np.concatenate([
            suit_remaining, my_highest, highest_unplayed,
            has_top, trump_rem_onehot, my_spades_onehot,
        ])

    # ── 第6类：队伍局势 Team Situation (154维) ──────────────────────────

    def _team_situation(self, state: GameState, player_id: int) -> np.ndarray:
        my_team = state.teams[player_id]
        opp_team = 1 - my_team

        # 计算队伍层面的统计
        my_team_tricks = 0
        opp_team_tricks = 0
        my_team_bid = 0
        opp_team_bid = 0
        teammate_id = None

        for pid in range(4):
            team = state.teams[pid]
            tricks = state.tricks_won[pid] if pid < len(state.tricks_won) else 0
            bid_val = state.max_bid[pid] if pid < len(state.max_bid) else None
            nb = self._get_numeric_bid(bid_val)

            if team == my_team:
                my_team_tricks += tricks
                my_team_bid += nb
                if pid != player_id:
                    teammate_id = pid
            else:
                opp_team_tricks += tricks
                opp_team_bid += nb

        # (6a) 我方已赢墩: 14 one-hot (0~13)
        my_tricks_onehot = np.zeros(14, dtype=np.float32)
        my_tricks_onehot[min(my_team_tricks, 13)] = 1.0

        # (6b) 对方已赢墩: 14 one-hot (0~13)
        opp_tricks_onehot = np.zeros(14, dtype=np.float32)
        opp_tricks_onehot[min(opp_team_tricks, 13)] = 1.0

        # (6c) 我方还差几墩完成叫品: 28 one-hot (-13..+14)
        # 正值=还差, 负值=已超
        my_remaining = my_team_bid - my_team_tricks
        my_remaining_clamped = max(min(my_remaining, 14), -13)
        my_rem_onehot = np.zeros(28, dtype=np.float32)
        my_rem_onehot[my_remaining_clamped + 13] = 1.0

        # (6d) 对方还差几墩完成叫品: 28 one-hot (-13..+14)
        opp_remaining = opp_team_bid - opp_team_tricks
        opp_remaining_clamped = max(min(opp_remaining, 14), -13)
        opp_rem_onehot = np.zeros(28, dtype=np.float32)
        opp_rem_onehot[opp_remaining_clamped + 13] = 1.0

        # (6e) 我个人 bid vs tricks: 28 one-hot (-13..+14)
        my_bid_val = self._get_numeric_bid(
            state.max_bid[player_id] if player_id < len(state.max_bid) else None
        )
        my_tricks = state.tricks_won[player_id] if player_id < len(state.tricks_won) else 0
        my_diff = my_tricks - my_bid_val  # 正值=超额, 负值=不足
        my_diff_clamped = max(min(my_diff, 14), -13)
        my_diff_onehot = np.zeros(28, dtype=np.float32)
        my_diff_onehot[my_diff_clamped + 13] = 1.0

        # (6f) 队友个人 bid vs tricks: 28 one-hot (-13..+14)
        if teammate_id is not None:
            tm_bid = self._get_numeric_bid(
                state.max_bid[teammate_id] if teammate_id < len(state.max_bid) else None
            )
            tm_tricks = state.tricks_won[teammate_id] if teammate_id < len(state.tricks_won) else 0
            tm_diff = tm_tricks - tm_bid
        else:
            tm_diff = 0
        tm_diff_clamped = max(min(tm_diff, 14), -13)
        tm_diff_onehot = np.zeros(28, dtype=np.float32)
        tm_diff_onehot[tm_diff_clamped + 13] = 1.0

        # (6g) 我方超墩数: 14 one-hot (0~13)
        overtricks = max(my_team_tricks - my_team_bid, 0)
        overtricks_onehot = np.zeros(14, dtype=np.float32)
        overtricks_onehot[min(overtricks, 13)] = 1.0

        return np.concatenate([
            my_tricks_onehot, opp_tricks_onehot,
            my_rem_onehot, opp_rem_onehot,
            my_diff_onehot, tm_diff_onehot,
            overtricks_onehot,
        ])

    # ── 第7类：全局标记 Global Flags (16维) ─────────────────────────────

    def _global_flags(self, state: GameState, player_id: int) -> np.ndarray:
        # (7a) Spades 已破
        spades_broken = np.float32(1.0 if state.spades_broken else 0.0)

        # (7b) 阶段 (叫牌/出牌): 2 one-hot
        phase_onehot = np.zeros(2, dtype=np.float32)
        if state.phase == Phase.BIDDING:
            phase_onehot[0] = 1.0
        else:
            # 默认视为"出牌"阶段
            phase_onehot[1] = 1.0

        # (7c) 我的座位: 4 one-hot
        seat_onehot = np.zeros(4, dtype=np.float32)
        seat_onehot[player_id % 4] = 1.0

        # (7d) 庄家座位: 4 one-hot
        dealer_seat = state.dealer_seat % 4 if hasattr(state, 'dealer_seat') else 0
        dealer_onehot = np.zeros(4, dtype=np.float32)
        dealer_onehot[dealer_seat] = 1.0

        # 检查各玩家是否为 nil / blind_nil
        my_team = state.teams[player_id]
        my_has_nil = False
        opp_has_nil = False
        i_am_nil = False
        teammate_is_nil = False
        i_am_blind_nil = False

        for pid in range(4):
            bid_val = state.max_bid[pid] if pid < len(state.max_bid) else None
            is_nil = (bid_val == "nil")
            is_blind_nil = (bid_val == "blind_nil")
            team = state.teams[pid]

            if is_nil or is_blind_nil:
                if team == my_team:
                    my_has_nil = True
                else:
                    opp_has_nil = True

            if pid == player_id:
                i_am_nil = is_nil
                i_am_blind_nil = is_blind_nil
            elif team == my_team:
                teammate_is_nil = is_nil or is_blind_nil

        # (7e) 我方有 nil/blind_nil 玩家
        # (7f) 对方有 nil/blind_nil 玩家
        # (7g) 我自己是 nil
        # (7h) 我队友是 nil/blind_nil
        # (7i) 我自己是 blind_nil
        flags = np.array([
            spades_broken,
            *phase_onehot,
            *seat_onehot,
            *dealer_onehot,
            float(my_has_nil),
            float(opp_has_nil),
            float(i_am_nil),
            float(teammate_is_nil),
            float(i_am_blind_nil),
        ], dtype=np.float32)

        return flags

    # ── 私有辅助方法 ─────────────────────────────────────────────────────

    @staticmethod
    def _bid_to_index(bid_value) -> int:
        """将叫品映射到 0~15 的索引"""
        if bid_value is None:
            return 0                                                      # 未叫
        if isinstance(bid_value, str):
            if bid_value == "nil":
                return 1
            if bid_value == "blind_nil":
                return 2
            if bid_value.startswith("bid_"):
                k = int(bid_value.split("_")[1])
                if 1 <= k <= 13:
                    return 2 + k
        if isinstance(bid_value, (int, float)):
            v = int(bid_value)
            if 1 <= v <= 13:
                return 2 + v
        return 0

    @staticmethod
    def _get_numeric_bid(bid_value) -> int:
        """获取叫牌的数值 (nil/blind_nil 视为 0)"""
        if bid_value is None:
            return 0
        if isinstance(bid_value, str):
            if bid_value == "nil" or bid_value == "blind_nil":
                return 0
            if bid_value.startswith("bid_"):
                return int(bid_value.split("_")[1])
        if isinstance(bid_value, (int, float)):
            return int(bid_value)
        return 0

    @staticmethod
    def _count_suit_played(played_bitset: int, suit: Suit) -> int:
        """统计指定花色已打出的牌数"""
        base = suit.value * 13
        mask = ((1 << 13) - 1) << base
        return (played_bitset & mask).bit_count()

    @staticmethod
    def _highest_unplayed(played_bitset: int, suit: Suit) -> Rank | None:
        """获取指定花色中最高未出牌 (无则返回 None)"""
        for rank in reversed(list(Rank)):  # 从 A 到 2 遍历
            card = Card(suit, rank)
            if not (played_bitset & card.bit):
                return rank
        return None

    @staticmethod
    def _estimate_trick_winner(state: GameState) -> int:
        """估计当前墩的赢家 (基于已出牌的牌面大小)"""
        table = state.table_cards
        if not table:
            return state.trick_leader

        spade_cards = [(pid, card) for pid, card in table if card.suit == Suit.SPADES]
        if spade_cards:
            winner_pid, _ = max(spade_cards, key=lambda x: x[1].rank.value)
        else:
            lead_suit = table[0][1].suit
            suit_cards = [(pid, card) for pid, card in table if card.suit == lead_suit]
            winner_pid, _ = max(suit_cards, key=lambda x: x[1].rank.value)

        return winner_pid
