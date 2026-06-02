"""
RL 特征编码器（170 维），仅用于 rl_exact 前 4 墩的强化学习模型。

Part 1 (164 维, 4+1+90+52+4+13):
  前 4 维: 叫牌（整数，nil 用 0 表示）
    - [0]: 我的叫牌
    - [1]: 我的上家的叫牌
    - [2]: 我的上上家的叫牌
    - [3]: 我的下家的叫牌
  第 5 维: 已出牌数 (0~15, 标量)
  之后 15×6=90 维: 每张打出来的牌 (最多 15 张, 时间倒序)
    - 每个牌 6 维: [点数(2~14), 花色S(0/1), 花色H, 花色D, 花色C, 出牌人(0=我,1=上家,2=上上家,3=下家)]
  之后 52 维: 手牌 bitmap (card_id 0~51, 1 表示在手)
  之后 4 维: 每门花色的张数 (S, H, D, C, 整数)
  之后 13 维: 当前墩领出花色跟踪（不是领出时，用 1/0 表示领出花色的每张牌是否已打过）

Part 2 (6 维):
  [164]: 本墩出牌位置 (1=第一个出, 2=第二个出, 3=第三个出, 4=第四个出)
  [165]: 队友是否大过当前墩（仅第3/4个出牌时有效, 0/1）
  [166]: 队友牌在领出花色是否不可击败（仅第3/4个出牌时有效, 0/1）
  [167-169]: 三个跟牌选项是否合法（领出时全0, 否则0/1）

编码方式（4+1+90+52+4+13=164 维）:
  前 4 维: 叫牌（整数，nil 用 0 表示）
    - [0]: 我的叫牌
    - [1]: 我的上家的叫牌
    - [2]: 我的上上家的叫牌
    - [3]: 我的下家的叫牌
  第 5 维: 已出牌数 (0~15, 标量)
  之后 15×6=90 维: 每张打出来的牌 (最多 15 张, 时间倒序)
    - 每个牌 6 维: [点数(2~14), 花色S(0/1), 花色H, 花色D, 花色C, 出牌人(0=我,1=上家,2=上上家,3=下家)]
  之后 52 维: 手牌 bitmap (card_id 0~51, 1 表示在手)
  之后 4 维: 每门花色的张数 (S, H, D, C, 整数)
  最后 13 维: 当前墩领出花色跟踪（不是领出时，用 1/0 表示领出花色的每张牌是否已打过）
"""

from __future__ import annotations

import numpy as np

from trick_taking.card import Card, Suit, Rank
from trick_taking.game_state import GameState


class RLFeatureEncoder:
    """前 4 墩 RL 策略的 170 维特征编码器。"""

    TOTAL_DIM = 170  # 164 (Part 1) + 6 (Part 2)

    # Part 1: 各分段起始位置
    BID_START = 0
    BID_DIM = 4

    PLAYED_COUNT_START = 4
    PLAYED_COUNT_DIM = 1

    CARDS_START = 5           # 15 × 6 dims
    CARD_TOTAL = 6            # 每张牌总维度
    CARDS_TOTAL = 15 * CARD_TOTAL  # 90

    HAND_START = CARDS_START + CARDS_TOTAL  # 95
    HAND_BITMAP_DIM = 52

    SUIT_COUNT_START = HAND_START + HAND_BITMAP_DIM  # 147
    SUIT_COUNT_DIM = 4

    LED_SUIT_START = SUIT_COUNT_START + SUIT_COUNT_DIM  # 151
    LED_SUIT_DIM = 13

    # Part 2: 额外 6 维
    TRICK_POS_START = 164
    PARTNER_BEAT_START = 165
    PARTNER_UNBEAT_START = 166
    OPTION_LEGAL_START = 167  # 3 dims (167, 168, 169)

    # 花色顺序: S=0, H=1, D=2, C=3

    @staticmethod
    def _numeric_bid(bid_val) -> int:
        """将叫牌值转为整数，nil/None 返回 0。"""
        if bid_val is None:
            return 0
        if isinstance(bid_val, str):
            if bid_val in ("nil", "blind_nil"):
                return 0
            if bid_val.startswith("bid_"):
                return int(bid_val.split("_")[1])
        if isinstance(bid_val, (int, float)):
            return int(bid_val)
        return 0

    def encode(self, state: GameState, player_id: int) -> np.ndarray:
        """将 GameState 编码为 164 维特征向量。"""
        feature = np.zeros(self.TOTAL_DIM, dtype=np.float32)

        # 1. 叫牌 (4 维): [自己, 上家, 上上家, 下家]
        relative_offsets = [0, 3, 2, 1]
        for i, offset in enumerate(relative_offsets):
            pid = (player_id + offset) % 4
            bid_val = state.max_bid[pid] if pid < len(state.max_bid) else None
            feature[self.BID_START + i] = self._numeric_bid(bid_val)

        # 2. 已出牌数 (1 维, 标量 0~15)
        n_completed = len(state.trick_history)  # 已完成墩数
        n_on_table = len(state.table_cards)     # 当前墩已出牌数
        n_played_total = n_completed * 4 + n_on_table
        feature[self.PLAYED_COUNT_START] = float(min(n_played_total, 15))

        # 3. 编码每张已出牌 (最多 15 张 × 6 维, 时间倒序)
        played_cards: list[tuple[Card, int]] = []

        for record in state.trick_history:
            for pid, card in record.cards:
                played_cards.append((card, pid))
        for pid, card in state.table_cards:
            played_cards.append((card, pid))

        played_cards.reverse()  # 时间倒序：最新出的牌在卡0
        for i, (card, pid) in enumerate(played_cards[:15]):
            start = self.CARDS_START + i * self.CARD_TOTAL

            # 点数 (2~14, 标量)
            feature[start] = card.rank.value

            # 花色 one-hot (S=0, H=1, D=2, C=3)
            suit_idx = card.suit.value
            feature[start + 1 + suit_idx] = 1.0

            # 出牌人 (0=我, 1=上家, 2=上上家, 3=下家)
            # (player_id - pid) 逆时针偏移: +1=上家, +2=上上家, +3=下家
            rel_idx = (player_id - pid + 4) % 4
            feature[start + 5] = rel_idx

        # 4. 手牌 bitmap (52 维)
        for card in state.hands[player_id]:
            feature[self.HAND_START + card.card_id] = 1.0

        # 5. 每门花色剩余张数 (4 维: S, H, D, C)
        suit_counts = {s: 0 for s in Suit}
        for card in state.hands[player_id]:
            suit_counts[card.suit] += 1
        for suit in Suit:
            feature[self.SUIT_COUNT_START + suit.value] = suit_counts[suit]

        # 6. 当前墩领出花色跟踪 (13 维, 0=领出/全0, 1=每张牌是否已打过)
        if state.table_cards:  # 当前墩已有牌 → 我不是领出
            led_suit = state.table_cards[0][1].suit  # 领出花色

            # 收集该花色所有已打出的牌的 rank
            led_suit_played_ranks = set()
            for record in state.trick_history:
                for pid, card in record.cards:
                    if card.suit == led_suit:
                        led_suit_played_ranks.add(card.rank.value)
            for pid, card in state.table_cards:
                if card.suit == led_suit:
                    led_suit_played_ranks.add(card.rank.value)

            # 编码 13 个 rank (2~14 → index 0~12)
            for rank_val in range(2, 15):
                if rank_val in led_suit_played_ranks:
                    feature[self.LED_SUIT_START + (rank_val - 2)] = 1.0
        # 领出时，13 维保持为 0（已在初始化为零时处理）

        # ── Part 2: 6 维额外特征 ──────────────────────────────────────
        # (1) 我是这墩第几个出牌的 (1~4)
        n_on_table = len(state.table_cards)
        trick_position = n_on_table + 1
        feature[self.TRICK_POS_START] = float(trick_position)

        # (2)(3) 队友信息（仅当第3或第4个出牌时有意义）
        partner_beat_all = 0.0
        partner_unbeatable = 0.0

        if n_on_table >= 2:
            partner_id = (player_id + 2) % 4
            partner_card = None
            for pid, card in state.table_cards:
                if pid == partner_id:
                    partner_card = card
                    break

            if partner_card is not None:
                # 找出桌上最大的牌
                best_card = state.table_cards[0][1]
                for _, card in state.table_cards[1:]:
                    if card.suit == Suit.SPADES:
                        if best_card.suit != Suit.SPADES or card.rank.value > best_card.rank.value:
                            best_card = card
                    elif card.suit == best_card.suit and card.rank.value > best_card.rank.value:
                        best_card = card

                # (2) 队友的牌是否大过当前所有牌
                if partner_card.suit == best_card.suit and partner_card.rank == best_card.rank:
                    partner_beat_all = 1.0

                # (3) 队友的牌在该花色是否不可击败
                # 检查所有更高牌是否都被看到（已出或在我手中）
                partner_suit = partner_card.suit
                partner_rank = partner_card.rank.value
                played_ranks_by_suit: dict[Suit, set[int]] = {s: set() for s in Suit}
                for record in state.trick_history:
                    for _, c in record.cards:
                        played_ranks_by_suit[c.suit].add(c.rank.value)
                for _, c in state.table_cards:
                    played_ranks_by_suit[c.suit].add(c.rank.value)

                higher_visible = False
                for rank_val in range(partner_rank + 1, 15):
                    if rank_val not in played_ranks_by_suit[partner_suit]:
                        in_my_hand = any(
                            c.suit == partner_suit and c.rank.value == rank_val
                            for c in state.hands[player_id]
                        )
                        if not in_my_hand:
                            higher_visible = True
                            break

                if not higher_visible:
                    partner_unbeatable = 1.0

        feature[self.PARTNER_BEAT_START] = partner_beat_all
        feature[self.PARTNER_UNBEAT_START] = partner_unbeatable

        # (4) 三个跟牌选项是否合法
        option_legal = [0.0, 0.0, 0.0]
        if n_on_table > 0:  # 不是领出
            lead_suit = state.table_cards[0][1].suit
            hand = state.hands[player_id]
            has_lead_suit = any(c.suit == lead_suit for c in hand)

            # 选项0: 出最小牌 — 总是合法
            option_legal[0] = 1.0

            # 找出桌上最大的牌
            best_card = state.table_cards[0][1]
            for _, card in state.table_cards[1:]:
                if card.suit == Suit.SPADES:
                    if best_card.suit != Suit.SPADES or card.rank.value > best_card.rank.value:
                        best_card = card
                elif card.suit == best_card.suit and card.rank.value > best_card.rank.value:
                    best_card = card

            # 选项1: 出能赢的最小牌
            beating_exists = False
            if has_lead_suit:
                if best_card.suit == lead_suit:
                    beating_exists = any(
                        c.suit == lead_suit and c.rank.value > best_card.rank.value
                        for c in hand
                    )
                elif best_card.suit == Suit.SPADES:
                    beating_exists = any(
                        c.suit == Suit.SPADES and c.rank.value > best_card.rank.value
                        for c in hand
                    )
            else:
                if best_card.suit == Suit.SPADES:
                    beating_exists = any(
                        c.suit == Suit.SPADES and c.rank.value > best_card.rank.value
                        for c in hand
                    )
                else:
                    beating_exists = (
                        any(c.suit == Suit.SPADES for c in hand)
                        or any(c.suit == best_card.suit and c.rank.value > best_card.rank.value for c in hand)
                    )
            option_legal[1] = 1.0 if beating_exists else 0.0

            # 选项2: 出最大牌 — 总是合法
            option_legal[2] = 1.0

        feature[self.OPTION_LEGAL_START] = option_legal[0]
        feature[self.OPTION_LEGAL_START + 1] = option_legal[1]
        feature[self.OPTION_LEGAL_START + 2] = option_legal[2]

        return feature

    def encode_dim_info(self) -> dict[str, tuple[int, int]]:
        """返回各分节的维度范围（用于调试）。"""
        info = {
            "bids": (self.BID_START, self.BID_START + self.BID_DIM),
            "played_count": (self.PLAYED_COUNT_START, self.PLAYED_COUNT_START + self.PLAYED_COUNT_DIM),
        }
        for i in range(15):
            start = self.CARDS_START + i * self.CARD_TOTAL
            info[f"card_{i}"] = (start, start + self.CARD_TOTAL)
        info["hand_bitmap"] = (self.HAND_START, self.HAND_START + self.HAND_BITMAP_DIM)
        info["suit_counts"] = (self.SUIT_COUNT_START, self.SUIT_COUNT_START + self.SUIT_COUNT_DIM)
        info["led_suit"] = (self.LED_SUIT_START, self.LED_SUIT_START + self.LED_SUIT_DIM)
        info["trick_position"] = (self.TRICK_POS_START, self.TRICK_POS_START + 1)
        info["partner_beat"] = (self.PARTNER_BEAT_START, self.PARTNER_BEAT_START + 1)
        info["partner_unbeatable"] = (self.PARTNER_UNBEAT_START, self.PARTNER_UNBEAT_START + 1)
        info["option_legal"] = (self.OPTION_LEGAL_START, self.OPTION_LEGAL_START + 3)
        return info
