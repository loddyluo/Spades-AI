"""
RL 特征编码器（387 维），仅用于 rl_exact 前 4 墩的强化学习模型。

编码方式（387 维）:
  前 16 维: 已出牌数的 one-hot (0~15)
  中间 315 维: 15 张牌 × 21 维/张
    - 前 13 维: 点数 one-hot (2,3,4,5,6,7,8,9,10,J,Q,K,A)
    - 中间 4 维: 花色 one-hot (S,H,D,C)
    - 后 4 维: 出牌人 one-hot (我=1000, 上家=0100, 上上家=0010, 下家=0001)
  后 56 维: 自己手牌信息
    - 前 52 维: 手牌 bitmap (card_id 0~51, 1 表示在手)
    - 后 4 维: 手牌中黑桃数 (one-hot, 0~3, ≥4 统一归入 3)

  忽略叫牌信息。

  出牌人相对编码: (abs_player_id - my_player_id + 4) % 4
    0 → 我, 1 → 上家, 2 → 上上家(对家), 3 → 下家
"""

from __future__ import annotations

import numpy as np

from trick_taking.card import Card, Suit, Rank
from trick_taking.game_state import GameState

_STANDARD_CARDS: list[Card] = [Card(s, r) for s in Suit for r in Rank]


class RLFeatureEncoder:
    """前 4 墩 RL 策略的 387 维特征编码器。"""

    DIM_PLAYED_COUNT = 16   # 已出牌数 one-hot
    DIM_PER_CARD = 21       # 每张牌: 13 rank + 4 suit + 4 player
    MAX_CARDS = 15          # 最多编码 15 张已出牌
    DIM_HAND = 56           # 手牌信息: 52 bitmap + 4 黑桃数

    TOTAL_DIM = DIM_PLAYED_COUNT + MAX_CARDS * DIM_PER_CARD + DIM_HAND  # 16 + 315 + 56 = 387

    RANK_OFFSET = 0                     # 13 维 rank one-hot 起始位置
    SUIT_OFFSET = 13                    # 4 维 suit one-hot 起始位置
    PLAYER_OFFSET = 17                  # 4 维 player one-hot 起始位置
    CARD_TOTAL = 21                     # 每张牌总维度

    # 手牌部分偏移
    HAND_START = DIM_PLAYED_COUNT + MAX_CARDS * DIM_PER_CARD  # 331
    HAND_BITMAP_OFFSET = 0              # 52 维手牌 bitmap
    HAND_SPADE_OFFSET = 52              # 4 维黑桃数 one-hot

    def encode(self, state: GameState, player_id: int) -> np.ndarray:
        """将 GameState 编码为 387 维特征向量。"""
        feature = np.zeros(self.TOTAL_DIM, dtype=np.float32)

        # 1. 收集所有已出牌（严格按时间顺序）
        played_cards: list[tuple[Card, int]] = []

        # 1a. 已完成的墩历史
        for record in state.trick_history:
            for pid, card in record.cards:
                played_cards.append((card, pid))

        # 1b. 当前墩的牌（已在桌上的）
        for pid, card in state.table_cards:
            played_cards.append((card, pid))

        # 只取前 MAX_CARDS 张（前 4 墩最多 16 张，此时最多已出 15 张）
        played_cards = played_cards[:self.MAX_CARDS]
        n_played = len(played_cards)

        # 2. 前 16 维：已出牌数 one-hot (0~15)
        feature[n_played] = 1.0

        # 3. 编码每张已出牌（最多 15 张）
        for i, (card, pid) in enumerate(played_cards):
            card_start = self.DIM_PLAYED_COUNT + i * self.CARD_TOTAL

            # 3a. 点数 one-hot (13 dims)
            # rank.value 范围 2~14, 映射到索引 0~12
            rank_idx = card.rank.value - 2
            feature[card_start + self.RANK_OFFSET + rank_idx] = 1.0

            # 3b. 花色 one-hot (4 dims)
            # suit.value: S=0, H=1, D=2, C=3
            feature[card_start + self.SUIT_OFFSET + card.suit.value] = 1.0

            # 3c. 出牌人相对编码 (4 dims)
            # rel = (abs_pid - my_pid + 4) % 4
            #   0 → 我 (1000)
            #   1 → 上家 (0100)
            #   2 → 上上家/对家 (0010)
            #   3 → 下家 (0001)
            rel_idx = (pid - player_id + 4) % 4
            feature[card_start + self.PLAYER_OFFSET + rel_idx] = 1.0

        # 4. 手牌 56 维
        hand = state.hands[player_id]
        hand_start = self.HAND_START

        # 4a. 52 维手牌 bitmap
        for card in hand:
            feature[hand_start + self.HAND_BITMAP_OFFSET + card.card_id] = 1.0

        # 4b. 4 维黑桃数 one-hot (0~3, ≥4 归入 3)
        n_spades = sum(1 for card in hand if card.suit == Suit.SPADES)
        spade_idx = min(n_spades, 3)
        feature[hand_start + self.HAND_SPADE_OFFSET + spade_idx] = 1.0

        return feature

    def encode_dim_info(self) -> dict[str, tuple[int, int]]:
        """返回各分节的维度范围（用于调试）。"""
        info = {
            "played_count": (0, self.DIM_PLAYED_COUNT),
        }
        for i in range(self.MAX_CARDS):
            start = self.DIM_PLAYED_COUNT + i * self.CARD_TOTAL
            info[f"card_{i}"] = (start, start + self.CARD_TOTAL)
        info["hand"] = (self.HAND_START, self.TOTAL_DIM)
        return info
