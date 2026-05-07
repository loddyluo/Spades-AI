"""黑桃王对局用的玩家程序封装。

文件作用：
- 提供一个随机出牌玩家程序；
- 提供一个接入 `strategy.truncated_mcts_strategy.TruncatedMCTSStrategy` 的实际出牌玩家程序；
- 这两个程序都可直接作为对局驱动中的四名玩家之一使用。

函数/类输入输出说明：
- RandomSpadesPlayer
    输入: seed (int | None)
    输出: AIPlayer 风格对象，`play_card` 会从合法动作中均匀随机选一个

- TruncatedMCTSPlayer
    输入:
      - config: TruncatedMCTSConfig | None，控制精确阈值、叶子阈值和模拟次数
      - checkpoint_path: str | None，MLP 权重文件路径
    输出: AIPlayer 风格对象，`play_card` 会调用截断 MCTS 策略选择动作
"""

from __future__ import annotations

import random
from typing import Any

from trick_taking.card import Card
from trick_taking.player import AIPlayer

from strategy.truncated_mcts_strategy import TruncatedMCTSConfig, TruncatedMCTSStrategy


class RandomSpadesPlayer(AIPlayer):
    """在所有合法动作中随机出牌的玩家程序。"""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self.position = -1
        self.hand: list[Card] = []

    def start_game(self, position: int, hand: list[Card], num_players: int) -> None:
        """记录自己的座位和起始牌。"""
        self.position = position
        self.hand = list(hand)

    def place_bid(self, legal_bids: list[Any], state_view: dict) -> Any:
        """从合法叫牌中随机选择一个。"""
        return self._rng.choice(legal_bids) if legal_bids else None

    def play_card(self, legal_cards: list[Card], state_view: dict) -> Card:
        """从合法出牌中随机选择一张。"""
        if not legal_cards:
            raise ValueError("随机玩家在没有合法牌时被要求出牌")
        return self._rng.choice(legal_cards)


class TruncatedMCTSPlayer(AIPlayer):
    """接入截断 MCTS 策略的可打牌玩家程序。"""

    def __init__(
        self,
        config: TruncatedMCTSConfig | None = None,
    ) -> None:
        self.strategy = TruncatedMCTSStrategy(config)
        self.position = -1
        self.hand: list[Card] = []

    def start_game(self, position: int, hand: list[Card], num_players: int) -> None:
        """记录自己的座位和起始牌。"""
        self.position = position
        self.hand = list(hand)

    def place_bid(self, legal_bids: list[Any], state_view: dict) -> Any:
        """叫牌阶段使用简单随机策略，避免影响出牌策略验证。"""
        if not legal_bids:
            return None
        return random.choice(legal_bids)

    def play_card(self, legal_cards: list[Card], state_view: dict) -> Card:
        """调用截断 MCTS 选择动作。

        输入 state_view 必须至少包含:
        - state: 当前完整 GameState 快照
        - feature: 1229 维特征向量（runner 会负责构造并打印）
        """
        state = state_view.get("state")
        if state is None:
            raise ValueError("TruncatedMCTSPlayer.play_card 需要 state_view['state']")

        action = self.strategy.choose_action(state)
        if action is None:
            raise ValueError("截断 MCTS 没有返回动作")
        return action
