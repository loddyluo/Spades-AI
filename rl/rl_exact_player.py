"""
RL + Exact 混合出牌玩家。

前 16 张牌（剩余牌数 > 36）：使用策略网络（PolicyMLP）采样出牌。
后 36 张牌（剩余牌数 <= 36）：使用精确双明手求解器出牌。

训练时会记录策略网络的所有决策轨迹（特征、动作、log_prob），
供训练脚本计算 policy gradient 更新。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trick_taking.card import Card
from trick_taking.game_state import GameState
from trick_taking.player import AIPlayer
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)
from trick_taking.utils.feature_encoder import SpadesFeatureEncoder
from rl.policy_network import PolicyMLP


class RLExactPlayer(AIPlayer):
    """RL + Exact 混合出牌玩家。

    属性:
        exact_threshold: 剩余牌数 <= 该值时使用精确求解器
        is_training: 训练模式（True=采样探索，False=argmax 贪心）
        trajectory: 当前对局中所有 RL 决策的记录
    """

    def __init__(
        self,
        policy_net: PolicyMLP,
        exact_solver: ExactDoubleDummyCppFastestSolver | None = None,
        encoder: SpadesFeatureEncoder | None = None,
        exact_threshold: int = 36,
        is_training: bool = True,
        bid_model=None,
        bid_device: str = "cpu",
    ) -> None:
        self.policy_net = policy_net
        self.encoder = encoder or SpadesFeatureEncoder()
        self.exact_threshold = exact_threshold
        self.is_training = is_training
        self._bid_model = bid_model
        self._bid_device = bid_device
        self.position = -1
        self.hand: list[Card] = []
        self.last_play_info: dict[str, Any] = {}
        self.last_bid_info: dict[str, Any] | None = None

        # 精确求解器
        if exact_solver is not None:
            self.exact_solver = exact_solver
        else:
            cpp_solver = ExactDoubleDummyCppFastestSolver()
            self.exact_solver = (
                cpp_solver if cpp_solver.native_available else None
            )

        # 训练轨迹（每局重置）
        self.trajectory: list[dict[str, Any]] = []

    def start_game(self, position: int, hand: list[Card], num_players: int) -> None:
        self.position = position
        self.hand = list(hand)
        self.last_play_info = {}
        self.trajectory = []

    def place_bid(self, legal_bids: list[Any], state_view: dict) -> Any:
        """使用 MLP bid model 叫牌（与 evaluate_cheat_mcts_vs_dds.py 中的 DDSPlayer 相同）。"""
        if self._bid_model is not None:
            try:
                go_mcts_dir = Path(__file__).resolve().parents[1] / "evaluate" / "GO-MCTS"
                if str(go_mcts_dir) not in sys.path:
                    sys.path.insert(0, str(go_mcts_dir))

                from bridge import normalize_bid_for_legal_options, to_go_state
                from models import MLPBidPlayer

                state = state_view.get("state")
                if state is None:
                    return legal_bids[0] if legal_bids else None

                mlp_bidder = MLPBidPlayer(self._bid_model, self._bid_device)
                go_state = to_go_state(state)
                raw_bid = mlp_bidder.choose_bid(go_state)
                normalized = normalize_bid_for_legal_options(raw_bid, legal_bids)
                self.last_bid_info = {
                    "chosen_bid": normalized,
                    "legal_bids": list(legal_bids),
                }
                return normalized
            except Exception:
                pass

        # Fallback: 简单启发式
        if not legal_bids:
            return None
        numeric_bids = [b for b in legal_bids if isinstance(b, str) and b.startswith("bid_")]
        if numeric_bids:
            return numeric_bids[0]
        return legal_bids[0]

    def play_card(self, legal_cards: list[Card], state_view: dict) -> Card:
        state: GameState | None = state_view.get("state")
        if state is None:
            self.last_play_info = {"mode": "no_state_fallback"}
            return legal_cards[0]

        # 计算剩余牌数
        remaining = sum(len(h) for h in state.hands)

        # 后 36 张牌：使用精确求解器
        if remaining <= self.exact_threshold:
            return self._exact_play(state, legal_cards)

        # 前 16 张牌：使用策略网络
        return self._policy_play(state, legal_cards)

    def _policy_play(self, state: GameState, legal_cards: list[Card]) -> Card:
        """使用策略网络选择动作（训练时采样，评估时 argmax）。"""
        feature = self.encoder.encode(state, self.position)
        feat_tensor = torch.from_numpy(feature).float().unsqueeze(0)

        # 训练时需要梯度流经 logits → log_prob，供 REINFORCE 使用；
        # 评估时不需要梯度以提高速度。
        if not self.is_training:
            with torch.no_grad():
                logits = self.policy_net(feat_tensor).squeeze(0)
        else:
            logits = self.policy_net(feat_tensor).squeeze(0)

        # 构造合法动作 mask（将非法牌 logits 设为 -inf）
        mask = torch.full((52,), float("-inf"))
        for card in legal_cards:
            mask[card.card_id] = 0.0

        masked_logits = logits + mask
        probs = torch.softmax(masked_logits, dim=0)

        if self.is_training:
            # 训练模式：采样
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)

            # 计算熵（用于熵奖励，鼓励探索）
            entropy = dist.entropy()

            # 根据 card_id 找到对应的 Card 对象
            chosen_card = None
            for c in legal_cards:
                if c.card_id == action_idx.item():
                    chosen_card = c
                    break
            if chosen_card is None:
                # 安全的 fallback（理论上不会执行到这里）
                chosen_card = legal_cards[0]

            # 记录轨迹
            self.trajectory.append({
                "feature": feature.copy(),
                "action": chosen_card,
                "log_prob": log_prob,
                "entropy": entropy,
                "legal_card_ids": [c.card_id for c in legal_cards],
            })

            self.last_play_info = {"mode": "rl_policy_sample"}
            return chosen_card
        else:
            # 评估模式：argmax
            action_idx = torch.argmax(probs).item()
            chosen_card = None
            for c in legal_cards:
                if c.card_id == action_idx:
                    chosen_card = c
                    break
            if chosen_card is None:
                chosen_card = legal_cards[0]

            self.last_play_info = {"mode": "rl_policy_argmax"}
            return chosen_card

    def _exact_play(self, state: GameState, legal_cards: list[Card]) -> Card:
        """使用精确求解器选择动作。"""
        if self.exact_solver is None:
            # 无精确求解器时的 fallback
            self.last_play_info = {"mode": "no_exact_solver_fallback"}
            return legal_cards[0]

        result = self.exact_solver.solve_with_q(state)
        best_action = result.get("best_action")
        if best_action is not None and best_action in legal_cards:
            self.last_play_info = {"mode": "exact"}
            return best_action

        # 精确求解返回的动作不在合法列表中（理论上不会发生）
        self.last_play_info = {"mode": "exact_no_match_fallback"}
        return legal_cards[0]

    def bid_placed(self, bidder: int, bid: Any) -> None:
        pass

    def set_teams(self, teams: list[int], bid_values: list[Any]) -> None:
        pass

    def card_played(self, player_id: int, card: Card) -> None:
        pass
