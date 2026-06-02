"""
RL + Exact 混合出牌玩家。

前 16 张牌（剩余牌数 > 36）：使用策略网络（PolicyMLP）采样出牌。
后 36 张牌（剩余牌数 <= 36）：使用精确双明手求解器出牌。

训练时会记录策略网络的所有决策轨迹（特征、动作、log_prob），
供训练脚本计算 policy gradient 更新。

模型输出维度支持 52 和 55:
- 52: 所有输出对应 52 张牌的 card_id（标准模式）
- 55: 前52维对应领出(card_id)，后3维对应跟牌的3个策略选项
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trick_taking.card import Card, Suit
from trick_taking.game_state import GameState
from trick_taking.player import AIPlayer
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)
from rl.policy_network import PolicyMLP
from rl.rl_feature_encoder import RLFeatureEncoder


class RLExactPlayer(AIPlayer):
    """RL + Exact 混合出牌玩家。

    支持单模型（所有牌位共享）或多模型（每张牌位一个独立策略网络）。

    属性:
        policy_nets: 策略网络列表。
            - 长度 1: 所有牌位共享此网络（兼容 train_rl_multicpu.py）
            - 长度 16: 每个牌位（0~15）对应独立网络
        exact_threshold: 剩余牌数 <= 该值时使用精确求解器
        is_training: 训练模式（True=采样探索，False=argmax 贪心）
        trajectory: 当前对局中所有 RL 决策的记录（含 card_idx）
    """

    def __init__(
        self,
        policy_nets: list[PolicyMLP] | None = None,
        policy_net: PolicyMLP | None = None,
        exact_solver: ExactDoubleDummyCppFastestSolver | None = None,
        encoder: RLFeatureEncoder | None = None,
        exact_threshold: int = 36,
        is_training: bool = True,
        bid_model=None,
        bid_device: str = "cpu",
    ) -> None:
        # 兼容旧调用方式（policy_net 单模型）和新调用方式（policy_nets 列表）
        if policy_nets is not None:
            self.policy_nets = policy_nets
            self.n_policies = len(policy_nets)
        elif policy_net is not None:
            self.policy_nets = [policy_net]
            self.n_policies = 1
        else:
            raise ValueError("必须提供 policy_nets 或 policy_net")
        self.encoder = encoder or RLFeatureEncoder()
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

        # 预训练模式（前4墩逐牌奖励）
        self.pretrain_mode = any(getattr(net, "pretrain_mode", False) for net in self.policy_nets)
        self._current_trick_cards: list[tuple[int, Card]] = []
        self._pending_entry_idx: int = -1

    def start_game(self, position: int, hand: list[Card], num_players: int) -> None:
        self.position = position
        self.hand = list(hand)
        self.last_play_info = {}
        self.trajectory = []
        self._current_trick_cards = []
        self._pending_entry_idx = -1

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

    def _find_best_card_on_table(self, table_cards: list[tuple[int, Card]]) -> Card | None:
        """找出桌上当前最大的牌（考虑将吃规则）。"""
        if not table_cards:
            return None
        lead_suit = table_cards[0][1].suit
        best_card = table_cards[0][1]
        for _, card in table_cards[1:]:
            if card.suit == Suit.SPADES:
                if best_card.suit != Suit.SPADES or card.rank.value > best_card.rank.value:
                    best_card = card
            elif card.suit == lead_suit and best_card.suit != Suit.SPADES:
                if card.rank.value > best_card.rank.value:
                    best_card = card
        return best_card

    def _compute_follow_options(
        self, state: GameState, legal_cards: list[Card]
    ) -> tuple[list[Card | None], list[bool]]:
        """跟牌时计算 3 个策略选项。

        返回:
            option_cards: 长度为 3 的列表，每个元素是对应的 Card 或 None
            option_exists: 长度为 3 的 bool 列表，表示选项是否可用
        """
        table_cards = state.table_cards
        lead_suit = table_cards[0][1].suit
        hand = state.hands[self.position]

        # 按花色分组
        hand_by_suit: dict[Suit, list[Card]] = {}
        for c in hand:
            hand_by_suit.setdefault(c.suit, []).append(c)

        options: list[Card | None] = [None, None, None]
        has_lead_suit = any(c.suit == lead_suit for c in hand)

        # ── 选项 0: 出最小牌 ──
        # 跟牌时出手里最小的该花色牌；垫牌时出最短花色中的最小牌
        if has_lead_suit:
            lead_cards = sorted(
                [c for c in hand if c.suit == lead_suit],
                key=lambda x: x.rank.value,
            )
            options[0] = lead_cards[0]
        else:
            # 垫牌: 找最短花色（非空）中的最小牌
            non_empty = {s: cards for s, cards in hand_by_suit.items() if cards}
            if non_empty:
                shortest_suit = min(non_empty.keys(), key=lambda s: len(non_empty[s]))
                options[0] = min(non_empty[shortest_suit], key=lambda x: x.rank.value)

        # ── 选项 1: 出能赢当前墩的最小牌 ──
        best_card = self._find_best_card_on_table(table_cards)
        beating_candidates: list[Card] = []
        for c in legal_cards:
            if best_card is None:
                beating_candidates.append(c)
            elif c.suit == Suit.SPADES and best_card.suit != Suit.SPADES:
                beating_candidates.append(c)
            elif c.suit == Suit.SPADES and best_card.suit == Suit.SPADES and c.rank.value > best_card.rank.value:
                beating_candidates.append(c)
            elif c.suit == best_card.suit and c.rank.value > best_card.rank.value:
                beating_candidates.append(c)
        if beating_candidates:
            options[1] = min(beating_candidates, key=lambda x: x.rank.value)

        # ── 选项 2: 出最大牌 ──
        if hand:
            options[2] = max(hand, key=lambda x: x.rank.value)

        # 检查每个选项是否合法可用
        option_exists: list[bool] = []
        for i in range(3):
            if options[i] is not None and options[i] in legal_cards:
                option_exists.append(True)
            else:
                option_exists.append(False)
                options[i] = None  # 不可用的置为 None

        return options, option_exists

    def _policy_play(self, state: GameState, legal_cards: list[Card]) -> Card:
        """使用策略网络选择动作（训练时采样，评估时 argmax）。

        根据模型输出维度决定行为:
        - 52 维: 标准模式，所有输出对应 card_id
        - 55 维: 前52维对应领出(card_id)，后3维对应跟牌的3个策略选项
        """
        feature = self.encoder.encode(state, self.position)
        feat_tensor = torch.from_numpy(feature).float().unsqueeze(0)

        # 计算全局出牌位置（0~15）
        n_completed = sum(len(record.cards) for record in state.trick_history)
        n_on_table = len(state.table_cards)
        card_idx = n_completed + n_on_table  # 0~15

        # 选择对应牌位的模型
        if self.n_policies == 1:
            policy_net = self.policy_nets[0]
        else:
            policy_net = self.policy_nets[card_idx]

        # 检测模型输出维度
        output_dim = getattr(policy_net, "output_dim", 52)
        is_55d = (output_dim == 55)

        # 训练时需要梯度流经 logits → log_prob，供 REINFORCE 使用
        if not self.is_training:
            with torch.no_grad():
                all_logits = policy_net(feat_tensor).squeeze(0)
        else:
            all_logits = policy_net(feat_tensor).squeeze(0)

        if is_55d:
            # ── 55 维模式 ──
            is_leading = (len(state.table_cards) == 0)

            if is_leading:
                # 领出: 使用前52维，对应 52 张牌的 card_id
                logits = all_logits[:52]
                mask = torch.full((52,), float("-inf"))
                for card in legal_cards:
                    mask[card.card_id] = 0.0
                masked_logits = logits + mask
                n_logit_dims = 52
                legal_logit_indices = [c.card_id for c in legal_cards]
            else:
                # 跟牌: 使用后3维，对应3个策略选项
                follow_logits = all_logits[52:55]
                option_cards, option_exists = self._compute_follow_options(state, legal_cards)

                # 构造 3 维 mask
                mask = torch.full((3,), float("-inf"))
                legal_logit_indices = []
                for i in range(3):
                    if option_exists[i]:
                        mask[i] = 0.0
                        legal_logit_indices.append(52 + i)

                masked_logits = follow_logits + mask
                n_logit_dims = 3
        else:
            # ── 52 维标准模式 ──
            is_leading = True  # 所有输出对应 card_id，视为"领出模式"
            logits = all_logits
            mask = torch.full((52,), float("-inf"))
            for card in legal_cards:
                mask[card.card_id] = 0.0
            masked_logits = logits + mask
            n_logit_dims = 52
            legal_logit_indices = [c.card_id for c in legal_cards]

        probs = torch.softmax(masked_logits, dim=0)

        if self.is_training:
            # 训练模式：采样
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            entropy = dist.entropy()

            # 根据模式获取实际打出的牌
            if is_55d and not is_leading:
                # 跟牌模式：将 action_idx (0,1,2) 映射到实际牌
                chosen_option = action_idx.item()
                chosen_card = option_cards[chosen_option]
                action_logit_idx = 52 + chosen_option
            else:
                # 领出或52维模式：action_idx 是 card_id
                chosen_card = None
                for c in legal_cards:
                    if c.card_id == action_idx.item():
                        chosen_card = c
                        break
                if chosen_card is None:
                    chosen_card = legal_cards[0]
                action_logit_idx = action_idx.item()

            # 记录轨迹（同时兼容新老格式）
            entry: dict[str, Any] = {
                "feature": feature.copy(),
                "action": chosen_card,
                "log_prob": log_prob,
                "entropy": entropy,
                "card_idx": card_idx,
                # 新格式（55 维模型使用）
                "action_logit_idx": action_logit_idx,
                "legal_logit_indices": list(legal_logit_indices),
                "is_leading": is_leading,
                # 老格式（52 维模型使用）
                "action_id": chosen_card.card_id,
                "legal_card_ids": [c.card_id for c in legal_cards],
            }
            if self.pretrain_mode:
                entry["reward"] = 0.0  # 占位，后续在 card_played 中更新
                self._pending_entry_idx = len(self.trajectory)
            self.trajectory.append(entry)

            if is_55d and not is_leading:
                chosen_option_name = ["最小牌", "能赢的最小牌", "最大牌"][chosen_option]
                self.last_play_info = {
                    "mode": "rl_policy_sample_follow",
                    "card_idx": card_idx,
                    "option": chosen_option_name,
                }
            else:
                self.last_play_info = {"mode": "rl_policy_sample", "card_idx": card_idx}
            return chosen_card
        else:
            # 评估模式：argmax
            action_idx = torch.argmax(probs).item()

            if is_55d and not is_leading:
                chosen_option = action_idx
                chosen_card = option_cards[chosen_option]
            else:
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
        if not self.pretrain_mode:
            return

        # 跟踪当前墩的所有出牌
        self._current_trick_cards.append((player_id, card))

        # 一墩结束（4张牌出完），计算奖励
        if len(self._current_trick_cards) == 4:
            self._compute_pretrain_reward()
            self._current_trick_cards = []

    def _compute_pretrain_reward(self) -> None:
        """计算当前已完成一墩的对抗性逐牌奖励。

        新规则：
        1. 每张牌的积分 = (赢墩? +18 : 0) - rank扣分 (A→17, K→12, Q→8, J→3, T→1)
        2. 我方两张牌的奖励 = 该牌积分 - 0.5 × 对方两张牌积分之和
        """
        entries = self._current_trick_cards  # list of (player_id, Card)

        # 找出该墩赢家
        lead_suit = entries[0][1].suit
        best_idx = 0
        best_card = entries[0][1]
        for i in range(1, 4):
            pid, card = entries[i]
            if card.suit == Suit.SPADES:
                if best_card.suit != Suit.SPADES or card.rank.value > best_card.rank.value:
                    best_idx = i
                    best_card = card
            elif card.suit == lead_suit and best_card.suit != Suit.SPADES:
                if card.rank.value > best_card.rank.value:
                    best_idx = i
                    best_card = card

        # 计算每张牌的积分 = 赢墩加分 - 点数扣分
        def card_score(card: Card, is_winner: bool) -> float:
            base = 18.0 if is_winner else 0.0
            deduction = {14: 17.0, 13: 12.0, 12: 8.0, 11: 3.0, 10: 1.0}.get(card.rank.value, 0.0)
            return base - deduction

        scores = [card_score(card, i == best_idx) for i, (_, card) in enumerate(entries)]

        # 确定对方队伍位置（基于本玩家的实际座位，适配队式赛互换座位）
        # 固定队伍分配：座位 {0, 2} 为队伍 0，{1, 3} 为队伍 1
        opp_positions = {1, 3} if self.position in (0, 2) else {0, 2}
        opp_sum = sum(scores[i] for i, (pid, _) in enumerate(entries) if pid in opp_positions)

        # 我方该牌 reward = 该牌积分 - 0.5 × 对方积分和
        our_score = None
        for i, (pid, _) in enumerate(entries):
            if pid == self.position:
                our_score = scores[i]
                break

        if our_score is None:
            return

        reward = our_score - 0.5 * opp_sum

        # 将奖励赋给对应的 trajectory 条目
        if 0 <= self._pending_entry_idx < len(self.trajectory):
            self.trajectory[self._pending_entry_idx]["reward"] = reward
            self._pending_entry_idx = -1
