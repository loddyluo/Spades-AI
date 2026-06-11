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

import copy
import importlib.util
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import math

from trick_taking.card import Card, Suit, _STANDARD_CARDS as STANDARD_52
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
        self._nil_bidders: set[int] = set()  # 叫 0 的玩家 id，用于奖励调整

        # 加载 rule_based_v2 和 bridge 模块（用于 exact 阶段平局时的出牌建议）
        self._prior_oracle = None
        self._bridge_mod = None
        try:
            collab_root = Path(__file__).resolve().parents[1] / "Spades_AI_GO-MCTS"
            if str(collab_root) not in sys.path:
                sys.path.insert(0, str(collab_root))
            from spades_ai.players.rule_based_v2.player import RuleBasedPlayer as _RBP  # type: ignore
            self._prior_oracle = _RBP()
        except Exception:
            self._prior_oracle = None

        try:
            base = os.path.dirname(__file__)
            bridge_path = os.path.normpath(os.path.join(base, "..", "evaluate", "GO-MCTS", "bridge.py"))
            if os.path.exists(bridge_path):
                spec = importlib.util.spec_from_file_location("_go_bridge", bridge_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    self._bridge_mod = mod
        except Exception:
            self._bridge_mod = None

    def start_game(self, position: int, hand: list[Card], num_players: int) -> None:
        self.position = position
        self.hand = list(hand)
        self.last_play_info = {}
        self.trajectory = []
        self._current_trick_cards = []
        self._pending_entry_idx = -1
        self._nil_bidders = set()

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

        # ── 选项 0: 出最小牌（默认）或 nil 安全策略 ──
        # 如果自己叫了 0 且目前吃了 0 墩，则出比桌上最大牌小的牌中最大的一张
        # （避免意外赢墩）；否则出最小牌
        is_nil = False
        if hasattr(state, "max_bid") and state.max_bid:
            my_bid = state.max_bid[self.position] if self.position < len(state.max_bid) else None
            is_nil = my_bid in ("nil", "blind_nil")

        nil_zero_tricks = False
        if is_nil:
            tricks_won = 0
            for record in state.trick_history:
                trick_cards = record.cards
                lead_suit_in_trick = trick_cards[0][1].suit
                best_idx = 0
                best_card_in_trick = trick_cards[0][1]
                for i in range(1, 4):
                    _, card = trick_cards[i]
                    if card.suit == Suit.SPADES:
                        if best_card_in_trick.suit != Suit.SPADES or card.rank.value > best_card_in_trick.rank.value:
                            best_idx = i
                            best_card_in_trick = card
                    elif card.suit == lead_suit_in_trick and best_card_in_trick.suit != Suit.SPADES:
                        if card.rank.value > best_card_in_trick.rank.value:
                            best_idx = i
                            best_card_in_trick = card
                winner_pid = trick_cards[best_idx][0]
                if winner_pid == self.position:
                    tricks_won += 1
            nil_zero_tricks = (tricks_won == 0)

        if nil_zero_tricks:
            # nil 安全策略：出比桌上最大牌小的牌中最大的一张
            best_on_table = self._find_best_card_on_table(table_cards)
            if has_lead_suit:
                candidates = [c for c in hand if c.suit == lead_suit]
            else:
                candidates = list(hand)
            # 找出不会赢墩的牌
            safe_cards = []
            for c in candidates:
                if best_on_table is None:
                    safe_cards.append(c)
                elif c.suit == Suit.SPADES and best_on_table.suit != Suit.SPADES:
                    pass  # 将吃会赢墩，排除
                elif c.suit != best_on_table.suit and c.suit != Suit.SPADES:
                    safe_cards.append(c)  # 垫不同花色不会赢
                elif c.suit == best_on_table.suit and c.rank.value < best_on_table.rank.value:
                    safe_cards.append(c)  # 同花色但更小
            if safe_cards:
                options[0] = max(safe_cards, key=lambda x: x.rank.value)
            else:
                # 没有安全牌，兜底出最小牌
                if has_lead_suit:
                    lead_cards = sorted(
                        [c for c in hand if c.suit == lead_suit],
                        key=lambda x: x.rank.value,
                    )
                    options[0] = lead_cards[0]
                else:
                    non_empty = {s: cards for s, cards in hand_by_suit.items() if cards}
                    if non_empty:
                        shortest_suit = min(non_empty.keys(), key=lambda s: len(non_empty[s]))
                        options[0] = min(non_empty[shortest_suit], key=lambda x: x.rank.value)
        else:
            # 原逻辑：出最小牌
            if has_lead_suit:
                lead_cards = sorted(
                    [c for c in hand if c.suit == lead_suit],
                    key=lambda x: x.rank.value,
                )
                options[0] = lead_cards[0]
            else:
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
        """使用精确求解器选择动作（对对手手牌做 32 次确定化采样后平均）。"""
        if self.exact_solver is None:
            self.last_play_info = {"mode": "no_exact_solver_fallback"}
            return legal_cards[0]

        rng = random.Random()
        K = 32
        agg_q: dict[int, float] = {}

        for _ in range(K):
            sim_state = copy.deepcopy(state)
            self._determinize_state(sim_state, state.turn, rng)
            result = self.exact_solver.solve_with_q(sim_state)
            for action, q in result.get("action_q_values", {}).items():
                aid = action.card_id
                agg_q[aid] = agg_q.get(aid, 0.0) + float(q)

        # 求平均
        for k in agg_q:
            agg_q[k] /= K

        # 选 Q 值最高的合法动作
        best_action = None
        best_q = float("-inf")
        for card in legal_cards:
            q = agg_q.get(card.card_id, float("-inf"))
            if q > best_q:
                best_q = q
                best_action = card

        if best_action is not None:
            self.last_play_info = {"mode": "exact_determinized", "samples": K}
            return best_action

        self.last_play_info = {"mode": "exact_no_match_fallback"}
        return legal_cards[0]

    def _determinize_state(self, state: GameState, observer_id: int,
                           rng: random.Random | None = None) -> None:
        """替换对手手牌为随机未露面牌（保留己方手牌和已打出的牌）。"""
        if rng is None:
            rng = random.Random()

        # 收集已占用的牌：观察者手牌 + 已打出牌
        used_ids: set[int] = set()
        for c in state.hands[observer_id]:
            used_ids.add(c.card_id)

        bitset = getattr(state, "played_bitset", 0)
        for cid in range(52):
            if bitset & (1 << cid):
                used_ids.add(cid)

        for pair in getattr(state, "table_cards", []):
            used_ids.add(pair[1].card_id)

        for record in getattr(state, "trick_history", []):
            for _, c in getattr(record, "cards", []):
                used_ids.add(c.card_id)

        pool = [c for c in STANDARD_52 if c.card_id not in used_ids]
        rng.shuffle(pool)

        indices = [pid for pid in range(state.num_players) if pid != observer_id]
        counts = {pid: len(state.hands[pid]) for pid in indices}

        pos = 0
        for pid in indices:
            n = counts[pid]
            assigned = pool[pos: pos + n]
            pos += n
            state.hands[pid] = list(assigned)

        if hasattr(state, "hand_bitsets"):
            for pid in range(state.num_players):
                bit = 0
                for c in state.hands[pid]:
                    bit |= (1 << c.card_id)
                state.hand_bitsets[pid] = bit

    # ── IS 确定化（重要性采样 + top-K 加权精确求解）─────────────────────────
    # 与 TruncatedMCTSStrategy._solve_with_determinization 一致

    @staticmethod
    def _build_play_sequence(state: GameState) -> list[tuple[int, Card]]:
        """Extract ordered (player_id, Card) sequence from trick_history + table_cards."""
        sequence: list[tuple[int, Card]] = []
        for record in state.trick_history:
            for pid, card in record.cards:
                sequence.append((pid, card))
        for pid, card in state.table_cards:
            sequence.append((pid, card))
        return sequence

    @staticmethod
    def _bid_str_to_mlp_index(bid_str: str) -> int:
        if bid_str == "nil":
            return 14
        if bid_str == "blind_nil":
            return 15
        if bid_str.startswith("bid_"):
            return int(bid_str.split("_")[1])
        return 0

    def _compute_bid_probs_product(
        self, initial_hands: list[list[Card]], max_bid: list[str],
    ) -> float:
        """∏ P(bid_p | hand_p) from BidMLP softmax."""
        # lazy-load BidMLP for IS weighting (separate from self._bid_model)
        if not hasattr(self, "_bid_model_is") or self._bid_model_is is None:
            ckpt_path = ""
            if hasattr(self, "_bid_model") and self._bid_model is not None:
                pass  # use the already-loaded bid model if compatible
            # Try to load from standard path
            try:
                go_dir = Path(__file__).resolve().parents[1] / "evaluate" / "GO-MCTS"
                if str(go_dir) not in sys.path:
                    sys.path.insert(0, str(go_dir))
                from spades_ai.models.bid_mlp import BidMLP
                from spades_ai.models.bid_encoder import BidEncoder

                # Find checkpoint
                possible_paths = [
                    Path("./Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt"),
                    Path(__file__).resolve().parents[1] / "Spades_AI_GO-MCTS" / "checkpoints" / "bid_nsfp.pt",
                ]
                ckpt = None
                for p in possible_paths:
                    if p.exists():
                        ckpt = str(p.resolve())
                        break
                if ckpt:
                    self._bid_model_is = BidMLP()
                    sd = torch.load(ckpt, weights_only=True, map_location="cpu")
                    self._bid_model_is.load_state_dict(sd)
                    self._bid_model_is.eval()
                    self._bid_encoder_is = BidEncoder()
                else:
                    self._bid_model_is = None
                    return 1.0
            except Exception:
                self._bid_model_is = None
                return 1.0

        if not hasattr(self, "_bid_encoder_is") or self._bid_encoder_is is None:
            return 1.0

        from spades_ai.game.card import Card as GoCard
        from spades_ai.game.card import Rank as GoRank, Suit as GoSuit
        from spades_ai.game.state import Bid as GoBid
        from spades_ai.game.scoring import BidType as GoBidType

        def _to_go_bid(bid_str: str) -> GoBid:
            if bid_str == "nil":
                return GoBid(value=0, bid_type=GoBidType.NIL)
            if bid_str == "blind_nil":
                return GoBid(value=0, bid_type=GoBidType.BLIND_NIL)
            if bid_str.startswith("bid_"):
                return GoBid(value=int(bid_str.split("_")[1]), bid_type=GoBidType.NORMAL)
            return GoBid(value=0, bid_type=GoBidType.NORMAL)

        go_bids = [_to_go_bid(b) for b in max_bid]

        features_list = []
        for p in range(4):
            hand = [GoCard(GoRank(c.rank.value), GoSuit[c.suit.name]) for c in initial_hands[p]]
            prev = go_bids[:p]
            position = min(p, 2)
            features = self._bid_encoder_is.encode(hand, prev, position)
            features_list.append(features.unsqueeze(0))

        x = torch.cat(features_list, dim=0)
        with torch.no_grad():
            logits = self._bid_model_is(x)
        probs = torch.softmax(logits, dim=-1)
        uniform = 1.0 / 14
        smoothed = 0.99 * probs + 0.01 * uniform

        product = 1.0
        for p in range(4):
            idx = self._bid_str_to_mlp_index(max_bid[p])
            product *= float(smoothed[p, idx].item())
        return product

    def _generate_proposal(
        self, all_cards: list[Card], observer_id: int,
        observer_current_hand: list[Card],
        played_by_player: dict[int, list[Card]], rng: random.Random,
    ) -> list[list[Card]]:
        """Generate one random initial deal consistent with observed play."""
        obs_set: set[int] = set(c.card_id for c in observer_current_hand)
        obs_set.update(c.card_id for c in played_by_player[observer_id])
        id_to_card = {c.card_id: c for c in all_cards}
        observer_initial = [id_to_card[cid] for cid in obs_set]

        used_ids: set[int] = set(obs_set)
        for p in range(4):
            if p != observer_id:
                used_ids.update(c.card_id for c in played_by_player[p])

        pool = [c for c in all_cards if c.card_id not in used_ids]
        rng.shuffle(pool)

        initial_hands: list[list[Card]] = [None] * 4  # type: ignore
        initial_hands[observer_id] = list(observer_initial)

        idx = 0
        for p in range(4):
            if p == observer_id:
                continue
            initial_hands[p] = list(played_by_player[p])
            need = 13 - len(played_by_player[p])
            initial_hands[p].extend(pool[idx: idx + need])
            idx += need

        return initial_hands

    def _compute_importance_weight(
        self, initial_hands: list[list[Card]],
        play_sequence: list[tuple[int, Card]],
        max_bid: list[str] | None = None,
    ) -> float:
        """Replay play_sequence against initial_hands; compute p = ∏(p_step).

        p = P_bid * ∏_{step} (1 / legal_count).
        Returns 0 if any move was illegal given this deal.
        """
        if max_bid is not None:
            bid_prod = self._compute_bid_probs_product(initial_hands, max_bid)
        else:
            bid_prod = 1.0

        hands = [list(h) for h in initial_hands]
        spades_broken = False
        pos_in_trick = 0
        led_suit: Suit | None = None
        weight = 1.0

        for step_idx, (player, card) in enumerate(play_sequence):
            hand = hands[player]
            try:
                idx = hand.index(card)
            except ValueError:
                return 0.0

            if pos_in_trick == 0:  # Leading
                if not spades_broken and card.suit == Suit.SPADES:
                    has_non_spade = any(c.suit != Suit.SPADES for c in hand)
                    if has_non_spade:
                        return 0.0
                if not spades_broken:
                    non_spades = [c for c in hand if c.suit != Suit.SPADES]
                    legal_count = len(non_spades) if non_spades else len(hand)
                else:
                    legal_count = len(hand)
                led_suit = card.suit
            else:  # Following
                has_led = any(c.suit == led_suit for c in hand)
                if has_led and card.suit != led_suit:
                    return 0.0
                legal_count = (sum(1 for c in hand if c.suit == led_suit)
                               if has_led else len(hand))

            weight *= 1.0 #/ legal_count
            hand.pop(idx)

            if card.suit == Suit.SPADES:
                spades_broken = True
            pos_in_trick = (pos_in_trick + 1) % 4
            if pos_in_trick == 0:
                led_suit = None
        ##print(weight)
        return weight * bid_prod * math.exp(random.uniform(0, 0.4))
        #return (weight* math.exp(random.uniform(0, 8)))**0.3 * bid_prod  # 0.3 是经验值，调小一些以增加多样性

    def _build_is_pool(
        self, state: GameState, observer_id: int, rng: random.Random,
        num_proposals: int = 6789,
    ) -> tuple[list[list[list[Card]]], list[float]]:
        """Build IS pool: generate proposals, compute weights."""
        play_sequence = self._build_play_sequence(state)

        max_bid: list[str] | None = None
        raw_bids = None
        if hasattr(state, "max_bid") and state.max_bid:
            raw_bids = state.max_bid
        elif hasattr(state, "bids") and state.bids:
            raw_bids = state.bids
        if raw_bids is not None and len(raw_bids) == 4:
            if all(isinstance(b, str) for b in raw_bids):
                max_bid = list(raw_bids)

        played_by_player: dict[int, list[Card]] = {p: [] for p in range(4)}
        for p, c in play_sequence:
            played_by_player[p].append(c)

        proposals: list[list[list[Card]]] = []
        prop_weights: list[float] = []

        for _ in range(num_proposals):
            initial_hands = self._generate_proposal(
                state.all_cards, observer_id, state.hands[observer_id],
                played_by_player, rng,
            )
            w = self._compute_importance_weight(
                initial_hands, play_sequence, max_bid=max_bid,
            )
            if w > 0.0:
                proposals.append(initial_hands)
                prop_weights.append(w)

        return proposals, prop_weights

    @staticmethod
    def _apply_proposal(
        state: GameState, observer_id: int, proposal: list[list[Card]],
    ) -> None:
        """Apply a proposal (set opponents' hands from proposal initial deal)."""
        played_by_player: dict[int, set[int]] = {i: set() for i in range(4)}
        for record in state.trick_history:
            for pid, card in record.cards:
                played_by_player[pid].add(card.card_id)
        for pid, card in state.table_cards:
            played_by_player[pid].add(card.card_id)

        for p in range(4):
            if p != observer_id:
                remaining = [c for c in proposal[p] if c.card_id not in played_by_player[p]]
                state.hands[p] = remaining

        if hasattr(state, "hand_bitsets"):
            for p in range(4):
                bit = 0
                for c in state.hands[p]:
                    bit |= (1 << c.card_id)
                state.hand_bitsets[p] = bit

    def _exact_play(self, state: GameState, legal_cards: list[Card]) -> Card:
        """使用精确求解器：IS proposal + top-K 加权平均（与 TruncatedMCTSStrategy 一致）。"""
        if self.exact_solver is None:
            self.last_play_info = {"mode": "no_exact_solver_fallback"}
            return legal_cards[0]

        rng = random.Random()
        K = 32
        id_to_card = {c.card_id: c for c in STANDARD_52}

        # Build IS pool
        pool_hands, pool_weights = self._build_is_pool(state, state.turn, rng)

        agg_q: dict[int, float] = {}
        my_team = 0 if self.position in (0, 2) else 1
        if not pool_hands:
            # Fallback: uniform determinization
            counts = 0
            for _ in range(K):
                sim_state = copy.deepcopy(state)
                self._determinize_state(sim_state, state.turn, rng)
                result = self.exact_solver.solve_with_q(sim_state)
                counts += 1
                for action, q in result.get("action_q_values", {}).items():
                    aid = action.card_id
                    agg_q[aid] = agg_q.get(aid, 0.0) + float(q)
            for k in agg_q:
                agg_q[k] /= max(1, counts)
        else:
            # Top-K by weight, weighted average (去重后取唯一的前 K 个)
            paired = list(zip(pool_hands, pool_weights))
            paired.sort(key=lambda x: x[1], reverse=True)
            seen = set()
            unique_paired = []
            for hand_proposal, w in paired:
                key = tuple(
                    tuple(c.card_id for c in hand)
                    for hand in hand_proposal
                )
                if key not in seen:
                    seen.add(key)
                    unique_paired.append((hand_proposal, w))
                    if len(unique_paired) >= K:
                        break
            top_hands, top_weights = zip(*unique_paired) if unique_paired else ([], [])
            weight_sum = sum(top_weights)
            norm_factors = [w / weight_sum for w in top_weights] if weight_sum > 0 else [1.0 / K] * K
            #print(norm_factors, "length= ", len(norm_factors))

            for hand_proposal, norm_w in zip(top_hands, norm_factors):
                sim_state = copy.deepcopy(state)
                self._apply_proposal(sim_state, state.turn, hand_proposal)
                result = self.exact_solver.solve_with_q(sim_state)
                action_q_dict = result.get("action_q_values", {})
                if action_q_dict:
                    max_q = max(float(q) for q in action_q_dict.values())
                    min_q = min(float(q) for q in action_q_dict.values())
                    for action, q in action_q_dict.items():
                        aid = action.card_id
                        if my_team == 0:
                            multiplier = float(q) - max_q  # Subtract max_q for numerical stability
                            if multiplier < -40.0:
                                multiplier *= 10.0
                            agg_q[aid] = agg_q.get(aid, 0.0) + norm_w * multiplier
                        else:
                            #print("the player use this way")
                            multiplier = float(q) - min_q
                            if multiplier > 40.0:
                                multiplier *= 10.0
                            agg_q[aid] = agg_q.get(aid, 0.0) + norm_w * multiplier

        # Reconstruct action -> q using Card objects
        action_q_values: dict[Card, float] = {}
        for aid, q in agg_q.items():
            if aid in id_to_card:
                action_q_values[id_to_card[aid]] = q

        # 根据玩家所在队伍选择动作：
        #   队伍 0 (座位 0,2) → max Q (Q 是 team0 - team1，越大越好)
        #   队伍 1 (座位 1,3) → min Q (分差越小，对 team1 越有利)

        if my_team == 0:
            best_q = max(action_q_values.values()) if action_q_values else None
        else:
            best_q = min(action_q_values.values()) if action_q_values else None

        #print(action_q_values)
        # 选出 Q 值并列第一的合法牌
        if best_q is not None:
            tied_cards = [c for c in legal_cards if c in action_q_values and action_q_values[c] == best_q]
        else:
            tied_cards = []
        if not tied_cards:
            # 兜底：取有任何 Q 值的合法牌
            tied_cards = [c for c in legal_cards if c in action_q_values]

        # 检查是否有人叫 0
        has_nil = False
        if hasattr(state, "max_bid") and state.max_bid:
            has_nil = any(
                isinstance(b, str) and b in ("nil", "blind_nil")
                for b in state.max_bid
            )

        # 花色优先级：S(0) > H(1) > D(2) > C(3)，同花色点数越大优先级越高
        # 有人叫 0 → 出优先级最高的牌（S大牌 > ... > C小牌）
        # 没人叫 0 → 出优先级最低的牌（C小牌 > ... > S大牌）
        def _card_priority_key(card: Card) -> tuple[int, int]:
            return (card.suit.value, -card.rank.value)

        if tied_cards:
            # 先看 rule_based_v2 的建议是否在 tied_cards 中
            #print("tied_cards:", tied_cards)
            rb_action = None
            if self._prior_oracle is not None and self._bridge_mod is not None:
                try:
                    go_state = self._bridge_mod.to_go_state(state)
                    go_card = self._prior_oracle.choose_card(go_state)
                    local_card = self._bridge_mod.to_local_card(go_card)
                    for c in tied_cards:
                        if c.card_id == local_card.card_id:
                            #print("action 1")
                            rb_action = c
                            break
                except Exception:
                    rb_action = None

            if rb_action is not None:
                best_action = rb_action
            elif has_nil:
                #print("has nil action 3")
                best_action = min(tied_cards, key=_card_priority_key)  # 优先级最高
            else:
                #print("action 2")
                best_action = max(tied_cards, key=_card_priority_key)  # 优先级最低
        else:
            best_action = None

        # 构造 action_scores 供 trace 日志记录（格式与 TruncatedMCTSStrategy 一致）
        action_scores = sorted(
            [{"action": card, "value": float(q)}
             for card, q in action_q_values.items()],
            key=lambda x: x["value"], reverse=True,
        )
        best_value = float(action_q_values.get(best_action, 0.0)) if best_action else 0.0

        if best_action is not None and best_action in legal_cards:
            self.last_play_info = {
                "mode": "exact_is_determinized",
                "samples": K,
                "best_value": best_value,
                "action_scores": action_scores,
            }
            return best_action

        self.last_play_info = {"mode": "exact_no_match_fallback"}
        return legal_cards[0]

    def bid_placed(self, bidder: int, bid: Any) -> None:
        if self.pretrain_mode:
            if isinstance(bid, str) and bid in ("nil", "blind_nil"):
                self._nil_bidders.add(bidder)

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
        2. 若出牌方队伍有人叫 0：叫 0 者赢墩则 -25，否则 +2
        3. 我方两张牌的奖励 = 该牌积分 - 0.5 × 对方两张牌积分之和
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

        # 计算每张牌的积分（含 nil 调整）
        scores = []
        for i, (pid, card) in enumerate(entries):
            # 基础分 = 赢墩加分 - 点数扣分 + 黑桃加分
            base = 18.0 if i == best_idx else 0.0
            deduction = {14: 16.0, 13: 12.0, 12: 8.0, 11: 3.0, 10: 1.0}.get(card.rank.value, 0.0)
            spade_bonus = 2.0 if card.suit == Suit.SPADES else 0.0
            score = base - deduction + spade_bonus

            # nil 调整：这张牌的出牌方队伍有人叫了 0
            for nil_pid in self._nil_bidders:
                if nil_pid % 2 == pid % 2:  # 同一队伍
                    # 找到叫 0 者在这墩中的位置
                    for j, (entry_pid, _) in enumerate(entries):
                        if entry_pid == nil_pid:
                            if j == best_idx:
                                score -= 50.0  # nil 者赢墩：扣 50
                            else:
                                score += 4.0   # nil 者没赢：加 4
                            break
                    break  # 同一队有多个叫 0 者也只调整一次

            scores.append(score)

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
