"""RuleExactFirst4Player —— 与 RLExactPlayer 等价的"前 4 墩 + 后 36 张精确"混合玩家,
但前 4 墩用规则式策略 (RuleBasedFirst4Player) 而不是 RL policy 网络。

设计目标:
- 评估 `RuleBasedFirst4Player` 的前 4 墩出牌质量, 在与 `rl_exact` 同等条件下
  (后 36 张同一个精确求解器), 直接对比两者作为前 4 墩的差异。
- 切换条件与 RLExactPlayer 完全一致: 用"剩余总牌数 remaining = sum(len(h)
  for h in state.hands)", `remaining > exact_threshold(默认 36)` 时走 rule-based,
  否则走 exact solver。

用法 (评估时):
    player = RuleExactFirst4Player(
        exact_solver=ExactDoubleDummyCppFastestSolver(),
        exact_threshold=36,
        bid_model=bid_model,    # 与 DDSPlayer / RLExactPlayer 用同一个叫牌模型
        bid_device="cpu",
    )

注意:
- 叫牌完全照抄 RLExactPlayer 的 MLP-bid 逻辑, 这样和 DDS / RL 公平。
- 规则式玩家自身只读自己的手牌+桌面+历史(盲眼); 后 36 张的 exact solver 对对手
  手牌做 32 次 determinize 采样后平均 Q 值, 与 RLExactPlayer 的 exact 阶段一致。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch

from trick_taking.card import Card, Rank, Suit, _STANDARD_CARDS as STANDARD_52
from trick_taking.game_state import GameState
from trick_taking.player import AIPlayer
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)
from strategy.hyperparam_config import HyperparamConfig
from strategy.rule_based_first4_player import (
    RuleBasedFirst4Player,
    _trick_current_winner,
)


# ── parallel solver worker (module-level for multiprocessing picklability) ──

def _parallel_solve_worker(args: tuple) -> dict[int, float]:
    """Solve one proposal in a worker process. Returns {card_id: q_value}.

    This function runs in a spawned child process.  It creates its own solver
    instance and deep-copies the state, so the process-global TT buffer is
    isolated per worker.

    Args:
        args: (state, observer_id, hand_proposal) where
              state is a GameState (dataclass, picklable),
              hand_proposal is list[list[Card]] (4 player's full starting hands).

    Returns:
        dict mapping card_id → q_value.  Returns empty dict on failure.
    """
    import copy as _copy
    from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
        ExactDoubleDummyCppFastestSolver as _WorkerSolver,
    )

    state, observer_id, hand_proposal = args

    try:
        sim_state = _copy.deepcopy(state)

        # ── inline _apply_proposal (avoid importing the whole module) ──
        played_by: dict[int, set] = {i: set() for i in range(4)}
        for record in sim_state.trick_history:
            for pid, card in record.cards:
                played_by[pid].add(card.card_id)
        for pid, card in sim_state.table_cards:
            played_by[pid].add(card.card_id)

        for p in range(4):
            if p != observer_id:
                remaining = [c for c in hand_proposal[p]
                             if c.card_id not in played_by[p]]
                sim_state.hands[p] = remaining

        if hasattr(sim_state, 'hand_bitsets'):
            for p in range(4):
                bit = 0
                for c in sim_state.hands[p]:
                    bit |= (1 << c.card_id)
                sim_state.hand_bitsets[p] = bit

        # ── solve ──
        solver = _WorkerSolver()
        return solver.solve_with_q_fast(sim_state)
    except Exception:
        return {}


# ── minimum batch size to trigger parallel solving ──
_MIN_PARALLEL_BATCH = 8

# ``fork`` from the threaded HTTP/WebSocket servers can inherit a locked
# Python mutex or partially initialized torch/native runtime.  ``spawn``
# starts each solver worker from a clean interpreter instead.
_SOLVER_MP_START_METHOD = "spawn"


class RuleExactFirst4Player(AIPlayer):
    """前 4 墩 rule-based + 后 36 张 exact solver 的混合玩家。

    与 `rl.rl_exact_player.RLExactPlayer` 一一对应, 仅前 4 墩的决策来源不同。

    当有人叫 nil 时，前 4 墩自动切换为使用 nil 策略网络 (55_2nil.pt) 出牌，
    与 RLExactPlayer 的行为保持一致。
    """

    _DECISION_SEED_VERSION = "rule-exact-visible-v1"

    def __init__(
        self,
        exact_solver: ExactDoubleDummyCppFastestSolver | None = None,
        exact_threshold: int = 36,
        bid_model=None,
        bid_device: str = "cpu",
        policy_net_nil: Any | None = None,
        encoder: Any | None = None,
        hyperparam_config: HyperparamConfig | None = None,
        num_workers: int = 0,
        debug: bool = False,
    ) -> None:
        # 内部规则式玩家. 不让它处理后 36 张 (我们自己路由)。
        self._rule_player = RuleBasedFirst4Player()
        self.exact_threshold = exact_threshold
        self._bid_model = bid_model
        self._bid_device = bid_device
        self.config = hyperparam_config or HyperparamConfig.default()

        # 并行 worker 数：构造参数优先，否则从 config 取，0 = auto (cpu_count)
        if num_workers > 0:
            self._num_workers = num_workers
        elif hasattr(self.config, 'num_workers') and self.config.num_workers > 0:
            self._num_workers = self.config.num_workers
        else:
            import os as _os
            self._num_workers = max(1, (_os.cpu_count() or 4) - 1)  # 给主进程留 1 核

        self.position: int = -1
        self.hand: list[Card] = []
        self.last_play_info: dict[str, Any] = {}
        self.last_bid_info: dict[str, Any] | None = None
        self._debug = debug  # 调试模式：在 last_play_info 中记录采样提案和 Q 值

        # nil 策略网络（有人叫 0 时替代 rule-based 打前 4 墩）
        self._policy_net_nil = policy_net_nil
        self._encoder = encoder
        self._has_nil_bid = False
        self._rl_nil_player: Any | None = None

        # 精确求解器
        if exact_solver is not None:
            self.exact_solver = exact_solver
        else:
            cpp_solver = ExactDoubleDummyCppFastestSolver()
            self.exact_solver = cpp_solver if cpp_solver.native_available else None

    # ─── 全部回调都需要双向转发到内部 _rule_player ─────────────────────

    def start_game(self, position: int, hand: list[Card], num_players: int) -> None:
        self.position = position
        self.hand = list(hand)
        self.last_play_info = {}
        self._has_nil_bid = False
        self._rl_nil_player = None
        self._rule_player.start_game(position, hand, num_players)

    def place_bid(self, legal_bids: list[Any], state_view: dict) -> Any:
        """与 RLExactPlayer.place_bid 完全相同的 MLP 叫牌逻辑。"""
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

    def bid_placed(self, bidder: int, bid: Any) -> None:
        # rule_player 不关心 bid, 但保持转发(以防未来扩展)
        try:
            self._rule_player.bid_placed(bidder, bid)
        except Exception:
            pass

    def set_teams(self, teams: list[int], bid_values: list[Any]) -> None:
        try:
            self._rule_player.set_teams(teams, bid_values)
        except Exception:
            pass
        # 检测是否有人叫 nil，若是则在 / 前 4 墩切换到策略网络
        nil_bid = any(
            isinstance(bv, str) and bv in ("nil", "blind_nil")
            for bv in bid_values
        )
        self._has_nil_bid = nil_bid
        if nil_bid and self._policy_net_nil is not None and self._rl_nil_player is None:
            # 导入 RLExactPlayer 并创建内部实例处理前 4 墩
            from rl.policy_network import PolicyMLP
            from rl.rl_feature_encoder import RLFeatureEncoder
            from rl.rl_exact_player import RLExactPlayer

            encoder = self._encoder or RLFeatureEncoder()
            self._rl_nil_player = RLExactPlayer(
                policy_nets=[self._policy_net_nil],
                exact_solver=self.exact_solver,
                encoder=encoder,
                exact_threshold=0,  # 只用 policy play
                is_training=False,
                bid_model=self._bid_model,
                bid_device=self._bid_device,
            )
            self._rl_nil_player.start_game(self.position, self.hand, 4)

    def card_played(self, player_id: int, card: Card) -> None:
        # 关键: 规则式玩家依赖这个回调跟踪历史, 必须转发
        self._rule_player.card_played(player_id, card)
        if self._rl_nil_player is not None:
            self._rl_nil_player.card_played(player_id, card)

    # ─── 出牌路由 ──────────────────────────────────────────────────

    def play_card(self, legal_cards: list[Card], state_view: dict) -> Card:
        state: GameState | None = state_view.get("state")
        if state is None:
            self.last_play_info = {"mode": "no_state_fallback"}
            return legal_cards[0]

        # 唯一合法动作无需决策；最后一张保留原有诊断标记。
        my_remaining = len(state.hands[self.position])
        if len(legal_cards) == 1:
            mode = "last_card_direct" if my_remaining <= 1 else "single_action_direct"
            self.last_play_info = {"mode": mode}
            return legal_cards[0]

        # 与 RLExactPlayer 完全相同的切换条件
        remaining = sum(len(h) for h in state.hands)

        if remaining <= self.exact_threshold:
            return self._exact_play(state, legal_cards)

        # 前 4 墩：有人叫 nil 且策略网络可用 → 使用 nil 策略网络
        if self._has_nil_bid and self._rl_nil_player is not None:
            card = self._rl_nil_player.play_card(legal_cards, state_view)
            self.last_play_info = {"mode": "nil_policy_first4"}
            return card

        return self._rule_play(legal_cards, state_view)

    def _rule_play(self, legal_cards: list[Card], state_view: dict) -> Card:
        card = self._rule_player.play_card(legal_cards, state_view)
        self.last_play_info = {"mode": "rule_first4"}
        return card

    @staticmethod
    def _canonical_card_choice(cards: list[Card]) -> Card:
        """Choose a stable fallback independent of the caller's list order."""
        return min(cards, key=lambda card: card.card_id)

    @staticmethod
    def _stable_bid_value(value: Any) -> str | int | None:
        """Return the stable primitive forms used by Spades max_bid."""
        if value is None or isinstance(value, (str, int)):
            return value
        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, (str, int)):
            return enum_value
        return type(value).__qualname__

    @classmethod
    def _decision_seed(
        cls,
        state: GameState,
        observer_id: int,
        legal_cards: list[Card],
    ) -> int:
        """Hash only the observer-visible decision state into a stable seed."""
        completed_tricks = []
        for record in getattr(state, "trick_history", []):
            cards = [
                [int(player_id), int(card.card_id)]
                for player_id, card in getattr(record, "cards", [])
            ]
            completed_tricks.append(
                {
                    "cards": cards,
                    "leader": int(getattr(record, "leader", cards[0][0] if cards else -1)),
                    "winner": int(getattr(record, "winner", -1)),
                }
            )

        phase = getattr(state, "phase", None)
        phase_name = getattr(phase, "name", str(phase))
        hands = getattr(state, "hands", [])
        own_hand = hands[observer_id] if observer_id < len(hands) else []
        snapshot = {
            "version": cls._DECISION_SEED_VERSION,
            "phase": phase_name,
            "observer": int(observer_id),
            "num_players": int(getattr(state, "num_players", len(hands))),
            "own_hand": sorted(int(card.card_id) for card in own_hand),
            "legal_cards": sorted(int(card.card_id) for card in legal_cards),
            "hand_sizes": [len(hand) for hand in hands],
            "max_bid": [
                cls._stable_bid_value(value)
                for value in getattr(state, "max_bid", [])
            ],
            "teams": [int(team) for team in getattr(state, "teams", [])],
            "turn": int(getattr(state, "turn", observer_id)),
            "trick_leader": int(getattr(state, "trick_leader", observer_id)),
            "tricks_played": int(getattr(state, "tricks_played", 0)),
            "tricks_won": [int(value) for value in getattr(state, "tricks_won", [])],
            "spades_broken": bool(
                getattr(state, "spades_broken", False)
                or getattr(state, "trump_broken", False)
            ),
            "completed_tricks": completed_tricks,
            "table_cards": [
                [int(player_id), int(card.card_id)]
                for player_id, card in getattr(state, "table_cards", [])
            ],
        }
        encoded = json.dumps(
            snapshot,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.blake2b(encoded, digest_size=16).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    def _exact_play(self, state: GameState, legal_cards: list[Card]) -> Card:
        """使用精确求解器：IS proposal + top-K 加权平均（与 RLExactPlayer 一致）。"""
        if self.exact_solver is None:
            self.last_play_info = {"mode": "no_exact_solver_fallback"}
            return self._canonical_card_choice(legal_cards)

        rng = random.Random(self._decision_seed(state, state.turn, legal_cards))
        # 计算剩余牌数，决定采样预算：越往后预算越大
        remaining_in = sum(len(h) for h in state.hands)
        top_k, max_samples = self.config.budget.lookup(remaining_in)
        K = max_samples
        id_to_card = {c.card_id: c for c in STANDARD_52}

        # Build IS pool (with config params)
        pool_hands, pool_weights = self._build_is_pool(
            state, state.turn, rng,
            num_proposals=self.config.num_proposals,
            num_proposals_limit=self.config.num_proposals_limit,
            min_pool_size=self.config.min_pool_size,
        )

        agg_q: dict[int, float] = {}
        my_team = 0 if self.position in (0, 2) else 1

        # ── debug: 记录 IS pool 原始统计 ──
        _debug_pool_info: dict[str, Any] | None = None
        _debug_unique_paired: list[dict[str, Any]] | None = None
        _debug_samples: list[dict[str, Any]] | None = None
        _debug_n_fill: int = 0
        _debug_n_is: int = 0
        if self._debug:
            _debug_pool_info = {
                "pool_size": len(pool_hands),
                "pool_weights_min": float(min(pool_weights)) if pool_weights else None,
                "pool_weights_max": float(max(pool_weights)) if pool_weights else None,
                "pool_weights_mean": float(sum(pool_weights) / len(pool_weights)) if pool_weights else None,
            }

        if not pool_hands:
            # Fallback: uniform determinization
            counts = 0
            _debug_fallback_qs: list[dict[str, Any]] = [] if self._debug else []
            for _ in range(K):
                sim_state = copy.deepcopy(state)
                self._determinize_state(sim_state, state.turn, rng)
                result = self.exact_solver.solve_with_q_fast(sim_state)
                counts += 1
                if self._debug:
                    _debug_fallback_qs.append({
                        "norm_weight": 1.0 / K,
                        "action_q_values": {str(id_to_card.get(cid, Card(Suit.SPADES, Rank.TWO))): float(q)
                                           for cid, q in result.items()} if result else {},
                        "all_hands": {
                            p: [str(c) for c in sim_state.hands[p]]
                            for p in range(4)
                        },
                    })
                for cid, q in result.items():
                    agg_q[cid] = agg_q.get(cid, 0.0) + float(q)
            for k in agg_q:
                agg_q[k] /= max(1, counts)
            n_samples_used = counts
            if self._debug:
                _debug_samples = _debug_fallback_qs
        else:
            # 去重并按权重降序排列
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
                    if len(unique_paired) >= 5120:
                        break

            # ── debug: 记录去重后的 proposal pool ──
            if self._debug and _debug_pool_info is not None:
                _debug_unique_paired = []
                for hp, w in unique_paired:
                    _debug_unique_paired.append({
                        "weight_raw": float(w),
                        "hands_summary": {
                            p: [str(c) for c in hp[p][:6]] + (
                                [f"...({len(hp[p]) - 6} more)"] if len(hp[p]) > 6 else []
                            )
                            for p in range(4)
                        },
                    })

            # 确定要检查分布的花色：
            # 如果不是领出且手中有领出花色，则检查该花色；否则检查黑桃
            check_suit = Suit.SPADES
            if len(state.table_cards) > 0:
                lead_suit = state.table_cards[0][1].suit
                if any(c.suit == lead_suit for c in state.hands[self.position]):
                    check_suit = lead_suit

            # ── 先把池中所有权重归一化为概率分布（和为1） ──
            pool_total = sum(w for _, w in unique_paired)
            if pool_total > 0:
                pool_probs = [w / pool_total for _, w in unique_paired]
            else:
                pool_probs = [1.0 / len(unique_paired)] * len(unique_paired)

            if self.config.swap_is_fill:
                # ── 先补全（按权重降序 + 花色多样性），后 IS ──
                fill_indices: list[int] = []
                fill_suit_dists: list[dict[int, int]] = []
                for i in range(len(unique_paired)):
                    if len(fill_indices) >= max_samples:
                        break
                    cand_hand = unique_paired[i][0]
                    cand_dist: dict[int, int] = {}
                    for player_idx, hand in enumerate(cand_hand):
                        for card in hand:
                            if card.suit == check_suit:
                                cand_dist[card.card_id] = player_idx
                    diff_from_all = True
                    for dist in fill_suit_dists:
                        same = all(cand_dist.get(cid) == dist.get(cid) for cid in cand_dist)
                        if same:
                            diff_from_all = False
                            break
                    if diff_from_all:
                        fill_indices.append(i)
                        fill_suit_dists.append(cand_dist)

                remaining_for_is = [i for i in range(len(unique_paired)) if i not in set(fill_indices)]
                if remaining_for_is:
                    k_is = min(top_k, len(remaining_for_is))
                    is_weights_for_pick = [unique_paired[i][1] for i in remaining_for_is]
                    is_raw = self._importance_sample_indices(is_weights_for_pick, k_is, rng)
                    is_idx_list = [remaining_for_is[idx] for idx in is_raw]  # actual pool indices
                else:
                    is_idx_list = []

                # Combine: fill first, IS second
                is_idx_list = fill_indices + is_idx_list
                top_hands = [unique_paired[i][0] for i in is_idx_list]
                n_selected = len(top_hands)
                n_fill = len(fill_indices)
                n_is = len(is_idx_list) - n_fill
                if self._debug:
                    _debug_n_fill = n_fill
                    _debug_n_is = n_is

                # ── 权重计算 ──
                # top_hands = [fill..., IS...], norm_factors 也与之匹配
                norm_factors = []
                if n_fill > 0:
                    fill_probs = [pool_probs[i] for i in fill_indices]
                    fill_total_prob = sum(fill_probs)
                else:
                    fill_total_prob = 0.0
                if n_fill > 0:
                    norm_factors.extend(fill_probs)  # fill 在前
                if n_is > 0:
                    is_weight = (1.0 - fill_total_prob) / n_is
                    norm_factors.extend([is_weight] * n_is)  # IS 在后
                n_samples_used = n_selected

            else:
                # ── 默认：先 IS，后补全（花色多样性）──
                # ── 重要性采样：有放回地抽取 top_k 个下标 ──
                k_is = min(top_k, len(unique_paired))
                is_idx = self._importance_sample_indices(
                    [w for _, w in unique_paired], k_is, rng
                )
                is_idx_set = set(is_idx)

                # is_idx_list 先放 IS 选中的下标（有放回，可重复），后续再 append 补全下标
                is_idx_list = list(is_idx)
                selected_suit_dists = []
                for i in is_idx_list:
                    hand_proposal = unique_paired[i][0]
                    dist: dict[int, int] = {}
                    for player_idx, hand in enumerate(hand_proposal):
                        for card in hand:
                            if card.suit == check_suit:
                                dist[card.card_id] = player_idx
                    selected_suit_dists.append(dist)

                # 补全候选：按权重降序排列中未被 IS 选中的下标
                fill_candidates = sorted(
                    [i for i in range(len(unique_paired)) if i not in is_idx_set],
                    key=lambda i: unique_paired[i][1], reverse=True,
                )

                # 从剩余中补全到 max_samples，做花色多样性过滤
                for cand_i in fill_candidates:
                    if len(is_idx_list) >= max_samples:
                        break
                    cand_hand = unique_paired[cand_i][0]
                    cand_dist = {}
                    for player_idx, hand in enumerate(cand_hand):
                        for card in hand:
                            if card.suit == check_suit:
                                cand_dist[card.card_id] = player_idx

                    diff_from_all = True
                    for dist in selected_suit_dists:
                        same = all(cand_dist.get(cid) == dist.get(cid) for cid in cand_dist)
                        if same:
                            diff_from_all = False
                            break

                    if diff_from_all:
                        is_idx_list.append(cand_i)
                        selected_suit_dists.append(cand_dist)

                top_hands = [unique_paired[i][0] for i in is_idx_list]
                n_selected = len(top_hands)
                n_is = len(is_idx)        # IS 样本数（含重复，可能 > len(is_idx_set)）
                n_fill = n_selected - n_is
                if self._debug:
                    _debug_n_fill = n_fill
                    _debug_n_is = n_is

                # ── 权重计算 ──
                # IS 组：等分剩余概率；补全组：直接用归一化概率 pool_probs
                # 注意顺序必须与 top_hands 一致（IS 在前，补全在后）
                norm_factors = []
                fill_total_prob = 0.0
                if n_fill > 0:
                    fill_probs = [pool_probs[is_idx_list[n_is + j]] for j in range(n_fill)]
                    fill_total_prob = sum(fill_probs)
                if n_is > 0:
                    is_weight = (1.0 - fill_total_prob) / n_is
                    norm_factors.extend([is_weight] * n_is)
                if n_fill > 0:
                    norm_factors.extend(fill_probs)
                n_samples_used = n_selected

            # ── solver calls (parallel if enough proposals) ──
            observer_id = state.turn
            n_proposals = len(top_hands)

            if n_proposals >= _MIN_PARALLEL_BATCH and self._num_workers > 1:
                work_items = [(state, observer_id, hp) for hp in top_hands]
                n_workers = min(self._num_workers, n_proposals)
                try:
                    import multiprocessing as _mp
                    _ctx = _mp.get_context(_SOLVER_MP_START_METHOD)
                    with _ctx.Pool(n_workers) as pool:
                        q_results: list[dict[int, float]] = pool.map(
                            _parallel_solve_worker, work_items
                        )
                except Exception:
                    # Fallback to sequential on multiprocessing errors
                    q_results = [
                        _parallel_solve_worker(item) for item in work_items
                    ]
            else:
                # Sequential (small batch or single worker configured)
                q_results = [
                    _parallel_solve_worker((state, observer_id, hp))
                    for hp in top_hands
                ]

            # ── debug: 记录每个采样的 Q 值 ──
            _debug_samples: list[dict[str, Any]] | None = [] if self._debug else None
            for (hand_proposal, norm_w), worker_result in zip(
                zip(top_hands, norm_factors), q_results
            ):
                # worker_result is {card_id: q_value} from the worker
                action_q_dict = worker_result
                if self._debug and _debug_samples is not None:
                    _debug_samples.append({
                        "norm_weight": float(norm_w),
                        "action_q_values": {
                            str(id_to_card.get(cid, Card(Suit.SPADES, Rank.TWO))): float(q)
                            for cid, q in action_q_dict.items()
                        } if action_q_dict else {},
                        "all_hands": {
                            p: [str(c) for c in hand_proposal[p]]
                            for p in range(4)
                        },
                    })
                if action_q_dict:
                    max_q = max(action_q_dict.values())
                    min_q = min(action_q_dict.values())
                    for card_id, q in action_q_dict.items():
                        if my_team == 0:
                            multiplier = q - max_q
                            if multiplier < -self.config.multiplier_clip:
                                multiplier *= self.config.multiplier_clip_factor
                            agg_q[card_id] = agg_q.get(card_id, 0.0) + norm_w * multiplier
                        else:
                            multiplier = q - min_q
                            if multiplier > self.config.multiplier_clip:
                                multiplier *= self.config.multiplier_clip_factor
                            agg_q[card_id] = agg_q.get(card_id, 0.0) + norm_w * multiplier

        # Reconstruct action -> q using Card objects
        action_q_values: dict[Card, float] = {}
        for aid, q in agg_q.items():
            if aid in id_to_card:
                action_q_values[id_to_card[aid]] = q

        # 根据玩家所在队伍选择动作：
        #   队伍 0 (座位 0,2) → max Q
        #   队伍 1 (座位 1,3) → min Q
        if my_team == 0:
            best_q = max(action_q_values.values()) if action_q_values else None
        else:
            best_q = min(action_q_values.values()) if action_q_values else None

        if best_q is not None:
            tied_cards = [c for c in legal_cards if c in action_q_values and action_q_values[c] == best_q]
        else:
            tied_cards = []
        if not tied_cards:
            tied_cards = [c for c in legal_cards if c in action_q_values]

        # 检查是否有人叫 0
        has_nil = False
        if hasattr(state, "max_bid") and state.max_bid:
            has_nil = any(
                isinstance(b, str) and b in ("nil", "blind_nil")
                for b in state.max_bid
            )

        # 所有黑桃优先级最大，非黑桃先比较点数再比较花色
        # 有人叫 0 → 出优先级最高的牌（S大牌 > ... > 小牌H/D/C）
        # 没人叫 0 → 出优先级最低的牌（小牌H/D/C > ... > S大牌）
        def _card_priority_key(card: Card) -> tuple:
            return (card.suit != Suit.SPADES, -card.rank.value, card.suit.value)

        if tied_cards:
            if has_nil:
                best_action = min(tied_cards, key=_card_priority_key)
            else:
                best_action = max(tied_cards, key=_card_priority_key)
        else:
            best_action = None

        if best_action is not None and best_action in legal_cards:
            # 构造 action_scores 供 trace 日志记录（格式与 RLExactPlayer 一致）
            action_scores = sorted(
                [{"action": card, "value": float(q)}
                 for card, q in action_q_values.items()],
                key=lambda x: x["value"], reverse=True,
            )
            best_value = float(action_q_values.get(best_action, 0.0))
            info = {
                "mode": "exact_is_determinized",
                "samples": n_samples_used,
                "best_value": best_value,
                "action_scores": action_scores,
            }
            if self._debug and _debug_pool_info is not None:
                info["debug"] = {
                    "pool": _debug_pool_info,
                    "unique_proposals": _debug_unique_paired,
                    "samples": _debug_samples,
                    "n_fill": _debug_n_fill,
                    "n_is": _debug_n_is,
                    "remaining_cards": sum(len(h) for h in state.hands),
                    "my_team": my_team,
                    "agg_q_raw": {str(id_to_card.get(aid, Card(Suit.SPADES, Rank.TWO))): float(q)
                                  for aid, q in agg_q.items()},
                }
            self.last_play_info = info
            return best_action

        self.last_play_info = {"mode": "exact_no_match_fallback"}
        return self._canonical_card_choice(legal_cards)

    def _determinize_state(
        self, state: GameState, observer_id: int,
        rng: random.Random | None = None,
    ) -> None:
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
            state.hands[pid] = sorted(assigned, key=lambda card: card.card_id)

        if hasattr(state, "hand_bitsets"):
            for pid in range(state.num_players):
                bit = 0
                for c in state.hands[pid]:
                    bit |= (1 << c.card_id)
                state.hand_bitsets[pid] = bit

    # ── IS 确定化（重要性采样 + top-K 加权精确求解）─────────────────────────

    @staticmethod
    def _importance_sample_indices(
        weights: list[float], k: int, rng: random.Random,
    ) -> list[int]:
        """从权重列表中**有放回**地重要性采样 k 个下标。

        将权重归一化为概率分布后独立抽取 k 次（允许重复）。
        权重 ≤ 0 的项概率为 0，永远不会被选中。
        当所有权重相等时，即使 k == len(weights) 也能产生随机性
        （区别于无放回时 k >= n 退化为确定性全选）。
        """
        # 截断负权为 0，归一化得概率分布
        pos_w = [max(w, 0.0) for w in weights]
        total = sum(pos_w)
        if total <= 0:
            # 所有权重都 ≤ 0，退化为均匀有放回
            return [rng.randint(0, len(weights) - 1) for _ in range(k)]
        # random.choices 会自行归一化权重，无需提前算概率
        return rng.choices(range(len(weights)), weights=pos_w, k=k)

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

    def _ensure_bid_model_loaded(self) -> bool:
        """Lazy-load BidMLP for IS weighting. Returns True if model is available."""
        if hasattr(self, "_bid_model_is") and self._bid_model_is is not None:
            return hasattr(self, "_bid_encoder_is") and self._bid_encoder_is is not None

        try:
            go_dir = Path(__file__).resolve().parents[1] / "Spades_AI_GO-MCTS"
            if str(go_dir) not in sys.path:
                sys.path.insert(0, str(go_dir))
            from spades_ai.models.bid_mlp import BidMLP
            from spades_ai.models.bid_encoder import BidEncoder

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
                device = getattr(self, "_bid_device", "cpu")
                if device == "cpu" and torch.cuda.is_available():
                    device = "cuda"
                self._bid_model_is = BidMLP().to(device)
                sd = torch.load(ckpt, weights_only=True, map_location=device)
                self._bid_model_is.load_state_dict(sd)
                self._bid_model_is.eval()
                if device != "cpu":
                    self._bid_model_is = torch.jit.optimize_for_inference(
                        torch.jit.script(self._bid_model_is)
                    )  # JIT compile for GPU inference
                self._bid_device_is = device   # 存储实际 device，避免 JIT 冻结后 parameters() 为空
                self._bid_encoder_is = BidEncoder()
                return True
            else:
                self._bid_model_is = None
                return False
        except Exception:
            self._bid_model_is = None
            return False

    def _compute_bid_probs_product(
        self, initial_hands: list[list[Card]], max_bid: list[str],
    ) -> float:
        """∏ P(bid_p | hand_p) from BidMLP softmax (single proposal)."""
        if not self._ensure_bid_model_loaded():
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
        x = x.to(self._bid_device_is)
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

    def _compute_batch_bid_prods(
        self, proposals: list[list[list[Card]]], max_bid: list[str],
    ) -> list[float]:
        """Compute ∏ P(bid_p | hand_p) for all proposals in one batched MLP forward."""
        if not self._ensure_bid_model_loaded():
            return [1.0] * len(proposals)

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
        all_features = []
        for initial_hands in proposals:
            for p in range(4):
                hand = [GoCard(GoRank(c.rank.value), GoSuit[c.suit.name]) for c in initial_hands[p]]
                prev = go_bids[:p]
                position = min(p, 2)
                features = self._bid_encoder_is.encode(hand, prev, position)
                all_features.append(features.unsqueeze(0))

        x = torch.cat(all_features, dim=0)
        x = x.to(self._bid_device_is)
        with torch.no_grad():
            logits = self._bid_model_is(x)
        probs = torch.softmax(logits, dim=-1)
        uniform = 1.0 / 14
        smoothed = 0.99 * probs + 0.01 * uniform

        n = len(proposals)
        bid_prods = []
        for i in range(n):
            product = 1.0
            for p in range(4):
                idx = self._bid_str_to_mlp_index(max_bid[p])
                product *= float(smoothed[i * 4 + p, idx].item())
            bid_prods.append(product)
        return bid_prods

    def _generate_proposal(
        self, all_cards: list[Card], observer_id: int,
        observer_current_hand: list[Card],
        played_by_player: dict[int, list[Card]], rng: random.Random,
        void_suits: dict[int, set[Suit]] | None = None,
    ) -> list[list[Card]]:
        """Generate one random initial deal consistent with observed play.

        使用拒绝采样保证概率学上条件均匀：
        1. 先生成无约束的均匀发牌（pool shuffle + 顺序切分）——每一种切分等概率
        2. 若有断门约束，检查新发牌是否包含断门花色，是则拒绝重试
        3. 由于无约束分布均匀且拒绝条件只取决于断门约束，
           最终分布 = 条件于断门约束的均匀分布。

        void_suits: dict[p] = set of suits player p has shown void in
                    (cards of those suits are excluded from their random fill).
        """
        all_cards = sorted(all_cards, key=lambda card: card.card_id)
        obs_set: set[int] = set(c.card_id for c in observer_current_hand)
        obs_set.update(c.card_id for c in played_by_player[observer_id])
        id_to_card = {c.card_id: c for c in all_cards}
        observer_initial = [id_to_card[cid] for cid in sorted(obs_set)]

        def canonicalize(hands: list[list[Card]]) -> list[list[Card]]:
            return [sorted(hand, key=lambda card: card.card_id) for hand in hands]

        used_ids: set[int] = set(obs_set)
        for p in range(4):
            if p != observer_id:
                used_ids.update(c.card_id for c in played_by_player[p])

        pool = [c for c in all_cards if c.card_id not in used_ids]

        # 预计算每位非观测者还需多少张牌
        needs: dict[int, int] = {}
        for p in range(4):
            if p != observer_id:
                needs[p] = 13 - len(played_by_player[p])

        # ── 快速路径：无断门约束，一次 shuffle + split 即均匀 ──
        if not void_suits or all(not s for s in void_suits.values()):
            rng.shuffle(pool)
            initial_hands: list[list[Card]] = [None] * 4  # type: ignore
            initial_hands[observer_id] = list(observer_initial)
            pos = 0
            for p in range(4):
                if p == observer_id:
                    continue
                n = needs[p]
                initial_hands[p] = list(played_by_player[p]) + pool[pos:pos + n]
                pos += n
            return canonicalize(initial_hands)

        # ── 拒绝采样：生成无约束均匀发牌，拒绝违反断门约束的样本 ──
        # 无约束均匀发牌 = shuffle(pool) → 按 needs 顺序切分
        # 每一种切分概率 = (n1! n2! ... nk!) / N!（只取决于组大小，与牌面无关）→ 均匀
        for _ in range(10000):
            rng.shuffle(pool)
            pool_copy = list(pool)
            initial_hands = [None] * 4  # type: ignore
            initial_hands[observer_id] = list(observer_initial)

            valid = True
            for p in range(4):
                if p == observer_id:
                    continue
                n = needs[p]
                new_cards = pool_copy[:n]
                pool_copy = pool_copy[n:]

                # 检查新发的牌是否包含断门花色
                if p in void_suits and void_suits[p]:
                    if any(c.suit in void_suits[p] for c in new_cards):
                        valid = False
                        break

                initial_hands[p] = list(played_by_player[p]) + new_cards

            if valid:
                return canonicalize(initial_hands)

        # 极罕见的情况：拒绝了 10000 次仍未找到可行解
        # （可能由于约束不可行，回退到带过滤的贪心分配）
        rng.shuffle(pool)
        initial_hands = [None] * 4  # type: ignore
        initial_hands[observer_id] = list(observer_initial)
        for p in range(4):
            if p == observer_id:
                continue
            initial_hands[p] = list(played_by_player[p])
            n = needs[p]
            if p in void_suits and void_suits[p]:
                valid_pool = [c for c in pool if c.suit not in void_suits[p]]
                selected = valid_pool[:n]
                selected_ids = {c.card_id for c in selected}
                pool = [c for c in pool if c.card_id not in selected_ids]
                initial_hands[p].extend(selected)
            else:
                initial_hands[p].extend(pool[:n])
                pool = pool[n:]

        return canonicalize(initial_hands)

    @staticmethod
    def _compute_void_suits(
        play_sequence: list[tuple[int, Card]],
    ) -> dict[int, set[Suit]]:
        """分析 play_sequence, 推断每个玩家在哪些花色上已断门。

        如果领出花色 X, 跟牌人打出的不是 X, 说明该跟牌人手中没有 X。
        返回 dict[p] = set of suits player p is void in.
        """
        void_suits: dict[int, set[Suit]] = {}

        # 完整墩 (每墩 4 张)
        n_full = len(play_sequence) // 4
        for ti in range(n_full):
            start = ti * 4
            lead_suit = play_sequence[start][1].suit
            for pos in range(1, 4):
                pid, card = play_sequence[start + pos]
                if card.suit != lead_suit:
                    void_suits.setdefault(pid, set()).add(lead_suit)

        # 处理可能的不完整尾墩
        tail = len(play_sequence) % 4
        if tail >= 1:
            start = n_full * 4
            lead_suit = play_sequence[start][1].suit
            for pos in range(1, tail):
                pid, card = play_sequence[start + pos]
                if card.suit != lead_suit:
                    void_suits.setdefault(pid, set()).add(lead_suit)

        return void_suits

    def _compute_importance_weight(
        self, initial_hands: list[list[Card]],
        play_sequence: list[tuple[int, Card]],
        max_bid: list[str] | None = None,
        bid_prod: float | None = None,
        original_state: GameState | None = None,
    ) -> float:
        """Replay play_sequence against initial_hands; compute p = ∏(p_step).

        p = P_bid * ∏_{step} (w_step).
        对于第 9~13 墩（step_idx // 4 >= 8），w_step 由精确求解器的 Q 值决定。
        Q 始终是 team 0 - team 1，因此 team 0 取 max，team 1 取 min：
          - 若动作是当前行动队伍的最优 Q，w = 1
          - 否则 w = B / (B + A)
            （A = 好动作数，B = 坏动作数）
        其余墩的 w = 1。

        Returns 0 if any move was illegal given this deal.
        """
        if bid_prod is not None:
            pass
        elif max_bid is not None:
            bid_prod = self._compute_bid_probs_product(initial_hands, max_bid)
        else:
            bid_prod = 1.0

        hands = [list(h) for h in initial_hands]
        spades_broken = False
        pos_in_trick = 0
        led_suit: Suit | None = None
        weight = 1.0

        # Track current incomplete trick's cards (for solver state construction)
        solver_table: list[tuple[int, Card]] = []
        replay_tricks_played = 0
        replay_tricks_won = [0, 0, 0, 0]

        # Pre-copy original state once for solver-based weighting (tricks 9-13)
        sim_state = None
        if original_state is not None and self.exact_solver is not None:
            sim_state = copy.deepcopy(original_state)

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
                led_suit = card.suit
            else:  # Following
                has_led = any(c.suit == led_suit for c in hand)
                if has_led and card.suit != led_suit:
                    return 0.0

            # ── 第 9~13 墩 (0-indexed: 8~12) 用精确求解器判定动作好坏 ──
            trick_num = step_idx // 4
            if trick_num >= self.config.trick_num_threshold and sim_state is not None:
                sim_state.hands = [list(h) for h in hands]
                for p in range(4):
                    bit = 0
                    for c in sim_state.hands[p]:
                        bit |= (1 << c.card_id)
                    sim_state.hand_bitsets[p] = bit
                sim_state.turn = player
                sim_state.table_cards = list(solver_table)
                sim_state.tricks_played = replay_tricks_played
                sim_state.tricks_won = list(replay_tricks_won)
                sim_state.spades_broken = spades_broken
                sim_state.trump_broken = spades_broken
                if solver_table:
                    sim_state.trick_leader = solver_table[0][0]
                else:
                    sim_state.trick_leader = player

                action_q = self.exact_solver.solve_with_q_fast(sim_state)  # {card_id: q_value}

                if action_q:
                    acting_team = sim_state.teams[player]
                    best_q_val = (
                        max(action_q.values())
                        if acting_team == 0
                        else min(action_q.values())
                    )
                    good_count = sum(1 for q in action_q.values() if q == best_q_val)
                    bad_count = len(action_q) - good_count

                    # 直接用 card_id 查找（避免 Card 对象创建和匹配）
                    q_val = action_q.get(card.card_id)
                    if q_val is None:
                        return 0.0  # 该动作在 proposal 下不合法

                    if q_val == best_q_val:
                        weight *= 1.0  # 好动作
                    else:
                        total = good_count + bad_count
                        if total > 0:
                            # bad_action_weight 解读:
                            #   "x" → 用比例 x = bad_count/total
                            #   数字字符串 → 用该常数
                            x = bad_count / total
                            expr = self.config.bad_action_weight.strip()
                            try:
                                mult = float(expr)
                            except ValueError:
                                mult = x  # "x" 或任何非数字字符串都按比例
                            weight *= mult

            # ── 执行动作 & 更新追踪 ──
            hand.pop(idx)
            solver_table.append((player, card))

            if card.suit == Suit.SPADES:
                spades_broken = True
            pos_in_trick = (pos_in_trick + 1) % 4
            if pos_in_trick == 0:
                winner, _ = _trick_current_winner(solver_table, Suit.SPADES)
                replay_tricks_won[winner] += 1
                replay_tricks_played += 1
                solver_table.clear()
                led_suit = None

        actual_weight = weight * bid_prod #* math.exp(random.uniform(0, 0.1)) # 0.4→0.6→0.1
        weight_ans = actual_weight #** 0.3
        return weight_ans

    def _build_is_pool(
        self, state: GameState, observer_id: int, rng: random.Random,
        num_proposals: int = 1234,
        num_proposals_limit: int = 5678,
        min_pool_size: int = 100,
    ) -> tuple[list[list[list[Card]]], list[float]]:
        """Build IS pool: generate proposals, compute weights.

        Keeps sampling (in batches of num_proposals) until the pool has at least
        min_pool_size valid (w > 0) proposals, or total attempts reach num_proposals_limit.
        """
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

        # 推断每人断门花色（领出 X 但跟牌人未出 X → 该人无 X）
        void_suits = self._compute_void_suits(play_sequence)

        proposals: list[list[list[Card]]] = []
        prop_weights: list[float] = []
        total_attempts = 0

        while total_attempts < num_proposals_limit:
            batch_size = min(num_proposals, num_proposals_limit - total_attempts)

            # Generate batch
            batch_proposals: list[list[list[Card]]] = []
            for _ in range(batch_size):
                initial_hands = self._generate_proposal(
                    state.all_cards, observer_id, state.hands[observer_id],
                    played_by_player, rng,
                    void_suits=void_suits,
                )
                batch_proposals.append(initial_hands)
            total_attempts += batch_size

            # Compute bid probability products for this batch
            if max_bid is not None:
                bid_prods = self._compute_batch_bid_prods(batch_proposals, max_bid)
            else:
                bid_prods = [1.0] * batch_size

            # Compute importance weights, filter zero-weight proposals
            for initial_hands, bid_prod in zip(batch_proposals, bid_prods):
                w = self._compute_importance_weight(
                    initial_hands, play_sequence, bid_prod=bid_prod,
                    original_state=state,
                )
                if w > 0.0:
                    proposals.append(initial_hands)
                    prop_weights.append(w)

            if len(proposals) >= min_pool_size:
                break

        # Log proposal count, remaining cards, and weight distribution
        remaining = sum(len(h) for h in state.hands)
        gt01 = sum(1 for w in prop_weights if w > 0.1)
        gt001 = sum(1 for w in prop_weights if w > 0.01)
        gt0001 = sum(1 for w in prop_weights if w > 0.001)
        gt00001 = sum(1 for w in prop_weights if w > 0.0001)
        sorted_w = sorted(prop_weights, reverse=True)
        A = sum(sorted_w[:3])
        B = sum(sorted_w[31:34]) if len(sorted_w) >= 34 else None
        C = sum(sorted_w[99:102]) if len(sorted_w) >= 102 else None
        ratio_ab = f"{A/B:.3f}" if B else "nan"
        ratio_ac = f"{A/C:.3f}" if C else "nan"
        with open("is_proposal_stats.txt", "a") as f:
            f.write(f"n_proposals={len(proposals)} remaining={remaining} w>0.1={gt01} w>0.01={gt001} w>0.001={gt0001} w>0.0001={gt00001} A/B={ratio_ab} A/C={ratio_ac}\n")

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
