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

import atexit
import copy
import functools
import hashlib
import json
import multiprocessing
import os
import random
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import _exact_solver_worker
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

_WORKER_SOLVER: ExactDoubleDummyCppFastestSolver | None = None


def _initialize_solver_worker() -> None:
    """Load and validate the native solver once for this spawned worker."""
    global _WORKER_SOLVER
    _WORKER_SOLVER = ExactDoubleDummyCppFastestSolver()


def _get_worker_solver() -> ExactDoubleDummyCppFastestSolver:
    """Return the process-local solver, lazily initializing nonstandard pools."""
    global _WORKER_SOLVER
    if _WORKER_SOLVER is None:
        _initialize_solver_worker()
    assert _WORKER_SOLVER is not None
    return _WORKER_SOLVER


def _solve_proposal(
    args: tuple,
    solver: Any,
) -> dict[int, float]:
    """Apply one hidden-hand proposal and solve it with an existing solver."""
    state, observer_id, hand_proposal = args
    sim_state = copy.deepcopy(state)

    played_by: dict[int, set[int]] = {i: set() for i in range(4)}
    for record in sim_state.trick_history:
        for pid, card in record.cards:
            played_by[pid].add(card.card_id)
    for pid, card in sim_state.table_cards:
        played_by[pid].add(card.card_id)

    for p in range(4):
        if p != observer_id:
            sim_state.hands[p] = [
                card
                for card in hand_proposal[p]
                if card.card_id not in played_by[p]
            ]

    if hasattr(sim_state, "hand_bitsets"):
        for p in range(4):
            bit = 0
            for card in sim_state.hands[p]:
                bit |= 1 << card.card_id
            sim_state.hand_bitsets[p] = bit

    return solver.solve_with_q_fast(sim_state)


def _parallel_solve_worker(args: tuple) -> dict[int, float]:
    """Solve one proposal in a worker process. Returns {card_id: q_value}.

    Each spawned process owns one persistent solver instance, so native-library
    validation and ctypes setup happen once per worker rather than once per
    proposal.  The state is still copied per task and the native process-global
    TT buffer remains isolated between workers.

    Args:
        args: (state, observer_id, hand_proposal) where
              state is a GameState (dataclass, picklable),
              hand_proposal is list[list[Card]] (4 player's full starting hands).

    Returns:
        dict mapping card_id → q_value.  Returns empty dict on failure.
    """
    try:
        solver = _get_worker_solver()
    except Exception:
        return {}
    return _solve_proposal_safely(args, solver)


def _solve_proposal_safely(
    args: tuple,
    solver: Any,
) -> dict[int, float]:
    """Keep one failed determinization from aborting the whole decision."""
    try:
        return _solve_proposal(args, solver)
    except Exception:
        return {}


# ── minimum batch size to trigger parallel solving ──
_MIN_PARALLEL_BATCH = 8
_BID_LIKELIHOOD_CACHE_SIZE = 65_536

# ``fork`` from the threaded HTTP/WebSocket servers can inherit a locked
# Python mutex or partially initialized torch/native runtime.  ``spawn``
# starts each solver worker from a clean interpreter instead.
_SOLVER_MP_START_METHOD = "spawn"


@dataclass
class _PersistentSolverPool:
    pool: Any
    map_lock: Any
    owner_pid: int


_SOLVER_POOLS: dict[int, _PersistentSolverPool] = {}
_SOLVER_POOLS_LOCK = threading.Lock()


def _get_persistent_solver_pool(num_workers: int) -> _PersistentSolverPool:
    """Return the process-wide pool for a worker count, creating it once."""
    with _SOLVER_POOLS_LOCK:
        entry = _SOLVER_POOLS.get(num_workers)
        if entry is not None and entry.owner_pid != os.getpid():
            # A forked child must never reuse or terminate the parent's Pool.
            _SOLVER_POOLS.pop(num_workers, None)
            entry = None
        if entry is None:
            context = multiprocessing.get_context(_SOLVER_MP_START_METHOD)
            entry = _PersistentSolverPool(
                pool=context.Pool(
                    num_workers,
                    initializer=(
                        _exact_solver_worker.initialize_solver_worker
                    ),
                ),
                map_lock=threading.Lock(),
                owner_pid=os.getpid(),
            )
            _SOLVER_POOLS[num_workers] = entry
        return entry


def _discard_persistent_solver_pool(
    num_workers: int,
    expected: _PersistentSolverPool,
) -> None:
    """Remove and terminate a failed pool without disturbing a replacement."""
    with _SOLVER_POOLS_LOCK:
        if _SOLVER_POOLS.get(num_workers) is expected:
            _SOLVER_POOLS.pop(num_workers, None)
    if expected.owner_pid != os.getpid():
        return
    try:
        expected.pool.terminate()
    finally:
        expected.pool.join()


def _map_persistent_solver_pool(
    num_workers: int,
    work_items: list[tuple],
) -> list[dict[int, float]]:
    """Map solver work through the reusable pool, serializing pool clients."""
    entry = _get_persistent_solver_pool(num_workers)
    try:
        with entry.map_lock:
            # Exact-search runtimes are highly skewed.  One item per chunk lets
            # an idle worker steal the next determinization instead of waiting
            # behind a slow item bundled with otherwise cheap work.
            return entry.pool.map(
                _exact_solver_worker.parallel_solve_worker,
                work_items,
                chunksize=1,
            )
    except Exception:
        _discard_persistent_solver_pool(num_workers, entry)
        raise


def _shutdown_persistent_solver_pools() -> None:
    """Terminate persistent workers during interpreter shutdown."""
    with _SOLVER_POOLS_LOCK:
        entries = list(_SOLVER_POOLS.values())
        _SOLVER_POOLS.clear()
    for entry in entries:
        if entry.owner_pid != os.getpid():
            continue
        try:
            entry.pool.terminate()
            entry.pool.join()
        except Exception:
            pass


atexit.register(_shutdown_persistent_solver_pools)


@dataclass
class _ReplaySnapshot:
    sequence_key: tuple[tuple[int, int], ...]
    hands: list[list[Card]]
    spades_broken: bool
    pos_in_trick: int
    led_suit: Suit | None
    play_weight: float
    completed_ranks_by_suit: dict[Suit, set[int]]
    solver_table: list[tuple[int, Card]]
    replay_tricks_played: int
    replay_tricks_won: list[int]


@dataclass
class _PosteriorEntry:
    initial_hands: list[list[Card]]
    bid_prod: float
    replay: _ReplaySnapshot


@dataclass
class _PosteriorCache:
    deal_key: tuple[int, tuple[int, ...]] | None
    observer_id: int
    max_bid: tuple[str, ...] | None
    sequence_key: tuple[tuple[int, int], ...]
    num_proposals: int
    num_proposals_limit: int
    entries: list[_PosteriorEntry]


@dataclass(frozen=True)
class _ProposalSamplerContext:
    """Deal-invariant data and conditional-completion counts for one IS pool."""

    observer_id: int
    observer_initial: tuple[Card, ...]
    played_by_player: tuple[tuple[Card, ...], ...]
    other_players: tuple[int, ...]
    pool: tuple[Card, ...]
    needs: tuple[int, ...]
    allowed_player_indices: tuple[tuple[int, ...], ...]
    constrained: bool
    completion_count: Any | None
    total_completions: int


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
        self._deal_key: tuple[int, tuple[int, ...]] | None = None
        self._posterior_cache: _PosteriorCache | None = None
        self._last_pool_cache_hit = False
        self._bid_likelihood_cache: OrderedDict[tuple[Any, ...], float] = (
            OrderedDict()
        )

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
        deal_key = (
            position,
            tuple(sorted(card.card_id for card in hand)),
        )
        if deal_key != self._deal_key:
            self._posterior_cache = None
        self._deal_key = deal_key
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

        decision_seed = self._decision_seed(state, state.turn, legal_cards)
        # Proposal construction and final sample selection use independent
        # deterministic streams.  A posterior-cache hit therefore cannot
        # change the selected samples merely because proposal generation was
        # skipped.
        proposal_rng = random.Random(decision_seed)
        selection_rng = random.Random(
            decision_seed ^ 0x9E3779B97F4A7C15D1B54A32D192ED03
        )
        # 计算剩余牌数，决定采样预算：越往后预算越大
        remaining_in = sum(len(h) for h in state.hands)
        top_k, max_samples = self.config.budget.lookup(remaining_in)
        max_samples = max(1, int(max_samples))
        top_k = min(max(0, int(top_k)), max_samples)
        K = max_samples
        id_to_card = {c.card_id: c for c in STANDARD_52}

        # Build IS pool (with config params)
        pool_hands, pool_weights = self._build_is_pool(
            state, state.turn, proposal_rng,
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
                "posterior_cache_hit": self._last_pool_cache_hit,
            }

        if not pool_hands:
            # Fallback: uniform determinization
            counts = 0
            _debug_fallback_qs: list[dict[str, Any]] = [] if self._debug else []
            for _ in range(K):
                sim_state = copy.deepcopy(state)
                self._determinize_state(sim_state, state.turn, selection_rng)
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
                # max_samples 是最终总上限；为后续 IS 预留 top_k 个位置。
                fill_limit = max(0, max_samples - top_k)
                fill_indices: list[int] = []
                fill_suit_dists: list[dict[int, int]] = []
                for i in range(len(unique_paired)):
                    if len(fill_indices) >= fill_limit:
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
                    k_is = min(
                        top_k,
                        max_samples - len(fill_indices),
                        len(remaining_for_is),
                    )
                    is_weights_for_pick = [unique_paired[i][1] for i in remaining_for_is]
                    is_raw = self._importance_sample_indices(
                        is_weights_for_pick,
                        k_is,
                        selection_rng,
                    )
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
                k_is = min(top_k, max_samples, len(unique_paired))
                is_idx = self._importance_sample_indices(
                    [w for _, w in unique_paired],
                    k_is,
                    selection_rng,
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
                    q_results = _map_persistent_solver_pool(
                        n_workers,
                        work_items,
                    )
                except Exception:
                    # Fallback to sequential on multiprocessing errors
                    q_results = [
                        _solve_proposal_safely(item, self.exact_solver)
                        for item in work_items
                    ]
            else:
                # Sequential (small batch or single worker configured)
                q_results = [
                    _solve_proposal_safely(
                        (state, observer_id, hp),
                        self.exact_solver,
                    )
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

        # ── 硬性约束：出牌必须是等大牌张中最大的 ──
        if best_action is not None and best_action in legal_cards:
            # 只统计已完成墩中的牌。当前桌面牌仍会影响本墩胜负，
            # 必须像 C++ equivalent-card filter 一样作为分组阻断牌。
            _played_by_suit: dict[Suit, set[int]] = {s: set() for s in Suit}
            for _rec in state.trick_history:
                for _pid, _c in _rec.cards:
                    _played_by_suit[_c.suit].add(_c.rank.value)
            best_action = self._enforce_largest_equal_magnitude(
                best_action, state.hands[self.position], legal_cards, _played_by_suit,
            )

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
        return self._compute_batch_bid_prods([initial_hands], max_bid)[0]

    def _compute_batch_bid_prods(
        self, proposals: list[list[list[Card]]], max_bid: list[str],
    ) -> list[float]:
        """Compute bid likelihoods with a bounded per-hand LRU cache.

        A proposal's observer hand is identical across the whole batch, and
        cached posterior particles keep all four initial hands across later
        decisions.  Encoding each distinct ``(bids, seat, hand)`` once avoids
        thousands of repeated Python feature-construction calls while retaining
        exactly the same model probabilities.
        """
        if not proposals:
            return []
        if not self._ensure_bid_model_loaded():
            return [1.0] * len(proposals)

        go_dir = Path(__file__).resolve().parents[1] / "Spades_AI_GO-MCTS"
        if str(go_dir) not in sys.path:
            sys.path.insert(0, str(go_dir))
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

        bids_key = tuple(max_bid)
        flat_keys: list[tuple[Any, ...]] = []
        missing: OrderedDict[
            tuple[Any, ...],
            tuple[int, list[Card]],
        ] = OrderedDict()

        for initial_hands in proposals:
            for player_id in range(4):
                hand_key = tuple(
                    sorted(card.card_id for card in initial_hands[player_id])
                )
                key = (bids_key, player_id, hand_key)
                flat_keys.append(key)
                if key in self._bid_likelihood_cache:
                    self._bid_likelihood_cache.move_to_end(key)
                elif key not in missing:
                    missing[key] = (player_id, initial_hands[player_id])

        if missing:
            go_bids = [_to_go_bid(bid) for bid in max_bid]
            go_cards = {
                card.card_id: GoCard(
                    GoRank(card.rank.value),
                    GoSuit[card.suit.name],
                )
                for card in STANDARD_52
            }
            missing_items = list(missing.items())
            features = []
            for _, (player_id, hand) in missing_items:
                go_hand = [go_cards[card.card_id] for card in hand]
                features.append(
                    self._bid_encoder_is.encode(
                        go_hand,
                        go_bids[:player_id],
                        min(player_id, 2),
                    )
                )

            x = torch.stack(features, dim=0).to(self._bid_device_is)
            with torch.no_grad():
                logits = self._bid_model_is(x)
            probs = torch.softmax(logits, dim=-1)
            smoothed = 0.99 * probs + 0.01 * (1.0 / 14)

            for row, (key, (player_id, _)) in enumerate(missing_items):
                bid_idx = self._bid_str_to_mlp_index(max_bid[player_id])
                self._bid_likelihood_cache[key] = float(
                    smoothed[row, bid_idx].item()
                )
                self._bid_likelihood_cache.move_to_end(key)

            while (
                len(self._bid_likelihood_cache)
                > _BID_LIKELIHOOD_CACHE_SIZE
            ):
                self._bid_likelihood_cache.popitem(last=False)

        bid_prods: list[float] = []
        offset = 0
        for _ in proposals:
            product = 1.0
            for key in flat_keys[offset:offset + 4]:
                product *= self._bid_likelihood_cache[key]
            bid_prods.append(product)
            offset += 4
        gamma = self.config.gamma
        if gamma != 1.0:
            bid_prods = [p ** gamma for p in bid_prods]
        return bid_prods

    @staticmethod
    def _prepare_proposal_sampler(
        all_cards: list[Card],
        observer_id: int,
        observer_current_hand: list[Card],
        played_by_player: dict[int, list[Card]],
        void_suits: dict[int, set[Suit]] | None = None,
    ) -> _ProposalSamplerContext:
        """Precompute everything shared by all proposals in one decision."""
        all_cards = sorted(all_cards, key=lambda card: card.card_id)
        obs_set: set[int] = set(c.card_id for c in observer_current_hand)
        obs_set.update(c.card_id for c in played_by_player[observer_id])
        id_to_card = {c.card_id: c for c in all_cards}
        observer_initial = tuple(
            id_to_card[cid] for cid in sorted(obs_set)
        )

        used_ids: set[int] = set(obs_set)
        for p in range(4):
            if p != observer_id:
                used_ids.update(c.card_id for c in played_by_player[p])

        pool = tuple(
            card for card in all_cards if card.card_id not in used_ids
        )
        other_players = tuple(
            player for player in range(4) if player != observer_id
        )
        needs = tuple(
            13 - len(played_by_player[player])
            for player in other_players
        )
        if sum(needs) != len(pool):
            raise ValueError(
                "observed cards and opponent hand capacities are inconsistent"
            )

        void_suits = void_suits or {}
        constrained = any(
            void_suits.get(player)
            for player in other_players
        )
        allowed_player_indices = tuple(
            tuple(
                index
                for index, player in enumerate(other_players)
                if card.suit not in void_suits.get(player, set())
            )
            for card in pool
        )

        completion_count = None
        total_completions = 1
        if constrained:
            @functools.lru_cache(maxsize=None)
            def count_completions(
                card_index: int,
                capacities: tuple[int, ...],
            ) -> int:
                if card_index == len(pool):
                    return int(all(capacity == 0 for capacity in capacities))
                if sum(capacities) != len(pool) - card_index:
                    return 0

                total = 0
                for player_index in allowed_player_indices[card_index]:
                    if capacities[player_index] <= 0:
                        continue
                    next_capacities = list(capacities)
                    next_capacities[player_index] -= 1
                    total += count_completions(
                        card_index + 1,
                        tuple(next_capacities),
                    )
                return total

            completion_count = count_completions
            total_completions = count_completions(0, needs)
            if total_completions == 0:
                raise ValueError(
                    "void-suit constraints admit no hidden-hand deal"
                )

        return _ProposalSamplerContext(
            observer_id=observer_id,
            observer_initial=observer_initial,
            played_by_player=tuple(
                tuple(played_by_player[player])
                for player in range(4)
            ),
            other_players=other_players,
            pool=pool,
            needs=needs,
            allowed_player_indices=allowed_player_indices,
            constrained=constrained,
            completion_count=completion_count,
            total_completions=total_completions,
        )

    def _generate_proposal(
        self, all_cards: list[Card], observer_id: int,
        observer_current_hand: list[Card],
        played_by_player: dict[int, list[Card]], rng: random.Random,
        void_suits: dict[int, set[Suit]] | None = None,
        sampler_context: _ProposalSamplerContext | None = None,
    ) -> list[list[Card]]:
        """Sample a uniformly random initial deal conditional on public facts.

        With no void information, shuffle-and-split remains the fastest exact
        sampler.  With void constraints, dynamic-programming completion counts
        choose each card owner in proportion to the number of valid suffix
        deals.  This is exact conditional sampling: no rejection loop, timeout,
        or biased greedy fallback.
        """
        context = sampler_context or self._prepare_proposal_sampler(
            all_cards,
            observer_id,
            observer_current_hand,
            played_by_player,
            void_suits,
        )

        assigned: list[list[Card]] = [
            [] for _ in context.other_players
        ]
        if not context.constrained:
            shuffled_pool = list(context.pool)
            rng.shuffle(shuffled_pool)
            position = 0
            for player_index, need in enumerate(context.needs):
                assigned[player_index].extend(
                    shuffled_pool[position:position + need]
                )
                position += need
        else:
            assert context.completion_count is not None
            capacities = context.needs
            for card_index, card in enumerate(context.pool):
                target = rng.randrange(
                    context.completion_count(card_index, capacities)
                )
                selected_index = -1
                for player_index in context.allowed_player_indices[card_index]:
                    if capacities[player_index] <= 0:
                        continue
                    next_capacities = list(capacities)
                    next_capacities[player_index] -= 1
                    next_capacities_tuple = tuple(next_capacities)
                    completions = context.completion_count(
                        card_index + 1,
                        next_capacities_tuple,
                    )
                    if target < completions:
                        selected_index = player_index
                        capacities = next_capacities_tuple
                        break
                    target -= completions
                if selected_index < 0:
                    raise RuntimeError(
                        "conditional deal sampler lost completion mass"
                    )
                assigned[selected_index].append(card)

        initial_hands: list[list[Card]] = [
            [] for _ in range(4)
        ]
        initial_hands[context.observer_id] = list(
            context.observer_initial
        )
        for player_index, player in enumerate(context.other_players):
            initial_hands[player] = [
                *context.played_by_player[player],
                *assigned[player_index],
            ]
        return [
            sorted(hand, key=lambda card: card.card_id)
            for hand in initial_hands
        ]

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

    @staticmethod
    def _compute_legal_cards_for_state(
        hand: list[Card], pos_in_trick: int,
        led_suit: Suit | None, spades_broken: bool,
    ) -> list[Card]:
        """Compute legal cards from a hand given the current trick state."""
        if pos_in_trick == 0:  # Leading
            if not spades_broken:
                non_spades = [c for c in hand if c.suit != Suit.SPADES]
                if non_spades:
                    return non_spades
            return list(hand)
        else:  # Following
            if led_suit is not None:
                led_cards = [c for c in hand if c.suit == led_suit]
                if led_cards:
                    return led_cards
            return list(hand)

    @staticmethod
    def _card_has_larger_equal_magnitude(
        card: Card, hand: list[Card],
        played_ranks_by_suit: dict[Suit, set[int]],
    ) -> bool:
        """Check if card has larger 等大牌张 in the same hand.

        "等大牌张": cards of the same suit where consecutive hand cards (sorted
        by rank) are in the same group if ALL standard ranks between them have
        been played.  Returns True if the card is NOT the largest in its group,
        meaning the teammate made a bad play.
        """
        suit = card.suit
        played = played_ranks_by_suit.get(suit, set())
        card_rank = card.rank.value

        # Get all cards of the same suit currently in hand
        suit_cards = [c for c in hand if c.suit == suit]
        if len(suit_cards) <= 1:
            return False

        # Sort descending by rank
        suit_cards.sort(key=lambda c: c.rank.value, reverse=True)

        # Form 等大 groups: consecutive sorted hand cards belong to the same
        # group iff all ranks between them in the standard ordering are played.
        groups: list[list[Card]] = []
        current_group = [suit_cards[0]]

        for i in range(1, len(suit_cards)):
            prev_rank = suit_cards[i-1].rank.value
            curr_rank = suit_cards[i].rank.value
            # All ranks strictly between curr_rank and prev_rank
            all_between_played = all(
                r in played
                for r in range(curr_rank + 1, prev_rank)
            )
            if all_between_played:
                current_group.append(suit_cards[i])
            else:
                groups.append(current_group)
                current_group = [suit_cards[i]]

        groups.append(current_group)

        # Check if the card is NOT the largest in its group
        for group in groups:
            group_max = max(c.rank.value for c in group)
            group_min = min(c.rank.value for c in group)
            if group_min <= card_rank <= group_max:
                return card_rank < group_max

        return False

    @staticmethod
    def _enforce_largest_equal_magnitude(
        card: Card, hand: list[Card], legal_cards: list[Card],
        played_ranks_by_suit: dict[Suit, set[int]],
    ) -> Card:
        """硬性规定：出的牌必须是其等大牌张组中最大的（在 legal_cards 范围内）。

        形成手牌同花色牌的等大组（与 _card_has_larger_equal_magnitude 同逻辑），
        如果选中的 card 不是组内最大且 legal 的牌，则替换为最大的那张。
        """
        suit = card.suit
        played = played_ranks_by_suit.get(suit, set())

        # 取手牌中同花色的牌
        suit_cards = [c for c in hand if c.suit == suit]
        if len(suit_cards) <= 1:
            return card

        suit_cards.sort(key=lambda c: c.rank.value, reverse=True)

        # 形成等大组
        groups: list[list[Card]] = []
        current_group = [suit_cards[0]]
        for i in range(1, len(suit_cards)):
            prev_rank = suit_cards[i-1].rank.value
            curr_rank = suit_cards[i].rank.value
            all_between_played = all(
                r in played
                for r in range(curr_rank + 1, prev_rank)
            )
            if all_between_played:
                current_group.append(suit_cards[i])
            else:
                groups.append(list(current_group))
                current_group = [suit_cards[i]]
        groups.append(list(current_group))

        # 找到 card 所在组，取 legal_cards 中最大的
        card_rank = card.rank.value
        for group in groups:
            if any(c.rank.value == card_rank for c in group):
                legal_in_group = [c for c in group if c in legal_cards]
                if legal_in_group:
                    best = max(legal_in_group, key=lambda c: c.rank.value)
                    return best
                return card
        return card

    def _compute_batch_replay_weights(
        self,
        proposals: list[list[list[Card]]],
        play_sequence: list[tuple[int, Card]],
        bid_prods: list[float],
        max_bid: list[str] | None,
        observer_id: int,
    ) -> list[tuple[float, _ReplaySnapshot | None]]:
        """Replay a proposal batch with bitset hands and shared public state.

        This fast path is used only before solver-based history weighting
        begins.  Trick position, table, winners, broken-trump state, and
        completed ranks depend solely on the observed public sequence, so they
        are advanced once per public action rather than once per proposal.
        Hidden-hand membership and follow-suit checks stay particle-specific.
        """
        if len(proposals) != len(bid_prods):
            raise ValueError("proposal and bid-product counts differ")
        if not proposals:
            return []

        sequence_key = tuple(
            (int(player), int(card.card_id))
            for player, card in play_sequence
        )
        teammate = (observer_id + 2) % 4
        has_nil_bid = (
            max_bid is not None
            and any(bid in ("nil", "blind_nil") for bid in max_bid)
        )

        hand_bits = [
            [
                sum(1 << card.card_id for card in hand)
                for hand in proposal
            ]
            for proposal in proposals
        ]
        valid = [True] * len(proposals)
        play_weights = [1.0] * len(proposals)
        rule_players: list[RuleBasedFirst4Player] = []

        for proposal in proposals:
            if has_nil_bid:
                from strategy.rule_based_first4_nil_player import (
                    RuleBasedFirst4NilPlayer,
                )
                rule_player = RuleBasedFirst4NilPlayer()
                rule_player.start_game(
                    teammate,
                    list(proposal[teammate]),
                    4,
                )
                assert max_bid is not None
                rule_player.set_teams([0, 1, 0, 1], max_bid)
            else:
                rule_player = RuleBasedFirst4Player()
                rule_player.start_game(
                    teammate,
                    list(proposal[teammate]),
                    4,
                )
            if max_bid is not None:
                for bidder, bid_value in enumerate(max_bid):
                    try:
                        rule_player.bid_placed(bidder, bid_value)
                    except Exception:
                        pass
            rule_players.append(rule_player)

        spades_broken = False
        pos_in_trick = 0
        led_suit: Suit | None = None
        completed_ranks_by_suit = {suit: set() for suit in Suit}
        solver_table: list[tuple[int, Card]] = []
        replay_tricks_played = 0
        replay_tricks_won = [0, 0, 0, 0]

        for step_idx, (player, card) in enumerate(play_sequence):
            if pos_in_trick == 0:
                led_suit = card.suit

            card_bit = 1 << card.card_id
            led_mask = (
                0x1FFF << (int(led_suit.value) * 13)
                if led_suit is not None
                else 0
            )

            for proposal_index, proposal in enumerate(proposals):
                if not valid[proposal_index]:
                    continue
                current_bits = hand_bits[proposal_index][player]
                if not current_bits & card_bit:
                    valid[proposal_index] = False
                    continue
                if (
                    pos_in_trick == 0
                    and not spades_broken
                    and card.suit == Suit.SPADES
                    and current_bits & ~0x1FFF
                ):
                    valid[proposal_index] = False
                    continue
                if (
                    pos_in_trick != 0
                    and current_bits & led_mask
                    and card.suit != led_suit
                ):
                    valid[proposal_index] = False
                    continue

                if step_idx < 16 and player == teammate:
                    current_hand = [
                        candidate
                        for candidate in proposal[player]
                        if current_bits & (1 << candidate.card_id)
                    ]
                    legal = self._compute_legal_cards_for_state(
                        current_hand,
                        pos_in_trick,
                        led_suit,
                        spades_broken,
                    )
                    state_view: dict[str, Any] = {
                        "table_cards": list(solver_table),
                        "tricks_played": replay_tricks_played,
                        "spades_broken": spades_broken,
                        "trump_broken": spades_broken,
                    }
                    try:
                        expected = rule_players[proposal_index].play_card(
                            legal,
                            state_view,
                        )
                        if (
                            expected is not None
                            and expected.card_id != card.card_id
                        ):
                            play_weights[proposal_index] *= self.config.bad_action_penalty_factor
                    except Exception:
                        pass

                if step_idx >= 16 and player == teammate:
                    current_hand = [
                        candidate
                        for candidate in proposal[player]
                        if current_bits & (1 << candidate.card_id)
                    ]
                    if self._card_has_larger_equal_magnitude(
                        card,
                        current_hand,
                        completed_ranks_by_suit,
                    ):
                        play_weights[proposal_index] *= self.config.bad_action_penalty_factor

                hand_bits[proposal_index][player] = (
                    current_bits & ~card_bit
                )
                try:
                    rule_players[proposal_index].card_played(player, card)
                except Exception:
                    pass

            solver_table.append((player, card))
            if card.suit == Suit.SPADES:
                spades_broken = True
            pos_in_trick = (pos_in_trick + 1) % 4
            if pos_in_trick == 0:
                winner, _ = _trick_current_winner(
                    solver_table,
                    Suit.SPADES,
                )
                replay_tricks_won[winner] += 1
                replay_tricks_played += 1
                for _, completed_card in solver_table:
                    completed_ranks_by_suit[completed_card.suit].add(
                        completed_card.rank.value
                    )
                solver_table.clear()
                led_suit = None

        results: list[tuple[float, _ReplaySnapshot | None]] = []
        for proposal_index, proposal in enumerate(proposals):
            if not valid[proposal_index]:
                results.append((0.0, None))
                continue
            remaining_hands = [
                [
                    card
                    for card in proposal[player]
                    if hand_bits[proposal_index][player]
                    & (1 << card.card_id)
                ]
                for player in range(4)
            ]
            snapshot = _ReplaySnapshot(
                sequence_key=sequence_key,
                hands=remaining_hands,
                spades_broken=spades_broken,
                pos_in_trick=pos_in_trick,
                led_suit=led_suit,
                play_weight=play_weights[proposal_index],
                completed_ranks_by_suit={
                    suit: set(ranks)
                    for suit, ranks in completed_ranks_by_suit.items()
                },
                solver_table=list(solver_table),
                replay_tricks_played=replay_tricks_played,
                replay_tricks_won=list(replay_tricks_won),
            )
            results.append(
                (
                    play_weights[proposal_index]
                    * bid_prods[proposal_index],
                    snapshot,
                )
            )
        return results

    def _compute_importance_weight(
        self, initial_hands: list[list[Card]],
        play_sequence: list[tuple[int, Card]],
        max_bid: list[str] | None = None,
        bid_prod: float | None = None,
        original_state: GameState | None = None,
        observer_id: int | None = None,
    ) -> float:
        weight, _ = self._compute_importance_weight_with_snapshot(
            initial_hands,
            play_sequence,
            max_bid=max_bid,
            bid_prod=bid_prod,
            original_state=original_state,
            observer_id=observer_id,
        )
        return weight

    def _compute_importance_weight_with_snapshot(
        self,
        initial_hands: list[list[Card]],
        play_sequence: list[tuple[int, Card]],
        max_bid: list[str] | None = None,
        bid_prod: float | None = None,
        original_state: GameState | None = None,
        observer_id: int | None = None,
        replay_snapshot: _ReplaySnapshot | None = None,
    ) -> tuple[float, _ReplaySnapshot | None]:
        """Replay play_sequence against initial_hands; compute p = ∏(p_step).

        p = P_bid * ∏_{step} (w_step).

        坏动作（CLAUDE.md）:
        1. 前 4 墩 (step < 16) 我的队友：用 rule_based 策略检查；不一致则 weight ×= bad_action_penalty_factor（默认 0.81）
        2. 第 5 墩起 (step >= 16) 我的队友：出牌的等大牌张中有更大的 → weight ×= bad_action_penalty_factor（默认 0.81）
        3. 第 9~13 墩 (step // 4 >= 8)：精确求解器 Q 值（保持原逻辑）

        When ``replay_snapshot`` is a matching prefix of at least four tricks,
        only the newly observed actions are evaluated.  Returns ``(0, None)``
        if any move is impossible under this deal.
        """
        if bid_prod is not None:
            pass
        elif max_bid is not None:
            bid_prod = self._compute_bid_probs_product(initial_hands, max_bid)
        else:
            bid_prod = 1.0
        assert bid_prod is not None

        sequence_key = tuple(
            (int(player), int(card.card_id))
            for player, card in play_sequence
        )

        # ── 确定队友位置 ──
        teammate_positions: set[int] = set()
        if observer_id is not None:
            teammate_positions.add((observer_id + 2) % 4)

        rule_players: dict[int, RuleBasedFirst4Player] = {}
        start_idx = 0

        can_resume = (
            replay_snapshot is not None
            and len(replay_snapshot.sequence_key) >= 16
            and len(replay_snapshot.sequence_key) <= len(sequence_key)
            and sequence_key[:len(replay_snapshot.sequence_key)]
            == replay_snapshot.sequence_key
        )
        if can_resume:
            assert replay_snapshot is not None
            start_idx = len(replay_snapshot.sequence_key)
            hands = [list(hand) for hand in replay_snapshot.hands]
            spades_broken = replay_snapshot.spades_broken
            pos_in_trick = replay_snapshot.pos_in_trick
            led_suit = replay_snapshot.led_suit
            weight = replay_snapshot.play_weight
            completed_ranks_by_suit = {
                suit: set(ranks)
                for suit, ranks
                in replay_snapshot.completed_ranks_by_suit.items()
            }
            solver_table = list(replay_snapshot.solver_table)
            replay_tricks_played = replay_snapshot.replay_tricks_played
            replay_tricks_won = list(replay_snapshot.replay_tricks_won)
        else:
            # ── 为队友创建 rule-based 玩家（前 4 墩检查用）──
            has_nil_bid = (
                max_bid is not None
                and any(bid in ("nil", "blind_nil") for bid in max_bid)
            )
            for teammate in teammate_positions:
                if has_nil_bid:
                    from strategy.rule_based_first4_nil_player import (
                        RuleBasedFirst4NilPlayer,
                    )
                    rp = RuleBasedFirst4NilPlayer()
                    rp.start_game(
                        teammate,
                        list(initial_hands[teammate]),
                        4,
                    )
                    # 标准 Spades 队伍分配 (0,2) vs (1,3)
                    rp.set_teams([0, 1, 0, 1], max_bid)
                else:
                    rp = RuleBasedFirst4Player()
                    rp.start_game(
                        teammate,
                        list(initial_hands[teammate]),
                        4,
                    )
                if max_bid is not None:
                    for bidder, bid_val in enumerate(max_bid):
                        try:
                            rp.bid_placed(bidder, bid_val)
                        except Exception:
                            pass
                rule_players[teammate] = rp

            hands = [list(hand) for hand in initial_hands]
            spades_broken = False
            pos_in_trick = 0
            led_suit: Suit | None = None
            weight = 1.0
            # Current-table ranks still affect this trick and cannot be treated
            # as gone by the equivalent-card grouping.
            completed_ranks_by_suit = {suit: set() for suit in Suit}
            solver_table: list[tuple[int, Card]] = []
            replay_tricks_played = 0
            replay_tricks_won = [0, 0, 0, 0]

        # Pre-copy original state once for solver-based weighting (tricks 9-13)
        sim_state = None
        if (
            max(
                start_idx,
                max(0, self.config.trick_num_threshold) * 4,
            )
            < len(play_sequence)
            and original_state is not None
            and self.exact_solver is not None
        ):
            sim_state = copy.deepcopy(original_state)

        for step_idx in range(start_idx, len(play_sequence)):
            player, card = play_sequence[step_idx]
            hand = hands[player]
            try:
                idx = hand.index(card)
            except ValueError:
                return 0.0, None

            if pos_in_trick == 0:  # Leading
                if not spades_broken and card.suit == Suit.SPADES:
                    has_non_spade = any(c.suit != Suit.SPADES for c in hand)
                    if has_non_spade:
                        return 0.0, None
                led_suit = card.suit
            else:  # Following
                has_led = any(c.suit == led_suit for c in hand)
                if has_led and card.suit != led_suit:
                    return 0.0, None

            # ── 前 4 墩（step < 16）：用 rule_based 检查队友动作 ──
            if step_idx < 16 and player in rule_players:
                rp = rule_players[player]
                legal = self._compute_legal_cards_for_state(
                    hand, pos_in_trick, led_suit, spades_broken,
                )
                state_view: dict[str, Any] = {
                    "table_cards": list(solver_table),
                    "tricks_played": replay_tricks_played,
                    "spades_broken": spades_broken,
                    "trump_broken": spades_broken,
                }
                try:
                    rp_card = rp.play_card(legal, state_view)
                    if rp_card is not None and rp_card.card_id != card.card_id:
                        weight *= self.config.bad_action_penalty_factor  # 坏动作
                except Exception:
                    pass  # rule player 异常不影响权重

            # ── 第 5 墩起（step >= 16）：等大牌张检查队友动作 ──
            if step_idx >= 16 and player in teammate_positions:
                if self._card_has_larger_equal_magnitude(
                    card, hand, completed_ranks_by_suit,
                ):
                    weight *= self.config.bad_action_penalty_factor  # 坏动作：出了非最大等大牌张

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
                        return 0.0, None  # 该动作在 proposal 下不合法

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

            # 将本步动作喂给 rule_players（保持内部状态一致，供后续前 4 墩检查）
            for rp in rule_players.values():
                try:
                    rp.card_played(player, card)
                except Exception:
                    pass

            if card.suit == Suit.SPADES:
                spades_broken = True
            pos_in_trick = (pos_in_trick + 1) % 4
            if pos_in_trick == 0:
                winner, _ = _trick_current_winner(solver_table, Suit.SPADES)
                replay_tricks_won[winner] += 1
                replay_tricks_played += 1
                for _pid, completed_card in solver_table:
                    completed_ranks_by_suit[completed_card.suit].add(
                        completed_card.rank.value
                    )
                solver_table.clear()
                led_suit = None

        snapshot = _ReplaySnapshot(
            sequence_key=sequence_key,
            hands=[list(hand) for hand in hands],
            spades_broken=spades_broken,
            pos_in_trick=pos_in_trick,
            led_suit=led_suit,
            play_weight=weight,
            completed_ranks_by_suit={
                suit: set(ranks)
                for suit, ranks in completed_ranks_by_suit.items()
            },
            solver_table=list(solver_table),
            replay_tricks_played=replay_tricks_played,
            replay_tricks_won=list(replay_tricks_won),
        )
        return weight * bid_prod, snapshot

    def _build_is_pool(
        self, state: GameState, observer_id: int, rng: random.Random,
        num_proposals: int = 1234,
        num_proposals_limit: int = 5678,
        min_pool_size: int = 100,
    ) -> tuple[list[list[list[Card]]], list[float]]:
        """Build IS pool: generate proposals, compute weights.

        Keeps sampling (in batches of num_proposals) until the pool has at least
        min_pool_size valid (w > 0) proposals, or total attempts reach num_proposals_limit.
        A compatible cache reuses the same posterior particles and evaluates
        only public actions appended since the previous decision.
        """
        play_sequence = self._build_play_sequence(state)
        sequence_key = tuple(
            (int(player), int(card.card_id))
            for player, card in play_sequence
        )
        observer_initial_ids = [
            card.card_id for card in state.hands[observer_id]
        ]
        observer_initial_ids.extend(
            card.card_id
            for player, card in play_sequence
            if player == observer_id
        )
        visible_deal_key = (
            observer_id,
            tuple(sorted(observer_initial_ids)),
        )
        self._last_pool_cache_hit = False

        def finish_pool(
            proposals: list[list[list[Card]]],
            weights: list[float],
        ) -> tuple[list[list[list[Card]]], list[float]]:
            remaining = sum(len(hand) for hand in state.hands)
            gt01 = sum(1 for weight in weights if weight > 0.1)
            gt001 = sum(1 for weight in weights if weight > 0.01)
            gt0001 = sum(1 for weight in weights if weight > 0.001)
            gt00001 = sum(1 for weight in weights if weight > 0.0001)
            sorted_weights = sorted(weights, reverse=True)
            a_value = sum(sorted_weights[:3])
            b_value = (
                sum(sorted_weights[31:34])
                if len(sorted_weights) >= 34
                else None
            )
            c_value = (
                sum(sorted_weights[99:102])
                if len(sorted_weights) >= 102
                else None
            )
            ratio_ab = f"{a_value / b_value:.3f}" if b_value else "nan"
            ratio_ac = f"{a_value / c_value:.3f}" if c_value else "nan"
            with open("is_proposal_stats.txt", "a") as stats_file:
                stats_file.write(
                    f"n_proposals={len(proposals)} remaining={remaining} "
                    f"w>0.1={gt01} w>0.01={gt001} "
                    f"w>0.001={gt0001} w>0.0001={gt00001} "
                    f"A/B={ratio_ab} A/C={ratio_ac}\n"
                )
            return proposals, weights

        max_bid: list[str] | None = None
        raw_bids = None
        if hasattr(state, "max_bid") and state.max_bid:
            raw_bids = state.max_bid
        elif hasattr(state, "bids") and state.bids:
            raw_bids = state.bids
        if raw_bids is not None and len(raw_bids) == 4:
            if all(isinstance(b, str) for b in raw_bids):
                max_bid = list(raw_bids)
        max_bid_key = tuple(max_bid) if max_bid is not None else None

        cache = self._posterior_cache
        cache_matches = (
            cache is not None
            and cache.deal_key == visible_deal_key
            and cache.observer_id == observer_id
            and cache.max_bid == max_bid_key
            and cache.num_proposals == num_proposals
            and cache.num_proposals_limit == num_proposals_limit
            and len(cache.sequence_key) >= 16
            and len(cache.sequence_key) <= len(sequence_key)
            and sequence_key[:len(cache.sequence_key)] == cache.sequence_key
        )
        if cache_matches:
            assert cache is not None
            updated_entries: list[_PosteriorEntry] = []
            updated_weights: list[float] = []

            def advance_cached_entry(entry: _PosteriorEntry) -> None:
                weight, replay = self._compute_importance_weight_with_snapshot(
                    entry.initial_hands,
                    play_sequence,
                    max_bid=max_bid,
                    bid_prod=entry.bid_prod,
                    original_state=state,
                    observer_id=observer_id,
                    replay_snapshot=entry.replay,
                )
                if weight > 0.0 and replay is not None:
                    updated_entries.append(
                        _PosteriorEntry(
                            initial_hands=entry.initial_hands,
                            bid_prod=entry.bid_prod,
                            replay=replay,
                        )
                    )
                    updated_weights.append(weight)

            # A newly observed exact-stage action can invalidate nearly the
            # whole particle set.  Probe a bounded prefix before spending the
            # full incremental-replay cost; if projected survivors cannot
            # clear the configured quality floor, rebuild immediately.
            pilot_count = min(
                len(cache.entries),
                max(min_pool_size, 128),
            )
            for entry in cache.entries[:pilot_count]:
                advance_cached_entry(entry)

            projected_survivors = (
                len(updated_entries) * len(cache.entries) / pilot_count
                if pilot_count
                else 0.0
            )
            should_finish_incremental = (
                pilot_count == len(cache.entries)
                or projected_survivors >= min_pool_size * 1.25
            )
            if should_finish_incremental:
                for entry in cache.entries[pilot_count:]:
                    advance_cached_entry(entry)

            # Rebuild rather than running with a depleted particle set.  This
            # preserves the configured quality floor while making the common
            # case incremental.
            if (
                should_finish_incremental
                and len(updated_entries) >= min_pool_size
            ):
                self._posterior_cache = _PosteriorCache(
                    deal_key=visible_deal_key,
                    observer_id=observer_id,
                    max_bid=max_bid_key,
                    sequence_key=sequence_key,
                    num_proposals=num_proposals,
                    num_proposals_limit=num_proposals_limit,
                    entries=updated_entries,
                )
                self._last_pool_cache_hit = True
                return finish_pool(
                    [entry.initial_hands for entry in updated_entries],
                    updated_weights,
                )
            self._posterior_cache = None

        played_by_player: dict[int, list[Card]] = {p: [] for p in range(4)}
        for p, c in play_sequence:
            played_by_player[p].append(c)

        # 推断每人断门花色（领出 X 但跟牌人未出 X → 该人无 X）
        void_suits = self._compute_void_suits(play_sequence)
        sampler_context = None
        generate_function = getattr(
            self._generate_proposal,
            "__func__",
            None,
        )
        try:
            sampler_context = self._prepare_proposal_sampler(
                state.all_cards,
                observer_id,
                state.hands[observer_id],
                played_by_player,
                void_suits,
            )
        except ValueError:
            # Test/custom subclasses may synthesize proposals from deliberately
            # incomplete states and ignore sampler_context altogether.  The
            # production implementation must still surface inconsistent facts.
            if generate_function is RuleExactFirst4Player._generate_proposal:
                raise

        proposals: list[list[list[Card]]] = []
        prop_weights: list[float] = []
        posterior_entries: list[_PosteriorEntry] = []
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
                    sampler_context=sampler_context,
                )
                batch_proposals.append(initial_hands)
            total_attempts += batch_size

            # Compute bid probability products for this batch
            if max_bid is not None:
                bid_prods = self._compute_batch_bid_prods(batch_proposals, max_bid)
            else:
                bid_prods = [1.0] * batch_size

            solver_weighting_starts = (
                max(0, self.config.trick_num_threshold) * 4
            )
            if (
                self.exact_solver is None
                or solver_weighting_starts >= len(play_sequence)
            ):
                weighted_replays = self._compute_batch_replay_weights(
                    batch_proposals,
                    play_sequence,
                    bid_prods,
                    max_bid,
                    observer_id,
                )
            else:
                weighted_replays = [
                    self._compute_importance_weight_with_snapshot(
                        initial_hands,
                        play_sequence,
                        max_bid=max_bid,
                        bid_prod=bid_prod,
                        original_state=state,
                        observer_id=observer_id,
                    )
                    for initial_hands, bid_prod in zip(
                        batch_proposals,
                        bid_prods,
                    )
                ]

            # Filter zero-weight proposals.
            for initial_hands, bid_prod, (weight, replay) in zip(
                batch_proposals,
                bid_prods,
                weighted_replays,
            ):
                if weight > 0.0 and replay is not None:
                    proposals.append(initial_hands)
                    prop_weights.append(weight)
                    posterior_entries.append(
                        _PosteriorEntry(
                            initial_hands=initial_hands,
                            bid_prod=bid_prod,
                            replay=replay,
                        )
                    )

            if len(proposals) >= min_pool_size:
                break

        if len(sequence_key) >= 16 and len(posterior_entries) >= min_pool_size:
            self._posterior_cache = _PosteriorCache(
                deal_key=visible_deal_key,
                observer_id=observer_id,
                max_bid=max_bid_key,
                sequence_key=sequence_key,
                num_proposals=num_proposals,
                num_proposals_limit=num_proposals_limit,
                entries=posterior_entries,
            )

        return finish_pool(proposals, prop_weights)

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
