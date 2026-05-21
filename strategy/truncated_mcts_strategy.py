"""截断式 MCTS 出牌策略。

文件作用：
- 在剩余牌数不超过 30 张时，直接调用精确求解器输出最优动作。
- 在剩余牌数大于 30 张时，运行一个全新的 PUCT/MCTS 搜索：
  - 搜索到剩余牌数 <= leaf_threshold 时，使用 value-head 进行局面估值；
  - 使用 policy-head 提供先验概率；
  - 使用队伍 0 视角的价值进行回传与动作比较；
  - 根节点动作选择时，队伍 0 取 argmax，队伍 1 取 argmin。

函数/类输入输出说明：
- TruncatedMCTSConfig:
    输入字段:
      - exact_threshold: int，剩余牌数 <= 该值时直接精确求解
      - leaf_threshold: int，MCTS 搜索到该剩余牌数时接入 MLP 估值
      - simulations_per_action: int，每个根动作模拟次数
      - exploration_constant: float，PUCT 探索系数
      - policy_temperature: float，policy head 温度
      - value_scale: float，模型输出缩放回真实 value_view 的倍率
      - checkpoint_path: str | None，MLP 权重文件路径
    输出: 配置对象本身

- TruncatedMCTSStrategy.choose_action(state: GameState) -> Card | None
    输入: 完整的 Spades 牌局状态（叫牌结束、出牌阶段、全信息）
    输出: 当前应出的最优牌；若无合法动作则返回 None

- TruncatedMCTSStrategy.play_full_game(state: GameState) -> list[Card]
    输入: 可执行到终局的完整牌局状态
    输出: 按出牌顺序排列的动作序列

- TruncatedMCTSStrategy.choose_action_with_info(state: GameState) -> dict
    输入: 同上
    输出: 包含 best_action、best_value、mode、action_scores、root_team 等信息的字典
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

from mlp.mlp_model import DoubleDummyMLP
from trick_taking.card import Card, Suit
from trick_taking.game_state import GameState
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver
from trick_taking.solvers.exact_double_dummy_cpp_opt1 import ExactDoubleDummyCppOpt1Solver
from trick_taking.utils.feature_encoder import SpadesFeatureEncoder
from trick_taking.card import _STANDARD_CARDS as STANDARD_52
import random
import copy
import time

from tqdm import tqdm


@dataclass
class TruncatedMCTSConfig:
    """截断 MCTS 的参数配置。"""

    exact_threshold: int = 24
    leaf_threshold: int = 24
    simulations_per_action: int = 20
    exploration_constant: float = 1.5
    policy_temperature: float = 1.0
    value_scale: float = 25.0
    checkpoint_path: str | None = None
    # Determinization options: when enabled, opponents' hands are sampled from
    # the unseen card pool instead of using their private ground-truth hands.
    use_determinization: bool = True
    determinization_count: int = 32
    # Number of IS-proposal determinizations to draw for each MCTS decision.
    # Each determinization gets its own independent MCTS tree (no sharing),
    # and results are averaged across determinizations.
    mcts_determinization_count: int = 4
    device: str = "cpu"
    # optional prior oracle spec (e.g. 'go_rule_2') to bias root priors
    prior_oracle_spec: str = "no"
    # Path to bid_nsfp.pt (BidMLP checkpoint) for bid probability in IS weights
    bid_checkpoint_path: str = ""


@dataclass
class SearchNode:
    """MCTS 树节点。

    输入:
    - state: 当前节点对应的 GameState
    - parent: 父节点，根节点为 None
    - action_from_parent: 从父节点到达当前节点所用的动作
    - prior: 该动作的先验概率

    输出:
    - 节点内部统计信息（访问次数、累积价值、孩子节点、未展开动作）
    """

    state: GameState
    parent: "SearchNode | None" = None
    action_from_parent: Card | None = None
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[Card, "SearchNode"] = field(default_factory=dict)
    unexpanded_actions: list[Card] = field(default_factory=list)
    action_priors: dict[int, float] = field(default_factory=dict)

    @property
    def average_value(self) -> float:
        """返回节点的平均价值（队伍 0 视角）。"""
        return self.value_sum / self.visits if self.visits > 0 else 0.0


class TruncatedMCTSStrategy:
    """基于 PUCT 的截断式出牌策略。"""

    def __init__(self, config: TruncatedMCTSConfig | None = None) -> None:
        self.config = config or TruncatedMCTSConfig()
        self.rules = SpadesRules()
        # Prefer C++ opt1 exact solver for speed; fall back to Python reference
        # only when native compilation/loading is unavailable.
        cpp_solver = ExactDoubleDummyCppOpt1Solver()
        self.exact_solver = cpp_solver if cpp_solver.native_available else ExactDoubleDummySolver()
        self.encoder = SpadesFeatureEncoder()
        self._leaf_value_cache: dict[int, float] = {}
        self._policy_priors_cache: dict[int, dict[int, float]] = {}
        # Per-decision caches avoid recomputing state hashes and legal-action
        # lists repeatedly inside one MCTS root search.
        self._decision_state_key_cache: dict[int, int] | None = None
        self._decision_legal_actions_cache: dict[int, list[Card]] | None = None
        self._decision_policy_priors_cache: dict[int, dict[int, float]] | None = None
        self._decision_leaf_value_cache: dict[int, float] | None = None
        # Diagnostics counters for performance analysis
        self._model_calls: int = 0
        self._policy_model_calls: int = 0
        self._exact_calls: int = 0
        self.model = self._load_model(self.config.checkpoint_path)
        # Ensure Spades_AI_GO-MCTS is on sys.path for oracle/BidMLP imports
        _collab_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "Spades_AI_GO-MCTS"))
        if _collab_root not in sys.path:
            sys.path.insert(0, _collab_root)

        # Lazy-loaded BidMLP for bid probability in importance sampling weights
        self._bid_model = None
        self._bid_encoder = None

        # optional external prior oracle (rule-based). Try to construct
        # if a spec was provided; keep None on failure.
        self._prior_oracle = None
        spec = getattr(self.config, "prior_oracle_spec", "")
        self._oracle_requested = spec not in {"", "no"}
        self._fallback_print_count = 0
        if self._oracle_requested:
            try:
                if spec == "go_rule_2":
                    from spades_ai.players.rule_based_v2.player import RuleBasedPlayer as _RBP  # type: ignore
                else:
                    from evaluate.GO_MCTS.models import RuleBasedPlayer as _RBP  # type: ignore
                self._prior_oracle = _RBP()
            except Exception:
                try:
                    from spades_ai.players.rule_based.player import RuleBasedPlayer as _RBP2  # type: ignore
                    self._prior_oracle = _RBP2()
                except Exception:
                    self._prior_oracle = None
        # Cache bridge module for converting local state -> GoGameState
        self._bridge_mod = None
        if self._prior_oracle is not None:
            try:
                from pathlib import Path
                import importlib.util
                _bp = Path(__file__).resolve().parent.parent / "evaluate" / "GO-MCTS" / "bridge.py"
                _spec = importlib.util.spec_from_file_location("_go_bridge", str(_bp))
                if _spec and _spec.loader:
                    _mod = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    self._bridge_mod = _mod
            except Exception:
                import traceback; traceback.print_exc()
                self._bridge_mod = None

    def _load_model(self, checkpoint_path: str | None) -> DoubleDummyMLP | None:
        """加载 value/policy 双头模型。

        输入:
        - checkpoint_path: 权重文件路径；为 None 时返回 None，表示不使用 MLP。

        输出:
        - DoubleDummyMLP 或 None
        """
        if not checkpoint_path:
            return None
        model = DoubleDummyMLP(input_dim=self.encoder.total_dim)
        model.load(checkpoint_path, device=self.config.device)
        return model

    def _state_cache_key(self, state: GameState) -> int:
        """生成用于缓存的状态键。

        输入:
        - state: 当前完整牌局状态。

        输出:
        - int: 用于叶子估值/先验缓存的稳定键。
        """
        if self._decision_state_key_cache is not None:
            state_id = id(state)
            cached = self._decision_state_key_cache.get(state_id)
            if cached is not None:
                return cached
            cache_key = self.exact_solver._state_hash(state) ^ self.exact_solver._tt_verify(state)
            self._decision_state_key_cache[state_id] = cache_key
            return cache_key
        return self.exact_solver._state_hash(state) ^ self.exact_solver._tt_verify(state)

    def choose_action(self, state: GameState) -> Card | None:
        """选择当前状态的最优动作。"""
        remaining_cards = self._remaining_cards(state)
        legal_actions = self._legal_actions(state)
        if not legal_actions:
            return None
        if len(legal_actions) == 1:
            return legal_actions[0]

        # 24张及以下直接用精确求解器。若启用 determinization，则对对手手牌做采样并汇总结果。
        if remaining_cards <= self.config.exact_threshold:
            self._exact_calls += 1
            if self.config.use_determinization:
                result = self._solve_with_determinization(state)
            else:
                result = self.exact_solver.solve_with_q(state)
            best_action = result["best_action"]
            if best_action is not None:
                return best_action
            return legal_actions[0]

        info = self.choose_action_with_info(state)
        return info["best_action"]

    def choose_action_with_info(self, state: GameState) -> dict[str, Any]:
        """返回带搜索统计的动作选择结果。"""
        self._decision_state_key_cache = {}
        self._decision_legal_actions_cache = {}
        self._decision_policy_priors_cache = {}
        self._decision_leaf_value_cache = {}
        try:
            return self._choose_action_with_info_impl(state)
        finally:
            self._decision_state_key_cache = None
            self._decision_legal_actions_cache = None
            self._decision_policy_priors_cache = None
            self._decision_leaf_value_cache = None

    def _choose_action_with_info_impl(self, state: GameState) -> dict[str, Any]:
        """Internal implementation for `choose_action_with_info`.

        Keeping the cache lifetime scoped to a single root decision avoids
        stale entries when the external game engine mutates `GameState`
        objects between turns.
        """
        remaining_cards = self._remaining_cards(state)
        legal_actions = self._legal_actions(state)
        if not legal_actions:
            return {
                "best_action": None,
                "best_value": 0.0,
                "mode": "no_legal_action",
                "root_team": self._current_team(state),
                "action_scores": [],
            }

        if len(legal_actions) == 1:
            return {
                "best_action": legal_actions[0],
                "best_value": 0.0,
                "mode": "single_action",
                "root_team": self._current_team(state),
                "action_scores": [{"action": legal_actions[0], "value": 0.0, "visits": 0}],
            }

        if remaining_cards <= self.config.exact_threshold:
            if self.config.use_determinization:
                exact_result = self._solve_with_determinization(state)
            else:
                exact_result = self.exact_solver.solve_with_q(state)
            action_scores = [
                {"action": action, "value": float(value)}
                for action, value in exact_result["action_q_values"].items()
            ]
            root_team = self._current_team(state)
            action_scores.sort(key=lambda item: item["value"], reverse=(root_team == 0))
            return {
                "best_action": exact_result["best_action"],
                "best_value": float(exact_result["value"]),
                "mode": "exact",
                "root_team": root_team,
                "action_scores": action_scores,
            }

        root_team = self._current_team(state)
        root_scores: list[dict[str, Any]] = []

        # --- Independent determinizations for MCTS ---
        # Build IS pool once, draw K distinct proposals.  Each proposal is a
        # complete initial-deal hypothesis.  We run a *separate* MCTS for each
        # proposal with the opponent hands fixed throughout — no tree sharing
        # across proposals.  This avoids the stale-state problem where different
        # determinizations pollute the same MCTS tree.
        det_states: list[GameState] = [state]
        if self.config.use_determinization:
            mcts_is_rng = random.Random()
            mcts_pool_hands, mcts_pool_weights = self._build_is_pool(state, state.turn, mcts_is_rng)
            if mcts_pool_hands:
                K = self.config.mcts_determinization_count
                det_states = []
                drawn_distinct: set[tuple] = set()
                for _ in range(K):
                    chosen = self._draw_is_sample(mcts_pool_hands, mcts_pool_weights, mcts_is_rng)
                    if chosen is not None:
                        try:
                            det_state = self.exact_solver._deep_copy_state(state)
                        except Exception:
                            det_state = copy.deepcopy(state)
                        self._apply_proposal(det_state, state.turn, chosen)
                        det_states.append(det_state)
                        # Track distinctness of drawn samples
                        drawn_key = tuple(
                            tuple(sorted(c.card_id for c in chosen[p]))
                            for p in range(4)
                        )
                        drawn_distinct.add(drawn_key)
                print(f"  [DEBUG sample] {K} draws, {len(drawn_distinct)} distinct initial-deal hands from pool of {len(mcts_pool_hands)}", flush=True)
                if not det_states:
                    det_states = [state]  # fallback: full-info (no determinization)

        total_sims = len(det_states) * len(legal_actions) * self.config.simulations_per_action
        pbar = tqdm(total=total_sims, desc=f"MCTS (rem={remaining_cards})", unit="sim",
                    leave=False, position=2)

        action_value_sum: dict[int, float] = {a.card_id: 0.0 for a in legal_actions}
        for det_state in det_states:
            for action in legal_actions:
                child_state = self._apply_action(det_state, action)
                child_node = self._build_root_child(child_state, action)
                for _ in range(self.config.simulations_per_action):
                    self._run_simulation(
                        child_node, root_observer_id=child_state.turn,
                        skip_determinization=True,
                    )
                    pbar.update(1)
                action_value_sum[action.card_id] += child_node.average_value

        pbar.close()

        for action in legal_actions:
            root_scores.append({
                "action": action,
                "value": action_value_sum[action.card_id] / len(det_states),
                "visits": 0,
            })

        best_score = root_scores[0]
        for item in root_scores[1:]:
            if root_team == 0:
                if item["value"] > best_score["value"]:
                    best_score = item
            else:
                if item["value"] < best_score["value"]:
                    best_score = item

        root_scores.sort(key=lambda item: item["value"], reverse=(root_team == 0))
        return {
            "best_action": best_score["action"],
            "best_value": float(best_score["value"]),
            "mode": "mcts",
            "root_team": root_team,
            "action_scores": root_scores,
        }

    def play_full_game(self, state: GameState) -> list[Card]:
        """从当前局面开始，一直出到终局，返回完整动作序列。"""
        actions: list[Card] = []
        safety = 0
        while not self.rules.end_trickgame(state):
            action = self.choose_action(state)
            if action is None:
                break
            actions.append(action)
            self._apply_action_in_place(state, action)
            safety += 1
            if safety > 52:
                raise RuntimeError("出牌序列超过 52 步，状态推进可能有误")
        return actions

    def _build_root_child(self, state: GameState, action: Card) -> SearchNode:
        """把根动作后的状态包装为搜索树根子节点。"""
        node = SearchNode(state=state, action_from_parent=action, prior=1.0)
        node.unexpanded_actions = self._legal_actions(state)
        return node

    def _run_simulation(
        self,
        root_node: SearchNode,
        root_observer_id: int | None = None,
        is_pool_hands: list[list[list[Card]]] | None = None,
        is_pool_weights: list[float] | None = None,
        is_rng: random.Random | None = None,
        skip_determinization: bool = False,
    ) -> float:
        """从给定节点向下做一次 PUCT 搜索，并回传队伍 0 视角价值。

        If determinization is enabled, this simulation will operate on a
        simulation-local copy of the state with opponents' hands sampled
        consistent with the public information for the given observer.
        When is_pool_hands is provided, uses importance-sampled determinization
        from a pre-built pool; otherwise falls back to uniform determinization.
        """
        path: list[SearchNode] = [root_node]
        node = root_node

        # Simulation-local state that follows the `node` as we advance.
        # Use solver's lightweight deep-copy to reduce Python object allocation.
        try:
            sim_state = self.exact_solver._deep_copy_state(node.state)
        except Exception:
            sim_state = copy.deepcopy(node.state)

        if self.config.use_determinization and root_observer_id is not None and not skip_determinization:
            if is_pool_hands is not None and is_pool_weights is not None:
                self._apply_is_determinization(
                    sim_state, root_observer_id, is_pool_hands, is_pool_weights,
                    is_rng or random.Random(),
                )
            else:
                self._determinize_state(sim_state, observer_id=root_observer_id)
            # `node` may have been created from a full-information state.
            # Rebuild its cached action list/prior distribution so they match
            # the determinized simulation state instead of the original state.
            node.unexpanded_actions = []
            node.action_priors = {}

        effective_leaf_threshold = self.config.leaf_threshold 
        #print("root-obs-id", root_observer_id)
        # if root_observer_id is not None:                                                    
        #     _bids = None                                                                    
        #     if hasattr(sim_state, "max_bid") and sim_state.max_bid:                         
        #         _bids = sim_state.max_bid    
        #         #print(sim_state.max_bid)
        #         #print("sim_state.max_bid")
        #     # if hasattr(sim_state, "bids") and sim_state.bids:   
        #     #     print("sim_state.bids")
        #     #     _bids = sim_state.bids                                                      
        #     if _bids and len(_bids) == 4:  
        #         _partner = (root_observer_id + 2) % 4      
        #         if _bids[0] == "nil":
        #             effective_leaf_threshold = 0
        #         if _bids[1] == "nil":
        #             effective_leaf_threshold = 0
        #         if _bids[2] == "nil":
        #             effective_leaf_threshold = 0
        #         if _bids[3] == "nil":
        #             effective_leaf_threshold = 0
        #         # if _bids[root_observer_id] == "nil":  
        #         #     #print("me nil, me is", root_observer_id)
        #         #     effective_leaf_threshold = 0
        #         # if _bids[_partner] == "nil":
        #         #     #print("partner nil， partner is", _partner)
        #         #     effective_leaf_threshold = 0
                    
        #print("effective_leaf_threshold=", effective_leaf_threshold)
        while True:
            if self._is_terminal(sim_state):
                value = self._terminal_value(sim_state)
                break

            remaining_cards = self._remaining_cards(sim_state)
            if remaining_cards <= effective_leaf_threshold:
                value = self._leaf_value(sim_state)
                break

            if not node.unexpanded_actions:
                if not node.children:
                    node.unexpanded_actions = self._legal_actions(sim_state)

            if node.unexpanded_actions:
                # --- decide which action to expand ---
                # 1st priority: oracle recommendation (if still unexpanded)
                # 2nd priority: first remaining by card_id order
                chosen_prior = None
                oracle_action_card = None
                if self._prior_oracle is not None and self._bridge_mod is not None:
                    try:
                        go_state = self._bridge_mod.to_go_state(node.state)
                        go_card = self._prior_oracle.choose_card(go_state)
                        local_rec = self._bridge_mod.to_local_card(go_card)
                        oracle_cid = local_rec.card_id
                        all_legal_full = self._legal_actions(node.state)
                        if any(oracle_cid == a.card_id for a in all_legal_full):
                            chosen_prior = oracle_cid
                            # Check if oracle's choice is still unexpanded
                            for i, a in enumerate(node.unexpanded_actions):
                                if a.card_id == oracle_cid:
                                    oracle_action_card = node.unexpanded_actions.pop(i)
                                    break
                    except Exception:
                        chosen_prior = None

                if oracle_action_card is not None:
                    action = oracle_action_card
                else:
                    action = node.unexpanded_actions.pop(0)

                child_state = self._apply_action(sim_state, action)
                n_actions = len(node.unexpanded_actions) + 1
                if chosen_prior is None:
                    # print("!!!!!!! CHOSEN PRIOR is none", flush=True)
                    uniform_prior = 1.0 / max(n_actions, 1)
                else:
                    # set prior for the chosen action to .75, others split 0.25
                    uniform_prior = 0.0
                child = SearchNode(
                    state=child_state,
                    parent=node,
                    action_from_parent=action,
                    prior=(0.75 if chosen_prior is not None and action.card_id == chosen_prior else (0.25 / max(n_actions - 1, 1)) if chosen_prior is not None else uniform_prior),
                )
                child.unexpanded_actions = self._legal_actions(child_state)
                node.children[action] = child
                path.append(child)
                node = child
                # advance sim_state to the child state's copy
                sim_state = child_state
                continue

            child = self._select_child_puct(node)
            if child is None:
                value = self._leaf_value(sim_state)
                break

            path.append(child)
            node = child
            # advance sim_state to match the chosen child
            sim_state = node.state

        for visited_node in path:
            visited_node.visits += 1
            visited_node.value_sum += value
        return value

    # -------------------- Determinization helpers --------------------

    def _precompute_go_step_data(
        self,
        play_sequence: list[tuple[int, Card]],
        original_state: GameState,
    ) -> list[dict]:
        """Precompute GoGameState invariants at each step of play_sequence.

        For each step k, returns a dict with everything needed to build
        a GoGameState *except* hands (which is proposal-dependent).
        Returns list of length len(play_sequence); entry k is the state
        *before* the k-th action.
        """
        bm = self._bridge_mod

        # Convert bids (same for all steps)
        go_bids = []
        for bv in original_state.max_bid:
            gb = bm._to_go_bid_value(bv)
            if gb is not None:
                go_bids.append(gb)
        go_bids_tuple = tuple(go_bids)

        steps: list[dict] = []
        completed_tricks_list: list = []
        current_trick_cards_list: list = []
        tricks_won = [0, 0, 0, 0]
        trick_number = 0
        spades_broken_flag = False
        void_sets: list[set] = [set(), set(), set(), set()]
        leader = play_sequence[0][0] if play_sequence else 0

        for step_idx, (player, card) in enumerate(play_sequence):
            pos_in_trick = step_idx % 4

            # --- record state BEFORE this action ---
            steps.append({
                "completed_tricks": tuple(completed_tricks_list),
                "current_trick_cards": tuple(current_trick_cards_list),
                "leader": leader,
                "trick_number": trick_number + 1,
                "tricks_won": tuple(tricks_won),
                "spades_broken": spades_broken_flag,
                "void_shown": tuple(frozenset(s) for s in void_sets),
                "bids": go_bids_tuple,
            })

            # --- apply this action to our tracking ---
            go_card = bm.to_go_card(card)
            go_tc = bm.GoTrickCard(player=player, card=go_card)

            # Void inference: follower who plays off-suit is void in led suit
            if pos_in_trick > 0 and current_trick_cards_list:
                led_suit_go = current_trick_cards_list[0].card.suit
                if go_card.suit != led_suit_go:
                    void_sets[player].add(led_suit_go)

            current_trick_cards_list.append(go_tc)

            if card.suit == Suit.SPADES:
                spades_broken_flag = True

            # Trick completion?
            if pos_in_trick == 3:
                led_suit_go = current_trick_cards_list[0].card.suit
                trick = bm.GoTrick(
                    cards=tuple(current_trick_cards_list),
                    led_suit=led_suit_go,
                )
                winner = trick.winner()
                tricks_won[winner] += 1
                completed_tricks_list.append(trick)
                current_trick_cards_list = []
                trick_number += 1
                leader = winner

        return steps

    def _build_play_sequence(self, state: GameState) -> list[tuple[int, Card]]:
        """Extract ordered (player_id, card) sequence from trick_history + table_cards.

        Input:
        - state: current game state.

        Output:
        - list of (player_id, Card) in the order cards were played so far.
        """
        sequence: list[tuple[int, Card]] = []
        for record in state.trick_history:
            for pid, card in record.cards:
                sequence.append((pid, card))
        for pid, card in state.table_cards:
            sequence.append((pid, card))
        return sequence

    def _generate_proposal(
        self,
        all_cards: list[Card],
        observer_id: int,
        observer_current_hand: list[Card],
        played_by_player: dict[int, list[Card]],
        rng: random.Random,
    ) -> list[list[Card]]:
        """Generate one random initial deal proposal consistent with observed play.

        Input:
        - all_cards: full 52-card deck.
        - observer_id: the player whose hand is fully known.
        - observer_current_hand: observer's current hand at this state.
        - played_by_player: dict mapping each player to cards they have played.
        - rng: seeded random generator.

        Output:
        - 4 initial hands (each list[Card], 13 cards each).
        """
        # Observer's initial = current hand + played cards
        obs_set: set[int] = set(c.card_id for c in observer_current_hand)
        obs_set.update(c.card_id for c in played_by_player[observer_id])
        id_to_card = {c.card_id: c for c in all_cards}
        observer_initial = [id_to_card[cid] for cid in obs_set]

        # Collect which cards are spoken-for
        used_ids: set[int] = set(obs_set)
        for p in range(4):
            if p != observer_id:
                used_ids.update(c.card_id for c in played_by_player[p])

        # Pool of remaining cards (those unaccounted for)
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

    @staticmethod
    def _bid_str_to_mlp_index(bid_str: str) -> int:
        """Convert a local bid string to a BidMLP output index (0-15).

        Input:
        - bid_str: e.g. "bid_3", "nil", "blind_nil"

        Output:
        - index 0-13 for normal bids, 14 for Nil, 15 for Blind Nil.
        """
        if bid_str == "nil":
            return 14
        if bid_str == "blind_nil":
            return 15
        if bid_str.startswith("bid_"):
            return int(bid_str.split("_")[1])
        return 0

    def _compute_bid_probs_product(
        self,
        initial_hands: list[list[Card]],
        max_bid: list[str],
    ) -> float:
        """Compute ∏_{p=0..3} P(bid_p | hand_p) from BidMLP softmax.

        Input:
        - initial_hands: 4 initial 13-card hands of the proposal.
        - max_bid: list of 4 bid strings indexed by player.

        Output:
        - Product of softmax probabilities of the actual bids (float).
          Returns 1.0 if the BidMLP is unavailable.
        """
        # Lazy-load BidMLP on first call
        if self._bid_model is None:
            ckpt = self.config.bid_checkpoint_path
            if not ckpt or not os.path.exists(ckpt):
                return 1.0
            try:
                import torch
                from spades_ai.models.bid_mlp import BidMLP
                from spades_ai.models.bid_encoder import BidEncoder
                self._bid_model = BidMLP()
                state_dict = torch.load(ckpt, weights_only=True, map_location="cpu")
                self._bid_model.load_state_dict(state_dict)
                self._bid_model.eval()
                self._bid_encoder = BidEncoder()
            except Exception:
                self._bid_model = None
                return 1.0

        if self._bid_encoder is None:
            return 1.0

        import torch
        from spades_ai.game.state import Bid as GoBid
        from spades_ai.game.scoring import BidType as GoBidType
        from spades_ai.game.card import Card as GoCard
        from spades_ai.game.card import Rank as GoRank, Suit as GoSuit
        
        # Convert local bid strings to Go Bid objects
        def _to_go_bid(bid_str: str) -> GoBid:
            if bid_str == "nil":
                return GoBid(value=0, bid_type=GoBidType.NIL)
            if bid_str == "blind_nil":
                return GoBid(value=0, bid_type=GoBidType.BLIND_NIL)
            if bid_str.startswith("bid_"):
                return GoBid(value=int(bid_str.split("_")[1]), bid_type=GoBidType.NORMAL)
            return GoBid(value=0, bid_type=GoBidType.NORMAL)

        go_bids = [_to_go_bid(b) for b in max_bid]

        # Encode each player's hand with appropriate prev_bids in seat order.
        # initial_hands contains local Card objects -> convert to Go Card via .card_id
        features_list = []
        for p in range(4):
            hand = [GoCard(GoRank(c.rank.value), GoSuit[c.suit.name]) for c in initial_hands[p]]
            prev = go_bids[:p]  # seat order 0→1→2→3
            position = min(p, 2)  # 0, 1, 2 for players 0,1,2; player 3 gets 2
            features = self._bid_encoder.encode(hand, prev, position)
            features_list.append(features.unsqueeze(0))

        x = torch.cat(features_list, dim=0)  # (4, 149)
        with torch.no_grad():
            logits = self._bid_model(x)  # (4, 16)

        probs = torch.softmax(logits, dim=-1)  # (4, 16)

        # Smooth: 0.75 * original + 0.25 * uniform over 14 legal bid types
        # (13 normal bid_1..bid_13 + nil; blind_nil disabled, bid_0 invalid)
        uniform = 1.0 / 14
        smoothed = 0.99 * probs + 0.01 * uniform

        product = 1.0
        for p in range(4):
            idx = self._bid_str_to_mlp_index(max_bid[p])
            product *= float(smoothed[p, idx].item())

        return product

    def _compute_importance_weight(
        self,
        initial_hands: list[list[Card]],
        play_sequence: list[tuple[int, Card]],
        step_contexts: list[dict] | None = None,
        max_bid: list[str] | None = None,
    ) -> float:
        """Replay play_sequence against initial_hands and compute p = ∏(p_i).

        The total weight is:
          p = P_bid * ∏_{step} p_step

        where P_bid = ∏_{player} P(bid_p | hand_p) from BidMLP softmax,
        and each p_step = 0.4*(1/D_i) + 0.6*oracle_match if legal, else 0.

        Input:
        - initial_hands: 4 initial hands (the proposal being evaluated).
        - play_sequence: ordered (player_id, Card) from actual game history.
        - step_contexts: optional pre-computed Go state invariants per step.
        - max_bid: actual bid strings per player (indexed by seat).

        Output:
        - probability weight p; 0 if any move was illegal given this deal.
        """
        # ── Bid probability from BidMLP ──
        if max_bid is not None:
            bid_prod = self._compute_bid_probs_product(initial_hands, max_bid)
        else:
            bid_prod = 1.0

        hands = [list(h) for h in initial_hands]  # mutable copies
        spades_broken = False
        pos_in_trick = 0
        led_suit: Suit | None = None
        weight = bid_prod

        use_oracle = (
            self._prior_oracle is not None
            and self._bridge_mod is not None
            and step_contexts is not None
        )

        if use_oracle:
            bm = self._bridge_mod
            # Maintain GO-format hands incrementally alongside local hands
            go_hands = tuple(
                frozenset(bm.to_go_card(c) for c in hands[p])
                for p in range(4)
            )

        for step_idx, (player, card) in enumerate(play_sequence):
            hand = hands[player]

            # Card must be in hand
            try:
                idx = hand.index(card)
            except ValueError:
                return 0.0

            if pos_in_trick == 0:  # Leading
                if not spades_broken and card.suit == Suit.SPADES:
                    has_non_spade = any(c.suit != Suit.SPADES for c in hand)
                    if has_non_spade:
                        return 0.0  # Can't lead spades before broken
                # Count legal actions at this step
                if not spades_broken:
                    non_spades = [c for c in hand if c.suit != Suit.SPADES]
                    legal_count = len(non_spades) if non_spades else len(hand)
                else:
                    legal_count = len(hand)
                led_suit = card.suit
            else:  # Following
                has_led = any(c.suit == led_suit for c in hand)
                if has_led and card.suit != led_suit:
                    return 0.0  # Must follow suit
                legal_count = (sum(1 for c in hand if c.suit == led_suit)
                               if has_led else len(hand))

            # --- compute step probability ---
            if use_oracle:
                ctx = step_contexts[step_idx]
                try:
                    go_state = bm.GoGameState(
                        hands=go_hands,
                        bids=ctx["bids"],
                        completed_tricks=ctx["completed_tricks"],
                        current_trick_cards=ctx["current_trick_cards"],
                        current_player=player,
                        leader=ctx["leader"],
                        trick_number=ctx["trick_number"],
                        tricks_won=ctx["tricks_won"],
                        spades_broken=spades_broken,
                        phase=bm.GoPhase.PLAYING,
                        void_shown=ctx["void_shown"],
                    )
                    oracle_card = self._prior_oracle.choose_card(go_state)
                    oracle_local = bm.to_local_card(oracle_card)
                    match = 1.0 if oracle_local.card_id == card.card_id else 0.0
                except Exception:
                    match = 0.0
                p_step = 0.4 * (1.0 / legal_count) + 0.6 * match
                # print(p_step)
            else:
                if self._oracle_requested and self._fallback_print_count < 5:
                    print("!!!!!!!!!!!!!!!! FALLBACK")
                    self._fallback_print_count += 1
                p_step = 1.0 / legal_count

            weight *= p_step

            # --- update tracking after the action ---
            hand.pop(idx)
            if use_oracle:
                go_card = bm.to_go_card(card)
                go_hands_list = list(go_hands)
                go_hands_list[player] = go_hands[player] - {go_card}
                go_hands = tuple(go_hands_list)

            if card.suit == Suit.SPADES:
                spades_broken = True

            pos_in_trick = (pos_in_trick + 1) % 4
            if pos_in_trick == 0:
                led_suit = None

        return weight

    def _build_is_pool(
        self,
        state: GameState,
        observer_id: int,
        rng: random.Random | None = None,
        num_proposals: int = 1234,
    ) -> tuple[list[list[list[Card]]], list[float]]:
        """Build importance-sampling pool once per decision.

        Generates num_proposals initial deal proposals, computes each
        proposal's probability weight — either uniform (1/D per step) or
        the oracle-blended 0.4*(1/D)+0.6*one_hot when the rule-based prior
        oracle is available — and returns (proposals, weights) for repeated
        weighted drawing.

        Input:
        - state: current game state.
        - observer_id: player whose hand is fully known.
        - rng: optional seeded random generator.
        - num_proposals: number of initial deal proposals to sample (default 1234).

        Output:
        - (proposals, weights) where proposals[i] is 4 initial hands, weights[i] is p.
          Only proposals with weight > 0 are included.
        """
        if rng is None:
            rng = random.Random()

        play_sequence = self._build_play_sequence(state)

        # Extract actual bids (per player, indexed by seat).
        # Only pass string bids — None or other types skip bid probability.
        max_bid: list[str] | None = None
        raw_bids = None
        if hasattr(state, "max_bid") and state.max_bid:
            raw_bids = state.max_bid
        elif hasattr(state, "bids") and state.bids:
            raw_bids = state.bids
        if raw_bids is not None and len(raw_bids) == 4:
            if all(isinstance(b, str) for b in raw_bids):
                max_bid = list(raw_bids)

        # Pre-compute which cards each player has played so far
        played_by_player: dict[int, list[Card]] = {p: [] for p in range(4)}
        for p, c in play_sequence:
            played_by_player[p].append(c)

        # Pre-compute Go step contexts when oracle is available
        step_contexts = None
        if self._prior_oracle is not None and self._bridge_mod is not None:
            try:
                step_contexts = self._precompute_go_step_data(play_sequence, state)
            except Exception:
                step_contexts = None

        proposals: list[list[list[Card]]] = []
        prop_weights: list[float] = []

        for _ in range(num_proposals):
            initial_hands = self._generate_proposal(
                state.all_cards, observer_id, state.hands[observer_id],
                played_by_player, rng,
            )
            w = self._compute_importance_weight(
                initial_hands, play_sequence, step_contexts,
                max_bid=max_bid,
            )
            if w > 0.0:
                proposals.append(initial_hands)
                prop_weights.append(w)

        if prop_weights:
            sorted_w = sorted(prop_weights, reverse=True)
            top3 = sorted_w[:3]
            min_w = sorted_w[-1]
        #     print(f"  [IS] {len(prop_weights)}/{num_proposals} proposals valid, "
        #           f"top3: {top3[0]:.6g}, {top3[1]:.6g}, {top3[2]:.6g}, "
        #           f"min: {min_w:.6g}")
        # else:
        #     print(f"  [IS] 0/{num_proposals} proposals valid — all weights zero")

        return proposals, prop_weights

    @staticmethod
    def _draw_is_sample(
        pool_hands: list[list[list[Card]]],
        pool_weights: list[float],
        rng: random.Random,
    ) -> list[list[Card]] | None:
        """Weighted random draw one proposal from the IS pool.

        Input:
        - pool_hands: list of proposals from _build_is_pool.
        - pool_weights: corresponding weights.
        - rng: seeded random generator.

        Output:
        - One proposal (4 initial hands), or None if pool is empty.
        """
        if not pool_hands or not pool_weights:
            return None

        total = sum(pool_weights)
        if total <= 0.0:
            return None
        #return pool_hands[2] ##############################NOTE: for testing, disable randomness and always pick the top proposal 
        r = rng.random() * total
        cumulative = 0.0
        for i, w in enumerate(pool_weights):
            cumulative += w
            if r < cumulative:
                return pool_hands[i]
        return pool_hands[-1]

    def _apply_proposal(
        self,
        state: GameState,
        observer_id: int,
        proposal: list[list[Card]],
    ) -> None:
        """Apply a specific initial-deal proposal to a state (opponents' hands in-place).

        Input:
        - state: deep-copied root state to modify
        - observer_id: player whose hand is kept unchanged
        - proposal: 4 initial 13-card hands from _build_is_pool
        """
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

    def _apply_is_determinization(
        self,
        state: GameState,
        observer_id: int,
        pool_hands: list[list[list[Card]]],
        pool_weights: list[float],
        rng: random.Random,
    ) -> None:
        """Apply one IS sample to state (modifies opponents' hands in-place).

        Falls back to uniform _determinize_state if pool is empty.
        """
        chosen = self._draw_is_sample(pool_hands, pool_weights, rng)
        if chosen is None:
            self._determinize_state(state, observer_id, rng)
            return
        self._apply_proposal(state, observer_id, chosen)

    def _determinize_state(self, state: GameState, observer_id: int, rng: random.Random | None = None) -> None:
        """Replace opponents' hands with a random deal consistent with public info.

        Modifies `state` in-place. Preserves observer's hand, played cards,
        table cards, trick_history, and hand sizes; fills other hands by
        randomly assigning from unseen cards.
        """
        if rng is None:
            rng = random.Random()

        # Collect used card ids: observer's hand + played cards + table + history
        used_ids: set[int] = set()
        for c in state.hands[observer_id]:
            used_ids.add(c.card_id)

        bitset = getattr(state, "played_bitset", 0)
        for cid in range(52):
            if bitset & (1 << cid):
                used_ids.add(cid)

        for pair in getattr(state, "table_cards", []):
            # table_cards is list[tuple[player_id, Card]]
            used_ids.add(pair[1].card_id)

        for record in getattr(state, "trick_history", []):
            for _, c in getattr(record, "cards", []):
                used_ids.add(c.card_id)

        # Pool of remaining cards
        pool = [c for c in STANDARD_52 if c.card_id not in used_ids]
        rng.shuffle(pool)

        # Assign to opponents preserving hand sizes
        indices = [pid for pid in range(state.num_players) if pid != observer_id]
        counts = {pid: len(state.hands[pid]) for pid in indices}

        pos = 0
        for pid in indices:
            n = counts[pid]
            assigned = pool[pos: pos + n]
            pos += n
            state.hands[pid] = list(assigned)

        # Recompute hand_bitsets if present
        if hasattr(state, "hand_bitsets"):
            for pid in range(state.num_players):
                bit = 0
                for c in state.hands[pid]:
                    bit |= (1 << c.card_id)
                state.hand_bitsets[pid] = bit

    def _solve_with_determinization(self, state: GameState) -> dict[str, Any]:
        """Approximate solve_with_q by averaging results across determinized samples."""
        t0 = time.time()
        agg_q: dict[int, float] = {}
        agg_value = 0.0
        counts = 0

        rng = random.Random()
        id_to_card = {c.card_id: c for c in STANDARD_52}

        drawn_distinct: set[tuple] = set()
        # Build IS pool once, then draw determinization_count samples from it
        pool_hands, pool_weights = self._build_is_pool(state, state.turn, rng)
        t1 = time.time()
        #print(f"  [TIMING] IS pool built: {len(pool_hands)} valid proposals in {t1-t0:.2f}s")

        for _ in range(self.config.determinization_count):
            sim_state = copy.deepcopy(state)
            observer = state.turn
            self._apply_is_determinization(sim_state, observer, pool_hands, pool_weights, rng)
            # Track distinctness: encode opponent remaining hands
            drawn_key = tuple(
                tuple(sorted(c.card_id for c in sim_state.hands[p]))
                for p in range(4) if p != observer
            )
            drawn_distinct.add(drawn_key)
            t2 = time.time()
            res = self.exact_solver.solve_with_q(sim_state)
            t3 = time.time()
            #print(f"  [TIMING]   determinization {_}: solve_with_q in {t3-t2:.2f}s")
            counts += 1
            agg_value += float(res.get("value", 0.0))
            for action, q in res.get("action_q_values", {}).items():
                aid = action.card_id
                agg_q[aid] = agg_q.get(aid, 0.0) + float(q)

        print(f"  [DEBUG sample exact] {self.config.determinization_count} draws, {len(drawn_distinct)} distinct opponent-hand configs from pool of {len(pool_hands)}", flush=True)

        # Average Qs
        for k in list(agg_q.keys()):
            agg_q[k] = agg_q[k] / max(1, counts)

        # Reconstruct action -> q mapping using Card objects
        action_q_values: dict[Card, float] = {}
        for aid, q in agg_q.items():
            if aid in id_to_card:
                action_q_values[id_to_card[aid]] = q

        avg_value = agg_value / max(1, counts)

        # Choose best action by averaged Q for root team
        root_team = state.teams[state.turn]
        if action_q_values:
            if root_team == 0:
                best_action = max(action_q_values.items(), key=lambda it: it[1])[0]
            else:
                best_action = min(action_q_values.items(), key=lambda it: it[1])[0]
        else:
            best_action = None

        return {"value": avg_value, "best_action": best_action, "action_q_values": action_q_values}

    def _select_child_puct(self, node: SearchNode) -> SearchNode | None:
        """按 PUCT 选择最值得继续探索的子节点。"""
        if not node.children:
            return None

        current_team = self._current_team(node.state)
        sign = 1.0 if current_team == 0 else -1.0
        sqrt_total = math.sqrt(max(1, node.visits))

        best_child = None
        best_score = -float("inf")

        for action, child in node.children.items():
            q_value = child.average_value
            prior = child.prior
            exploration = self.config.exploration_constant * prior * sqrt_total / (1.0 + child.visits)
            score = sign * q_value + exploration
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _policy_priors(self, state: GameState, legal_actions: list[Card]) -> dict[int, float]:
        """返回合法动作的均匀先验概率。

        输入:
        - state: 当前状态（仅用于缓存键，不触发模型前向）
        - legal_actions: 当前合法动作列表

        输出:
        - dict[int, float]: `card_id -> prior_prob`，所有合法动作等概率

        说明:
        - 为避免在全局阶段误用仅针对残局训练的 policy head，这里固定
          使用均匀先验，不调用模型。
        """
        if not legal_actions:
            return {}

        if self._decision_policy_priors_cache is not None:
            state_id = id(state)
            cached = self._decision_policy_priors_cache.get(state_id)
            if cached is not None:
                return dict(cached)

        prob = 1.0 / len(legal_actions)
        priors: dict[int, float] = {action.card_id: prob for action in legal_actions}
        if self._decision_policy_priors_cache is not None:
            self._decision_policy_priors_cache[id(state)] = dict(priors)
        else:
            cache_key = self._state_cache_key(state)
            self._policy_priors_cache[cache_key] = dict(priors)
        return priors

    def _legal_actions(self, state: GameState) -> list[Card]:
        """Cached wrapper around `rules.playable` to avoid repeated allocation."""
        if self._decision_legal_actions_cache is not None:
            state_id = id(state)
            cached = self._decision_legal_actions_cache.get(state_id)
            if cached is not None:
                return list(cached)

        hand = state.hands[state.turn]
        legal_actions = self.rules.playable(state, hand, state.turn)
        legal_actions = sorted(legal_actions, key=lambda card: card.card_id)
        if self._decision_legal_actions_cache is not None:
            # store canonical list to avoid repeated work; callers get a copy
            self._decision_legal_actions_cache[id(state)] = list(legal_actions)
        return list(legal_actions)

    def _leaf_value(self, state: GameState) -> float:
        """在 leaf_threshold 处使用 MLP 估值，并换算到队伍 0 视角。"""
        if self._decision_leaf_value_cache is not None:
            state_id = id(state)
            cached = self._decision_leaf_value_cache.get(state_id)
            if cached is not None:
                return cached

        cache_key = self._state_cache_key(state)
        cached = self._leaf_value_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._is_terminal(state):
            value = self._terminal_value(state)
            self._leaf_value_cache[cache_key] = value
            if self._decision_leaf_value_cache is not None:
                self._decision_leaf_value_cache[id(state)] = value
            return value

        if self.model is None:
            # checkpoint_path is None: directly use the exact solver for Q value.
            # The exact solver operates on the full-information state (including
            # determinized opponent hands if called from within a simulation).
            self._exact_calls += 1
            result = self.exact_solver.solve_with_q(state)
            value = float(result["value"])
            self._leaf_value_cache[cache_key] = value
            if self._decision_leaf_value_cache is not None:
                self._decision_leaf_value_cache[id(state)] = value
            return value

        feature = self.encoder.encode(state, state.turn)
        self._model_calls += 1
        pred_value_view_scaled = float(self.model.predict(feature))
        pred_value_view = pred_value_view_scaled * self.config.value_scale
        value = pred_value_view if self._current_team(state) == 0 else -pred_value_view
        self._leaf_value_cache[cache_key] = value
        if self._decision_leaf_value_cache is not None:
            self._decision_leaf_value_cache[id(state)] = value
        return value

    def get_diagnostics(self) -> dict[str, int]:
        """Return simple diagnostics counters for this strategy instance.

        Output:
        - dict with keys: model_calls, policy_model_calls, exact_calls
        """
        return {
            "model_calls": int(self._model_calls),
            "policy_model_calls": int(self._policy_model_calls),
            "exact_calls": int(self._exact_calls),
        }

    def _terminal_value(self, state: GameState) -> float:
        """终局价值（队伍 0 视角）。"""
        return float(self.rules.score(state)[0])

    def _is_terminal(self, state: GameState) -> bool:
        """判断是否已经进入终局。"""
        return self.rules.end_trickgame(state)

    def _current_team(self, state: GameState) -> int:
        """返回当前行动方所属队伍。"""
        return state.teams[state.turn]

    def _remaining_cards(self, state: GameState) -> int:
        """返回当前局面的剩余牌数。"""
        return sum(len(hand) for hand in state.hands)

    def _legal_actions(self, state: GameState) -> list[Card]:
        """返回当前状态下的合法动作，并按 card_id 排序。"""
        hand = state.hands[state.turn]
        legal_actions = self.rules.playable(state, hand, state.turn)
        return sorted(legal_actions, key=lambda card: card.card_id)

    def _apply_action(self, state: GameState, action: Card) -> GameState:
        """返回应用动作后的新状态。"""
        return self.exact_solver._apply_action(state, action, state.turn)

    def _apply_action_in_place(self, state: GameState, action: Card) -> None:
        """在原状态上执行出牌推进。"""
        player_id = state.turn
        state.play_card_to_table(player_id, action)
        if action.suit == Suit.SPADES:
            state.spades_broken = True
            state.trump_broken = True
        state.turn = (player_id + 1) % state.num_players
        if state.trick_complete:
            winner = self.rules.winner_trick(state)
            state.complete_trick(winner)
            state.trick_leader = winner
            state.turn = winner


def _build_demo_state(seed: int) -> GameState:
    """构造一个从 52 张牌开始的可出牌测试局面。

    输入:
    - seed: int，发牌和叫牌随机种子

    输出:
    - GameState，已经完成叫牌并进入出牌阶段，52 张牌都还在各自手中
    """
    from data.training_data import build_state_with_remaining_cards

    return build_state_with_remaining_cards(target_remaining=52, seed=seed)


def main() -> None:
    """命令行入口：从 52 张牌开始跑完整出牌序列。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False, default=None, help="MLP 权重路径")
    parser.add_argument("--seed", type=int, default=0, help="牌局随机种子")
    parser.add_argument("--exact_threshold", type=int, default=30, help="剩余牌数 <= 该值时直接精确求解")
    parser.add_argument("--leaf_threshold", type=int, default=24, help="MCTS 搜索到该剩余牌数时接入 MLP")
    parser.add_argument("--simulations_per_action", type=int, default=5, help="每个根动作模拟次数")
    parser.add_argument("--exploration_constant", type=float, default=1.5, help="PUCT 探索系数")
    parser.add_argument("--policy_temperature", type=float, default=1.0, help="policy 先验温度")
    args = parser.parse_args()

    config = TruncatedMCTSConfig(
        exact_threshold=args.exact_threshold,
        leaf_threshold=args.leaf_threshold,
        simulations_per_action=args.simulations_per_action,
        exploration_constant=args.exploration_constant,
        policy_temperature=args.policy_temperature,
        checkpoint_path=args.checkpoint,
    )

    strategy = TruncatedMCTSStrategy(config)
    state = _build_demo_state(args.seed)

    print("初始局面：")
    print(f"  当前玩家: {state.turn}")
    print(f"  剩余牌数: {sum(len(hand) for hand in state.hands)}")
    print(f"  策略配置: exact_threshold={config.exact_threshold}, leaf_threshold={config.leaf_threshold}, simulations_per_action={config.simulations_per_action}")
    print()

    action_sequence = strategy.play_full_game(state)

    print("完整出牌序列：")
    print(" -> ".join(str(card) for card in action_sequence))
    print()
    print("终局统计：")
    print(f"  总动作数: {len(action_sequence)}")
    print(f"  最终得分: {strategy.rules.score(state)}")


if __name__ == "__main__":
    main()
