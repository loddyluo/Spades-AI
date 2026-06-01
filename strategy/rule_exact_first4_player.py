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
- 规则式玩家自身只读自己的手牌+桌面+历史(盲眼); exact_solver 会读 state.hands(全开),
  这与 RLExactPlayer 一样 —— "后 36 张作弊"是评估用的标准设定。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from trick_taking.card import Card
from trick_taking.game_state import GameState
from trick_taking.player import AIPlayer
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)
from strategy.rule_based_first4_player import RuleBasedFirst4Player


class RuleExactFirst4Player(AIPlayer):
    """前 4 墩 rule-based + 后 36 张 exact solver 的混合玩家。

    与 `rl.rl_exact_player.RLExactPlayer` 一一对应, 仅前 4 墩的决策来源不同。
    """

    def __init__(
        self,
        exact_solver: ExactDoubleDummyCppFastestSolver | None = None,
        exact_threshold: int = 36,
        bid_model=None,
        bid_device: str = "cpu",
    ) -> None:
        # 内部规则式玩家. 不让它处理后 36 张 (我们自己路由)。
        self._rule_player = RuleBasedFirst4Player()
        self.exact_threshold = exact_threshold
        self._bid_model = bid_model
        self._bid_device = bid_device

        self.position: int = -1
        self.hand: list[Card] = []
        self.last_play_info: dict[str, Any] = {}
        self.last_bid_info: dict[str, Any] | None = None

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

    def card_played(self, player_id: int, card: Card) -> None:
        # 关键: 规则式玩家依赖这个回调跟踪历史, 必须转发
        self._rule_player.card_played(player_id, card)

    # ─── 出牌路由 ──────────────────────────────────────────────────

    def play_card(self, legal_cards: list[Card], state_view: dict) -> Card:
        state: GameState | None = state_view.get("state")
        if state is None:
            self.last_play_info = {"mode": "no_state_fallback"}
            return legal_cards[0]

        # 与 RLExactPlayer 完全相同的切换条件
        remaining = sum(len(h) for h in state.hands)

        if remaining <= self.exact_threshold:
            return self._exact_play(state, legal_cards)
        return self._rule_play(legal_cards, state_view)

    def _rule_play(self, legal_cards: list[Card], state_view: dict) -> Card:
        card = self._rule_player.play_card(legal_cards, state_view)
        self.last_play_info = {"mode": "rule_first4"}
        return card

    def _exact_play(self, state: GameState, legal_cards: list[Card]) -> Card:
        """与 RLExactPlayer._exact_play 完全相同的实现。"""
        if self.exact_solver is None:
            self.last_play_info = {"mode": "no_exact_solver_fallback"}
            return legal_cards[0]

        result = self.exact_solver.solve_with_q(state)
        best_action = result.get("best_action")
        if best_action is not None and best_action in legal_cards:
            self.last_play_info = {"mode": "exact"}
            return best_action

        self.last_play_info = {"mode": "exact_no_match_fallback"}
        return legal_cards[0]
