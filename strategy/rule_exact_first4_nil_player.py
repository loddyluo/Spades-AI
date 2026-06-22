"""RuleExactFirst4NilPlayer —— 继承 RuleExactFirst4Player 的 nil 规则前 4 墩玩家。

设计意图:
- **后 36 张完全继承**:`_exact_play` + IS pool + budget table + determinization + 加权 Q
  都通过继承零代码复用(协作者新版 strategy/rule_exact_first4_player.py 提供)。
- **唯一行为差异**:前 4 墩。父类在 nil 局可选走 RL net (55_2nil.pt) 或规则非 nil;
  本子类在 nil 局走**我们的规则 nil 策略**(RuleBasedFirst4NilPlayer),非 nil 局
  继承父类行为(走 RuleBasedFirst4Player)。
- 父类的 `policy_net_nil` / `encoder` / `_rl_nil_player` 路径**不启用**(我们不传
  policy_net_nil,使 `_rl_nil_player` 恒为 None)。
"""

from __future__ import annotations

from typing import Any

from trick_taking.card import Card
from trick_taking.game_state import GameState
from strategy.rule_based_first4_nil_player import RuleBasedFirst4NilPlayer
from strategy.rule_exact_first4_player import RuleExactFirst4Player


class RuleExactFirst4NilPlayer(RuleExactFirst4Player):
    """前 4 墩 nil 规则 + 后 36 张父类 IS pool 的混合玩家。

    继承结构:
      - `__init__` / `start_game` / `place_bid` / `bid_placed` / `card_played` 全继承,
        额外创建并维护 `_nil_rule_player`(我们的规则 nil 策略)。
      - `set_teams` 重写:既调父类(让 `_has_nil_bid` 等状态正常),也转发我们的 nil 规则
        玩家(让它做角色分派)。
      - `play_card` 重写:remaining > threshold(前 4 墩)且 has_nil_bid → 用规则 nil;
        无 nil 时继承父类行为(走规则非 nil);remaining ≤ threshold 时调 `_exact_play`。
    """

    def __init__(
        self,
        exact_solver: Any | None = None,
        exact_threshold: int = 36,
        bid_model=None,
        bid_device: str = "cpu",
        hyperparam_config: Any | None = None,
    ) -> None:
        # 不传 policy_net_nil / encoder:父类的 RL nil fallback 路径被禁用
        super().__init__(
            exact_solver=exact_solver,
            exact_threshold=exact_threshold,
            bid_model=bid_model,
            bid_device=bid_device,
            policy_net_nil=None,
            encoder=None,
            hyperparam_config=hyperparam_config,
        )
        # 我们的规则 nil 策略(独立于父类的 `_rule_player`,后者是非 nil 版)
        self._nil_rule_player = RuleBasedFirst4NilPlayer()

    # ─── 回调:在父类基础上额外转发到 _nil_rule_player ─────────────────────

    def start_game(self, position: int, hand: list[Card], num_players: int) -> None:
        super().start_game(position, hand, num_players)
        self._nil_rule_player.start_game(position, hand, num_players)

    def set_teams(self, teams: list[int], bid_values: list[Any]) -> None:
        # 父类:计算 _has_nil_bid(可能启动 _rl_nil_player,但我们没传 policy_net_nil → 不启)
        super().set_teams(teams, bid_values)
        # 我们的 nil 规则玩家:做角色分派 + 建 trackers
        try:
            self._nil_rule_player.set_teams(teams, bid_values)
        except Exception:
            pass

    def card_played(self, player_id: int, card: Card) -> None:
        super().card_played(player_id, card)
        self._nil_rule_player.card_played(player_id, card)

    # ─── 出牌路由(唯一行为差异) ────────────────────────────────────

    def play_card(self, legal_cards: list[Card], state_view: dict) -> Card:
        state: GameState | None = state_view.get("state")
        if state is None:
            self.last_play_info = {"mode": "no_state_fallback"}
            return legal_cards[0]

        # 后 36 张:完全继承父类的 IS pool _exact_play
        remaining = sum(len(h) for h in state.hands)
        if remaining <= self.exact_threshold:
            return self._exact_play(state, legal_cards)

        # 前 4 墩:nil 局走我们的规则 nil;非 nil 局退回父类行为(走规则非 nil)
        if self._has_nil_bid:
            card = self._nil_rule_player.play_card(legal_cards, state_view)
            self.last_play_info = {"mode": "rule_nil_first4"}
            return card
        # 非 nil 局:复用父类的非 nil 规则路径(self._rule_player = RuleBasedFirst4Player)
        return self._rule_play(legal_cards, state_view)
