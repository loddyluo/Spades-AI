"""RuleBasedFirst4NilPlayer — 含 Nil 的前 4 墩规则式策略

实现 strategy/nil_first4_strategy_design.md 描述的策略。完全独立于
strategy/rule_based_first4_player.py(后者只处理非 nil 情形),
但 import 复用其顶层工具函数 (_by_suit / _trick_current_winner / _NON_SPADE_SUITS)。

角色分派 (§0.3,每墩判定):
  1. 我自己叫 nil           → Nil 本人策略 (§3)
  2. 我队友 (pos+2)%4 叫 nil → Nil 队友策略 (§4)
  3. 某个对手叫 nil          → Nil 对手策略 (§5)
  4. 都不是                  → 非 nil 兜底 (复用 RuleBasedFirst4Player)

作用范围:
  - 前 4 墩 (tricks_played < 4) 由本类处理;
  - 第 5 墩起调用 fallback_player.play_card (类似 RuleBasedFirst4Player 的设计)。

核心原理 (P1-P10, 详见设计文档 §2):
  P1 Nil 输是吸收态  P2 三层决策 (硬护栏⊃铁律⊃算账)  P3 相对大小 lo
  P4 资源=深度  P5 门槛由我顶  P6 动作多义性  P7 座次几何
  P8 空门双向资源  P9 将吃=目标耦合  P10 前 4 墩 = handoff
"""

from __future__ import annotations

from typing import Any, Optional

from trick_taking.card import Card, Rank, Suit
from trick_taking.player import AIPlayer
from strategy.rule_based_first4_player import (
    RuleBasedFirst4Player,
    _NON_SPADE_SUITS,
    _by_suit,
    _trick_current_winner,
)


# ─── 阈值常量 (设计文档附录 A,可后续挂 hyperparam_config) ──────────────────────


class _Thresholds:
    """所有 lo 阈值 / 分档参数 (附录 A.1 / A.2)。"""
    # §3.1 nil 本人安全牌
    SAFE_LOW_MAX: int = 3       # safe_low: lo ≤ 3
    SAFE_SECOND_MAX: int = 6    # safe_second: lo ≤ 6

    # §4.1 nil 队友"不小的牌"
    COVER_CARD_LO_MIN: int = 6  # 能盖低位领出: lo ≥ 6

    # §5.4 nil 对手 ATTACK 公式
    DEPTH_LOW_MAX: int = 2      # 攻方小牌弹药: lo ≤ 2
    MY_BURDEN_MIN: int = 7      # 自救负担: lo ≥ 7 (非 boss)
    ATTACK_W_DEPTH_LOW: float = 2.0
    ATTACK_W_NIL_LEN: float = 1.0
    ATTACK_W_BURDEN: float = -0.5

    # §5.5 多义试探"中 lo"
    MID_LO_MIN: int = 5
    MID_LO_MAX: int = 8

    # §5.1 bid_sum 重心
    BID_SUM_LOW_MAX: int = 10    # ≤ 偏 A
    BID_SUM_HIGH_MIN: int = 12   # ≥ 偏 C

    # §4.5 队友 "偏高/偏低" 分界 (领出牌 rank ≥ 此值算偏高)
    HIGH_LEAD_RANK: int = Rank.JACK.value

    # §3.1 将吃折扣 outstanding 分档
    COVER_OUTSTANDING_HARD: int = 1   # ≤ 1 降 2 档
    COVER_OUTSTANDING_SOFT: int = 2   # ≤ 2 降 1 档

    # §5.6 B 动态 small_threshold
    SMALL_THRESHOLD_DIVISOR: int = 3   # small_threshold = unseen // 3

    # §5.6 B small_depth 二档分界
    SMALL_DEPTH_GOOD: int = 2          # ≥ 2 不盖 / ≤ 1 盖

    # §3.1 escapes 二档分界
    ESCAPES_GOOD: int = 2              # ≥ 2 才算真有逃生力


# ─── 度量原子量 ────────────────────────────────────────────────────────────────


def _lo(card: Card, unseen_in_suit: list[Card]) -> int:
    """C 能压过的 unseen 张数 = unseen 里 rank < C 的张数 (P3 核心刻度)。"""
    return sum(1 for u in unseen_in_suit if u.rank.value < card.rank.value)


def _hi(card: Card, unseen_in_suit: list[Card]) -> int:
    """unseen 里比 C 大的张数。hi=0 ⇒ boss。"""
    return sum(1 for u in unseen_in_suit if u.rank.value > card.rank.value)


def _is_boss(card: Card, unseen_in_suit: list[Card]) -> bool:
    return _hi(card, unseen_in_suit) == 0


def _suit_unseen(suit: Suit, my_hand: list[Card], seen: set[int]) -> list[Card]:
    """该门 unseen = 全部 13 张 − 我手里这门 − 已出现过的(seen 含 rank.value)。"""
    mine_ranks = {c.rank.value for c in my_hand if c.suit == suit}
    return [
        Card(suit, Rank(r))
        for r in range(2, 15)
        if r not in mine_ranks and r not in seen
    ]


# ─── nil 信号 / 推断量 ────────────────────────────────────────────────────────


class _NilTracker:
    """跟踪 nil 玩家的可观测信号(整局累积,在 card_played 中更新)。

    属性 (针对单个目标 nil 座位):
      - nil_pid:               目标 nil 座位号
      - voids:                 set[Suit] = nil 已确定空门的花色
      - led_suits:             set[Suit] = nil 作为领出者开过的花色
      - seen_by_suit:           dict[Suit, set[int]] = 该门已出现过的 rank
                               (跨所有玩家,用于算 unseen / outstanding / L)
    """

    def __init__(self, nil_pid: int) -> None:
        self.nil_pid = nil_pid
        self.voids: set[Suit] = set()
        self.led_suits: set[Suit] = set()
        self.seen_by_suit: dict[Suit, set[int]] = {s: set() for s in Suit}
        # 本墩起始时,记录领出者+领出花色,用于探测 nil 是否跟出
        self._trick_lead_suit: Optional[Suit] = None
        self._trick_lead_pid: Optional[int] = None
        # 本墩 nil 是否已出过牌 (含本墩从 trick 开始累积)
        self._nil_played_this_trick: bool = False
        self._trick_card_count: int = 0

    def on_card_played(self, player_id: int, card: Card) -> None:
        # 跟踪本墩第一张 = 领出
        if self._trick_card_count == 0:
            self._trick_lead_suit = card.suit
            self._trick_lead_pid = player_id
            if player_id == self.nil_pid:
                self.led_suits.add(card.suit)
        # nil 跟出非领花 → 它该门空门
        else:
            if (
                player_id == self.nil_pid
                and self._trick_lead_suit is not None
                and card.suit != self._trick_lead_suit
            ):
                self.voids.add(self._trick_lead_suit)
        # 累积 seen
        self.seen_by_suit[card.suit].add(card.rank.value)
        if player_id == self.nil_pid:
            self._nil_played_this_trick = True
        # 本墩第 4 张后重置
        self._trick_card_count += 1
        if self._trick_card_count >= 4:
            self._trick_card_count = 0
            self._trick_lead_suit = None
            self._trick_lead_pid = None
            self._nil_played_this_trick = False

    def reset_for_new_trick_if_needed(self, table_cards: list[tuple[int, Card]]) -> None:
        """driver 可能在 trick 之间清空 table_cards;同步重置本墩状态。"""
        if not table_cards:
            self._trick_card_count = 0
            self._trick_lead_suit = None
            self._trick_lead_pid = None
            self._nil_played_this_trick = False

    def outstanding(self, suit: Suit, my_hand: list[Card]) -> int:
        """该门还在别人手里的张数 = 13 − 我张数 − seen 数。"""
        mine = sum(1 for c in my_hand if c.suit == suit)
        return max(0, 13 - mine - len(self.seen_by_suit[suit]))

    def nil_inferred_lowest(self, suit: Suit, my_hand: list[Card]) -> Optional[int]:
        """L = 该门 unseen 里的最小 rank。空门或全部见光 → None。"""
        if suit in self.voids:
            return None
        unseen = _suit_unseen(suit, my_hand, self.seen_by_suit[suit])
        if not unseen:
            return None
        return min(u.rank.value for u in unseen)

    def nil_estimated_length(self, suit: Suit, my_hand: list[Card]) -> float:
        if suit in self.voids:
            return 0.0
        return self.outstanding(suit, my_hand) / 3.0


# ─── Player ────────────────────────────────────────────────────────────────────


class _DefaultFallback:
    """5 墩起的占位 fallback(只在外部没注入真 fallback 时使用)。"""

    def play_card(self, legal_cards: list[Card], state_view: dict) -> Card:
        return min(legal_cards, key=lambda c: (c.suit.value, c.rank.value))


class RuleBasedFirst4NilPlayer(AIPlayer):
    """含 nil 的前 4 墩规则式策略。

    与 AIPlayer 接口完全兼容;可被 RuleExactFirst4NilPlayer 包装,后 36 张交
    精确求解器。如果该局没人叫 nil,内部 fall through 到 RuleBasedFirst4Player
    的非 nil 策略 (§0.3 优先级 4),保证行为退化等于纯非 nil 玩家。
    """

    TRUMP_SUIT: Suit = Suit.SPADES

    def __init__(
        self,
        fallback_player: Optional[Any] = None,
        bid_strategy: str = "random",
        bid_seed: int | None = None,
    ) -> None:
        self.fallback_player = fallback_player or _DefaultFallback()
        self._non_nil_player = RuleBasedFirst4Player(
            fallback_player=self.fallback_player,
            bid_strategy=bid_strategy,
            bid_seed=bid_seed,
        )

        # 每局重置
        self.position: int = -1
        self.hand: list[Card] = []

        # bids / 角色
        self._bid_values: list[Any] = []
        self._role: str = "non_nil"   # nil_self / nil_teammate / nil_opponent / non_nil
        self._my_nil_pid: int = -1    # 我自己/队友 nil 时是它的 pid;对手 nil 时是目标 nil pid
        self._nil_trackers: dict[int, _NilTracker] = {}  # 所有叫 nil 的座位(支持双 nil)
        self._target_nil_pid: int = -1  # 当前回合的"目标 nil"(对手策略主要 vs 谁)
        self._bid_sum: int = 0

        # 用于 partner 策略的 acquire 跟踪 (§4.2)
        self._my_numeric_bid: int = 0
        # tricks_won 由 state_view 在 play_card 时读取,不需自己维护

    # ─── 接口 ────────────────────────────────────────────────────────────

    def start_game(self, position: int, hand: list[Card], num_players: int) -> None:
        self.position = position
        self.hand = list(hand)
        self._bid_values = []
        self._role = "non_nil"
        self._my_nil_pid = -1
        self._nil_trackers = {}
        self._target_nil_pid = -1
        self._bid_sum = 0
        self._my_numeric_bid = 0
        self._non_nil_player.start_game(position, hand, num_players)

    def place_bid(self, legal_bids: list[Any], state_view: dict) -> Any:
        # 不关心叫牌,交给非 nil 玩家的随机策略(评测脚本会用 MLP 包装层覆盖)
        return self._non_nil_player.place_bid(legal_bids, state_view)

    def bid_placed(self, bidder: int, bid: Any) -> None:
        try:
            self._non_nil_player.bid_placed(bidder, bid)
        except Exception:
            pass

    def set_teams(self, teams: list[int], bid_values: list[Any]) -> None:
        try:
            self._non_nil_player.set_teams(teams, bid_values)
        except Exception:
            pass
        self._bid_values = list(bid_values)
        self._compute_role_and_bidsum()

    def card_played(self, player_id: int, card: Card) -> None:
        self._non_nil_player.card_played(player_id, card)
        for tracker in self._nil_trackers.values():
            tracker.on_card_played(player_id, card)
        # 出牌也要同步移走自己的牌
        if player_id == self.position:
            try:
                self.hand.remove(card)
            except ValueError:
                pass

    def play_card(self, legal_cards: list[Card], state_view: dict) -> Card:
        tricks_played = state_view.get("tricks_played", 0)
        if tricks_played >= 4:
            return self.fallback_player.play_card(legal_cards, state_view)

        # 同步 trackers 本墩状态(若 driver 清了 table_cards)
        table_cards = list(state_view.get("table_cards", []))
        for tracker in self._nil_trackers.values():
            tracker.reset_for_new_trick_if_needed(table_cards)

        # 角色分派
        if self._role == "nil_self":
            return self._play_nil_self(legal_cards, state_view)
        if self._role == "nil_teammate":
            return self._play_nil_teammate(legal_cards, state_view)
        if self._role == "nil_opponent":
            return self._play_nil_opponent(legal_cards, state_view)
        # non_nil 兜底
        return self._non_nil_player.play_card(legal_cards, state_view)

    # ─── 角色判定 (§0.3) ─────────────────────────────────────────────────

    def _compute_role_and_bidsum(self) -> None:
        nil_class = {"nil", "blind_nil"}
        partner_pid = (self.position + 2) % 4
        opp_pids = [(self.position + 1) % 4, (self.position + 3) % 4]

        # bid_sum: 四家叫牌数字之和(nil 计 0)
        total = 0
        for bv in self._bid_values:
            if isinstance(bv, str) and bv.startswith("bid_"):
                try:
                    total += int(bv.split("_")[1])
                except ValueError:
                    pass
        self._bid_sum = total

        # 收集所有 nil 座位
        nil_pids: list[int] = []
        for pid, bv in enumerate(self._bid_values):
            if isinstance(bv, str) and bv in nil_class:
                nil_pids.append(pid)
                if pid not in self._nil_trackers:
                    self._nil_trackers[pid] = _NilTracker(pid)

        # 我自己的数字叫牌
        my_bid = self._bid_values[self.position] if self.position < len(self._bid_values) else None
        if isinstance(my_bid, str) and my_bid.startswith("bid_"):
            try:
                self._my_numeric_bid = int(my_bid.split("_")[1])
            except ValueError:
                self._my_numeric_bid = 0

        # 优先级分派
        if self.position in nil_pids:
            self._role = "nil_self"
            self._my_nil_pid = self.position
        elif partner_pid in nil_pids:
            self._role = "nil_teammate"
            self._my_nil_pid = partner_pid
            self._target_nil_pid = partner_pid
        elif any(p in nil_pids for p in opp_pids):
            self._role = "nil_opponent"
            # 默认目标 = 第一个对手 nil(双对手 nil 时 play_card 内部会按本墩处境挑)
            self._target_nil_pid = next(p for p in opp_pids if p in nil_pids)
            self._my_nil_pid = self._target_nil_pid
        else:
            self._role = "non_nil"

    # ─── 队伍 / 座次工具 ─────────────────────────────────────────────────

    def _is_us(self, pid: int) -> bool:
        return (pid - self.position) % 2 == 0

    def _is_opponent(self, pid: int) -> bool:
        return not self._is_us(pid)

    # =================================================================
    # §3 Nil 本人策略
    # =================================================================

    def _play_nil_self(self, legal_cards: list[Card], state_view: dict) -> Card:
        """Nil 本人 — 前 4 墩取 0 墩。"""
        table_cards = list(state_view.get("table_cards", []))
        spades_broken = bool(state_view.get("trump_broken", state_view.get("spades_broken", False)))

        # 死 nil 回退 (§6.2):我自己已赢过一墩 → 切非 nil
        tricks_won = state_view.get("tricks_won", [0, 0, 0, 0])
        if self.position < len(tricks_won) and tricks_won[self.position] > 0:
            return self._non_nil_player.play_card(legal_cards, state_view)

        seen = self._all_seen()
        # 没领花
        if not table_cards:
            return self._nil_self_lead(legal_cards, spades_broken, seen)

        # 跟牌
        lead_suit = table_cards[0][1].suit
        legal_by_suit = _by_suit(legal_cards)
        same_suit = legal_by_suit[lead_suit]
        n_on_table = len(table_cards)

        if same_suit:
            return self._nil_self_follow_in_suit(same_suit, table_cards, n_on_table, seen)
        return self._nil_self_void(legal_cards, legal_by_suit, table_cards, seen)

    def _nil_self_lead(self, legal_cards: list[Card], spades_broken: bool, seen: dict[Suit, set[int]]) -> Card:
        """§3.3 首攻 4 桶。"""
        legal_by_suit = _by_suit(legal_cards)
        my_by_suit = _by_suit(self.hand)
        # 合法可领花色
        legal_suits = [s for s in Suit if legal_by_suit[s]]
        if not spades_broken and any(s != Suit.SPADES for s in legal_suits):
            legal_suits = [s for s in legal_suits if s != Suit.SPADES]

        def safe_low(card: Card, unseen: list[Card]) -> bool:
            return _lo(card, unseen) <= _Thresholds.SAFE_LOW_MAX and not _is_boss(card, unseen)

        def safe_second(card: Card, unseen: list[Card]) -> bool:
            return _lo(card, unseen) <= _Thresholds.SAFE_SECOND_MAX and not _is_boss(card, unseen)

        # 桶 1: 安全单张
        bucket1: list[tuple[int, Card]] = []  # (lo, card)
        for s in legal_suits:
            if len(my_by_suit[s]) == 1:
                c = my_by_suit[s][0]
                unseen = _suit_unseen(s, self.hand, seen[s])
                if safe_low(c, unseen):
                    bucket1.append((_lo(c, unseen), c))
        if bucket1:
            return min(bucket1, key=lambda x: x[0])[1]

        # 桶 2: 安全双张(高张满足 safe_low)
        bucket2: list[tuple[int, Card]] = []
        for s in legal_suits:
            cards = my_by_suit[s]
            if len(cards) == 2:
                high = cards[1]  # 升序 → [1] 是高张
                unseen = _suit_unseen(s, self.hand, seen[s])
                if safe_low(high, unseen):
                    bucket2.append((_lo(high, unseen), high))
        if bucket2:
            return min(bucket2, key=lambda x: x[0])[1]

        # 桶 3: 最不危险长门 ≥3 张,领第二小(需 safe_second)
        danger_ranked: list[tuple[float, Suit, Card]] = []  # (danger_score, suit, second_smallest)
        for s in legal_suits:
            cards = my_by_suit[s]
            if len(cards) < 3:
                continue
            unseen = _suit_unseen(s, self.hand, seen[s])
            second = cards[1]
            if not safe_second(second, unseen):
                continue
            # 危险度:取该门最大 lo + 长度惩罚 + 黑桃加重(spades-last)
            max_lo = max((_lo(c, unseen) for c in cards), default=0)
            length_bonus = len(cards) * 0.1
            spade_penalty = 100.0 if s == Suit.SPADES else 0.0
            outstanding = max(0, 13 - len(cards) - len(seen[s]))
            cover_discount = self._cover_discount_factor(outstanding, s)
            # 黑桃不享将吃折扣
            danger = (max_lo + length_bonus + spade_penalty) * cover_discount
            danger_ranked.append((danger, s, second))
        if danger_ranked:
            danger_ranked.sort(key=lambda x: x[0])
            return danger_ranked[0][2]

        # 桶 4: 兜底 — 去 boss 牌出 lo 最小;全 boss 出全局 lo 最小
        all_with_unseen: list[tuple[Card, list[Card]]] = []
        for s in legal_suits:
            unseen = _suit_unseen(s, self.hand, seen[s])
            for c in legal_by_suit[s]:
                all_with_unseen.append((c, unseen))
        non_boss = [(c, u) for c, u in all_with_unseen if not _is_boss(c, u)]
        pool = non_boss if non_boss else all_with_unseen
        return min(pool, key=lambda x: _lo(x[0], x[1]))[0]

    @staticmethod
    def _cover_discount_factor(outstanding: int, suit: Suit) -> float:
        """§3.1 将吃折扣 → 转成连续乘数 (用于桶 3 比较)。黑桃不享。"""
        if suit == Suit.SPADES:
            return 1.0
        if outstanding <= _Thresholds.COVER_OUTSTANDING_HARD:
            return 0.4  # 降 2 档
        if outstanding <= _Thresholds.COVER_OUTSTANDING_SOFT:
            return 0.7  # 降 1 档
        return 1.0

    def _nil_self_follow_in_suit(
        self,
        same_suit: list[Card],
        table_cards: list[tuple[int, Card]],
        n_on_table: int,
        seen: dict[Suit, set[int]],
    ) -> Card:
        """§3.4 跟牌(有领花)— 保证输的牌里最大;若全压过赢家,看座次。"""
        winner_pid, winner_card = _trick_current_winner(table_cards, self.TRUMP_SUIT)
        lead_suit = table_cards[0][1].suit

        # 保证输的牌(W 压过 C)
        losers = [c for c in same_suit if not self._a_beats_b(c, winner_card, lead_suit)]
        if losers:
            return max(losers, key=lambda c: c.rank.value)

        # 全压过 → 看座次
        if n_on_table == 3:
            # 末家必赢 → 切吃墩模式,出最小赢张 (§7.3 已定)
            return min(same_suit, key=lambda c: c.rank.value)
        # 非末家 → 出最小(等后手盖)
        return min(same_suit, key=lambda c: c.rank.value)

    def _nil_self_void(
        self,
        legal_cards: list[Card],
        legal_by_suit: dict[Suit, list[Card]],
        table_cards: list[tuple[int, Card]],
        seen: dict[Suit, set[int]],
    ) -> Card:
        """§3.5 空门 — 一律垫非黑桃,绝不将吃。垫 DANGER 最大门的最大牌。"""
        # 收集所有非黑桃合法牌
        non_spades_by_suit: dict[Suit, list[Card]] = {
            s: legal_by_suit[s] for s in _NON_SPADE_SUITS if legal_by_suit[s]
        }
        if not non_spades_by_suit:
            # 满手只剩黑桃被迫出将
            spades = legal_by_suit[Suit.SPADES]
            spades_on_table = [c for _, c in table_cards if c.suit == Suit.SPADES]
            lead_suit = table_cards[0][1].suit
            # 在黑桃间用"保证输"口径
            if spades_on_table:
                top_trump = max(spades_on_table, key=lambda c: c.rank.value)
                losers = [c for c in spades if c.rank.value < top_trump.rank.value]
                if losers:
                    return max(losers, key=lambda c: c.rank.value)
                # 全大,看座次
                if len(table_cards) == 3:
                    return min(spades, key=lambda c: c.rank.value)
                return min(spades, key=lambda c: c.rank.value)
            return min(spades, key=lambda c: c.rank.value)

        # 选 DANGER 最大门,出它的最大牌
        def suit_danger(s: Suit) -> float:
            cards = non_spades_by_suit[s]
            unseen = _suit_unseen(s, self.hand, seen[s])
            max_lo = max((_lo(c, unseen) for c in cards), default=0)
            length_bonus = len(cards) * 0.1
            outstanding = max(0, 13 - len(cards) - len(seen[s]))
            cover = self._cover_discount_factor(outstanding, s)
            return (max_lo + length_bonus) * cover

        danger_suits = sorted(non_spades_by_suit.keys(), key=suit_danger, reverse=True)
        target_suit = danger_suits[0]
        return max(non_spades_by_suit[target_suit], key=lambda c: c.rank.value)

    # =================================================================
    # §4 Nil 队友策略 (Coverer)
    # =================================================================

    def _play_nil_teammate(self, legal_cards: list[Card], state_view: dict) -> Card:
        """Nil 队友 — 盖 nil + 兜合约。"""
        table_cards = list(state_view.get("table_cards", []))
        spades_broken = bool(state_view.get("trump_broken", state_view.get("spades_broken", False)))
        tricks_won = state_view.get("tricks_won", [0, 0, 0, 0])
        nil_pid = self._target_nil_pid

        # 死 nil 回退 (§6.2):nil 已赢 ≥1 墩 → 撤掉盖 nil,跑纯合约
        if 0 <= nil_pid < len(tricks_won) and tricks_won[nil_pid] > 0:
            return self._non_nil_player.play_card(legal_cards, state_view)

        seen = self._all_seen()

        # 模式 (§4.2): tricks_needed
        my_team_won = tricks_won[self.position] + (tricks_won[nil_pid] if 0 <= nil_pid < len(tricks_won) else 0)
        tricks_needed = self._my_numeric_bid - my_team_won
        acquire_mode = tricks_needed > 0

        # 没出牌 = 首攻
        if not table_cards:
            return self._cover_lead(legal_cards, spades_broken, seen, acquire_mode)

        n_on_table = len(table_cards)
        winner_pid, winner_card = _trick_current_winner(table_cards, self.TRUMP_SUIT)
        is_nil_winning = (winner_pid == nil_pid)

        if n_on_table == 1:
            return self._cover_2nd_hand(legal_cards, table_cards, seen, acquire_mode)
        if n_on_table == 2:
            return self._cover_3rd_hand(
                legal_cards, table_cards, winner_card, is_nil_winning, acquire_mode, seen,
            )
        # n_on_table == 3
        return self._cover_4th_hand(
            legal_cards, table_cards, winner_pid, winner_card, is_nil_winning, acquire_mode, seen,
        )

    def _cover_lead(
        self, legal_cards: list[Card], spades_broken: bool,
        seen: dict[Suit, set[int]], acquire_mode: bool,
    ) -> Card:
        """§4.4 首攻 — 默认领高造避风港。"""
        legal_by_suit = _by_suit(legal_cards)
        nil_tracker = self._nil_trackers.get(self._target_nil_pid)
        nil_voids = nil_tracker.voids if nil_tracker else set()

        # 候选花色
        candidate_suits = [s for s in Suit if legal_by_suit[s]]
        # 黑桃未破限制
        if not spades_broken and any(s != Suit.SPADES for s in candidate_suits):
            candidate_suits = [s for s in candidate_suits if s != Suit.SPADES]

        # 禁忌:绝不向已知对手空门领出(对手空门需要跟踪,这里我们没显式跟踪 partner/opp,只跟 nil)
        # 简化:nil 还活时不主动领黑桃,除非 nil 黑桃空门或只剩黑桃
        if Suit.SPADES not in nil_voids and any(s != Suit.SPADES for s in candidate_suits):
            candidate_suits = [s for s in candidate_suits if s != Suit.SPADES]

        # 优先级 1: 已知 nil 旁花空门且我那门有 boss → 领那张赢张
        for s in candidate_suits:
            if s in nil_voids and s != Suit.SPADES:
                unseen = _suit_unseen(s, self.hand, seen[s])
                # 我的 boss(hi=0)
                for c in sorted(legal_by_suit[s], key=lambda x: -x.rank.value):
                    if _is_boss(c, unseen):
                        return c

        # 优先级 2: 已知 nil 空门那门 → 领最小
        for s in candidate_suits:
            if s in nil_voids and s != Suit.SPADES:
                return min(legal_by_suit[s], key=lambda c: c.rank.value)

        # 优先级 3: 最高 DEF_STRENGTH(深度)那门里的赢墩高牌
        suit_strengths: list[tuple[int, Suit]] = []
        for s in candidate_suits:
            if s == Suit.SPADES:
                continue
            unseen = _suit_unseen(s, self.hand, seen[s])
            depth = sum(1 for c in legal_by_suit[s] if _lo(c, unseen) >= _Thresholds.COVER_CARD_LO_MIN)
            suit_strengths.append((depth, s))
        if suit_strengths:
            suit_strengths.sort(key=lambda x: -x[0])
            # 优先深门(深度≥2 = STRONG);脆门只在没有更深门时才用
            for depth, s in suit_strengths:
                if depth == 0:
                    continue
                # 用该门里最大的"非 boss"赢墩牌(造避风港,留更深的牌续盖)
                cards = legal_by_suit[s]
                unseen = _suit_unseen(s, self.hand, seen[s])
                # 若我的最大牌就是 boss,且深度 ≥2,可以领第二大(留 boss)
                if depth >= 2:
                    sorted_cards = sorted(cards, key=lambda c: -c.rank.value)
                    # 找最高的"已能赢"的牌(非 boss 也可,只要 lo 高)
                    for c in sorted_cards:
                        if _lo(c, unseen) >= _Thresholds.COVER_CARD_LO_MIN:
                            # 用该门次大(留最大续盖)
                            if c == sorted_cards[0] and len(sorted_cards) > 1:
                                # 检查第二大是否也能赢
                                second = sorted_cards[1]
                                if _lo(second, unseen) >= _Thresholds.COVER_CARD_LO_MIN:
                                    return second
                            return c
                # 脆门(深度 1):用唯一盖牌
                # 仅在没有更深门可选时落到此处
                for c in sorted(cards, key=lambda c: -c.rank.value):
                    if _lo(c, unseen) >= _Thresholds.COVER_CARD_LO_MIN:
                        return c

        # 优先级 4 兜底:所有候选门都 LOW → 领最弱门最小牌(没办法,造不了港)
        if candidate_suits:
            return min(legal_cards, key=lambda c: (c.suit == Suit.SPADES, c.suit.value, c.rank.value))
        return legal_cards[0]

    def _cover_2nd_hand(
        self,
        legal_cards: list[Card],
        table_cards: list[tuple[int, Card]],
        seen: dict[Suit, set[int]],
        acquire_mode: bool,
    ) -> Card:
        """§4.5 2nd 手 — 对手领出,nil 最后出。按门槛高低顶 / 让。"""
        lead_card = table_cards[0][1]
        lead_suit = lead_card.suit
        legal_by_suit = _by_suit(legal_cards)
        same_suit = legal_by_suit[lead_suit]
        threshold_low = lead_card.rank.value < _Thresholds.HIGH_LEAD_RANK

        if same_suit:
            if threshold_low:
                # 门槛低 → 顶门槛造港
                bigger = [c for c in same_suit if c.rank.value > lead_card.rank.value]
                if bigger:
                    # 用较便宜的盖牌(够顶门槛即可,留更深的)
                    return min(bigger, key=lambda c: c.rank.value)
                # 我无更大 → LOW 门,出该门最高的小牌
                return max(same_suit, key=lambda c: c.rank.value)
            else:
                # 门槛已够 nil 躲 → 出最小,保深度,清自己低牌
                return min(same_suit, key=lambda c: c.rank.value)

        # 空门
        nil_tracker = self._nil_trackers.get(self._target_nil_pid)
        nil_void_in_lead = nil_tracker is not None and lead_suit in nil_tracker.voids

        if threshold_low and not nil_void_in_lead:
            # 将吃造港 (§4.5 空门 → 对手小我大)
            spades = legal_by_suit[Suit.SPADES]
            if spades:
                return min(spades, key=lambda c: c.rank.value)
        # 门槛高 / nil 也空门 → 垫,垫最弱门(简化:垫整体最小非黑桃)
        non_spades = [c for s in _NON_SPADE_SUITS for c in legal_by_suit[s]]
        if non_spades:
            return min(non_spades, key=lambda c: c.rank.value)
        return min(legal_cards, key=lambda c: c.rank.value)

    def _cover_3rd_hand(
        self,
        legal_cards: list[Card],
        table_cards: list[tuple[int, Card]],
        winner_card: Card,
        is_nil_winning: bool,
        acquire_mode: bool,
        seen: dict[Suit, set[int]],
    ) -> Card:
        """§4.6 3rd 手 — 领出者就是 nil。铁律:nil 在赢且我能盖 → 盖无条件。"""
        lead_suit = table_cards[0][1].suit
        legal_by_suit = _by_suit(legal_cards)
        same_suit = legal_by_suit[lead_suit]

        if is_nil_winning:
            # 铁律:能盖就盖,无条件,不算深度账
            if same_suit:
                bigger = [c for c in same_suit if c.rank.value > winner_card.rank.value]
                if bigger:
                    return min(bigger, key=lambda c: c.rank.value)
                return min(same_suit, key=lambda c: c.rank.value)
            # 空门 → 用最小够用黑桃将吃
            spades = legal_by_suit[Suit.SPADES]
            if spades:
                spades_on_table = [c for _, c in table_cards if c.suit == Suit.SPADES]
                if spades_on_table:
                    top = max(spades_on_table, key=lambda c: c.rank.value)
                    bigger = [c for c in spades if c.rank.value > top.rank.value]
                    if bigger:
                        return min(bigger, key=lambda c: c.rank.value)
                    # 将不过 → 垫
                    return self._cover_discard(legal_cards, legal_by_suit, seen)
                return min(spades, key=lambda c: c.rank.value)
            # 既无领花也无黑桃(已被法律 above) → 垫
            return self._cover_discard(legal_cards, legal_by_suit, seen)

        # nil 已安全 → 深度账
        if same_suit:
            if acquire_mode:
                # 只用 STRONG 门最便宜赢张拿
                bigger = [c for c in same_suit if c.rank.value > winner_card.rank.value]
                if bigger:
                    unseen = _suit_unseen(lead_suit, self.hand, seen[lead_suit])
                    depth = sum(1 for c in same_suit if _lo(c, unseen) >= _Thresholds.COVER_CARD_LO_MIN)
                    if depth >= 2:
                        return min(bigger, key=lambda c: c.rank.value)
                return min(same_suit, key=lambda c: c.rank.value)
            return min(same_suit, key=lambda c: c.rank.value)
        return self._cover_discard(legal_cards, legal_by_suit, seen)

    def _cover_4th_hand(
        self,
        legal_cards: list[Card],
        table_cards: list[tuple[int, Card]],
        winner_pid: int,
        winner_card: Card,
        is_nil_winning: bool,
        acquire_mode: bool,
        seen: dict[Suit, set[int]],
    ) -> Card:
        """§4.7 4th 手 — 末家。"""
        lead_suit = table_cards[0][1].suit
        legal_by_suit = _by_suit(legal_cards)
        same_suit = legal_by_suit[lead_suit]

        if is_nil_winning:
            # 铁律盖(脆 honor / 单张黑桃也花)
            if same_suit:
                bigger = [c for c in same_suit if c.rank.value > winner_card.rank.value]
                if bigger:
                    return min(bigger, key=lambda c: c.rank.value)
                return min(same_suit, key=lambda c: c.rank.value)
            spades = legal_by_suit[Suit.SPADES]
            if spades:
                spades_on_table = [c for _, c in table_cards if c.suit == Suit.SPADES]
                if spades_on_table:
                    top = max(spades_on_table, key=lambda c: c.rank.value)
                    bigger = [c for c in spades if c.rank.value > top.rank.value]
                    if bigger:
                        return min(bigger, key=lambda c: c.rank.value)
                    return self._cover_discard(legal_cards, legal_by_suit, seen)
                return min(spades, key=lambda c: c.rank.value)
            return self._cover_discard(legal_cards, legal_by_suit, seen)

        # nil 安全 → 深度账重新生效
        if acquire_mode:
            if same_suit:
                bigger = [c for c in same_suit if c.rank.value > winner_card.rank.value]
                if bigger:
                    unseen = _suit_unseen(lead_suit, self.hand, seen[lead_suit])
                    depth = sum(1 for c in same_suit if _lo(c, unseen) >= _Thresholds.COVER_CARD_LO_MIN)
                    if depth >= 2:
                        return min(bigger, key=lambda c: c.rank.value)
                    # 脆门 → 不花脆 honor,出最小
                    return min(same_suit, key=lambda c: c.rank.value)
                return min(same_suit, key=lambda c: c.rank.value)
            return self._cover_discard(legal_cards, legal_by_suit, seen)
        # PROTECT-ONLY
        if same_suit:
            return min(same_suit, key=lambda c: c.rank.value)
        return self._cover_discard(legal_cards, legal_by_suit, seen)

    def _cover_discard(
        self,
        legal_cards: list[Card],
        legal_by_suit: dict[Suit, list[Card]],
        seen: dict[Suit, set[int]],
    ) -> Card:
        """§4.8 空门垫牌 — 最弱门(LOW)的最小牌,留黑桃。"""
        non_spades = [(s, legal_by_suit[s]) for s in _NON_SPADE_SUITS if legal_by_suit[s]]
        if not non_spades:
            return min(legal_cards, key=lambda c: c.rank.value)
        # 选 DEF_STRENGTH 最低(无盖牌)的门的最小牌
        def def_strength(s: Suit, cards: list[Card]) -> int:
            unseen = _suit_unseen(s, self.hand, seen[s])
            return sum(1 for c in cards if _lo(c, unseen) >= _Thresholds.COVER_CARD_LO_MIN)
        non_spades.sort(key=lambda x: def_strength(x[0], x[1]))
        return min(non_spades[0][1], key=lambda c: c.rank.value)

    # =================================================================
    # §5 Nil 对手策略
    # =================================================================

    def _play_nil_opponent(self, legal_cards: list[Card], state_view: dict) -> Card:
        """Nil 对手 — 三目标 + bid_sum 重心 + 三层架构。"""
        table_cards = list(state_view.get("table_cards", []))
        spades_broken = bool(state_view.get("trump_broken", state_view.get("spades_broken", False)))
        tricks_won = state_view.get("tricks_won", [0, 0, 0, 0])
        nil_pid = self._target_nil_pid

        # 死 nil 回退 (§6.2):目标 nil 已赢 ≥1 墩 → 切非 nil
        if 0 <= nil_pid < len(tricks_won) and tricks_won[nil_pid] > 0:
            return self._non_nil_player.play_card(legal_cards, state_view)

        seen = self._all_seen()

        if not table_cards:
            # 我领出 → §5.5
            return self._opp_lead(legal_cards, spades_broken, seen)

        n_on_table = len(table_cards)
        winner_pid, winner_card = _trick_current_winner(table_cards, self.TRUMP_SUIT)
        is_nil_winning = (winner_pid == nil_pid)

        # 判定座次: nil 是否已在桌上出过?
        nil_played = any(pid == nil_pid for pid, _ in table_cards)

        if nil_played:
            # King 座跟牌路径 (§5.6)
            return self._opp_king_seat_follow(
                legal_cards, table_cards, winner_pid, winner_card,
                is_nil_winning, seen,
            )
        # Bitch 座路径 (§5.7) - nil 还没出
        return self._opp_bitch_seat_follow(
            legal_cards, table_cards, winner_pid, winner_card, seen,
        )

    def _opp_lead(self, legal_cards: list[Card], spades_broken: bool, seen: dict[Suit, set[int]]) -> Card:
        """§5.5 首攻 — 选门 + 选牌。"""
        legal_by_suit = _by_suit(legal_cards)
        nil_tracker = self._nil_trackers.get(self._target_nil_pid)
        nil_voids = nil_tracker.voids if nil_tracker else set()
        nil_led_suits = nil_tracker.led_suits if nil_tracker else set()

        # 候选(硬护栏过滤):非 nil 空门、非 nil 已领过、黑桃合法且达门槛
        candidates: list[Suit] = []
        for s in Suit:
            if not legal_by_suit[s]:
                continue
            if s in nil_voids:
                continue
            if s in nil_led_suits:
                continue
            if s == Suit.SPADES:
                if not spades_broken and any(s2 != Suit.SPADES and legal_by_suit[s2] for s2 in Suit):
                    continue
                if Suit.SPADES in nil_voids:
                    continue
                # 黑桃门槛: 我有 lo ≤ 2 的低黑桃
                unseen = _suit_unseen(Suit.SPADES, self.hand, seen[Suit.SPADES])
                if not any(_lo(c, unseen) <= _Thresholds.DEPTH_LOW_MAX for c in legal_by_suit[Suit.SPADES]):
                    continue
            candidates.append(s)

        # 兜底:全被过滤 → 退到合法集
        if not candidates:
            candidates = [s for s in Suit if legal_by_suit[s]]
            if not spades_broken:
                non_sp = [s for s in candidates if s != Suit.SPADES]
                if non_sp:
                    candidates = non_sp

        # ATTACK 公式
        nil_estimated_len = (
            nil_tracker.nil_estimated_length if nil_tracker
            else (lambda s, h: max(0, 13 - sum(1 for c in h if c.suit == s)) / 3.0)
        )

        def attack_score(s: Suit) -> float:
            cards = legal_by_suit[s]
            unseen = _suit_unseen(s, self.hand, seen[s])
            depth_low = sum(1 for c in cards if _lo(c, unseen) <= _Thresholds.DEPTH_LOW_MAX)
            my_burden = sum(
                1 for c in cards
                if _lo(c, unseen) >= _Thresholds.MY_BURDEN_MIN and not _is_boss(c, unseen)
            )
            nil_len = nil_estimated_len(s, self.hand)
            return (
                _Thresholds.ATTACK_W_DEPTH_LOW * depth_low
                + _Thresholds.ATTACK_W_NIL_LEN * nil_len
                + _Thresholds.ATTACK_W_BURDEN * my_burden
            )

        # 选 ATTACK 最高门
        best_suit = max(candidates, key=attack_score)
        unseen = _suit_unseen(best_suit, self.hand, seen[best_suit])
        cards = legal_by_suit[best_suit]

        # 选牌:bid_sum ≥ 12 启用多义试探(中 lo);≤ 11 默认领 lo 最小
        if self._bid_sum >= _Thresholds.BID_SUM_HIGH_MIN:
            # 多义:领"中 lo"(5~8 且非 boss)
            mid_cards = [
                c for c in cards
                if _Thresholds.MID_LO_MIN <= _lo(c, unseen) <= _Thresholds.MID_LO_MAX
                and not _is_boss(c, unseen)
            ]
            if mid_cards:
                return min(mid_cards, key=lambda c: c.rank.value)
            # 兜底: 领 lo 最小非 boss
            non_boss = [c for c in cards if not _is_boss(c, unseen)]
            if non_boss:
                return min(non_boss, key=lambda c: _lo(c, unseen))
            return min(cards, key=lambda c: c.rank.value)

        # 默认: lo 最小,非 boss
        non_boss = [c for c in cards if not _is_boss(c, unseen)]
        if non_boss:
            return min(non_boss, key=lambda c: _lo(c, unseen))
        return min(cards, key=lambda c: c.rank.value)

    def _opp_king_seat_follow(
        self,
        legal_cards: list[Card],
        table_cards: list[tuple[int, Card]],
        winner_pid: int,
        winner_card: Card,
        is_nil_winning: bool,
        seen: dict[Suit, set[int]],
    ) -> Card:
        """§5.6 King 座 — nil 已出。按 is_nil_winning 二分。"""
        lead_suit = table_cards[0][1].suit
        legal_by_suit = _by_suit(legal_cards)
        same_suit = legal_by_suit[lead_suit]
        nil_card = next((c for pid, c in table_cards if pid == self._target_nil_pid), None)

        if is_nil_winning:
            # A 分支:维持 nil 赢家。硬护栏 1: 绝不盖。
            if same_suit:
                under = [c for c in same_suit if c.rank.value < winner_card.rank.value]
                if under:
                    return max(under, key=lambda c: c.rank.value)  # lo 最大且 < nil
                # 全 > nil 被迫盖 → 出最大 (切吃墩模式抢领牌权)
                return max(same_suit, key=lambda c: c.rank.value)
            # 空门 → 出最大非黑桃 (绝不将吃)
            non_spades = [c for s in _NON_SPADE_SUITS for c in legal_by_suit[s]]
            if non_spades:
                return max(non_spades, key=lambda c: c.rank.value)
            # 满手黑桃被迫将 (违硬护栏 1 + 2 但无选择) → 最小黑桃
            return min(legal_by_suit[Suit.SPADES], key=lambda c: c.rank.value)

        # B 分支:nil 不是赢家 → small_depth 深度账
        partner_pid = (self.position + 2) % 4
        is_partner_winner = (winner_pid == partner_pid)

        if same_suit:
            top_in_suit = max(same_suit, key=lambda c: c.rank.value)
            if not self._a_beats_b(top_in_suit, winner_card, lead_suit):
                # 盖不过 → 出最大非黑桃 (扔最大 burden)
                return top_in_suit
            # 能盖
            if is_partner_winner:
                # small_depth 动态计算
                unseen = _suit_unseen(lead_suit, self.hand, seen[lead_suit])
                small_th = max(0, len(unseen) // _Thresholds.SMALL_THRESHOLD_DIVISOR)
                # 不盖时出 ≤ W 最大;算"留下后" small_depth
                under_W = [c for c in same_suit if c.rank.value <= winner_card.rank.value]
                if under_W:
                    candidate = max(under_W, key=lambda c: c.rank.value)
                    small_depth_after = sum(
                        1 for c in same_suit
                        if c != candidate and _lo(c, unseen) <= small_th
                    )
                    if small_depth_after >= _Thresholds.SMALL_DEPTH_GOOD:
                        return candidate  # 不盖,扔最大 burden
                # small_depth ≤ 1 或没有 ≤W 的牌 → 盖,出最大非黑桃
                return top_in_suit
            # 对方赢 → 几乎总盖
            return top_in_suit
        # 空门 → 几乎不将吃,垫最大非黑桃
        non_spades = [c for s in _NON_SPADE_SUITS for c in legal_by_suit[s]]
        if non_spades:
            return max(non_spades, key=lambda c: c.rank.value)
        return min(legal_by_suit[Suit.SPADES], key=lambda c: c.rank.value)

    def _opp_bitch_seat_follow(
        self,
        legal_cards: list[Card],
        table_cards: list[tuple[int, Card]],
        winner_pid: int,
        winner_card: Card,
        seen: dict[Suit, set[int]],
    ) -> Card:
        """§5.7 Bitch 座 — nil 还没出。"""
        lead_suit = table_cards[0][1].suit
        legal_by_suit = _by_suit(legal_cards)
        same_suit = legal_by_suit[lead_suit]
        nil_tracker = self._nil_trackers.get(self._target_nil_pid)

        # P 是否已出过?(等价于"nil 队友 pid 在 table_cards 里")
        nil_partner_pid = (self._target_nil_pid + 2) % 4
        p_played = any(pid == nil_partner_pid for pid, _ in table_cards)

        if p_played:
            # 情形 ①:干净下套
            L = nil_tracker.nil_inferred_lowest(lead_suit, self.hand) if nil_tracker else None
            W_rank = winner_card.rank.value
            W_is_trump = winner_card.suit == Suit.SPADES
            W_is_lead = winner_card.suit == lead_suit

            # 下套不可能的条件
            trap_impossible = (
                L is None
                or (nil_tracker is not None and lead_suit in nil_tracker.voids)
                or (W_is_trump and lead_suit != Suit.SPADES)
                or (W_is_lead and W_rank >= L)
            )
            if trap_impossible:
                if same_suit:
                    return min(same_suit, key=lambda c: c.rank.value)
                return self._opp_discard(legal_cards, legal_by_suit, seen)

            # W 是领花且 W < L → 贴 L 下方
            if same_suit:
                under_L = [c for c in same_suit if c.rank.value < L]
                if under_L:
                    return max(under_L, key=lambda c: c.rank.value)
                return min(same_suit, key=lambda c: c.rank.value)
            return self._opp_discard(legal_cards, legal_by_suit, seen)

        # 情形 ②: 我领出 / P 未出 → 多目标试探
        if same_suit:
            # 底线施压: 出中 lo 的牌
            unseen = _suit_unseen(lead_suit, self.hand, seen[lead_suit])
            mid = [
                c for c in same_suit
                if _Thresholds.MID_LO_MIN <= _lo(c, unseen) <= _Thresholds.MID_LO_MAX
                and not _is_boss(c, unseen)
            ]
            if mid:
                # 选中 lo 中 rank 最小的(节省高牌)
                return min(mid, key=lambda c: c.rank.value)
            # 没中 lo 牌 → 出最小(不白吃墩躲 -9)
            return min(same_suit, key=lambda c: c.rank.value)
        return self._opp_discard(legal_cards, legal_by_suit, seen)

    def _opp_discard(
        self,
        legal_cards: list[Card],
        legal_by_suit: dict[Suit, list[Card]],
        seen: dict[Suit, set[int]],
    ) -> Card:
        """§5.8 空门统一处理."""
        # 黑桃未破 / 已破检测
        # 简化:此处只读 seen 黑桃 + 桌面 — driver 的 trump_broken 已在更上层考虑
        nil_tracker = self._nil_trackers.get(self._target_nil_pid)
        spades_seen = len(seen[Suit.SPADES]) > 0
        nil_void_spades = nil_tracker is not None and Suit.SPADES in nil_tracker.voids

        # 本墩还能下套(简化: nil 还没出 → 表示这墩 §5.7 ② 路径,保陷阱)
        # 但 _opp_discard 是 §5.6 B 或 §5.7 ② trap_impossible 调用 → 简化为统一垫
        non_spades = [(s, legal_by_suit[s]) for s in _NON_SPADE_SUITS if legal_by_suit[s]]

        # bid_sum ≤ 11 且黑桃未破 → 用 lo 最大的低黑桃破黑桃
        if (
            self._bid_sum <= _Thresholds.BID_SUM_LOW_MAX + 1  # ≤ 11
            and not spades_seen
            and not nil_void_spades
            and legal_by_suit[Suit.SPADES]
        ):
            unseen_sp = _suit_unseen(Suit.SPADES, self.hand, seen[Suit.SPADES])
            low_spades = [
                c for c in legal_by_suit[Suit.SPADES]
                if _lo(c, unseen_sp) <= _Thresholds.DEPTH_LOW_MAX
            ]
            if low_spades:
                # lo 最大的低黑桃(最能赢这墩的低黑桃)
                return max(low_spades, key=lambda c: _lo(c, unseen_sp))

        # 否则: 垫我最弱门的最大牌(消耗 Coverer 深度)/ 或最小非黑桃(默认)
        if self._bid_sum >= _Thresholds.BID_SUM_HIGH_MIN and non_spades:
            # 偏 C: 垫最弱门最大牌
            def my_def(s: Suit) -> int:
                unseen = _suit_unseen(s, self.hand, seen[s])
                return sum(
                    1 for c in legal_by_suit[s]
                    if _lo(c, unseen) >= _Thresholds.COVER_CARD_LO_MIN
                )
            non_spades.sort(key=lambda x: my_def(x[0]))
            target_suit, cards = non_spades[0]
            return max(cards, key=lambda c: c.rank.value)

        # 偏 A / 居中默认: 垫最小非黑桃
        if non_spades:
            all_non_spades = [c for _, cards in non_spades for c in cards]
            return min(all_non_spades, key=lambda c: c.rank.value)
        # 满手黑桃被迫
        return min(legal_by_suit[Suit.SPADES], key=lambda c: c.rank.value)

    # =================================================================
    # 工具
    # =================================================================

    def _all_seen(self) -> dict[Suit, set[int]]:
        """合并所有 trackers 的 seen + 当前桌面(若 trackers 落后)。"""
        result: dict[Suit, set[int]] = {s: set() for s in Suit}
        for tracker in self._nil_trackers.values():
            for s in Suit:
                result[s] |= tracker.seen_by_suit[s]
        return result

    def _a_beats_b(self, a: Card, b: Card, lead_suit: Suit) -> bool:
        """a 是否比 b 赢这墩(已知 lead_suit)。"""
        a_t = a.suit == self.TRUMP_SUIT
        b_t = b.suit == self.TRUMP_SUIT
        if a_t and not b_t:
            return True
        if b_t and not a_t:
            return False
        if a_t and b_t:
            return a.rank.value > b.rank.value
        if a.suit == lead_suit and b.suit != lead_suit:
            return True
        if b.suit == lead_suit and a.suit != lead_suit:
            return False
        if a.suit == lead_suit and b.suit == lead_suit:
            return a.rank.value > b.rank.value
        return False
