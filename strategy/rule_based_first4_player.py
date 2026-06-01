"""RuleBasedFirst4Player —— 前 4 墩的规则式黑桃出牌玩家。

文件作用:
- 实现用户给定的规则式策略，覆盖前 4 墩的所有出牌情况;
- 严格盲眼: 仅使用自己的手牌 + 桌面已出牌 + 通过 `card_played` 自己累积的历史;
- 第 5 墩及之后由可注入的 `fallback_player` 接管(默认实现会出最小合法牌, 仅做兜底)。

核心规则(逐条对应用户描述):

1) "花色喜好排序": 在拿到 13 张牌时一次性计算, 全局保留, 不会随出牌变化。
   规则等级(从高到低):
     A) 单张 (该花色仅 1 张)
     B) 双张 且最大牌 <= J (即最大不超过 J)
     C) 含连张大牌 AK / KQ / QJ 的花色
     D) 双张 且最大牌为 A/K/Q
     E) 最长的花色
     F) 其他 (按花色长度从长到短随便排; 不重要)
   同等级内部: 黑桃永远放在最后(攻黑桃风险大), 其余按长度从短到长。

2) 攻牌(我方首攻):
   - 我方"第一次"获得攻牌权 -> 出"我喜好序最高且未被对手攻出"的花色
   - 我方非第一次获得攻牌权 -> 优先级:
       a. 我方第一次攻的那门花色
       b. 其他旁花(按当前出牌人初始喜好排序)
       c. 黑桃(若合法)
       d. 对手攻过的花色
   - 同一门花色内的取舍("打牌原则 1"):
       * 持有 1~2 张 -> 攻较大的那张
       * 持有 AK / KQ / QJ -> 攻这组连张中的最大张
       * 其余情况 -> 攻最小张

3) 第二家(防守, 跟牌):
   - 若有 lead suit 牌:
       lead 牌 >= 10  -> 出"刚好比 lead 大"的最小牌, 没有则出该花色最小
       lead 牌 <= 9   -> 出 lead 花色最小
   - 没有 lead suit 牌(将吃 / 垫牌):
       将吃合法且能将过 -> 用"最小将吃"原则(永远用最小将牌)
       否则 -> 走"垫牌原则"(规则 6)

4) 第三家:
   - 若 (lead 牌 > 第二家牌) 且 lead 牌 >= Q  -> 跟最小(因为我方"上家"已经压住)
   - 否则:
       * 有 lead suit 牌:
           手里有比"前两家最大牌"更大的同花色牌 -> 出最大那张; 否则出最小
       * 没有 lead suit 牌:
           将吃合法且能将过且 lead 牌不是 lead 花色"必赢" -> 用最小将吃
           否则 -> 垫牌(规则 6)
     (说明: 用户定义"第三家若第一家牌一定最大, 不将吃" -> 解释为
      第二家没盖过 lead 时, 若 lead 花色我们已知是必赢的 (即比 lead 高的同花色都已出现过),
      就不浪费将牌)

5) 第四家:
   - 若 第二家在当前桌面是最大方(即"我方"上家是当前赢家)
       -> 出最小(浪费最少)
   - 否则:
       * 有 lead suit 牌:
           手里有比这一轮当前最大牌更大的同花色 -> 出"那些牌中的最小张"; 否则出最小
       * 没有 lead suit 牌:
           当前最大不是黑桃 且能将过 -> 用"最小压制将吃"
           否则 -> 垫牌

6) 将吃原则:
   - 一律使用"能压过当前已出将牌的最小将牌"
   - 若不能压过则不将吃 -> 走垫牌

7) 垫牌原则(没有 lead suit 牌、并决定不将吃时):
   - 优先垫"最短花色"中的最小张
   - 若该花色最小张 >= Q, 换下一个最短花色再试
   - 兜底(罕见): 全部都 >= Q -> 出整手最小牌

== 队友判定 ==
Spades 固定 0&2 vs 1&3, 即 teammate = (self.position + 2) % 4。
"对手"是 (self.position + 1) % 4 与 (self.position + 3) % 4。

== 输入接口 ==
和 AIPlayer 一致: start_game / play_card(legal_cards, state_view) / card_played。
state_view 至少需要: hand, table_cards, trump_broken / spades_broken, tricks_played。
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Iterable, Optional

from trick_taking.card import Card, Rank, Suit
from trick_taking.player import AIPlayer


# ─── 通用小工具 ────────────────────────────────────────────────────────────

_NON_SPADE_SUITS: tuple[Suit, Suit, Suit] = (Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS)


def _by_suit(cards: Iterable[Card]) -> dict[Suit, list[Card]]:
    """把一手牌按花色分组, 每组按 rank 升序。"""
    out: dict[Suit, list[Card]] = {s: [] for s in Suit}
    for c in cards:
        out[c.suit].append(c)
    for s in Suit:
        out[s].sort(key=lambda c: c.rank.value)
    return out


def _has_consec_pair(ranks_set: set[int], a: int, b: int) -> bool:
    """rank 集合中是否同时含 a 和 b。"""
    return a in ranks_set and b in ranks_set


def _trick_current_winner(
    table_cards: list[tuple[int, Card]],
    trump_suit: Suit,
) -> tuple[int, Card]:
    """根据 game_rules.winner_trick 同样的规则, 计算桌上当前最大方。

    入参 table_cards 为 (player_id, Card) 列表(按出牌时间顺序); 至少含一张。
    """
    lead_suit = table_cards[0][1].suit
    best_pid, best_card = table_cards[0]
    best_is_trump = best_card.suit == trump_suit
    for pid, card in table_cards[1:]:
        is_trump = card.suit == trump_suit
        if is_trump and not best_is_trump:
            best_pid, best_card, best_is_trump = pid, card, True
        elif is_trump and best_is_trump:
            if card.rank.value > best_card.rank.value:
                best_pid, best_card = pid, card
        elif not is_trump and not best_is_trump:
            if card.suit == lead_suit and (
                best_card.suit != lead_suit
                or card.rank.value > best_card.rank.value
            ):
                best_pid, best_card = pid, card
    return best_pid, best_card


# ─── 喜好排序 ──────────────────────────────────────────────────────────────


def _suit_preference_order(hand: list[Card]) -> list[Suit]:
    """按用户定义的等级算每门花色的"喜好序", 数字越小越想先攻。

    返回的 list 长度恒为 4 (4 门花色), 顺序即喜好优先级。
    平级时: 黑桃放最后, 其他按长度从短到长。
    """
    by_suit = _by_suit(hand)

    def tier(suit: Suit) -> int:
        cards = by_suit[suit]
        n = len(cards)
        if n == 0:
            # 空门没什么"喜好攻"的意义, 一律放到最后档(F)
            return 5
        if n == 1:
            return 0  # A) 单张

        ranks_set = {c.rank.value for c in cards}
        max_rank = max(ranks_set)
        has_ak_kq_qj = (
            _has_consec_pair(ranks_set, Rank.ACE.value, Rank.KING.value)
            or _has_consec_pair(ranks_set, Rank.KING.value, Rank.QUEEN.value)
            or _has_consec_pair(ranks_set, Rank.QUEEN.value, Rank.JACK.value)
        )

        if n == 2:
            # 双张:
            #   B) 最大牌 <= J
            #   C) AK/KQ/QJ 双张本身就"含连张大牌" -> 优先 C
            #   D) 双张且最大为 A/K/Q 但非 AK/KQ/QJ 连张 (e.g. K3, A5, Q7)
            if max_rank <= Rank.JACK.value:
                return 1  # B
            if has_ak_kq_qj:
                return 2  # C
            if max_rank in (Rank.ACE.value, Rank.KING.value, Rank.QUEEN.value):
                return 3  # D
            return 4  # 兜底

        # n >= 3
        if has_ak_kq_qj:
            return 2  # C

        # E) 最长 / F) 其他 —— 都先记 4, 后续把最长升级为 3.5
        return 4

    # 第一轮 tier
    raw = [(s, tier(s)) for s in Suit]

    # 在 tier == 4 中找出最长那一门, 升级为 E (4) -> 实际上我们想区分 E 和 F
    # 但用户给的 E "最长花色" 只比 F 优先, 二者都比 D 差。
    # 简化: 把 "最长花色" 在 tier=4 的子集里降为 3.5 (让它优先)
    tier4 = [s for (s, t) in raw if t == 4]
    if tier4:
        max_len = max(len(by_suit[s]) for s in tier4)
        # 把同样最长且 tier==4 的花色升级为 3.5
        upgraded = {s for s in tier4 if len(by_suit[s]) == max_len}
    else:
        upgraded = set()

    def final_tier(item: tuple[Suit, int]) -> float:
        suit, t = item
        if t == 4 and suit in upgraded:
            return 3.5  # E) 最长
        return float(t)

    # 排序键: (tier, 黑桃排最后, 长度短的优先, 花色枚举值保证稳定)
    def sort_key(item: tuple[Suit, int]) -> tuple[float, int, int, int]:
        suit, _ = item
        cards = by_suit[suit]
        spade_last = 1 if suit == Suit.SPADES else 0
        return (final_tier(item), spade_last, len(cards), suit.value)

    raw_sorted = sorted(raw, key=sort_key)
    return [s for (s, _) in raw_sorted]


# ─── Player ────────────────────────────────────────────────────────────────


class _DefaultFallback:
    """非常笨的兜底: 出最小合法牌。仅在没传 fallback_player 时使用, 第 5 墩起。"""

    def play_card(self, legal_cards: list[Card], state_view: dict) -> Card:
        return min(legal_cards, key=lambda c: (c.suit.value, c.rank.value))


class RuleBasedFirst4Player(AIPlayer):
    """前 4 墩规则式玩家。

    第 5 墩及以后的出牌交给 `fallback_player.play_card(legal_cards, state_view)`,
    默认是出最小合法牌(仅作占位, 真用法应外部注入精确求解器)。
    """

    TRUMP_SUIT: Suit = Suit.SPADES

    def __init__(
        self,
        fallback_player: Optional[Any] = None,
        bid_strategy: str = "random",
        bid_seed: int | None = None,
    ) -> None:
        self.fallback_player = fallback_player or _DefaultFallback()
        # 叫牌策略: 这里不是重点, 留个开关; 规则式比赛只想做出牌
        self._bid_strategy = bid_strategy
        self._bid_rng = random.Random(bid_seed)

        # 每局重置
        self.position: int = -1
        self.hand: list[Card] = []
        self.preference_order: list[Suit] = []  # 13 张牌确定后填充

        # 历史 / 跟踪
        # _history[t] = list of (player_id, card), 第 t 墩按出牌顺序
        self._history: list[list[tuple[int, Card]]] = []
        # 当前正在进行的墩, 4 张满了之后会 flush 到 _history
        self._current_trick: list[tuple[int, Card]] = []
        # 对手"攻过"的花色(每墩起首一张就记一次, 仅在对手起首时记)
        self._opp_led_suits: set[Suit] = set()
        # 我方第一次起首时攻的那门花色(我方 = self.position 或 (self+2)%4 )
        self._our_first_led_suit: Optional[Suit] = None

    # ─── 接口 ────────────────────────────────────────────────────────

    def start_game(self, position: int, hand: list[Card], num_players: int) -> None:
        self.position = position
        self.hand = list(hand)
        self.preference_order = _suit_preference_order(self.hand)
        self._history = []
        self._current_trick = []
        self._opp_led_suits = set()
        self._our_first_led_suit = None

    def place_bid(self, legal_bids: list[Any], state_view: dict) -> Any:
        # 规则式策略不关心叫牌 —— 默认随机, 调用者通常会用更好的策略
        if not legal_bids:
            return None
        return self._bid_rng.choice(legal_bids)

    def card_played(self, player_id: int, card: Card) -> None:
        """所有玩家落一张牌时都会被调用 -> 更新历史 + 自己的手牌副本。"""
        self._current_trick.append((player_id, card))
        if player_id == self.position:
            # 同步从手中移走
            try:
                self.hand.remove(card)
            except ValueError:
                pass

        # 检查是否该墩开始(只有第一张时, 记 lead 花色)
        if len(self._current_trick) == 1:
            leader = player_id
            if self._is_opponent(leader):
                self._opp_led_suits.add(card.suit)
            elif self._is_us(leader) and self._our_first_led_suit is None:
                self._our_first_led_suit = card.suit

        # 一墩满了 -> flush
        if len(self._current_trick) == 4:
            self._history.append(list(self._current_trick))
            self._current_trick = []

    def play_card(self, legal_cards: list[Card], state_view: dict) -> Card:
        # 第 5 墩起交给 fallback
        tricks_played = state_view.get("tricks_played", len(self._history))
        if tricks_played >= 4:
            return self.fallback_player.play_card(legal_cards, state_view)

        # 我们必须保持手牌副本和真实 legal_cards 一致(防御: 如果 start_game 传的
        # 手牌和 runner 的不一致, 直接以 legal_cards 推断)
        # 在前 4 墩, legal_cards 总是合法子集, 所以下面用 self.hand 时只取
        # legal_cards 里的卡作为可选项(并以 self.hand 的整体形状做策略推断)。
        table_cards: list[tuple[int, Card]] = list(state_view.get("table_cards", []))
        spades_broken: bool = bool(
            state_view.get("trump_broken", state_view.get("spades_broken", False))
        )

        if not table_cards:
            # 我是首攻
            return self._lead(legal_cards, spades_broken)

        n_on_table = len(table_cards)
        if n_on_table == 1:
            return self._second_hand(legal_cards, table_cards)
        if n_on_table == 2:
            return self._third_hand(legal_cards, table_cards)
        if n_on_table == 3:
            return self._fourth_hand(legal_cards, table_cards)

        # 不应到这里
        return min(legal_cards, key=lambda c: c.rank.value)

    # ─── 队伍判定 ────────────────────────────────────────────────────

    def _is_us(self, pid: int) -> bool:
        return (pid - self.position) % 2 == 0

    def _is_opponent(self, pid: int) -> bool:
        return not self._is_us(pid)

    # ─── 首攻 ────────────────────────────────────────────────────────

    def _is_first_lead_for_us(self) -> bool:
        """到目前为止, 我方还没有起过首攻(包括当前这次)。"""
        return self._our_first_led_suit is None

    def _lead(self, legal_cards: list[Card], spades_broken: bool) -> Card:
        """选择首攻的花色 + 具体那张牌。"""
        legal_by_suit = _by_suit(legal_cards)
        # 当前手中按花色统计(以 self.hand 为准, 用于"是否单张/连张"等判断)
        my_by_suit = _by_suit(self.hand)

        # 候选花色: 在 legal_cards 出现过的花色集合
        legal_suits = {s for s in Suit if legal_by_suit[s]}

        is_first = self._is_first_lead_for_us()

        if is_first:
            # 喜好序最高 & 未被对手攻出 & 在 legal 中
            candidate_suits = [
                s for s in self.preference_order
                if s in legal_suits and s not in self._opp_led_suits
            ]
            # 兜底: 若全部排除掉(理论上罕见), 退回喜好序里第一个 legal
            if not candidate_suits:
                candidate_suits = [s for s in self.preference_order if s in legal_suits]
        else:
            # 非第一次: a 我方首攻花色 -> b 其他旁花 -> c 黑桃 -> d 对手攻过的花色
            order: list[Suit] = []
            # a) 我方第一次攻的花色
            if (
                self._our_first_led_suit is not None
                and self._our_first_led_suit in legal_suits
            ):
                order.append(self._our_first_led_suit)
            # b) 其他旁花(按当前喜好排序)
            for s in self.preference_order:
                if (
                    s != Suit.SPADES
                    and s != self._our_first_led_suit
                    and s in legal_suits
                ):
                    order.append(s)
            # c) 黑桃
            if Suit.SPADES in legal_suits and Suit.SPADES not in order:
                order.append(Suit.SPADES)
            # d) 对手攻过的花色(可能已经在前面出现, 这里仅作为兜底)
            for s in self._opp_led_suits:
                if s in legal_suits and s not in order:
                    order.append(s)

            candidate_suits = order or list(legal_suits)

        chosen_suit = candidate_suits[0]
        return self._pick_attack_card(chosen_suit, legal_by_suit, my_by_suit)

    def _pick_attack_card(
        self,
        suit: Suit,
        legal_by_suit: dict[Suit, list[Card]],
        my_by_suit: dict[Suit, list[Card]],
    ) -> Card:
        """我已经决定攻这门花色, 用"打牌原则 1"挑出哪张。"""
        legal_cards_in_suit = legal_by_suit[suit]
        if not legal_cards_in_suit:
            # 不应发生, 兜底
            return min(
                [c for s in legal_by_suit for c in legal_by_suit[s]],
                key=lambda c: c.rank.value,
            )

        my_cards_in_suit = my_by_suit[suit]
        n = len(my_cards_in_suit)
        ranks_set = {c.rank.value for c in my_cards_in_suit}

        # 1) 1~2 张 -> 攻较大的那张(在 legal 中)
        if n <= 2:
            return max(legal_cards_in_suit, key=lambda c: c.rank.value)

        # 2) 含连张大牌 AK/KQ/QJ -> 攻该组中最大的(legal 内)
        # 优先级: AK > KQ > QJ
        for hi, lo in (
            (Rank.ACE.value, Rank.KING.value),
            (Rank.KING.value, Rank.QUEEN.value),
            (Rank.QUEEN.value, Rank.JACK.value),
        ):
            if hi in ranks_set and lo in ranks_set:
                # 攻 hi(若 legal); 否则 lo; 否则 legal 内最大
                hi_card = next(
                    (c for c in legal_cards_in_suit if c.rank.value == hi), None
                )
                if hi_card is not None:
                    return hi_card
                lo_card = next(
                    (c for c in legal_cards_in_suit if c.rank.value == lo), None
                )
                if lo_card is not None:
                    return lo_card
                break  # 没法选连张, 落到下面的"最小"逻辑

        # 3) 其余情况 -> 攻最小张
        return min(legal_cards_in_suit, key=lambda c: c.rank.value)

    # ─── 第二家 ──────────────────────────────────────────────────────

    def _second_hand(
        self,
        legal_cards: list[Card],
        table_cards: list[tuple[int, Card]],
    ) -> Card:
        lead_card = table_cards[0][1]
        lead_suit = lead_card.suit
        legal_by_suit = _by_suit(legal_cards)
        same_suit = legal_by_suit[lead_suit]

        if same_suit:
            # 必须跟 lead 花色
            if lead_card.rank.value >= Rank.TEN.value:
                # 出"刚好比 lead 大"的最小牌, 没有则该花色最小
                bigger = [c for c in same_suit if c.rank.value > lead_card.rank.value]
                if bigger:
                    return min(bigger, key=lambda c: c.rank.value)
                return min(same_suit, key=lambda c: c.rank.value)
            else:
                # lead <=9, 出 lead 花色最小
                return min(same_suit, key=lambda c: c.rank.value)

        # 没有 lead 花色 -> 将吃 or 垫牌
        return self._ruff_or_discard(
            legal_cards=legal_cards,
            legal_by_suit=legal_by_suit,
            table_cards=table_cards,
            lead_suit=lead_suit,
            position_in_trick=2,
        )

    # ─── 第三家 ──────────────────────────────────────────────────────

    def _third_hand(
        self,
        legal_cards: list[Card],
        table_cards: list[tuple[int, Card]],
    ) -> Card:
        lead_pid, lead_card = table_cards[0]
        second_pid, second_card = table_cards[1]
        lead_suit = lead_card.suit
        legal_by_suit = _by_suit(legal_cards)

        # 当前桌面最大方
        cur_winner_pid, cur_winner_card = _trick_current_winner(
            table_cards, self.TRUMP_SUIT
        )

        # 用户规则: "若 lead 比第二家大 且 lead >= Q -> 出最小"
        # (= 我方上家也就是第二家被压住, 而 lead 已经 >= Q, 我跟最小)
        # 这里"出最小"指: 整手最小的合法牌(不分花色, 因为可能没 lead 花色)
        # 但若必须跟 lead 花色 -> 跟 lead 花色最小
        lead_beats_second = self._cards_higher_in_trick(
            lead_card, second_card, lead_suit
        )
        if lead_beats_second and lead_card.rank.value >= Rank.QUEEN.value:
            same_suit = legal_by_suit[lead_suit]
            if same_suit:
                return min(same_suit, key=lambda c: c.rank.value)
            return min(legal_cards, key=lambda c: c.rank.value)

        # 否则:
        same_suit = legal_by_suit[lead_suit]
        if same_suit:
            # 比"前两家最大"还大的同花色
            top_so_far = max(
                [lead_card, second_card] if second_card.suit == lead_suit else [lead_card],
                key=lambda c: c.rank.value,
            )
            bigger = [c for c in same_suit if c.rank.value > top_so_far.rank.value]
            if bigger:
                return max(bigger, key=lambda c: c.rank.value)  # "出最大那张"
            return min(same_suit, key=lambda c: c.rank.value)

        # 没 lead 花色 -> 将吃 / 垫牌
        # 用户特别规定: 若 lead 是 lead 花色"必赢" -> 不将吃
        if self._lead_is_already_winning(lead_card, lead_suit):
            # 不将吃, 垫牌
            return self._discard(legal_cards, legal_by_suit)
        return self._ruff_or_discard(
            legal_cards=legal_cards,
            legal_by_suit=legal_by_suit,
            table_cards=table_cards,
            lead_suit=lead_suit,
            position_in_trick=3,
        )

    # ─── 第四家 ──────────────────────────────────────────────────────

    def _fourth_hand(
        self,
        legal_cards: list[Card],
        table_cards: list[tuple[int, Card]],
    ) -> Card:
        lead_pid, lead_card = table_cards[0]
        second_pid, second_card = table_cards[1]
        third_pid, third_card = table_cards[2]
        lead_suit = lead_card.suit
        legal_by_suit = _by_suit(legal_cards)

        cur_winner_pid, cur_winner_card = _trick_current_winner(
            table_cards, self.TRUMP_SUIT
        )

        # 第二家是不是当前赢家? -> 第二家是我方(self+1 之前那个 = (self-1)%4)? 不是,
        # 第四家轮到我时, 桌上 leader=lead_pid, 第二家=(lead_pid+1)%4,
        # 第三家=(lead_pid+2)%4, 第四家=(lead_pid+3)%4=self.position
        # 第二家 pid = (lead_pid+1)%4
        second_pid_calc = (lead_pid + 1) % 4
        if cur_winner_pid == second_pid_calc:
            # 第二家正在赢 —— 注意第二家是对手(因为我是第四家, lead_pid 是我对家+1
            # 之类... 实际上座次顺序与队伍是 0,1,2,3, 我=lead_pid+3. 队友 = self+2 =
            # lead_pid+1 = 第二家). 等下, 队友 = (self+2)%4 = (lead_pid+3+2)%4 =
            # (lead_pid+1)%4 = 第二家. 所以第二家是我队友。
            # 用户规则: "如果这一轮第二家大了 -> 出最小"
            # (因为队友已经赢了, 我留实力)
            same_suit = legal_by_suit[lead_suit]
            if same_suit:
                return min(same_suit, key=lambda c: c.rank.value)
            return self._discard(legal_cards, legal_by_suit)

        # 否则: 队友没赢 -> 我尽量赢
        same_suit = legal_by_suit[lead_suit]
        if same_suit:
            bigger = [c for c in same_suit if c.rank.value > cur_winner_card.rank.value
                      and cur_winner_card.suit != self.TRUMP_SUIT]
            # 注意: 当前最大如果是黑桃, 用同花色赢不了 (除非 lead suit==SPADES)
            if cur_winner_card.suit == self.TRUMP_SUIT and lead_suit != self.TRUMP_SUIT:
                # 用同花色注定赢不了 -> 出最小
                return min(same_suit, key=lambda c: c.rank.value)
            if bigger:
                # 用户规则(第四家): "出这些牌中的最小张"
                return min(bigger, key=lambda c: c.rank.value)
            return min(same_suit, key=lambda c: c.rank.value)

        # 没 lead 花色 -> 将吃 / 垫牌
        return self._ruff_or_discard(
            legal_cards=legal_cards,
            legal_by_suit=legal_by_suit,
            table_cards=table_cards,
            lead_suit=lead_suit,
            position_in_trick=4,
        )

    # ─── 将吃 / 垫牌 ─────────────────────────────────────────────────

    def _ruff_or_discard(
        self,
        legal_cards: list[Card],
        legal_by_suit: dict[Suit, list[Card]],
        table_cards: list[tuple[int, Card]],
        lead_suit: Suit,
        position_in_trick: int,
    ) -> Card:
        """没有 lead 花色时, 决定将吃还是垫牌。

        用户的将吃细则: 第二家"一定要将吃"(若合法且能将过); 第三家在第一家不是必赢
        时将吃; 第四家在第二家未大时将吃。这里 caller 已经按位置过滤过特殊"不将吃"
        条件, 我们只负责: "能将过就用最小将牌; 否则垫牌"。
        """
        spades = legal_by_suit[Suit.SPADES]
        if not spades:
            return self._discard(legal_cards, legal_by_suit)

        # 当前桌面已有的将牌(黑桃)中, 最大那张是多少?
        spades_on_table = [
            c for _, c in table_cards if c.suit == self.TRUMP_SUIT
        ]
        if spades_on_table:
            top_trump = max(spades_on_table, key=lambda c: c.rank.value)
            higher = [c for c in spades if c.rank.value > top_trump.rank.value]
            if not higher:
                # 将不过 -> 不将吃, 垫牌
                return self._discard(legal_cards, legal_by_suit)
            return min(higher, key=lambda c: c.rank.value)

        # 桌面无将 -> 用最小的将牌
        return min(spades, key=lambda c: c.rank.value)

    def _discard(
        self,
        legal_cards: list[Card],
        legal_by_suit: dict[Suit, list[Card]],
    ) -> Card:
        """垫牌: 优先垫最短花色的最小张; 若该花色最小张 >= Q, 换下一个最短。"""
        # 候选: 非黑桃花色, 因为黑桃要保留(若黑桃也得垫, 走兜底)
        non_spade_groups: list[tuple[int, Suit, list[Card]]] = []
        for s in _NON_SPADE_SUITS:
            cards = legal_by_suit[s]
            if cards:
                non_spade_groups.append((len(cards), s, cards))

        # 按长度从短到长排; 同长度按花色枚举值稳定排
        non_spade_groups.sort(key=lambda x: (x[0], x[1].value))

        for _, _, cards in non_spade_groups:
            smallest = min(cards, key=lambda c: c.rank.value)
            if smallest.rank.value < Rank.QUEEN.value:
                return smallest

        # 全部非黑桃花色最小都 >= Q, 或者根本没有非黑桃 -> 整体最小
        # (这里也算一种"不得已"垫牌, 包含黑桃也行)
        return min(legal_cards, key=lambda c: c.rank.value)

    # ─── helper: 比较 ─────────────────────────────────────────────────

    def _cards_higher_in_trick(
        self,
        a: Card,
        b: Card,
        lead_suit: Suit,
    ) -> bool:
        """在当前墩(已知 lead_suit)中, a 是否比 b 大。

        等同 winner_trick 的局部判定(忽略其他玩家的牌)。
        """
        a_trump = a.suit == self.TRUMP_SUIT
        b_trump = b.suit == self.TRUMP_SUIT
        if a_trump and not b_trump:
            return True
        if b_trump and not a_trump:
            return False
        if a_trump and b_trump:
            return a.rank.value > b.rank.value
        # 都不是将
        if a.suit == lead_suit and b.suit != lead_suit:
            return True
        if b.suit == lead_suit and a.suit != lead_suit:
            return False
        if a.suit == lead_suit and b.suit == lead_suit:
            return a.rank.value > b.rank.value
        return False  # 两张都不是 lead 也不是将, 都不算赢

    def _lead_is_already_winning(self, lead_card: Card, lead_suit: Suit) -> bool:
        """lead_card 是不是该花色"必胜"(更高的同花色都已被打出 / 自家有, 但作为
        盲眼玩家只能看"已出过"和"自家有"的合并)。

        判断: 把已出过的和自己手里的合并, 比 lead_card 大的同花色是否都没了。
        """
        seen: set[int] = set()
        # 已出过的(包含历史 + 当前墩)
        for trick in self._history:
            for _, c in trick:
                if c.suit == lead_suit:
                    seen.add(c.rank.value)
        for _, c in self._current_trick:
            if c.suit == lead_suit:
                seen.add(c.rank.value)
        # 自己手里的
        for c in self.hand:
            if c.suit == lead_suit:
                seen.add(c.rank.value)
        # 比 lead_card 大的所有 rank 是否都在 seen 中?
        higher = [r for r in range(lead_card.rank.value + 1, Rank.ACE.value + 1)]
        return all(r in seen for r in higher)
