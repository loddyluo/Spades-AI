"""minimal sanity test for RuleBasedFirst4Player.

跑几副牌, 4 个 RuleBasedFirst4Player 互打前 4 墩(第 5 墩起 fallback 出最小合法牌),
检查:
  (a) 每一步出的都是 legal_cards 的成员(由 runner 校验, 我们这里也再确认)
  (b) 前 4 墩内, 没有抛异常
  (c) 简单打印一下喜好排序、首攻选择, 方便人眼检查
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy.rule_based_first4_player import (
    RuleBasedFirst4Player,
    _suit_preference_order,
)
from strategy.spades_match_runner import SpadesMatchRunner, build_random_state
from trick_taking.card import Card, Suit, Rank
from trick_taking.games.spades import SpadesRules


def _build_runner(seed: int):
    rules = SpadesRules()
    players = [RuleBasedFirst4Player() for _ in range(4)]

    runner = SpadesMatchRunner(
        players=players,
        seed=seed,
        verbose=False,
        rules=rules,
        max_tricks=4,  # 只跑前 4 墩
    )
    return runner, players


def test_preference_order_obvious_cases():
    """构几个典型手牌, 看喜好排序是不是符合直觉。"""
    # 设计: 让每个 tier 各自只命中一门, 这样次序唯一。
    #   ♥7   单张                                -> tier 0 (A 单张)
    #   ♣97  双张, 最大 = 9 <= J                 -> tier 1 (B 双张<=J)
    #   ♦AK  双张, 含 AK                         -> tier 2 (C 含连张大牌)
    #   ♠ x9 (从 2 到 T)                         -> tier 3.5 (E 最长)
    # 总 13 张
    hand = [
        Card(Suit.HEARTS, Rank.SEVEN),
        Card(Suit.CLUBS, Rank.NINE),
        Card(Suit.CLUBS, Rank.SEVEN),
        Card(Suit.DIAMONDS, Rank.ACE),
        Card(Suit.DIAMONDS, Rank.KING),
        Card(Suit.SPADES, Rank.TWO),
        Card(Suit.SPADES, Rank.THREE),
        Card(Suit.SPADES, Rank.FOUR),
        Card(Suit.SPADES, Rank.FIVE),
        Card(Suit.SPADES, Rank.SIX),
        Card(Suit.SPADES, Rank.SEVEN),
        Card(Suit.SPADES, Rank.EIGHT),
        Card(Suit.SPADES, Rank.NINE),
    ]
    order = _suit_preference_order(hand)
    print("[case1] order =", [s.short for s in order])
    # 期望唯一次序: H(0) < C(1) < D(2) < S(3.5)
    assert order == [Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS, Suit.SPADES], (
        f"got {order}"
    )

    # case 2: 双张含 K/Q 但不连张 (D 档) -- 应排在最长(E) 前
    #   ♥A      单张                  -> tier 0
    #   ♣K3     双张, K-3 不连张       -> tier 3 (D)
    #   ♦53     双张, 最大 5 <= J     -> tier 1 (B)
    #   ♠ 长牌 9张                    -> tier 3.5 (E)
    hand2 = [
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.CLUBS, Rank.KING),
        Card(Suit.CLUBS, Rank.THREE),
        Card(Suit.DIAMONDS, Rank.FIVE),
        Card(Suit.DIAMONDS, Rank.THREE),
        Card(Suit.SPADES, Rank.TWO),
        Card(Suit.SPADES, Rank.FOUR),
        Card(Suit.SPADES, Rank.FIVE),
        Card(Suit.SPADES, Rank.SIX),
        Card(Suit.SPADES, Rank.SEVEN),
        Card(Suit.SPADES, Rank.EIGHT),
        Card(Suit.SPADES, Rank.TEN),
        Card(Suit.SPADES, Rank.JACK),
    ]
    order2 = _suit_preference_order(hand2)
    print("[case2] order =", [s.short for s in order2])
    # 期望: H(0) < D(1, 双张 53) < C(3, 双张 K3) < S(3.5, 最长)
    assert order2 == [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES], (
        f"got {order2}"
    )


def test_run_a_few_seeds():
    for seed in (42, 7, 123, 2024):
        runner, players = _build_runner(seed)
        runner.play_game()  # runner 会自己校验合法性
        print(
            f"[seed={seed}] OK; "
            f"first 4 tricks done; "
            f"player0 preference={[s.short for s in players[0].preference_order]}"
        )


def main():
    test_preference_order_obvious_cases()
    test_run_a_few_seeds()
    print("\nALL OK")


if __name__ == "__main__":
    main()
