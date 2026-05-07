#!/usr/bin/env python3
"""
简单双明手求解器测试

设计一个简单牌局，手动计算正确结果，验证求解器输出。
"""

import sys
sys.path.insert(0, '.')

from trick_taking.card import Card, Suit, Rank, cards_to_bitset
from trick_taking.deck import Deck, STANDARD_52
from trick_taking.game_state import GameState, Bid
from trick_taking.solvers.double_dummy import DoubleDummySolver
from trick_taking.games.spades import SpadesRules


def create_one_trick_state():
    """
    创建一个只有一墩牌的简单牌局：
    玩家0: 黑桃A
    玩家1: 黑桃K
    玩家2: 黑桃Q
    玩家3: 黑桃J

    叫牌均为nil（0）
    黑桃已破
    已玩墩数：12（只剩最后一墩）
    """
    # 创建牌堆
    deck = Deck(STANDARD_52)

    # 选取4张黑桃：A, K, Q, J
    spade_cards = []
    for rank_str in ["A", "K", "Q", "J"]:
        # 找到黑桃A等
        for card in deck.all_cards:
            if card.suit == Suit.SPADES and card.rank.short == rank_str:
                spade_cards.append(card)
                break

    if len(spade_cards) != 4:
        raise ValueError("未找到所需的黑桃牌")

    # 每玩家手牌：一张黑桃 + 12张已打出的牌（虚拟）
    hands = []
    for i in range(4):
        hand = [spade_cards[i]]  # 每人一张不同的黑桃
        # 填充12张已打出的牌（从牌堆中取其他牌，但标记为已打出）
        # 我们稍后设置played_bitset
        hands.append(hand)

    # 创建游戏状态
    state = GameState()
    # 需要52张牌，但我们只关心这4张牌
    all_cards = deck.all_cards
    state.init_for_deal(4, hands, [], all_cards)

    # 设置叫牌：均为nil
    state.bids = [
        Bid(player_id=0, value='nil'),
        Bid(player_id=1, value='nil'),
        Bid(player_id=2, value='nil'),
        Bid(player_id=3, value='nil'),
    ]
    state.max_bid = ['nil', 'nil', 'nil', 'nil']

    # 设置队伍（0&2 vs 1&3）
    state.teams = [0, 1, 0, 1]

    # 设置出牌阶段
    state.phase = state.phase.PLAYING
    state.turn = 0  # 玩家0首攻
    state.trick_leader = 0

    # 设置黑桃已破
    state.spades_broken = True
    state.trump_broken = True

    # 设置已玩墩数：12
    state.tricks_played = 12

    # 设置已打出牌位图：除了这4张牌以外的所有牌都已打出
    state.played_bitset = 0
    for card in all_cards:
        if card not in spade_cards:
            state.played_bitset |= card.bit

    # 从手牌中移除已打出的牌（手牌中只留一张牌）
    # 但init_for_deal已经设置了手牌，我们需要更新hand_bitsets
    state.hand_bitsets = [cards_to_bitset(hand) for hand in hands]

    return state


# 使用trick_taking.card中的cards_to_bitset函数


def manual_calculation():
    """
    手动计算预期结果：
    玩家0: 黑桃A
    玩家1: 黑桃K
    玩家2: 黑桃Q
    玩家3: 黑桃J

    叫牌均为nil
    赢墩：玩家0赢得1墩（黑桃A最大），其他玩家0墩

    Nil结算：
    玩家0: 失败（-50）
    玩家1: 成功（+50）
    玩家2: 成功（+50）
    玩家3: 成功（+50）

    队伍0（玩家0和2）得分：-50 + 50 = 0
    队伍1（玩家1和3）得分：50 + 50 = 100

    得分差（队伍0 - 队伍1）: 0 - 100 = -100
    """
    return -100.0


def test_with_mcts():
    """使用MCTS求解器测试"""
    print("创建简单牌局...")
    state = create_one_trick_state()

    print(f"当前玩家: {state.turn}")
    print(f"各玩家手牌: {[[str(c) for c in hand] for hand in state.hands]}")
    print(f"已玩墩数: {state.tricks_played}")
    print(f"黑桃已破: {state.spades_broken}")

    print("\n使用MCTS求解器...")
    solver = DoubleDummySolver(max_iterations=1000)
    result = solver.solve(state, current_player=state.turn)

    print(f"最优动作: {result['best_action']}")
    print(f"预期得分差: {result['state_evaluation']['expected_score_diff']}")

    # 手动计算预期结果
    expected = manual_calculation()
    print(f"手动计算得分差: {expected}")

    # 比较
    diff = result['state_evaluation']['expected_score_diff']
    if abs(diff - expected) < 1e-5:
        print("✓ 测试通过！")
        return True
    else:
        print(f"✗ 测试失败：期望{expected}，得到{diff}")
        return False


if __name__ == "__main__":
    success = test_with_mcts()
    sys.exit(0 if success else 1)