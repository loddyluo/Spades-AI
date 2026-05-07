"""
test_9.py 的 MCTS 求解器版本
功能与 test_9.py 完全一致，仅求解器替换为 DoubleDummySolver
"""

import sys
sys.path.insert(0, '.')

from trick_taking.card import Card, Suit, Rank, cards_to_bitset
from trick_taking.deck import Deck, STANDARD_52
from trick_taking.game_state import GameState, Bid
from trick_taking.solvers.double_dummy import DoubleDummySolver


def create_two_trick_state():

    deck = Deck(STANDARD_52)
    all_cards = deck.all_cards

    target_strs = ["SA", "SK", "SQ", "H2", "H3", "C4", "C5", "H6", "HT", "HJ", "H8", "H9"]
    target_cards = []
    for s in target_strs:
        card = Card.from_str(s)
        if card not in all_cards:
            raise ValueError(f"牌 {s} 不在标准牌组中")
        target_cards.append(card)

    if len(target_cards) != 12:
        raise ValueError("未找到所需的12张牌")

    hands = [
        [target_cards[0], target_cards[11], target_cards[5]],  # P0: SA, H9, C4
        [target_cards[2], target_cards[4], target_cards[3]],  # P1: SQ, H3, H2
        [target_cards[1], target_cards[8], target_cards[9]],  # P2: SK, HT, HJ
        [target_cards[7], target_cards[10], target_cards[6]],  # P3: C5, H8, H6
    ]

    state = GameState()
    state.init_for_deal(4, hands, [], all_cards)

    state.bids = [
        Bid(player_id=0, value='bid_1'),
        Bid(player_id=1, value='bid_1'),
        Bid(player_id=2, value='bid_1'),
        Bid(player_id=3, value='nil'), # 02队:20，13队：60 或 【02队: 11，13队：40】
        # 0如果先吊将，然后再出C4，只能拿到2墩
    ]
    state.max_bid = ['bid_1', 'bid_1', 'bid_1', 'nil']

    state.teams = [0, 1, 0, 1]

    state.phase = state.phase.PLAYING
    state.turn = 0
    state.trick_leader = 0

    state.spades_broken = True
    state.trump_broken = True

    state.tricks_played = 10

    state.played_bitset = 0
    for card in all_cards:
        if card not in target_cards:
            state.played_bitset |= card.bit

    state.hand_bitsets = [cards_to_bitset(hand) for hand in hands]

    return state


def manual_calculation():
    return -29.0


def main():
    print("创建牌局（最后三墩，12张牌）...")
    state = create_two_trick_state()

    print(f"当前玩家: {state.turn}")
    print(f"各玩家手牌:")
    for i in range(4):
        hand_str = ', '.join(str(c) for c in state.hands[i])
        print(f"  玩家{i}（队伍{state.teams[i]}，叫牌{state.max_bid[i]}）: {hand_str}")
    print(f"已玩墩数: {state.tricks_played}")
    print(f"黑桃已破: {state.spades_broken}")

    print("\n使用MCTS求解器...")
    solver = DoubleDummySolver(max_iterations=30000)
    result = solver.solve(state, current_player=state.turn)

    print(f"最优出牌: {result['best_action']}")
    score_diff = result['state_evaluation']['expected_score_diff']
    print(f"最优得分差（队伍0 - 队伍1）: {score_diff}")

    expected = manual_calculation()
    print(f"手动计算预期得分差: {expected}")

    if abs(score_diff - expected) < 1.0:
        print(f"✓ 测试通过！（MCTS 估计值 {score_diff:.1f} ≈ 期望 {expected}）")
        return True
    else:
        print(f"✗ 测试失败：期望 {expected}，得到 {score_diff}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
