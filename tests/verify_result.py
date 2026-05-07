"""
用精确求解器验证正确结果
"""
import sys
sys.path.insert(0, '.')

from trick_taking.card import Card, Suit, Rank, cards_to_bitset
from trick_taking.deck import Deck, STANDARD_52
from trick_taking.game_state import GameState, Bid
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver


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
        [target_cards[0], target_cards[2], target_cards[4]],  # P0: SA, SQ, H3
        [target_cards[1], target_cards[5], target_cards[6]],  # P1: SK, C4, C5
        [target_cards[3], target_cards[8], target_cards[9]],  # P2: H2, HT, HJ
        [target_cards[7], target_cards[10], target_cards[11]],  # P3: H6, H8, H9
    ]

    state = GameState()
    state.init_for_deal(4, hands, [], all_cards)

    state.bids = [
        Bid(player_id=0, value='bid_1'),
        Bid(player_id=1, value='bid_3'),
        Bid(player_id=2, value='nil'),
        Bid(player_id=3, value='bid_3'),
    ]
    state.max_bid = ['bid_1', 'bid_3', 'nil', 'bid_3']

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


def main():
    state = create_two_trick_state()

    print("当前牌局：")
    print(f"轮到：玩家{state.turn}")
    for i in range(4):
        hand_str = ', '.join(str(c) for c in state.hands[i])
        print(f"  P{i}（队伍{state.teams[i]}，叫{state.max_bid[i]}）: {hand_str}")

    solver = ExactDoubleDummySolver()
    score_diff = solver.solve(state)

    print(f"\n精确求解器结果：{score_diff}")
    print(f"预期值（manual_calculation）：120.0")
    print(f"差距：{abs(score_diff - 120.0)}")

    # 手动计算验证
    from trick_taking.games.spades import SpadesRules
    rules = SpadesRules()

    # SA->SQ->H3 路线
    # 墩1: SA, SK, dis-HJ, dis-H6 => P0胜
    # 墩2: SQ, dis-C4, dis-HT, dis-H8 => P0胜
    # 墩3: H3, dis-C5, H2, H9 => P3胜
    test_state = create_two_trick_state()
    test_state.tricks_won = [2, 0, 0, 1]
    scores = rules.score(test_state)
    print(f"\n手动模拟得分：P0={scores[0]}")
    print(f"  team0=P0(2tricks,bid_1)+P2(0tricks,nil)")
    print(f"  team1=P3(1trick,bid_3)+P1(0tricks,bid_3)")
    print(f"  预期得分差（P0视角）：{scores[0]}")


if __name__ == "__main__":
    main()
