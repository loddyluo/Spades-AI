"""
更精确地模拟和分析精确求解器的结果
"""
import sys
sys.path.insert(0, '.')

from trick_taking.card import Card, Suit, Rank, cards_to_bitset
from trick_taking.deck import Deck, STANDARD_52
from trick_taking.game_state import GameState, Bid, Phase
from trick_taking.games.spades import SpadesRules
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
    state.phase = Phase.PLAYING
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


def compute_score(tricks_won):
    """直接根据tricks_won计算P0的得分差"""
    from copy import deepcopy
    state = create_two_trick_state()
    state.tricks_won = list(tricks_won)
    rules = SpadesRules()
    scores = rules.score(state)
    return scores[0]


# 分析所有可能的tricks_won组合和对应的得分
rules = SpadesRules()
print("各可能结果得分分析：")
for p0 in range(4):
    for p1 in range(4):
        for p2 in range(4):
            for p3 in range(4):
                if p0 + p1 + p2 + p3 == 3:  # 只有3墩
                    tricks = [p0, p1, p2, p3]
                    score = compute_score(tricks)
                    if score >= 100:  # 只看高分
                        print(f"  tricks_won={tricks}: P0得分差={score}")

# 分析精确求解器的第一次动作
print("\n\n分析精确求解器的选择...")
solver = ExactDoubleDummySolver()

# 修改：追踪第一次动作
import copy
state = create_two_trick_state()
rules = SpadesRules()

hand0 = state.hands[0]
legal = rules.playable(state, hand0, 0)
print(f"P0的合法动作: {[str(c) for c in legal]}")

for action in legal:
    new_state = solver._apply_action(state, action, 0)
    value = solver.solve(new_state)
    print(f"  出 {action}: 后续最优值 = {value}")
