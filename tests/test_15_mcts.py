"""
测试35张剩余牌的局面（其中一墩的第一张牌已被打出）
"""

import sys
sys.path.insert(0, '.')

from trick_taking.card import Card, Suit, Rank, cards_to_bitset
from trick_taking.deck import Deck, STANDARD_52
from trick_taking.game_state import GameState, Bid
from trick_taking.solvers.double_dummy import DoubleDummySolver


def create_state():
    deck = Deck(STANDARD_52)
    all_cards = deck.all_cards

    # 36张目标牌：35张在手牌中 + 1张在桌上（玩家0首攻CK）
    # 已完成的4墩 = 16张牌不在目标中
    target_strs = [
        "SA", "SK", "SQ", "SJ", "HA", "HK", "H2", "CA",     # P0 8张
        "DA", "DK", "DQ", "DJ", "DT", "D9", "D8", "D7", "D6",  # P1 9张
        "S3", "S4", "S5", "S6", "S7", "S8", "S9", "ST", "H4",  # P2 9张
        "D5", "D4", "D3", "D2", "C3", "C4", "C5", "C6", "H3",   # P3 9张
        "CK",                                                   # 桌上 1张
    ]

    target_cards = []
    for s in target_strs:
        card = Card.from_str(s)
        if card not in all_cards:
            raise ValueError(f"牌 {s} 不在标准牌组中")
        target_cards.append(card)

    if len(target_cards) != 36:
        raise ValueError(f"需要35张目标牌，实际有{len(target_cards)}张")

    hands = [
        target_cards[0:8],    # P0: SA, SK, SQ, SJ, HA, HK, H2, CA
        target_cards[8:17],   # P1: DA, DK, DQ, DJ, DT, D9, D8, D7, D6
        target_cards[17:26],  # P2: S3~ST, H4
        target_cards[26:35],  # P3: D5~D2, C3~C6, H3
    ]

    state = GameState()
    state.init_for_deal(4, hands, [], all_cards)

    state.bids = [
        Bid(player_id=0, value='bid_2'),
        Bid(player_id=1, value='bid_2'),
        Bid(player_id=2, value='bid_2'),
        Bid(player_id=3, value='bid_2'),
    ]
    state.max_bid = ['bid_2', 'bid_2', 'bid_2', 'bid_2']

    state.teams = [0, 1, 0, 1]

    state.phase = state.phase.PLAYING
    state.trick_leader = 0    # 当前墩由玩家0首攻
    state.turn = 1            # 玩家0已出牌，轮到玩家1

    state.spades_broken = False
    state.trump_broken = False

    state.tricks_played = 4
    state.tricks_won = [1, 1, 1, 1] # 02队：4超7, -23分， 13队：-40

    # 桌上已有玩家0打出的CK
    state.table_cards = [(0, target_cards[35])]

    # 标记已完成的16张牌为已打
    state.played_bitset = 0
    for card in all_cards:
        if card not in target_cards:
            state.played_bitset |= card.bit

    state.hand_bitsets = [cards_to_bitset(hand) for hand in hands]

    return state


def main():
    print("创建牌局（最后35张牌，CK已在桌上）...")
    state = create_state()

    print(f"当前出牌玩家: {state.turn}")
    print(f"当前墩首攻玩家: {state.trick_leader}")
    print(f"桌上牌: {[(pid, str(c)) for pid, c in state.table_cards]}")
    print(f"各玩家手牌:")
    for i in range(4):
        hand_str = ', '.join(str(c) for c in state.hands[i])
        print(f"  玩家{i}（队伍{state.teams[i]}，叫牌{state.max_bid[i]}）: {hand_str}")
    print(f"已玩墩数: {state.tricks_played}")
    print(f"赢墩情况: {state.tricks_won}")
    print(f"黑桃已破: {state.spades_broken}")

    print("\n使用MCTS求解器...")
    solver = DoubleDummySolver(max_iterations=5000)
    result = solver.solve(state, current_player=state.turn)

    print(f"最优出牌: {result['best_action']}")
    score_diff = result['state_evaluation']['expected_score_diff']
    print(f"最优得分差（玩家{state.turn}视角）: {score_diff}")

    print("\n动作评估:")
    for action_info in result['action_values'][:5]:
        print(f"  {action_info['action']}: 价值={action_info['value']:.2f}, "
              f"访问={action_info['visits']}, 置信度={action_info['confidence']:.2f}")

    print(f"\n局面评估:")
    eval_info = result['state_evaluation']
    print(f"  预期得分差: {eval_info['expected_score_diff']:.2f}")
    print(f"  团队胜率: {eval_info['team_win_probability']:.2f}")
    print(f"  确定性: {eval_info['certainty']:.2f}")

    print(f"\n搜索统计:")
    stats = result['search_statistics']
    print(f"  迭代次数: {stats['iterations']}")
    print(f"  耗时: {stats['time_elapsed']:.2f}秒")
    print(f"  扩展节点: {stats['nodes_expanded']}")
    print(f"  缓存命中: {stats['cache_hits']}")


if __name__ == "__main__":
    main()
