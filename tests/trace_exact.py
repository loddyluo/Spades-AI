"""
追踪精确求解器的具体出牌路线
"""
import sys
sys.path.insert(0, '.')

from trick_taking.card import Card, Suit, Rank, cards_to_bitset
from trick_taking.deck import Deck, STANDARD_52
from trick_taking.game_state import GameState, Bid
from trick_taking.games.spades import SpadesRules


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
        [target_cards[0], target_cards[2], target_cards[4]],
        [target_cards[1], target_cards[5], target_cards[6]],
        [target_cards[3], target_cards[8], target_cards[9]],
        [target_cards[7], target_cards[10], target_cards[11]],
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


def simulate_line(plays):
    """模拟一条完整的出牌路线，打印每一步并计算得分"""
    state = create_two_trick_state()
    rules = SpadesRules()

    print(f"{'='*60}")
    print(f"模拟出牌路线 {'->'.join(str(p[1]) for p in plays)}")
    print(f"{'='*60}")

    turn = 0
    trick_num = 11  # 当前已是第11墩（从1开始）

    for pid, action_str in plays:
        action = Card.from_str(action_str)
        print(f"\n第{state.tricks_played+1}墩，玩家{pid}出 {action}")

        # 检查合法性
        hand = state.hands[pid]
        # 打印所有玩家的手牌
        if len(state.table_cards) == 0:
            for i in range(4):
                print(f"  玩家{i}手牌: {[str(c) for c in state.hands[i]]}")

        state.play_card_to_table(pid, action)
        print(f"  桌面: {[(p, str(c)) for p, c in state.table_cards]}")

        if action.suit == Suit.SPADES:
            state.spades_broken = True
            state.trump_broken = True

        if len(state.table_cards) == 4:
            # 确定赢家
            spades = [(p, c) for p, c in state.table_cards if c.suit == Suit.SPADES]
            if spades:
                winner_pid, _ = max(spades, key=lambda x: x[1].rank.value)
            else:
                lead_suit = state.table_cards[0][1].suit
                suit_cards = [(p, c) for p, c in state.table_cards if c.suit == lead_suit]
                winner_pid, _ = max(suit_cards, key=lambda x: x[1].rank.value)

            print(f"  玩家{winner_pid}赢墩")
            state.complete_trick(winner_pid)
            state.trick_leader = winner_pid

    print(f"\n最终 tricks_won: {state.tricks_won}")
    scores = rules.score(state)
    print(f"各玩家得分: {scores}")
    print(f"P0得分差（队伍0-队伍1）: {scores[0]}")
    return scores[0]


def main():
    rules = SpadesRules()

    # 路线1: SA -> SQ -> H3 （我分析的最优路线）
    score1 = simulate_line([
        (0, "SA"), (1, "SK"), (2, "HJ"), (3, "H9"),   # 墩1
        (0, "SQ"), (1, "C4"), (2, "HT"), (3, "H8"),   # 墩2
        (0, "H3"), (1, "C5"), (2, "H2"), (3, "H6"),   # 墩3
    ])

    # 路线2: 先出H3
    score2 = simulate_line([
        (0, "H3"), (1, "C4"), (2, "H2"), (3, "H6"),   # 墩1
        (3, "H8"), (0, "SA"), (1, "SK"), (2, "HT"),   # 墩2
        (2, "HJ"), (0, "SQ"), (1, "C5"), (3, "H9"),   # 墩3
    ])

    # 路线3: SA -> H3
    score3 = simulate_line([
        (0, "SA"), (1, "SK"), (2, "HJ"), (3, "H9"),   # 墩1
        (0, "H3"), (1, "C4"), (2, "HT"), (3, "H6"),   # 墩2
        (3, "H8"), (0, "SQ"), (1, "C5"), (2, "H2"),   # 墩3
    ])

    # 路线4: SA -> SQ, but P3 discards low hearts
    score4 = simulate_line([
        (0, "SA"), (1, "SK"), (2, "HJ"), (3, "H6"),   # 墩1 P3留H9,H8
        (0, "SQ"), (1, "C4"), (2, "HT"), (3, "H8"),   # 墩2 P3留H9
        (0, "H3"), (1, "C5"), (2, "H2"), (3, "H9"),   # 墩3 P3用H9赢
    ])

    print(f"\n{'='*60}")
    print(f"结果汇总:")
    print(f"  路线1 (SA->SQ->H3): {score1}")
    print(f"  路线2 (先出H3): {score2}")
    print(f"  路线3 (SA->H3): {score3}")
    print(f"  路线4 (SA->SQ->H3, P3保高牌): {score4}")


if __name__ == "__main__":
    main()
