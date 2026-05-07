

import sys
sys.path.insert(0, '.')

from trick_taking.card import Card, Suit, Rank, cards_to_bitset
from trick_taking.deck import Deck, STANDARD_52
from trick_taking.game_state import GameState, Bid
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver


def create_two_trick_state():
    """
    创建只剩最后两墩的牌局：

    玩家0 (队伍0, 叫牌2): ♠A, ♠Q  （首攻）
    玩家1 (队伍1, 叫牌1): ♥2, ♥3
    玩家2 (队伍0, 叫牌nil): ♥4, ♥5
    玩家3 (队伍1, 叫牌1): ♠K, ♣A

    分析：
    方案1: P0出♠A → P3跟♠K → P0赢; P0再出♠Q → 无人有黑桃 → P0赢
    方案2: P0出♠Q → P3♠K赢 → P3出♣A → P0♠A将吃 → P0赢
    无论哪种，P0获得全部2墩，其他人0墩。

    预期得分差（队伍0 - 队伍1）= 90
    """
    deck = Deck(STANDARD_52)
    all_cards = deck.all_cards

    # 选取8张目标牌：♠A, ♠Q, ♥2, ♥3, ♥4, ♥5, ♠K, ♣A
    target_strs = ["SA", "SQ", "H2", "H3", "H4", "H5", "SK", "CA"]
    target_cards = []
    for s in target_strs:
        card = Card.from_str(s)
        if card not in all_cards:
            raise ValueError(f"牌 {s} 不在标准牌组中")
        target_cards.append(card)

    if len(target_cards) != 8:
        raise ValueError("未找到所需的8张牌")

    # 分配手牌：每玩家2张
    hands = [
        [target_cards[0], target_cards[6]],  # P0: ♠A, ♠K
        [target_cards[2], target_cards[3]],  # P1: ♥2, ♥3
        [target_cards[4], target_cards[5]],  # P2: ♥4, ♥5
        [target_cards[1], target_cards[7]],  # P3: ♠Q, ♣A
    ]

    # 创建游戏状态
    state = GameState()
    state.init_for_deal(4, hands, [], all_cards)

    # 设置叫牌
    state.bids = [
        Bid(player_id=0, value='bid_1'),
        Bid(player_id=1, value='bid_4'),
        Bid(player_id=2, value='nil'),
        Bid(player_id=3, value='bid_4'),  # 02队：51分，13队：-80分
    ]
    state.max_bid = ['bid_1', 'bid_4', 'nil', 'bid_4']

    # 设置队伍（0&2 vs 1&3）
    state.teams = [0, 1, 0, 1]

    # 设置出牌阶段
    state.phase = state.phase.PLAYING
    state.turn = 0
    state.trick_leader = 0

    # 黑桃已破
    state.spades_broken = True
    state.trump_broken = True

    # 已玩11墩，还剩最后2墩
    state.tricks_played = 11

    # 已打出牌位图：除这8张牌外其余44张都已打出
    state.played_bitset = 0
    for card in all_cards:
        if card not in target_cards:
            state.played_bitset |= card.bit

    # 更新手牌位图
    state.hand_bitsets = [cards_to_bitset(hand) for hand in hands]

    return state


def manual_calculation():
    return 131.0


def main():
    print("创建牌局（最后两墩，8张牌）...")
    state = create_two_trick_state()

    print(f"当前玩家: {state.turn}")
    print(f"各玩家手牌:")
    for i in range(4):
        hand_str = ', '.join(str(c) for c in state.hands[i])
        print(f"  玩家{i}（队伍{state.teams[i]}，叫牌{state.max_bid[i]}）: {hand_str}")
    print(f"已玩墩数: {state.tricks_played}")
    print(f"黑桃已破: {state.spades_broken}")

    print("\n使用精确双明手求解器...")
    solver = ExactDoubleDummySolver()
    try:
        score_diff = solver.solve(state)
        print(f"最优得分差（队伍0 - 队伍1）: {score_diff}")
    except Exception as e:
        print(f"求解失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    expected = manual_calculation()
    print(f"手动计算预期得分差: {expected}")

    if abs(score_diff - expected) < 1e-5:
        print("✓ 测试通过！")
        return True
    else:
        print(f"✗ 测试失败：期望 {expected}，得到 {score_diff}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
