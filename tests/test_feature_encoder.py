"""
测试特征编码器

验证:
1. 编码维度是否与设计一致 (1229维)
2. 对实际牌局的编码结果是否合理
3. 手牌/叫牌/花色分析等特征的正确性

使用说明:
    python test_feature_encoder.py
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from trick_taking.card import Card, Suit, Rank
from trick_taking.game_state import GameState, Bid, Phase
from trick_taking.utils.feature_encoder import SpadesFeatureEncoder


def create_17card_state():
    """
    创建 17 张手牌的测试状态 (与 test_both_solvers_17.py 一致)
    队伍分配: 玩家0&2 为队伍0 (我方), 玩家1&3 为队伍1 (对方)
    8 墩已完成, 第 9 墩进行中
    """
    from trick_taking.deck import Deck, STANDARD_52
    from trick_taking.card import cards_to_bitset

    deck = Deck(STANDARD_52)
    all_cards = deck.all_cards

    hand_strs = [
        ["SA", "SQ", "HA", "CQ"],
        ["HK", "DA", "C3", "H2"],
        ["S3", "H3", "D3", "C5"],
        ["DK", "C4", "D4", "H4", "H5"],
    ]

    hands = []
    for player_hand_strs in hand_strs:
        hand = [Card.from_str(s) for s in player_hand_strs]
        hands.append(hand)

    table_cards = [
        (0, Card.from_str("SK")),
        (1, Card.from_str("D2")),
        (2, Card.from_str("S4")),
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

    state.phase = Phase.PLAYING
    state.trick_leader = 0
    state.turn = 3
    state.spades_broken = True
    state.trump_broken = True
    state.tricks_played = 8
    state.tricks_won = [3, 2, 2, 1]
    state.table_cards = table_cards

    # 计算 played_bitset
    all_remaining = []
    for hand in hands:
        all_remaining.extend(hand)
    for _, card in table_cards:
        all_remaining.append(card)

    state.played_bitset = 0
    for card in all_cards:
        if card not in all_remaining:
            state.played_bitset |= card.bit

    state.hand_bitsets = [cards_to_bitset(hand) for hand in hands]

    return state


def test_dimensions():
    """测试总体和各类别的维度"""
    print("=" * 60)
    print("【测试1: 特征维度验证】")
    print("=" * 60)

    encoder = SpadesFeatureEncoder()
    assert encoder.total_dim == 1229, f"总维度应为 1229, 实际 {encoder.total_dim}"

    # 检查各分类维度
    sections = {
        "hand": encoder.DIM_HAND,
        "bidding": encoder.DIM_BIDDING,
        "current_trick": encoder.DIM_CURRENT_TRICK,
        "history": encoder.DIM_HISTORY,
        "suit_analysis": encoder.DIM_SUIT_ANALYSIS,
        "team_situation": encoder.DIM_TEAM_SITUATION,
        "global_flags": encoder.DIM_GLOBAL_FLAGS,
    }
    total = sum(sections.values())
    assert total == 1229, f"分类维度之和 {total} != 1229"

    for name, dim in sections.items():
        print(f"  {name:20s}: {dim:3d} 维")

    print(f"  {'总计':20s}: {total:3d} 维")
    print("  ✓ 维度验证通过\n")


def test_encode_output():
    """测试编码器输出形状和数值范围"""
    print("=" * 60)
    print("【测试2: 编码输出验证】")
    print("=" * 60)

    encoder = SpadesFeatureEncoder()
    state = create_17card_state()

    for pid in range(4):
        features = encoder.encode(state, pid)
        assert features.shape == (1229,), f"玩家{pid} 特征形状 {features.shape} != (1229,)"
        assert features.dtype == np.float32, f"数据类型应为 float32, 实际 {features.dtype}"
        assert np.isfinite(features).all(), "特征中不应出现 NaN/Inf"
        assert features.min() >= -1.0 and features.max() <= 1.0, "特征值应在[-1,1]范围内"

    print("  ✓ 编码输出形状正确 (1229,)")
    print("  ✓ 数据类型为 float32")
    print("  ✓ 所有特征值均在[-1,1]范围\n")


def test_hand_section():
    """测试手牌编码"""
    print("=" * 60)
    print("【测试3: 手牌信息编码验证】")
    print("=" * 60)

    encoder = SpadesFeatureEncoder()
    state = create_17card_state()

    # 玩家0 (队伍0): SA, SQ, HA, CQ
    features = encoder.encode(state, 0)

    # (1a) 手牌 52-bit: card_id = suit*13 + (rank-2)
    # SA(suit=0,rank=14) → 0*13+(14-2)=12
    # SQ(suit=0,rank=12) → 0*13+(12-2)=10
    # HA(suit=1,rank=14) → 1*13+(14-2)=25
    # CQ(suit=3,rank=12) → 3*13+(12-2)=49
    hand_start = 0
    hand_52 = features[hand_start:hand_start + 52]
    assert hand_52[12] == 1.0, f"SA (card_id=12) 应为 1"
    assert hand_52[10] == 1.0, f"SQ (card_id=10) 应为 1"
    assert hand_52[25] == 1.0, f"HA (card_id=25) 应为 1"
    assert hand_52[49] == 1.0, f"CQ (card_id=49) 应为 1"
    assert hand_52[0] == 0.0, "S2 (card_id=0) 不应在手中"
    assert hand_52.sum() == 4.0, f"玩家0应有4张手牌, 但 sum={hand_52.sum()}"

    # (1b) 每花色手牌数量: 4 × 14 one-hot
    suit_count_start = 52
    for suit_idx in range(4):
        seg = features[suit_count_start + suit_idx * 14: suit_count_start + (suit_idx + 1) * 14]
        cnt = int(seg.argmax())
        print(f"  花色{suit_idx} 手牌数: {cnt}")
    # ♠: SA, SQ -> 2张
    assert features[suit_count_start + 0 * 14 + 2] == 1.0, "♠ 应有2张"
    # ♥: HA -> 1张
    assert features[suit_count_start + 1 * 14 + 1] == 1.0, "♥ 应有1张"
    # ♦: 0张
    assert features[suit_count_start + 2 * 14 + 0] == 1.0, "♦ 应有0张"
    # ♣: CQ -> 1张
    assert features[suit_count_start + 3 * 14 + 1] == 1.0, "♣ 应有1张"

    # (1c) 缺门标记: 4-bit
    void_start = suit_count_start + 4 * 14
    void_flags = features[void_start:void_start + 4]
    assert void_flags[2] == 1.0, "♦ 应缺门"
    assert void_flags[0] == 0.0, "♠ 不应缺门"

    print("  ✓ 手牌编码正确\n")


def test_bidding_section():
    """测试叫牌编码"""
    print("=" * 60)
    print("【测试4: 叫牌信息编码验证】")
    print("=" * 60)

    encoder = SpadesFeatureEncoder()
    state = create_17card_state()

    features = encoder.encode(state, 0)

    # (2a) 每玩家叫品: 4 × 16 one-hot
    bid_start = 112  # 112 = DIM_HAND
    for pid in range(4):
        seg = features[bid_start + pid * 16: bid_start + (pid + 1) * 16]
        idx = int(seg.argmax())
        # bid_2 → index = 2 + 2 = 4
        assert idx == 4, f"玩家{pid} 叫品索引应为 4 (bid_2), 实际 {idx}"

    # (2d) 我方总叫品: 玩家0+2 各 bid_2 → 4
    team_bid_self_start = bid_start + 4 * 16 + 4 + 4  # bids + nil + blind_nil
    team_bid_self = features[team_bid_self_start:team_bid_self_start + 27]
    assert team_bid_self[4] == 1.0, "我方总叫品应为 4"

    # (2e) 对方总叫品: 玩家1+3 各 bid_2 → 4
    team_bid_opp_start = team_bid_self_start + 27
    team_bid_opp = features[team_bid_opp_start:team_bid_opp_start + 27]
    assert team_bid_opp[4] == 1.0, "对方总叫品应为 4"

    print("  ✓ 叫牌编码正确\n")


def test_current_trick_section():
    """测试当前墩编码"""
    print("=" * 60)
    print("【测试5: 当前墩信息编码验证】")
    print("=" * 60)

    encoder = SpadesFeatureEncoder()
    state = create_17card_state()

    sections = encoder.encode_sections(state, 3)  # 玩家3视角
    trick = sections["current_trick"]

    # (3a) 桌面牌: SK(card_id=11), D2(card_id=26), S4(card_id=2)
    # card_id = suit*13 + (rank-2)
    sk_id = Card(Suit.SPADES, Rank.KING).card_id     # 0*13+(13-2)=11
    d2_id = Card(Suit.DIAMONDS, Rank.TWO).card_id     # 2*13+(2-2)=26
    s4_id = Card(Suit.SPADES, Rank.FOUR).card_id      # 0*13+(4-2)=2
    assert trick[sk_id] == 1.0, f"SK (card_id={sk_id}) 应在桌面"
    assert trick[d2_id] == 1.0, f"D2 (card_id={d2_id}) 应在桌面"
    assert trick[s4_id] == 1.0, f"S4 (card_id={s4_id}) 应在桌面"
    assert trick[0:52].sum() == 3.0, f"桌面牌52-bit区应有3张牌, 实际 {trick[0:52].sum()}"

    # (3b) 出牌位置: 玩家0,1,2 已出
    played_pos = trick[52:56]
    assert played_pos[0] == 1.0
    assert played_pos[1] == 1.0
    assert played_pos[2] == 1.0
    assert played_pos[3] == 0.0  # 玩家3未出

    # (3c) 引牌花色: ♠ (Suit.SPADES=0)
    lead = trick[56:60]
    assert lead[0] == 1.0, f"引牌应为♠, 实际 {lead.argmax()}"

    # (3d) 玩家3是第4个出牌
    order = trick[60:64]
    assert order[3] == 1.0, f"玩家3应是第4个出牌, 实际 index={order.argmax()}"

    # (3e) 玩家3是否领牌: 否 (trick_leader=0)
    assert trick[64] == 0.0

    # (3f) 当前墩赢家: ♠K 最大 → 玩家0
    winner = trick[65:70]
    assert winner[0] == 1.0, f"赢家应为玩家0, 实际 {winner.argmax()}"

    # (3g) 赢家是否我方队友: 玩家3视角, 队友是玩家1(队伍1)
    # 赢家是玩家0(队伍0), 所以是对方
    mate = trick[70:73]
    assert mate[1] == 1.0, "赢家应为对方"

    print("  ✓ 当前墩编码正确\n")


def test_card_trace_history():
    """测试新增历史轨迹：每张牌由谁在第几轮打出"""
    print("=" * 60)
    print("【测试6: 出牌轨迹编码验证】")
    print("=" * 60)

    encoder = SpadesFeatureEncoder()
    state = create_17card_state()
    sections = encoder.encode_sections(state, 0)
    hist = sections["history"]

    # history 分块偏移：
    # 52 + 56 + 56 + 14 + 14 = 192
    # card_player_onehot = 52 * 6
    card_player_start = 192
    card_player_len = 52 * 6
    card_round_start = card_player_start + card_player_len

    # 当前桌面牌（SK, D2, S4）应被编码为：对应玩家 + 第9轮（8墩已完成，当前第9墩）
    table_expected = [
        (Card.from_str("SK"), 0),
        (Card.from_str("D2"), 1),
        (Card.from_str("S4"), 2),
    ]
    expected_round = (state.tricks_played + 1) / 13.0

    for card, pid in table_expected:
        cid = card.card_id
        player_seg = hist[card_player_start + cid * 6: card_player_start + cid * 6 + 6]
        round_val = hist[card_round_start + cid]
        assert int(player_seg.argmax()) == pid, f"{card} 应由玩家{pid}打出"
        assert abs(round_val - expected_round) < 1e-6, f"{card} 轮次编码应为第9轮"

    # 手牌中的牌应标记为“未出牌”（类别4），轮次为0
    sa = Card.from_str("SA")
    sa_seg = hist[card_player_start + sa.card_id * 6: card_player_start + sa.card_id * 6 + 6]
    sa_round = hist[card_round_start + sa.card_id]
    assert int(sa_seg.argmax()) == 4, "未出的 SA 应标记为未出牌"
    assert abs(sa_round - 0.0) < 1e-6, "未出的 SA 轮次应为0"

    # 已出但无历史明细的牌应标记为“未知”（类别5），轮次为-1
    table_ids = {card.card_id for _, card in state.table_cards}
    hand_ids = {card.card_id for hand in state.hands for card in hand}
    unknown_cid = None
    for cid in range(52):
        is_played = (state.played_bitset & (1 << cid)) != 0
        if is_played and cid not in table_ids and cid not in hand_ids:
            unknown_cid = cid
            break

    assert unknown_cid is not None, "测试状态中应存在已出且轨迹未知的牌"
    unknown_seg = hist[card_player_start + unknown_cid * 6: card_player_start + unknown_cid * 6 + 6]
    unknown_round = hist[card_round_start + unknown_cid]
    assert int(unknown_seg.argmax()) == 5, "轨迹未知牌应标记为未知类别"
    assert abs(unknown_round - (-1.0)) < 1e-6, "轨迹未知牌轮次应为-1"

    print("  ✓ 出牌轨迹编码正确\n")


def test_suit_analysis():
    """测试花色分析编码"""
    print("=" * 60)
    print("【测试7: 花色分析编码验证】")
    print("=" * 60)

    encoder = SpadesFeatureEncoder()
    state = create_17card_state()

    # 从玩家1视角看
    # 玩家1持有: HK, DA, C3, H2
    # SA, SQ, SK, S4 已出 (from table + 手牌)
    sections = encoder.encode_sections(state, 1)
    suit = sections["suit_analysis"]

    # (5d) 是否持有花色最高牌: 4-bit
    # ♠: 最高已出? SA不在桌上, 在玩家0手中. 未出最高应是SA, 玩家1无♠ → 0
    # ♥: 未出最高? HA在玩家0手中, 玩家1有HK → 玩家1不是最高, 所以应为0
    # ♦: 未出最高? DA在玩家1手中 → 玩家1最高! 应=1
    # ♣: 未出最高? 所有♣都未出 → CQ在玩家0手中, 不是玩家1 → 0
    has_top = suit[56 + 52 + 52:56 + 52 + 52 + 4]  # after suit_remaining(56), my_highest(52), highest_unplayed(52)
    # Wait let me recalculate the offset
    # (5a): 56
    # (5b): 52
    # (5c): 52
    # (5d): 4
    # (5e): 14
    # (5f): 14
    # Total: 56+52+52+4+14+14 = 192
    has_top_start = 56 + 52 + 52
    assert suit[has_top_start + 2] == 1.0, f"玩家1应持有♦最高牌(D A), 实际 {suit[has_top_start:has_top_start+4]}"
    assert suit[has_top_start + 0] == 0.0, "玩家1不应持有♠最高牌"

    # (5e) ♠ 剩余张数: 初始13张
    # 玩家1手中有0♠, 已打出
    played_mask = state.played_bitset
    spades_played = 0
    for rank in Rank:
        c = Card(Suit.SPADES, rank)
        if played_mask & c.bit:
            spades_played += 1
    # 桌上还有SK, S4
    # 手中SA, SQ在玩家0, S3在玩家2
    # 已打出: 8墩*4=32张, 其中♠? 至少SK, S4. 还有其他墩的♠
    remaining = 13 - spades_played
    # 我的♠手牌: 玩家1手中有0♠
    trump_rem_start = has_top_start + 4
    remaining_actual = int(suit[trump_rem_start:trump_rem_start + 14].argmax())
    print(f"  ♠已打出: {spades_played}, 剩余: {remaining}, 编码: {remaining_actual}")
    assert remaining_actual == remaining, f"♠剩余张数编码 {remaining_actual} != 实际剩余 {remaining}"

    print("  ✓ 花色分析编码通过\n")


def test_team_situation():
    """测试队伍局势编码"""
    print("=" * 60)
    print("【测试8: 队伍局势编码验证】")
    print("=" * 60)

    encoder = SpadesFeatureEncoder()
    state = create_17card_state()
    # 队伍: [0,1,0,1] 玩家0&2 vs 1&3
    # 赢墩: [3,2,2,1] → 队伍0=5, 队伍1=3
    # 叫品: 每人bid_2 → 各队叫品和都是4

    sections = encoder.encode_sections(state, 0)
    team = sections["team_situation"]

    # (6a) 我方已赢墩: 玩家0视角, 队伍0 = 3+2 = 5
    my_tricks = team[0:14]
    assert my_tricks[5] == 1.0, f"我方已赢墩应为5, 实际 {my_tricks.argmax()}"

    # (6b) 对方已赢墩: 队伍1 = 2+1 = 3
    opp_tricks = team[14:28]
    assert opp_tricks[3] == 1.0, f"对方已赢墩应为3, 实际 {opp_tricks.argmax()}"

    # (6c) 我方还差几墩: bid=4(2+2), tricks=5, 差值 = 4-5 = -1 → 已超1墩
    my_rem = team[28:56]
    # index = -1 + 13 = 12
    assert my_rem[12] == 1.0, f"我方还差几墩应为 -1 (index=12), 实际 index={my_rem.argmax()}"

    # (6d) 对方还差几墩: bid=4, tricks=3, 差值=4-3=1 → 还差1墩
    opp_rem = team[56:84]
    # index = 1 + 13 = 14
    assert opp_rem[14] == 1.0, f"对方还差几墩应为 1 (index=14), 实际 index={opp_rem.argmax()}"

    # (6e) 我个人 bid vs tricks: 玩家0 bid=2, tricks=3, diff=1
    my_diff = team[84:112]
    assert my_diff[14] == 1.0, f"我个人 diff=1 (index=14), 实际 index={my_diff.argmax()}"

    # (6f) 队友 (玩家2) bid vs tricks: bid=2, tricks=2, diff=0
    tm_diff = team[112:140]
    assert tm_diff[13] == 1.0, f"队友 diff=0 (index=13), 实际 index={tm_diff.argmax()}"

    # (6g) 我方超墩数: tricks=5, bid=4 → 1
    overtricks = team[140:154]
    assert overtricks[1] == 1.0, f"超墩数应为1, 实际 {overtricks.argmax()}"

    print("  ✓ 队伍局势编码正确\n")


def test_global_flags():
    """测试全局标记编码"""
    print("=" * 60)
    print("【测试9: 全局标记编码验证】")
    print("=" * 60)

    encoder = SpadesFeatureEncoder()
    state = create_17card_state()
    state.max_bid = ["bid_2", "nil", "blind_nil", "bid_2"]
    state.teams = [0, 1, 0, 1]
    state.dealer_seat = 1

    sections = encoder.encode_sections(state, 0)
    flags = sections["global_flags"]

    # (7a) spades broken
    assert flags[0] == 1.0, "♠ 已破"

    # (7b) 阶段: PLAYING
    assert flags[2] == 1.0, "应为出牌阶段"

    # (7c) 我的座位: 玩家0
    assert flags[3] == 1.0, "我的座位应为 0"

    # (7d) 庄家座位: 1 (index = 7 + 1 = 8)
    assert flags[8] == 1.0, f"庄家座位应为 1, 但 index 7={flags[7]}, index 8={flags[8]}"

    # (7e) 我方有 nil: 队友(玩家2)是blind_nil
    assert flags[11] == 1.0, "我方有 nil/blind_nil"
    # (7f) 对方有 nil: 玩家1是nil
    assert flags[12] == 1.0, "对方有 nil/blind_nil"
    # (7g) 我自己是 nil: false (bid_2)
    assert flags[13] == 0.0, "我自己不是 nil"
    # (7h) 队友是 nil/blind_nil: 玩家2是 blind_nil
    assert flags[14] == 1.0, "队友是 nil/blind_nil"
    # (7i) 我自己是 blind_nil: false
    assert flags[15] == 0.0, "我自己不是 blind_nil"

    print("  ✓ 全局标记编码正确\n")


def test_nil_bidding():
    """测试 nil/blind_nil 叫牌场景"""
    print("=" * 60)
    print("【测试10: nil/blind_nil 叫牌场景】")
    print("=" * 60)

    encoder = SpadesFeatureEncoder()
    state = create_17card_state()
    state.max_bid = ["nil", "blind_nil", "bid_5", "bid_3"]
    state.teams = [0, 1, 0, 1]

    sections = encoder.encode_sections(state, 0)

    # 叫品编码
    bidding = sections["bidding"]
    # 玩家0: nil → index 1
    # 玩家1: blind_nil → index 2
    # 玩家2: bid_5 → index 2+5 = 7
    # 玩家3: bid_3 → index 2+3 = 5
    assert bidding[0 * 16 + 1] == 1.0, "P0 应为 nil"
    assert bidding[1 * 16 + 2] == 1.0, "P1 应为 blind_nil"
    assert bidding[2 * 16 + 7] == 1.0, "P2 应为 bid_5"
    assert bidding[3 * 16 + 5] == 1.0, "P3 应为 bid_3"

    # nil_flags: P0=1, others=0
    nil_start = 4 * 16
    assert bidding[nil_start + 0] == 1.0, "P0 nil 标记应为1"

    # blind_nil_flags: P1=1, others=0
    bn_start = nil_start + 4
    assert bidding[bn_start + 1] == 1.0, "P1 blind_nil 标记应为1"

    # 我方队伍总叫品: 玩家0(nil→0) + 玩家2(bid_5→5) = 5
    team_self_start = bn_start + 4
    assert bidding[team_self_start + 5] == 1.0, "我方总叫品应为5"

    # 对方队伍总叫品: 玩家1(blind_nil→0) + 玩家3(bid_3→3) = 3
    team_opp_start = team_self_start + 27
    assert bidding[team_opp_start + 3] == 1.0, "对方总叫品应为3"

    print("  ✓ nil/blind_nil 叫牌编码正确\n")


def print_feature_structure():
    """打印完整的特征结构"""
    print("=" * 60)
    print("【特征结构总览】")
    print("=" * 60)

    encoder = SpadesFeatureEncoder()
    state = create_17card_state()
    features = encoder.encode(state, 0)

    sections = encoder.encode_sections(state, 0)
    offset = 0
    for name, arr in sections.items():
        dim = len(arr)
        # 历史分块中含标量值，使用非零计数更稳妥
        ones = int(np.count_nonzero(arr))
        print(f"  {name:20s}  [{offset:3d}:{offset+dim-1:3d}]  dim={dim:3d}  "
              f"非零={ones:3d}  ({ones/dim*100:.0f}%)")
        offset += dim

    total_nonzero = int(np.count_nonzero(features))
    print(f"  {'总计':20s}  [  0:{features.shape[0]-1}]  dim={features.shape[0]:3d}  "
          f"非零={total_nonzero:3d}  ({total_nonzero/features.shape[0]*100:.0f}%)")


def main():
    print("特征编码器测试\n", "=" * 60, sep="")

    test_dimensions()
    test_encode_output()
    test_hand_section()
    test_bidding_section()
    test_current_trick_section()
    test_card_trace_history()
    test_suit_analysis()
    test_team_situation()
    test_global_flags()
    test_nil_bidding()

    print_feature_structure()

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
