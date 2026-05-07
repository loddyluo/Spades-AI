"""
同时测试两种双明手求解器（MCTS 和 精确求解器 Alpha-Beta）

牌局状态（参照 test_15_mcts.py 的构造方式）：
- 8 墩已完成（32 张牌已打出），第 9 墩正在进行
- 玩家 0 首攻 ♠K，玩家 1 跟 ♦2，玩家 2 跟 ♠4
- 桌上 3 张牌，轮到玩家 3 出牌
- 手牌共 17 张（玩家 0/1/2 各 4 张，玩家 3 有 5 张）

设计意图：
- 玩家 3 无黑桃（首攻为黑桃，玩家 3 无同花色可跟），因此有 5 个合法动作
- 既能让 MCTS 真正展开搜索（>1 动作），又可与精确求解器对比
"""

import sys
sys.path.insert(0, '.')

from trick_taking.card import Card, Suit, Rank, cards_to_bitset
from trick_taking.deck import Deck, STANDARD_52
from trick_taking.game_state import GameState, Bid, Phase
from trick_taking.solvers.double_dummy import DoubleDummySolver
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver


def create_17card_state():
    """
    创建 17 张手牌的测试状态

    队伍分配: 玩家0&2 为队伍0（己方），玩家1&3 为队伍1（对方）
    """
    deck = Deck(STANDARD_52)
    all_cards = deck.all_cards

    # ---- 17 张手牌（不包含已打到桌上的牌） ----
    # 玩家 0（队伍0）：♠A ♠Q ♥A ♣Q，已打出 ♠K
    # 玩家 1（队伍1）：♥K ♦A ♣3 ♥2，已打出 ♦2
    # 玩家 2（队伍0）：♠3 ♥3 ♦3 ♣5，已打出 ♠4
    # 玩家 3（队伍1）：♦K ♣4 ♦4 ♥4 ♥5（无黑桃！无法跟黑桃，可任选）
    hand_strs = [
        ["SA", "SQ", "HA", "CQ"],        # P0 队伍0 4 张
        ["HK", "DA", "C3", "H2"],        # P1 队伍1 4 张
        ["S3", "H3", "D3", "C5"],        # P2 队伍0 4 张
        ["DK", "C4", "D4", "H4", "H5"],  # P3 队伍1 5 张 (无黑桃)
    ]

    hands = []
    for player_hand_strs in hand_strs:
        hand = []
        for s in player_hand_strs:
            card = Card.from_str(s)
            if card not in all_cards:
                raise ValueError(f"牌 {s} 不在标准牌组中")
            hand.append(card)
        hands.append(hand)

    # ---- 桌上已有 3 张牌 ----
    table_cards = [
        (0, Card.from_str("SK")),   # P0 首攻 ♠K
        (1, Card.from_str("D2")),   # P1 无黑桃垫 ♦2
        (2, Card.from_str("S4")),   # P2 跟 ♠4
    ]

    # ---- 创建状态 ----
    state = GameState()
    state.init_for_deal(4, hands, [], all_cards)

    # 叫牌
    state.bids = [
        Bid(player_id=0, value='bid_2'),
        Bid(player_id=1, value='bid_2'),
        Bid(player_id=2, value='bid_2'),
        Bid(player_id=3, value='bid_2'),
    ]
    state.max_bid = ['bid_2', 'bid_2', 'bid_2', 'bid_2']
    state.teams = [0, 1, 0, 1]

    # 出牌阶段
    state.phase = Phase.PLAYING
    state.trick_leader = 0
    state.turn = 3    # 轮到玩家 3

    state.spades_broken = True
    state.trump_broken = True

    state.tricks_played = 8
    # 赢墩：P0=3 P1=2 P2=2 P3=1 → 队伍0=5 队伍1=3
    # 队伍0会得到10墩，队伍1只能得到3墩，队伍0得分-14，队伍1-40，结果：-26
    state.tricks_won = [3, 2, 2, 1]

    # 桌面牌
    state.table_cards = table_cards

    # ---- 计算 played_bitset（已打完的牌） ----
    all_remaining = []   # 手牌 + 桌面牌 = 还"存活"的牌
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


def print_state_info(state):
    """打印牌局信息"""
    print("=" * 60)
    print("牌局信息")
    print("=" * 60)
    print(f"当前出牌玩家: {state.turn}")
    print(f"当前墩首攻玩家: {state.trick_leader}")
    print(f"桌上牌: {[(pid, str(c)) for pid, c in state.table_cards]}")
    print(f"各玩家手牌:")
    for i in range(4):
        hand_str = ', '.join(str(c) for c in state.hands[i])
        bid = state.max_bid[i]
        print(f"  玩家{i}（队伍{state.teams[i]}，叫牌{bid}）: {hand_str}")
    print(f"已玩墩数: {state.tricks_played}")
    print(f"赢墩情况: {state.tricks_won}")
    total_hand_cards = sum(len(h) for h in state.hands)
    table_cnt = len(state.table_cards)
    played_total = 52 - total_hand_cards - table_cnt
    print(f"手牌数: {total_hand_cards}, 桌面牌数: {table_cnt}, "
          f"已打出（完成墩）: {played_total}, "
          f"共计剩余（手牌+桌面）: {total_hand_cards + table_cnt}")
    print()

    # 检查玩家3的合法动作（预期应有5个，因无黑桃）
    from trick_taking.games.spades import SpadesRules
    rules = SpadesRules()
    legal = rules.playable(state, state.hands[3], 3)
    print(f"玩家3 合法动作数: {len(legal)}")
    print(f"  可出: {', '.join(str(c) for c in legal)}")
    print()


def test_exact_solver(state):
    """测试精确求解器"""
    print("=" * 60)
    print("【精确求解器 (Alpha-Beta 剪枝)】")
    print("=" * 60)

    solver = ExactDoubleDummySolver()
    score_diff = solver.solve(state)

    print(f"最优得分差（队伍0 - 队伍1）: {score_diff}")
    print(f"转置表大小: {len(solver.tt)}")
    print()
    return score_diff


def test_mcts_solver(state):
    """测试 MCTS 求解器"""
    print("=" * 60)
    print("【MCTS 求解器 (蒙特卡洛树搜索)】")
    print("=" * 60)

    solver = DoubleDummySolver(max_iterations=2000, exploration_weight=1.4)
    result = solver.solve(state, current_player=state.turn)

    best_action = result['best_action']
    score_diff = result['state_evaluation']['expected_score_diff']

    print(f"最优出牌: {best_action}")
    print(f"最优得分差（玩家{state.turn}视角）: {score_diff:.2f}")

    print("\n动作评估（全部）:")
    for action_info in result['action_values']:
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
    print()

    # MCTS 返回的是 current_player 视角的得分差
    # 将该值转换为队伍0视角，以便与精确求解器公平对比
    player = state.turn
    player_team = state.teams[player]  # player=3 → team=1
    mcts_team0_score = score_diff if player_team == 0 else -score_diff

    return result, mcts_team0_score


def main():
    print("创建测试牌局（17 张手牌，8 墩已完成，第 9 墩进行中）...\n")

    state = create_17card_state()
    print_state_info(state)

    # --- 测试精确求解器 ---
    exact_score = test_exact_solver(state)

    # --- 测试 MCTS 求解器 ---
    mcts_result, mcts_team0 = test_mcts_solver(state)
    mcts_score = mcts_result['state_evaluation']['expected_score_diff']

    # --- 结果对比 ---
    print("=" * 60)
    print("结果对比")
    print("=" * 60)
    print(f"精确求解器: 得分差（队伍0-队伍1）= {exact_score}")
    print(f"MCTS 求解器: 得分差（玩家{state.turn}视角）= {mcts_score:.2f}")
    print(f"MCTS 求解器: 得分差（队伍0-队伍1）  = {mcts_team0:.2f}")
    print(f"差距: {abs(exact_score - mcts_team0):.2f}")
    if abs(exact_score - mcts_team0) < 1:
        print("结论: 两种求解器结果完全一致 ✓✓")
    elif abs(exact_score - mcts_team0) < 10:
        print("结论: 两种求解器结果非常接近 ✓")
    else:
        print("结论: 有差距（MCTS 为近似解），可增加迭代次数")

    best_action = mcts_result['best_action']
    if best_action:
        print(f"\nMCTS 建议最优出牌: {best_action}")

    # 简要分析
    print("\n--- 简要分析 ---")
    print(f"当前桌面 ♠K/♦2/♠4，玩家3 无黑桃可垫任意花色")
    print(f"♠K 是当前最大将牌，本墩大概率由 队伍0 赢得")


if __name__ == "__main__":
    main()
