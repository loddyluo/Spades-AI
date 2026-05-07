"""
双明手求解器工具函数

提供一些实用函数来帮助使用双明手求解器：
1. 状态创建和验证
2. 结果分析和可视化
3. 性能测试
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Any, Optional

from trick_taking.card import Card, Suit, Rank
from trick_taking.deck import Deck, STANDARD_52
from trick_taking.game_state import GameState, Bid
from trick_taking.games.spades import SpadesRules


def create_random_state() -> GameState:
    """
    创建随机游戏状态（叫牌已结束，出牌阶段刚开始）
    
    返回:
        随机的游戏状态，包含：
        - 4名玩家各13张牌
        - 随机叫牌（包含nil/blind nil可能性）
        - 队伍分配（0&2 vs 1&3）
        - 准备开始第一墩
    """
    # 创建牌堆并洗牌
    deck = Deck(STANDARD_52)
    
    # 发牌
    hands = []
    for i in range(4):
        hand = deck.deal(13)
        hands.append(hand)
    
    # 创建游戏状态
    state = GameState()
    state.init_for_deal(4, hands, [], deck.all_cards)
    
    # 设置随机叫牌
    bids = []
    max_bid = []
    for i in range(4):
        # 随机选择叫牌类型
        bid_type = random.choice(["nil", "bid"])
        
        if bid_type == "nil":
            # 10%概率叫blind nil
            if random.random() < 0.1:
                bids.append(Bid(player_id=i, value="blind_nil"))
                max_bid.append("blind_nil")
            else:
                bids.append(Bid(player_id=i, value="nil"))
                max_bid.append("nil")
        else:
            # 随机叫1-13墩
            bid_value = random.randint(1, 13)
            bids.append(Bid(player_id=i, value=f"bid_{bid_value}"))
            max_bid.append(f"bid_{bid_value}")
    
    state.bids = bids
    state.max_bid = max_bid
    
    # 设置队伍（0&2 vs 1&3）
    state.teams = [0, 1, 0, 1]
    
    # 设置出牌阶段
    state.phase = state.phase.PLAYING
    state.turn = 0  # 玩家0首攻
    state.trick_leader = 0
    
    return state


def create_state_from_hands(hands: List[List[Card]], bids: List[str]) -> GameState:
    """
    从指定手牌和叫牌创建游戏状态
    
    参数:
        hands: 4名玩家的手牌列表，每名玩家13张牌
        bids: 4名玩家的叫牌列表，格式如 ["bid_3", "nil", "bid_5", "blind_nil"]
    
    返回:
        游戏状态
    """
    # 验证输入
    if len(hands) != 4:
        raise ValueError("必须提供4名玩家的手牌")
    if len(bids) != 4:
        raise ValueError("必须提供4名玩家的叫牌")
    
    for i, hand in enumerate(hands):
        if len(hand) != 13:
            raise ValueError(f"玩家{i}必须有13张牌，实际有{len(hand)}张")
    
    # 创建游戏状态
    state = GameState()
    
    # 收集所有卡牌
    all_cards = []
    for hand in hands:
        all_cards.extend(hand)
    
    state.init_for_deal(4, hands, [], all_cards)
    
    # 设置叫牌
    state.bids = [Bid(player_id=i, value=bid) for i, bid in enumerate(bids)]
    state.max_bid = bids.copy()
    
    # 设置队伍（0&2 vs 1&3）
    state.teams = [0, 1, 0, 1]
    
    # 设置出牌阶段
    state.phase = state.phase.PLAYING
    state.turn = 0  # 玩家0首攻
    state.trick_leader = 0
    
    return state


def state_to_dict(state: GameState) -> Dict[str, Any]:
    """
    将游戏状态转换为字典（便于JSON序列化）
    """
    return {
        "hands": [[str(card) for card in hand] for hand in state.hands],
        "bids": [str(bid.value) for bid in state.bids],
        "teams": state.teams,
        "turn": state.turn,
        "trick_leader": state.trick_leader,
        "table_cards": [(pid, str(card)) for pid, card in state.table_cards],
        "tricks_won": state.tricks_won,
        "tricks_played": state.tricks_played,
        "spades_broken": state.spades_broken,
    }


def dict_to_state(data: Dict[str, Any]) -> GameState:
    """
    从字典恢复游戏状态
    """
    from trick_taking.card import Card
    
    # 解析手牌
    hands = []
    for hand_strs in data["hands"]:
        hand = [Card.from_string(s) for s in hand_strs]
        hands.append(hand)
    
    # 创建状态
    all_cards = []
    for hand in hands:
        all_cards.extend(hand)
    
    state = GameState()
    state.init_for_deal(4, hands, [], all_cards)
    
    # 设置叫牌
    bids = []
    max_bid = []
    for i, bid_str in enumerate(data["bids"]):
        bids.append(Bid(player_id=i, value=bid_str))
        max_bid.append(bid_str)
    
    state.bids = bids
    state.max_bid = max_bid
    
    # 设置其他属性
    state.teams = data["teams"]
    state.turn = data["turn"]
    state.trick_leader = data["trick_leader"]
    state.tricks_won = data["tricks_won"]
    state.tricks_played = data["tricks_played"]
    state.spades_broken = data["spades_broken"]
    state.trump_broken = data["spades_broken"]
    
    # 设置出牌阶段
    state.phase = state.phase.PLAYING
    
    # 解析桌上牌
    table_cards = []
    for pid, card_str in data["table_cards"]:
        card = Card.from_string(card_str)
        table_cards.append((pid, card))
    
    state.table_cards = table_cards
    
    return state


def save_state_to_file(state: GameState, filename: str) -> None:
    """
    保存游戏状态到JSON文件
    """
    data = state_to_dict(state)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def load_state_from_file(filename: str) -> GameState:
    """
    从JSON文件加载游戏状态
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return dict_to_state(data)


def analyze_result(result: Dict[str, Any]) -> str:
    """
    分析求解器结果，生成易读的报告
    """
    lines = []
    
    # 最优动作
    best_action = result.get("best_action")
    lines.append(f"最优动作: {best_action}")
    lines.append("")
    
    # 动作评估
    action_values = result.get("action_values", [])
    if action_values:
        lines.append("动作评估:")
        lines.append("-" * 40)
        for i, action_info in enumerate(action_values[:10]):  # 只显示前10个
            action = action_info["action"]
            value = action_info["value"]
            visits = action_info["visits"]
            confidence = action_info["confidence"]
            lines.append(f"{i+1:2d}. {action:6s} 价值: {value:7.2f}  访问: {visits:5d}  置信度: {confidence:.2f}")
        lines.append("")
    
    # 局面评估
    state_eval = result.get("state_evaluation", {})
    if state_eval:
        lines.append("局面评估:")
        lines.append(f"  预期得分差: {state_eval.get('expected_score_diff', 0):.2f}")
        lines.append(f"  团队胜率: {state_eval.get('team_win_probability', 0.5):.2f}")
        lines.append(f"  确定性: {state_eval.get('certainty', 0.5):.2f}")
        lines.append("")
    
    # 搜索统计
    stats = result.get("search_statistics", {})
    if stats:
        lines.append("搜索统计:")
        lines.append(f"  迭代次数: {stats.get('iterations', 0)}")
        lines.append(f"  耗时: {stats.get('time_elapsed', 0):.2f}秒")
        lines.append(f"  扩展节点: {stats.get('nodes_expanded', 0)}")
        lines.append(f"  缓存命中: {stats.get('cache_hits', 0)}")
        lines.append("")
    
    return "\n".join(lines)


def compare_actions(state: GameState, current_player: int, 
                   solver_iterations: int = 1000) -> Dict[Card, Dict[str, float]]:
    """
    比较所有合法动作的价值（快速评估）
    
    返回:
        每个动作的评估信息字典
    """
    from trick_taking.solvers.double_dummy import DoubleDummySolver
    
    solver = DoubleDummySolver(max_iterations=solver_iterations)
    hand = state.hands[current_player]
    legal_actions = SpadesRules().playable(state, hand, current_player)
    
    results = {}
    
    for action in legal_actions:
        # 应用动作
        new_state = _apply_action_copy(state, action, current_player)
        
        # 快速模拟评估
        value = _quick_evaluate(new_state, current_player)
        
        results[action] = {
            "value": value,
            "description": _describe_action_impact(action, state, current_player)
        }
    
    # 按价值排序
    sorted_results = dict(sorted(
        results.items(), 
        key=lambda x: x[1]["value"], 
        reverse=True
    ))
    
    return sorted_results


def _apply_action_copy(state: GameState, action: Card, player_id: int) -> GameState:
    """应用动作并返回新状态的深拷贝"""
    # 简单深拷贝
    new_state = GameState()
    
    # 复制基本属性
    for attr in ["num_players", "phase", "dealer_seat", "turn", "trick_leader",
                 "trump_broken", "spades_broken", "tricks_played", "played_bitset"]:
        setattr(new_state, attr, getattr(state, attr))
    
    # 深拷贝列表
    new_state.hands = [hand.copy() for hand in state.hands]
    new_state.hand_bitsets = state.hand_bitsets.copy()
    new_state.bids = state.bids.copy()
    new_state.max_bid = state.max_bid.copy()
    new_state.teams = state.teams.copy()
    new_state.table_cards = state.table_cards.copy()
    new_state.tricks_won = state.tricks_won.copy()
    new_state.cards_won = [cards.copy() for cards in state.cards_won]
    new_state.points = state.points.copy()
    new_state.trick_history = state.trick_history.copy()
    
    # 应用动作
    new_state.play_card_to_table(player_id, action)
    new_state.turn = (player_id + 1) % new_state.num_players
    
    if action.suit == Suit.SPADES:
        new_state.spades_broken = True
        new_state.trump_broken = True
    
    return new_state


def _quick_evaluate(state: GameState, root_player: int) -> float:
    """快速评估局面价值（启发式）"""
    # 计算团队剩余黑桃数量
    teams = state.teams
    team_spades = [0, 0]
    
    for pid in range(4):
        team = teams[pid]
        spades_count = sum(1 for card in state.hands[pid] if card.suit == Suit.SPADES)
        team_spades[team] += spades_count
    
    # 计算团队剩余高牌（A、K、Q）数量
    team_high_cards = [0, 0]
    high_ranks = {Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK}
    
    for pid in range(4):
        team = teams[pid]
        high_count = sum(1 for card in state.hands[pid] if card.rank in high_ranks)
        team_high_cards[team] += high_count
    
    root_team = teams[root_player]
    
    # 组合评估
    spade_advantage = (team_spades[root_team] - team_spades[1 - root_team]) / max(1, sum(team_spades))
    high_card_advantage = (team_high_cards[root_team] - team_high_cards[1 - root_team]) / max(1, sum(team_high_cards))
    
    # 当前墩优势
    trick_advantage = 0.0
    if state.table_cards:
        current_leader = _estimate_trick_winner(state)
        if teams[current_leader] == root_team:
            trick_advantage = 0.1
    
    # 综合评估
    total_advantage = 0.5 * spade_advantage + 0.3 * high_card_advantage + 0.2 * trick_advantage
    
    return total_advantage * 100  # 转换为分数范围


def _estimate_trick_winner(state: GameState) -> int:
    """估计当前墩的赢家"""
    if not state.table_cards:
        return state.trick_leader
    
    # 检查是否有黑桃
    spades_cards = [(pid, card) for pid, card in state.table_cards if card.suit == Suit.SPADES]
    
    if spades_cards:
        # 有黑桃：黑桃中点数最大者赢
        winner_pid, _ = max(spades_cards, key=lambda x: x[1].rank.value)
    else:
        # 无黑桃：首攻花色中点数最大者赢
        lead_suit = state.table_cards[0][1].suit
        suit_cards = [(pid, card) for pid, card in state.table_cards if card.suit == lead_suit]
        winner_pid, _ = max(suit_cards, key=lambda x: x[1].rank.value)
    
    return winner_pid


def _describe_action_impact(action: Card, state: GameState, player_id: int) -> str:
    """描述动作的影响"""
    descriptions = []
    
    # 花色信息
    suit_name = {Suit.SPADES: "黑桃", Suit.HEARTS: "红心", 
                 Suit.DIAMONDS: "方块", Suit.CLUBS: "梅花"}[action.suit]
    descriptions.append(f"{suit_name}{action.rank}")
    
    # 首攻判断
    if not state.table_cards:
        if action.suit == Suit.SPADES and not state.spades_broken:
            descriptions.append("首攻黑桃（破禁）")
        elif action.suit == Suit.SPADES:
            descriptions.append("首攻黑桃")
        else:
            descriptions.append("首攻")
    else:
        # 跟牌判断
        lead_suit = state.table_cards[0][1].suit
        if action.suit == lead_suit:
            descriptions.append("跟牌")
        else:
            descriptions.append("垫牌")
    
    # 牌力判断
    if action.rank in {"A", "K", "Q"}:
        descriptions.append("大牌")
    elif action.rank in {"2", "3", "4"}:
        descriptions.append("小牌")
    
    return " | ".join(descriptions)