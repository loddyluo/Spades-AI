"""
黑桃王（Spades）双明手求解器

实现原理：
1. 双明手求解器拥有完全信息（知道所有玩家的手牌）
2. 使用蒙特卡洛树搜索（MCTS）寻找最优策略
3. 优化目标：最大化（己方队伍得分 - 对方队伍得分）
4. 支持叫牌阶段后的任意局面（0-51张牌已打出）

使用场景：
- 叫牌已结束，出牌阶段中
- 知道所有玩家的剩余手牌
- 需要为当前玩家选择最优出牌
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from trick_taking.card import Card, Suit
from trick_taking.game_state import GameState
from trick_taking.games.spades import SpadesRules


@dataclass
class SearchNode:
    """蒙特卡洛树搜索节点"""
    state: GameState  # 当前游戏状态（深拷贝）
    parent: Optional[SearchNode] = None
    action: Optional[Card] = None  # 从父节点到达此节点的动作
    children: Dict[Card, SearchNode] = field(default_factory=dict)
    
    # MCTS统计信息
    visits: int = 0  # 访问次数
    total_value: float = 0.0  # 累计价值（团队得分差）
    
    # 扩展信息
    untried_actions: List[Card] = field(default_factory=list)  # 未尝试的动作
    
    def __repr__(self) -> str:
        return f"SearchNode(visits={self.visits}, value={self.average_value:.2f}, untried={len(self.untried_actions)})"
    
    @property
    def average_value(self) -> float:
        """节点的平均价值"""
        return self.total_value / self.visits if self.visits > 0 else 0.0
    
    @property
    def is_fully_expanded(self) -> bool:
        """是否已完全扩展（所有可能动作都已尝试）"""
        return len(self.untried_actions) == 0
    
    @property
    def is_terminal(self) -> bool:
        """是否为终止节点（游戏结束）"""
        rules = SpadesRules()
        return rules.end_trickgame(self.state)


class DoubleDummySolver:
    """
    双明手求解器
    
    使用蒙特卡洛树搜索在完全信息下寻找最优策略
    优化目标：最大化（己方队伍得分 - 对方队伍得分）
    """
    
    def __init__(self, max_iterations: int = 1000, exploration_weight: float = 1.4,
                 rollout_epsilon: float = 0.0):
        """
        初始化求解器
        
        参数:
            max_iterations: 最大搜索迭代次数
            exploration_weight: 探索权重（UCT公式中的C）
            rollout_epsilon: rollout阶段随机探索概率（0表示纯策略rollout）
        """
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight
        self.rollout_epsilon = rollout_epsilon
        self.rules = SpadesRules()
        
        # 缓存已评估的状态
        self.state_cache: Dict[str, Tuple[float, int]] = {}
    
    def solve(self, state: GameState, current_player: int) -> Dict[str, Any]:
        """
        求解当前局面，返回最优动作和评估信息
        
        参数:
            state: 当前游戏状态（必须包含所有玩家手牌信息）
            current_player: 当前需要行动的玩家ID
            
        返回:
            包含最优动作和评估信息的字典
        """
        print(f"开始求解：玩家{current_player}行动，剩余牌数={sum(len(h) for h in state.hands)}")
        
        # 获取合法动作
        hand = state.hands[current_player]
        legal_actions = self.rules.playable(state, hand, current_player)
        
        if not legal_actions:
            return self._empty_result(current_player)
        
        # 如果只有一个合法动作，直接返回
        if len(legal_actions) == 1:
            return self._single_action_result(legal_actions[0], current_player)
        
        # 创建根节点
        root_state = self._deep_copy_state(state)
        root_node = SearchNode(state=root_state)
        root_node.untried_actions = legal_actions.copy()
        
        # 执行蒙特卡洛树搜索
        start_time = time.time()
        
        for i in range(self.max_iterations):
            # 选择阶段
            node = self._tree_policy(root_node, current_player)
            
            # 模拟阶段
            reward = self._simulate(node.state, current_player)
            
            # 回溯阶段（使用根玩家视角传播 reward）
            self._backpropagate(node, reward, current_player)
            
            # 每100次迭代打印进度
            #if (i + 1) % 100 == 0:
                # print(f"  进度：{i+1}/{self.max_iterations}次迭代")
        
        elapsed_time = time.time() - start_time
        
        # 收集结果
        return self._collect_results(root_node, current_player, elapsed_time)
    
    def _tree_policy(self, node: SearchNode, root_player: int) -> SearchNode:
        """
        树策略：选择要扩展或模拟的节点
        
        使用UCT公式平衡探索和利用：
        UCT = Q(s,a) + C * sqrt(log(N(s)) / N(s,a))
        其中Q是动作的平均价值，N是访问次数
        """
        current_node = node
        
        while not current_node.is_terminal:
            if not current_node.is_fully_expanded:
                # 扩展一个未尝试的动作
                return self._expand(current_node, root_player)
            else:
                # 选择最优子节点（UCT公式）
                next_node = self._best_child(current_node, root_player)
                if next_node is None or next_node is current_node:
                    # 无法继续下探时返回当前节点，避免死循环
                    return current_node
                current_node = next_node

        return current_node
    
    def _expand(self, node: SearchNode, root_player: int) -> SearchNode:
        """
        扩展节点：选择一个未尝试的动作，创建新子节点
        """
        if not node.untried_actions:
            return node

        # 随机选择一个未尝试的动作
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        # 应用动作，创建新状态
        new_state = self._apply_action(node.state, action, node.state.turn)
        
        # 创建子节点
        child_node = SearchNode(
            state=new_state,
            parent=node,
            action=action,
            untried_actions=self._get_legal_actions(new_state, new_state.turn)
        )
        
        node.children[action] = child_node
        return child_node
    
    def _best_child(self, node: SearchNode, root_player: int) -> Optional[SearchNode]:
        """
        使用UCT公式选择最优子节点
        """
        if not node.children:
            return None
        
        # UCT公式：Q + C * sqrt(log(N_parent) / N_child)
        best_score = -float('inf')
        best_child = None

        root_team = node.state.teams[root_player]
        node_team = node.state.teams[node.state.turn]
        is_root_team_turn = node_team == root_team
        
        for action, child in node.children.items():
            if child.visits == 0:
                # 未访问的子节点优先探索
                return child

            # 所有节点统一使用根玩家视角价值。
            # 根方行动时最大化，对手行动时最小化。
            exploit = child.average_value if is_root_team_turn else -child.average_value
            explore = self.exploration_weight * math.sqrt(math.log(node.visits) / child.visits)
            score = exploit + explore
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _simulate(self, state: GameState, root_player: int) -> float:
        """
        模拟：从当前状态开始按默认策略模拟直到游戏结束
        
        返回:
            团队得分差（己方-对方）
        """
        sim_state = self._deep_copy_state(state)
        
        # 检查缓存
        state_key = self._state_key(sim_state)
        if state_key in self.state_cache:
            return self.state_cache[state_key][0]
        
        # 默认策略模拟完成剩余牌局
        max_steps = max(1, sum(len(h) for h in sim_state.hands) + 8)
        steps = 0
        while not self.rules.end_trickgame(sim_state):
            steps += 1
            if steps > max_steps:
                # 兜底保护：状态异常时避免无限循环
                break

            current_player = sim_state.turn
            hand = sim_state.hands[current_player]
            
            # 获取合法动作
            legal_actions = self.rules.playable(sim_state, hand, current_player)
            if not legal_actions:
                break

            # rollout策略：当前行动方按其目标（我方最大化/对手最小化）选择动作
            action = self._rollout_select_action(sim_state, legal_actions, current_player, root_player)
            
            # 应用动作
            self._apply_action_in_place(sim_state, action, current_player)
        
        # 计算最终得分
        scores = self.rules.score(sim_state)
        
        # SpadesRules.score 返回的是玩家视角 payoff（己方队伍分 - 对方队伍分）
        # 直接取根玩家视角即可，避免重复聚合导致数值放大。
        score_diff = scores[root_player]
        
        # 缓存结果
        self.state_cache[state_key] = (score_diff, 1)
        return score_diff
    
    def _backpropagate(self, node: SearchNode, reward: float, root_player: int) -> None:
        """
        回溯：更新从叶子节点到根节点的统计信息
        """
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            # 统一维护根玩家视角价值，避免在回溯阶段重复翻转符号。
            current_node.total_value += reward
            current_node = current_node.parent
    
    def _collect_results(self, root_node: SearchNode, current_player: int, elapsed_time: float) -> Dict[str, Any]:
        """
        收集搜索结果
        """
        # 选择平均价值最高的动作；当价值相同，用访问次数和字符串稳定打破平局。
        best_action = None
        best_value = -float('inf')
        best_visits = -1
        
        action_values = []
        
        for action, child in root_node.children.items():
            action_values.append({
                "action": action,
                "value": child.average_value,
                "visits": child.visits,
                "confidence": min(1.0, child.visits / root_node.visits) if root_node.visits > 0 else 0.0
            })
            # 根据平均价值选择最优动作；同值时优先访问更多的动作。
            if (child.average_value > best_value or
                (abs(child.average_value - best_value) < 1e-12 and child.visits > best_visits) or
                (abs(child.average_value - best_value) < 1e-12 and child.visits == best_visits and
                 (best_action is None or str(action) < str(best_action)))):
                best_value = child.average_value
                best_visits = child.visits
                best_action = action
        
        # 如果没有子节点，选择第一个合法动作
        if best_action is None and root_node.untried_actions:
            best_action = root_node.untried_actions[0]
        
        # 计算局面整体评估（与搜索回报同一量纲）
        state_evaluation = self._evaluate_state(root_node, current_player)
        
        return {
            "best_action": best_action,
            "action_values": sorted(action_values, key=lambda x: x["value"], reverse=True),
            "state_evaluation": state_evaluation,
            "search_statistics": {
                "iterations": self.max_iterations,
                "time_elapsed": elapsed_time,
                "nodes_expanded": self._count_nodes(root_node),
                "cache_hits": sum(hits for _, hits in self.state_cache.values())
            }
        }
    
    # ─── 辅助方法 ────────────────────────────────────────────────────
    
    def _deep_copy_state(self, state: GameState) -> GameState:
        """
        深度复制游戏状态
        
        注意：这不是完全的深拷贝，但足以用于搜索
        """
        # 创建新状态对象
        new_state = GameState()
        
        # 复制基本属性
        new_state.num_players = state.num_players
        new_state.phase = state.phase
        new_state.dealer_seat = state.dealer_seat
        new_state.current_bidder = state.current_bidder
        new_state.turn = state.turn
        new_state.trick_leader = state.trick_leader
        new_state.trump_suit = state.trump_suit
        new_state.declaration = state.declaration
        new_state.declarer = state.declarer
        new_state.trump_broken = state.trump_broken
        new_state.spades_broken = state.spades_broken
        new_state.tricks_played = state.tricks_played
        new_state.played_bitset = state.played_bitset
        new_state.round_number = state.round_number
        
        # 深拷贝手牌
        new_state.hands = [hand.copy() for hand in state.hands]
        new_state.hand_bitsets = state.hand_bitsets.copy()
        new_state.dog = state.dog.copy()
        new_state.all_cards = state.all_cards.copy()
        
        # 深拷贝其他列表
        new_state.bids = state.bids.copy()
        new_state.max_bid = state.max_bid.copy()
        new_state.teams = state.teams.copy()
        new_state.table_cards = state.table_cards.copy()
        new_state.tricks_won = state.tricks_won.copy()
        new_state.cards_won = [cards.copy() for cards in state.cards_won]
        new_state.points = state.points.copy()
        new_state.trick_history = state.trick_history.copy()
        
        return new_state
    
    def _apply_action(self, state: GameState, action: Card, player_id: int) -> GameState:
        """
        应用动作并返回新状态（不修改原状态）
        """
        new_state = self._deep_copy_state(state)
        self._apply_action_in_place(new_state, action, player_id)
        return new_state
    
    def _apply_action_in_place(self, state: GameState, action: Card, player_id: int) -> None:
        """
        在原地应用动作（修改状态）
        """
        # 打出牌
        state.play_card_to_table(player_id, action)
        
        # 更新黑桃破禁状态
        if action.suit == Suit.SPADES:
            state.spades_broken = True
            state.trump_broken = True
        
        # 更新当前玩家
        state.turn = (player_id + 1) % state.num_players

        # 当前墩满4张后立即结墩并切到赢家，保证状态可持续推进
        if state.trick_complete:
            winner = self._determine_trick_winner(state)
            state.complete_trick(winner)
            state.trick_leader = winner
            state.turn = winner
    
    def _get_legal_actions(self, state: GameState, player_id: int) -> List[Card]:
        """获取合法动作"""
        hand = state.hands[player_id]
        return self.rules.playable(state, hand, player_id)
    
    def _determine_trick_winner(self, state: GameState) -> int:
        """确定当前墩的赢家"""
        if not state.table_cards:
            return state.trick_leader
        
        # 检查是否有黑桃
        spades_cards = [(pid, card) for pid, card in state.table_cards if card.suit == Suit.SPADES]
        
        if spades_cards:
            # 有黑桃：黑桃中点数最大者赢
            winner_pid, winner_card = max(spades_cards, key=lambda x: x[1].rank.value)
        else:
            # 无黑桃：首攻花色中点数最大者赢
            lead_suit = state.table_cards[0][1].suit
            suit_cards = [(pid, card) for pid, card in state.table_cards if card.suit == lead_suit]
            winner_pid, winner_card = max(suit_cards, key=lambda x: x[1].rank.value)
        
        return winner_pid
    
    def _state_key(self, state: GameState) -> str:
        """生成状态缓存键"""
        # 使用手牌位图 + 完整桌面牌 + 回合信息，避免不同局面误命中缓存。
        key_parts = []
        key_parts.append(f"hands:{state.hand_bitsets}")
        table_repr = ",".join(f"{pid}:{card}" for pid, card in state.table_cards)
        key_parts.append(f"table:{table_repr}")
        key_parts.append(f"tricks:{state.tricks_won}")
        key_parts.append(f"broken:{int(state.spades_broken)}")
        key_parts.append(f"leader:{state.trick_leader}")
        key_parts.append(f"turn:{state.turn}")
        return "|".join(key_parts)

    def _rollout_select_action(
        self,
        state: GameState,
        legal_actions: List[Card],
        current_player: int,
        root_player: int,
    ) -> Card:
        """rollout默认策略：一步前瞻贪心，等值时随机+牌面偏好打破平局。

        对nil玩家偏好低牌（避免意外赢墩），对非nil偏好高牌（争取赢墩）。
        """
        if len(legal_actions) == 1:
            return legal_actions[0]

        if self.rollout_epsilon > 0.0 and random.random() < self.rollout_epsilon:
            return random.choice(legal_actions)

        root_team = state.teams[root_player]
        acting_team = state.teams[current_player]
        maximize = acting_team == root_team

        is_nil = state.max_bid[current_player] in ('nil', 'blind_nil')
        # nil玩家偏好低牌（rank.value越小越好），非nil偏好高牌
        card_bias = -1.0 if is_nil else 1.0

        best_actions = [legal_actions[0]]
        best_value = -float('inf') if maximize else float('inf')

        for action in legal_actions:
            next_state = self._apply_action(state, action, current_player)
            value = self._rollout_state_value(next_state, root_player)

            # 域特定启发式：避免己方玩家在非黑桃领出时浪费高花将牌（黑桃）。
            # 在一步前瞻中，用将牌赢墩看起来很好（分值变高），但长远看
            # 可能因为浪费了高花将牌导致后面输掉关键墩或nil失败。
            # 惩罚量需要大于一步前瞻中"赢墩 vs 输墩"的表观差异。
            # 赢墩使trick_diff不变（own_tricks↑, opp_tricks不变），
            # 输墩使trick_diff-1（own_tricks不变, opp_tricks↑），
            # 差值约为2*10=20分。因此惩罚应 > 2*10=20才能改变决策。
            if (maximize and not is_nil
                    and action.suit == Suit.SPADES
                    and state.table_cards
                    and state.table_cards[0][1].suit != Suit.SPADES):
                value -= 2 * action.rank.value  # SA=28, SQ=24, S2=4

            # 等值时加入微小牌面偏好（不影响主要信号）
            value += action.rank.value * card_bias * 1e-6

            better = value > best_value if maximize else value < best_value
            tied = abs(value - best_value) < 1e-12

            if better:
                best_value = value
                best_actions = [action]
            elif tied:
                best_actions.append(action)

        return random.choice(best_actions)

    def _rollout_state_value(self, state: GameState, root_player: int) -> float:
        """rollout中一步前瞻的快速估值（统一根玩家视角）。

        对终局使用精确得分函数。对非终局不使用 score() 是因为当前得分
        包含合约/nil奖分，这些奖分在后续出牌中可能无法实现（如当前nil看似
        成功但后续可能意外赢墩），会严重误导一步前瞻贪心决策。
        """
        if self.rules.end_trickgame(state):
            scores = self.rules.score(state)
            return scores[root_player]

        # 非终局：只使用赢墩差 × 每墩期望分值的粗估计
        root_team = state.teams[root_player]
        own_tricks = sum(state.tricks_won[pid] for pid in range(4) if state.teams[pid] == root_team)
        opp_tricks = sum(state.tricks_won[pid] for pid in range(4) if state.teams[pid] != root_team)
        return 10.0 * (own_tricks - opp_tricks)
    
    def _evaluate_state(self, root_node: SearchNode, current_player: int) -> Dict[str, float]:
        """评估当前局面（基于搜索统计，而非独立启发式）。"""
        if root_node.visits == 0:
            return {
                "expected_score_diff": 0.0,
                "team_win_probability": 0.5,
                "certainty": 0.0,
            }

        expected_score_diff = root_node.average_value

        # 使用已探索动作中正收益占比作为胜率近似。
        child_values = [child.average_value for child in root_node.children.values() if child.visits > 0]
        if child_values:
            win_ratio = sum(1 for v in child_values if v > 0) / len(child_values)
            certainty = min(1.0, root_node.visits / max(1, self.max_iterations))
        else:
            win_ratio = 0.5
            certainty = 0.0

        return {
            "expected_score_diff": expected_score_diff,
            "team_win_probability": win_ratio,
            "certainty": certainty,
        }
    
    def _count_nodes(self, node: SearchNode) -> int:
        """统计搜索树中的节点数量"""
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count
    
    def _empty_result(self, current_player: int) -> Dict[str, Any]:
        """返回空结果"""
        return {
            "best_action": None,
            "action_values": [],
            "state_evaluation": {
                "expected_score_diff": 0.0,
                "team_win_probability": 0.5,
                "certainty": 0.0
            },
            "search_statistics": {
                "iterations": 0,
                "time_elapsed": 0.0,
                "nodes_expanded": 0,
                "cache_hits": 0
            }
        }
    
    def _single_action_result(self, action: Card, current_player: int) -> Dict[str, Any]:
        """只有一个合法动作时的结果"""
        return {
            "best_action": action,
            "action_values": [{
                "action": action,
                "value": 0.0,
                "visits": 1,
                "confidence": 1.0
            }],
            "state_evaluation": {
                "expected_score_diff": 0.0,
                "team_win_probability": 0.5,
                "certainty": 1.0
            },
            "search_statistics": {
                "iterations": 0,
                "time_elapsed": 0.0,
                "nodes_expanded": 1,
                "cache_hits": 0
            }
        }


def create_test_state() -> GameState:
    """创建测试游戏状态"""
    from trick_taking.deck import Deck, STANDARD_52
    
    # 创建牌堆
    deck = Deck(STANDARD_52)
    
    # 发牌
    hands = []
    for i in range(4):
        hand = deck.deal(13)
        hands.append(hand)
    
    # 创建游戏状态
    state = GameState()
    state.init_for_deal(4, hands, [], deck.all_cards)
    
    # 设置叫牌（示例：玩家0叫3，玩家1叫4，玩家2叫0，玩家3叫2）
    state.bids = []
    state.max_bid = ["bid_3", "bid_4", "nil", "bid_2"]
    
    # 设置队伍（0&2 vs 1&3）
    state.teams = [0, 1, 0, 1]
    
    # 设置出牌阶段
    state.phase = state.phase.PLAYING
    state.turn = 0  # 玩家0行动
    state.trick_leader = 0
    
    return state


def main():
    """测试主函数"""
    print("测试双明手求解器...")
    
    # 创建测试状态
    state = create_test_state()
    
    # 创建求解器
    solver = DoubleDummySolver(max_iterations=500)  # 减少迭代次数以加速测试
    
    # 求解
    result = solver.solve(state, current_player=0)
    
    # 打印结果
    print(f"\n最优动作: {result['best_action']}")
    print(f"\n动作评估:")
    for action_info in result['action_values'][:5]:  # 只显示前5个
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