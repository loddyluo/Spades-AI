"""
黑桃王（Spades）精确双明手求解器

实现原理：
1. 双明手求解器拥有完全信息（知道所有玩家的手牌）
2. 使用极小极大搜索（alpha-beta剪枝）寻找最优策略
3. 优化目标：最大化（己方队伍得分 - 对方队伍得分）
4. 支持叫牌阶段后的任意局面（0-51张牌已打出）

使用场景：
- 叫牌已结束，出牌阶段中
- 知道所有玩家的剩余手牌
- 需要计算双方最优玩法下的最终得分差

模块函数说明（输入/输出）:

- ExactDoubleDummySolver.solve(state: GameState) -> float
    输入: `GameState`（处于出牌阶段，包含所有玩家手牌）
    输出: float，表示在完全信息下双方最优策略时的得分差（队伍0得分 - 队伍1得分）

- ExactDoubleDummySolver.solve_with_q(state: GameState) -> dict
    输入: 同上
    输出: 字典，包括键 `value`(float), `best_action`(Card|None), `action_q_values`(Dict[Card,float]),
                `action_values`(list), `current_player`(int), `optimize_for_team`(int)

- 该模块同时提供 Python 参考实现和对 C++ 原生实现的运行时切换，
    当需要对照或回归测试时可使用 Python 实现。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trick_taking.card import Card, Suit
from trick_taking.game_state import GameState
from trick_taking.games.spades import SpadesRules


# 转置表标签
TT_EXACT = 0       # 精确值（所有子节点搜索完毕）
TT_LOWER_BOUND = 1  # 上界（MAX节点被beta剪枝，真实值 >= 缓存值）
TT_UPPER_BOUND = 2  # 下界（MIN节点被alpha剪枝，真实值 <= 缓存值）


def expand_equivalent_root_q_values(
    state: GameState,
    action_q_values: Mapping[int, float],
    legal_actions: Sequence[Card] | None = None,
) -> Dict[int, float]:
    """Expand native root-Q representatives to every legal card.

    The native solver deliberately removes strategically equivalent lower
    cards at the root.  That is ideal for search, but replay diagnostics need
    one Q value for every action the player could click.  Cards in one native
    equivalence class have the same continuation value, so omitted cards can
    safely inherit the retained (highest-card) representative's Q value.
    """
    current_player = int(state.turn)
    if legal_actions is None:
        legal_actions = SpadesRules().playable(
            state,
            state.hands[current_player],
            current_player,
        )

    expanded = {int(card_id): float(q) for card_id, q in action_q_values.items()}
    legal_ids = {card.card_id for card in legal_actions}
    if not legal_ids or legal_ids.issubset(expanded):
        return {card_id: expanded[card_id] for card_id in legal_ids if card_id in expanded}

    blocking_ids = {
        card.card_id
        for player_id, hand in enumerate(state.hands)
        if player_id != current_player
        for card in hand
    }
    blocking_ids.update(card.card_id for _, card in state.table_cards)

    for suit in Suit:
        own_cards = sorted(
            (card for card in state.hands[current_player] if card.suit == suit),
            key=lambda card: card.rank.value,
            reverse=True,
        )
        if not own_cards:
            continue

        representative = own_cards[0]
        if representative.card_id in legal_ids and representative.card_id in expanded:
            expanded[representative.card_id] = float(expanded[representative.card_id])

        for card in own_cards[1:]:
            between_is_blocked = any(
                suit.value * 13 + (rank_value - 2) in blocking_ids
                for rank_value in range(card.rank.value + 1, representative.rank.value)
            )
            if between_is_blocked:
                representative = card

            if card.card_id not in legal_ids:
                continue
            representative_q = expanded.get(representative.card_id)
            if representative_q is not None:
                expanded[card.card_id] = float(representative_q)

    return {card_id: expanded[card_id] for card_id in legal_ids if card_id in expanded}


class ExactDoubleDummySolver:
    """
    精确双明手求解器

    使用极小极大搜索在完全信息下计算最优得分差
    优化目标：最大化（己方队伍得分 - 对方队伍得分）
    假设队伍0为己方（玩家0和2），队伍1为对方（玩家1和3）
    """

    def __init__(self):
        """初始化求解器"""
        self.rules = SpadesRules()
        # 转置表：{hash: (value, flag, verify)}，flag指示值是精确值还是边界
        # verify = hand_bitsets之和，用于检测哈希碰撞
        self.tt: Dict[int, Tuple[float, int, int]] = {}
        self._cpp_solver = None

    def _get_cpp_solver(self):
        """懒加载 C++ 求解器，避免模块导入循环。"""
        if self._cpp_solver is None:
            from trick_taking.solvers.exact_double_dummy_cpp_fastest import ExactDoubleDummyCppFastestSolver

            self._cpp_solver = ExactDoubleDummyCppFastestSolver()
        return self._cpp_solver

    def solve(self, state: GameState) -> float:
        """
        求解当前局面，返回最优得分差（队伍0得分 - 队伍1得分）

        兼容性说明：
        - 该接口保持原始返回类型 float，不破坏现有调用方。
        - 若需要 best_action 与 action->Q，请使用 solve_with_q。

        参数:
            state: 当前游戏状态（必须包含所有玩家手牌信息）

        返回:
            双方都按最优策略时的最终得分差
        """
        # 默认入口切换到 C++ 实现；子类可以复用 Python 实现。
        if type(self) is ExactDoubleDummySolver:
            return self._get_cpp_solver().solve(state)
        return self._solve_python(state)

    def solve_with_q(self, state: GameState) -> Dict[str, Any]:
        """
        求解当前局面并返回动作Q值信息。

        返回字段：
        - value: float，当前状态最优值（队伍0视角）
        - best_action: Card | None，当前行动方的最优动作
        - action_q_values: Dict[Card, float]，每个合法动作的Q值（队伍0视角）
        - action_values: list[dict]，便于日志打印/序列化的动作列表
        - current_player: int，当前行动玩家
        - optimize_for_team: int，当前行动方队伍（0=最大化，1=最小化）
        """
        # 默认入口切换到 C++ 实现；子类可以复用 Python 实现。
        if type(self) is ExactDoubleDummySolver:
            return self._get_cpp_solver().solve_with_q(state)
        return self._solve_with_q_python(state)

    def _solve_python(self, state: GameState) -> float:
        """Python 参考实现（保留用于回归测试和对照）。"""
        self._validate_state(state)
        score_diff = self._minimax(state, alpha=-float('inf'), beta=float('inf'))
        return score_diff

    def _solve_with_q_python(self, state: GameState) -> Dict[str, Any]:
        """Python 参考实现：返回根节点动作 Q 值。"""
        self._validate_state(state)

        if self.rules.end_trickgame(state):
            terminal_value = self._compute_score_diff(state)
            return {
                "value": terminal_value,
                "best_action": None,
                "action_q_values": {},
                "action_values": [],
                "current_player": state.turn,
                "optimize_for_team": state.teams[state.turn],
            }

        current_player = state.turn
        current_team = state.teams[current_player]
        legal_actions = self.rules.playable(state, state.hands[current_player], current_player)

        if not legal_actions:
            value = self._minimax(state, alpha=-float('inf'), beta=float('inf'))
            return {
                "value": value,
                "best_action": None,
                "action_q_values": {},
                "action_values": [],
                "current_player": current_player,
                "optimize_for_team": current_team,
            }

        action_q_values: Dict[Card, float] = {}
        best_action = None
        best_value = -float('inf') if current_team == 0 else float('inf')

        for action in legal_actions:
            next_state = self._apply_action(state, action, current_player)
            q_value = self._minimax(next_state, alpha=-float('inf'), beta=float('inf'))
            action_q_values[action] = q_value

            if current_team == 0:
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
            else:
                if q_value < best_value:
                    best_value = q_value
                    best_action = action

        action_values = [{"action": action, "q_value": q} for action, q in action_q_values.items()]
        action_values.sort(key=lambda x: x["q_value"], reverse=(current_team == 0))

        return {
            "value": best_value,
            "best_action": best_action,
            "action_q_values": action_q_values,
            "action_values": action_values,
            "current_player": current_player,
            "optimize_for_team": current_team,
        }

    def _validate_state(self, state: GameState) -> None:
        """统一状态校验，供 solve / solve_with_q 复用。"""
        # 验证状态：叫牌已结束，处于出牌阶段
        if state.phase != state.phase.PLAYING:
            raise ValueError("求解器只适用于出牌阶段")

        # 验证队伍分配
        if state.teams != [0, 1, 0, 1]:
            raise ValueError("队伍分配必须为[0, 1, 0, 1]")

    def _minimax(self, state: GameState, alpha: float, beta: float) -> float:
        """
        极小极大搜索（alpha-beta剪枝）

        返回:
            当前状态下，双方最优玩法时的得分差（队伍0视角）
        """
        # 检查游戏是否结束
        if self.rules.end_trickgame(state):
            return self._compute_score_diff(state)

        # 转置表查找：带边界类型检查和碰撞检测
        state_hash = self._state_hash(state)
        if state_hash in self.tt:
            cached_value, tt_flag, verify = self.tt[state_hash]
            # 碰撞检测：验证 hand_bitsets 一致
            if verify == self._tt_verify(state):
                if tt_flag == TT_EXACT:
                    return cached_value
                if tt_flag == TT_LOWER_BOUND and cached_value >= beta:
                    return cached_value
                if tt_flag == TT_UPPER_BOUND and cached_value <= alpha:
                    return cached_value
        current_player = state.turn
        hand = state.hands[current_player]
        legal_actions = self.rules.playable(state, hand, current_player)

        # 如果没有合法动作（不应该发生），返回当前得分差
        if not legal_actions:
            return self._compute_score_diff(state)

        # 确定当前玩家所属队伍
        current_team = state.teams[current_player]

        # 队伍0最大化，队伍1最小化
        pruned = False
        if current_team == 0:
            value = -float('inf')
            for action in legal_actions:
                new_state = self._apply_action(state, action, current_player)
                child_value = self._minimax(new_state, alpha, beta)
                value = max(value, child_value)
                alpha = max(alpha, value)
                if value >= beta:
                    pruned = True
                    break  # beta剪枝
            # MAX节点被剪枝 → 缓存为LOWER_BOUND（真实值 >= value）
            if pruned:
                self.tt[state_hash] = (value, TT_LOWER_BOUND, self._tt_verify(state))
            else:
                self.tt[state_hash] = (value, TT_EXACT, self._tt_verify(state))
        else:  # current_team == 1
            value = float('inf')
            for action in legal_actions:
                new_state = self._apply_action(state, action, current_player)
                child_value = self._minimax(new_state, alpha, beta)
                value = min(value, child_value)
                beta = min(beta, value)
                if value <= alpha:
                    pruned = True
                    break  # alpha剪枝
            # MIN节点被剪枝 → 缓存为UPPER_BOUND（真实值 <= value）
            if pruned:
                self.tt[state_hash] = (value, TT_UPPER_BOUND, self._tt_verify(state))
            else:
                self.tt[state_hash] = (value, TT_EXACT, self._tt_verify(state))

        return value

    def _compute_score_diff(self, state: GameState) -> float:
        """
        计算当前状态的得分差（队伍0得分 - 队伍1得分）

        使用SpadesRules.score方法，取玩家0的payoff（己方队伍分 - 对方队伍分）
        """
        scores = self.rules.score(state)
        # 玩家0属于队伍0，其payoff就是队伍0得分 - 队伍1得分
        return scores[0]

    def _apply_action(self, state: GameState, action: Card, player_id: int) -> GameState:
        """
        应用动作并返回新状态（深拷贝）

        参数:
            state: 原始状态
            action: 要打出的牌
            player_id: 出牌玩家

        返回:
            新状态
        """
        # 深拷贝状态
        new_state = self._deep_copy_state(state)

        # 打出牌
        new_state.play_card_to_table(player_id, action)

        # 更新黑桃破禁状态
        if action.suit == Suit.SPADES:
            new_state.spades_broken = True
            new_state.trump_broken = True

        # 更新当前玩家
        new_state.turn = (player_id + 1) % new_state.num_players

        # 当前墩满4张后立即结墩并切到赢家
        if new_state.trick_complete:
            winner = self._determine_trick_winner(new_state)
            new_state.complete_trick(winner)
            new_state.trick_leader = winner
            new_state.turn = winner

        return new_state

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

    def _state_hash(self, state: GameState) -> int:
        """
        生成状态哈希值

        使用手牌位图、桌面牌、回合、赢墩数等关键信息
        """
        hash_val = 0
        for bitset in state.hand_bitsets:
            hash_val ^= bitset
            hash_val = self._rotate(hash_val, 13)

        for pid, card in state.table_cards:
            hash_val ^= (pid << 16) | card.bit
            hash_val = self._rotate(hash_val, 7)

        hash_val ^= state.turn
        hash_val = self._rotate(hash_val, 11)
        hash_val ^= state.trick_leader << 8
        hash_val = self._rotate(hash_val, 11)
        hash_val ^= state.tricks_played << 16
        hash_val = self._rotate(hash_val, 11)
        hash_val ^= state.spades_broken << 24
        hash_val = self._rotate(hash_val, 11)

        # 赢墩数也参与旋转，避免与手牌哈希抵消
        for i, tricks in enumerate(state.tricks_won):
            hash_val ^= (tricks << (i * 4))
            hash_val = self._rotate(hash_val, 13)

        return hash_val & ((1 << 64) - 1)

    @staticmethod
    def _rotate(x: int, n: int) -> int:
        """64位循环左移"""
        return ((x << n) | (x >> (64 - n))) & ((1 << 64) - 1)

    def _tt_verify(self, state: GameState) -> int:
        """
        转置表碰撞检测验证码

        使用与_state_hash不同的mix，降低碰撞概率
        """
        v = 0
        for bitset in state.hand_bitsets:
            v ^= bitset
            v = self._rotate(v, 17)
        for t in state.tricks_won:
            v ^= t
            v = self._rotate(v, 19)
        v ^= state.tricks_played
        v = self._rotate(v, 23)
        v ^= state.turn
        return v & ((1 << 64) - 1)  # 限制为64位


def test_solver():
    """测试求解器"""
    from trick_taking.deck import Deck, STANDARD_52
    from trick_taking.utils.state_tools import create_state_from_hands
    print("测试精确双明手求解器...")

    # 创建一个简单的测试牌局，每玩家只有2张牌
    # 使用Deck发牌，但只取前8张牌
    deck = Deck(STANDARD_52)
    all_cards = deck.all_cards[:8]  # 取前8张牌
    hands = [all_cards[i*2:(i+1)*2] for i in range(4)]  # 每玩家2张

    # 补齐到13张牌（用虚拟牌填充，但实际不会用到，因为游戏会提前结束）
    # 我们修改状态，使游戏提前结束：设置tricks_played为12，只剩最后1墩
    bids = ["bid_1", "bid_1", "bid_1", "bid_1"]

    try:
        state = create_state_from_hands(hands, bids)
        # 修改状态，模拟已经完成12墩，只剩最后4张牌
        state.tricks_played = 12
        # 设置当前回合为玩家0
        state.turn = 0
        state.trick_leader = 0

        solver = ExactDoubleDummySolver()
        score_diff = solver.solve(state)
        print(f"最优得分差（队伍0 - 队伍1）: {score_diff}")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_solver()


class ExactDoubleDummyPythonSolver(ExactDoubleDummySolver):
    """保留 Python 参考求解器，专用于正确性回归对照。"""

    def solve(self, state: GameState) -> float:
        return self._solve_python(state)

    def solve_with_q(self, state: GameState) -> Dict[str, Any]:
        return self._solve_with_q_python(state)
