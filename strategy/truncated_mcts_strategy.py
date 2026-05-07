"""截断式 MCTS 出牌策略。

文件作用：
- 在剩余牌数不超过 30 张时，直接调用精确求解器输出最优动作。
- 在剩余牌数大于 30 张时，运行一个全新的 PUCT/MCTS 搜索：
  - 搜索到剩余牌数 <= leaf_threshold 时，使用 value-head 进行局面估值；
  - 使用 policy-head 提供先验概率；
  - 使用队伍 0 视角的价值进行回传与动作比较；
  - 根节点动作选择时，队伍 0 取 argmax，队伍 1 取 argmin。

函数/类输入输出说明：
- TruncatedMCTSConfig:
    输入字段:
      - exact_threshold: int，剩余牌数 <= 该值时直接精确求解
      - leaf_threshold: int，MCTS 搜索到该剩余牌数时接入 MLP 估值
      - simulations_per_action: int，每个根动作模拟次数
      - exploration_constant: float，PUCT 探索系数
      - policy_temperature: float，policy head 温度
      - value_scale: float，模型输出缩放回真实 value_view 的倍率
      - checkpoint_path: str | None，MLP 权重文件路径
    输出: 配置对象本身

- TruncatedMCTSStrategy.choose_action(state: GameState) -> Card | None
    输入: 完整的 Spades 牌局状态（叫牌结束、出牌阶段、全信息）
    输出: 当前应出的最优牌；若无合法动作则返回 None

- TruncatedMCTSStrategy.play_full_game(state: GameState) -> list[Card]
    输入: 可执行到终局的完整牌局状态
    输出: 按出牌顺序排列的动作序列

- TruncatedMCTSStrategy.choose_action_with_info(state: GameState) -> dict
    输入: 同上
    输出: 包含 best_action、best_value、mode、action_scores、root_team 等信息的字典
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

from mlp.mlp_model import DoubleDummyMLP
from trick_taking.card import Card, Suit
from trick_taking.game_state import GameState
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver
from trick_taking.utils.feature_encoder import SpadesFeatureEncoder


@dataclass
class TruncatedMCTSConfig:
    """截断 MCTS 的参数配置。"""

    exact_threshold: int = 30
    leaf_threshold: int = 24
    simulations_per_action: int = 5
    exploration_constant: float = 1.5
    policy_temperature: float = 1.0
    value_scale: float = 25.0
    checkpoint_path: str | None = None


@dataclass
class SearchNode:
    """MCTS 树节点。

    输入:
    - state: 当前节点对应的 GameState
    - parent: 父节点，根节点为 None
    - action_from_parent: 从父节点到达当前节点所用的动作
    - prior: 该动作的先验概率

    输出:
    - 节点内部统计信息（访问次数、累积价值、孩子节点、未展开动作）
    """

    state: GameState
    parent: "SearchNode | None" = None
    action_from_parent: Card | None = None
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[Card, "SearchNode"] = field(default_factory=dict)
    unexpanded_actions: list[Card] = field(default_factory=list)
    action_priors: dict[int, float] = field(default_factory=dict)

    @property
    def average_value(self) -> float:
        """返回节点的平均价值（队伍 0 视角）。"""
        return self.value_sum / self.visits if self.visits > 0 else 0.0


class TruncatedMCTSStrategy:
    """基于 PUCT 的截断式出牌策略。"""

    def __init__(self, config: TruncatedMCTSConfig | None = None) -> None:
        self.config = config or TruncatedMCTSConfig()
        self.rules = SpadesRules()
        self.exact_solver = ExactDoubleDummySolver()
        self.encoder = SpadesFeatureEncoder()
        self.model = self._load_model(self.config.checkpoint_path)

    def _load_model(self, checkpoint_path: str | None) -> DoubleDummyMLP | None:
        """加载 value/policy 双头模型。

        输入:
        - checkpoint_path: 权重文件路径；为 None 时返回 None，表示不使用 MLP。

        输出:
        - DoubleDummyMLP 或 None
        """
        if not checkpoint_path:
            return None
        model = DoubleDummyMLP(input_dim=self.encoder.total_dim)
        model.load(checkpoint_path)
        return model

    def choose_action(self, state: GameState) -> Card | None:
        """选择当前状态的最优动作。"""
        remaining_cards = self._remaining_cards(state)
        legal_actions = self._legal_actions(state)
        if not legal_actions:
            return None
        if len(legal_actions) == 1:
            return legal_actions[0]

        # 30张及以下直接用精确求解器。
        if remaining_cards <= self.config.exact_threshold:
            result = self.exact_solver.solve_with_q(state)
            best_action = result["best_action"]
            if best_action is not None:
                return best_action
            return legal_actions[0]

        info = self.choose_action_with_info(state)
        return info["best_action"]

    def choose_action_with_info(self, state: GameState) -> dict[str, Any]:
        """返回带搜索统计的动作选择结果。"""
        remaining_cards = self._remaining_cards(state)
        legal_actions = self._legal_actions(state)
        if not legal_actions:
            return {
                "best_action": None,
                "best_value": 0.0,
                "mode": "no_legal_action",
                "root_team": self._current_team(state),
                "action_scores": [],
            }

        if len(legal_actions) == 1:
            return {
                "best_action": legal_actions[0],
                "best_value": 0.0,
                "mode": "single_action",
                "root_team": self._current_team(state),
                "action_scores": [{"action": legal_actions[0], "value": 0.0, "visits": 0}],
            }

        if remaining_cards <= self.config.exact_threshold:
            exact_result = self.exact_solver.solve_with_q(state)
            action_scores = [
                {"action": action, "value": float(value)}
                for action, value in exact_result["action_q_values"].items()
            ]
            root_team = self._current_team(state)
            action_scores.sort(key=lambda item: item["value"], reverse=(root_team == 0))
            return {
                "best_action": exact_result["best_action"],
                "best_value": float(exact_result["value"]),
                "mode": "exact",
                "root_team": root_team,
                "action_scores": action_scores,
            }

        root_team = self._current_team(state)
        root_scores: list[dict[str, Any]] = []

        for action in legal_actions:
            child_state = self._apply_action(state, action)
            child_node = self._build_root_child(child_state, action)
            for _ in range(self.config.simulations_per_action):
                self._run_simulation(child_node)
            root_scores.append(
                {
                    "action": action,
                    "value": child_node.average_value,
                    "visits": child_node.visits,
                }
            )

        best_score = root_scores[0]
        for item in root_scores[1:]:
            if root_team == 0:
                if item["value"] > best_score["value"]:
                    best_score = item
            else:
                if item["value"] < best_score["value"]:
                    best_score = item

        root_scores.sort(key=lambda item: item["value"], reverse=(root_team == 0))
        return {
            "best_action": best_score["action"],
            "best_value": float(best_score["value"]),
            "mode": "mcts",
            "root_team": root_team,
            "action_scores": root_scores,
        }

    def play_full_game(self, state: GameState) -> list[Card]:
        """从当前局面开始，一直出到终局，返回完整动作序列。"""
        actions: list[Card] = []
        safety = 0
        while not self.rules.end_trickgame(state):
            action = self.choose_action(state)
            if action is None:
                break
            actions.append(action)
            self._apply_action_in_place(state, action)
            safety += 1
            if safety > 52:
                raise RuntimeError("出牌序列超过 52 步，状态推进可能有误")
        return actions

    def _build_root_child(self, state: GameState, action: Card) -> SearchNode:
        """把根动作后的状态包装为搜索树根子节点。"""
        node = SearchNode(state=state, action_from_parent=action, prior=1.0)
        node.unexpanded_actions = self._legal_actions(state)
        node.action_priors = self._policy_priors(state, node.unexpanded_actions)
        node.unexpanded_actions.sort(key=lambda card: node.action_priors.get(card.card_id, 0.0), reverse=True)
        return node

    def _run_simulation(self, root_node: SearchNode) -> float:
        """从给定节点向下做一次 PUCT 搜索，并回传队伍 0 视角价值。"""
        path: list[SearchNode] = [root_node]
        node = root_node

        while True:
            if self._is_terminal(node.state):
                value = self._terminal_value(node.state)
                break

            remaining_cards = self._remaining_cards(node.state)
            if remaining_cards <= self.config.leaf_threshold:
                value = self._leaf_value(node.state)
                break

            if not node.unexpanded_actions:
                node.unexpanded_actions = self._legal_actions(node.state)
                node.action_priors = self._policy_priors(node.state, node.unexpanded_actions)
                node.unexpanded_actions.sort(
                    key=lambda card: node.action_priors.get(card.card_id, 0.0),
                    reverse=True,
                )

            if node.unexpanded_actions:
                action = node.unexpanded_actions.pop(0)
                child_state = self._apply_action(node.state, action)
                child = SearchNode(
                    state=child_state,
                    parent=node,
                    action_from_parent=action,
                    prior=node.action_priors.get(action.card_id, 1.0),
                )
                child.unexpanded_actions = self._legal_actions(child_state)
                child.action_priors = self._policy_priors(child_state, child.unexpanded_actions)
                child.unexpanded_actions.sort(
                    key=lambda card: child.action_priors.get(card.card_id, 0.0),
                    reverse=True,
                )
                node.children[action] = child
                path.append(child)
                node = child
                continue

            child = self._select_child_puct(node)
            if child is None:
                value = self._leaf_value(node.state)
                break

            path.append(child)
            node = child

        for visited_node in path:
            visited_node.visits += 1
            visited_node.value_sum += value
        return value

    def _select_child_puct(self, node: SearchNode) -> SearchNode | None:
        """按 PUCT 选择最值得继续探索的子节点。"""
        if not node.children:
            return None

        current_team = self._current_team(node.state)
        sign = 1.0 if current_team == 0 else -1.0
        sqrt_total = math.sqrt(max(1, node.visits))

        best_child = None
        best_score = -float("inf")

        for action, child in node.children.items():
            q_value = child.average_value
            prior = child.prior
            exploration = self.config.exploration_constant * prior * sqrt_total / (1.0 + child.visits)
            score = sign * q_value + exploration
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _policy_priors(self, state: GameState, legal_actions: list[Card]) -> dict[int, float]:
        """从 policy head 提取合法动作先验概率。

        若没有加载模型，则返回均匀分布。
        """
        if not legal_actions:
            return {}

        if self.model is None:
            prob = 1.0 / len(legal_actions)
            return {action.card_id: prob for action in legal_actions}

        logits = self.model.predict_policy_logits(self.encoder.encode(state, state.turn))
        if logits.ndim > 1:
            logits = logits[0]

        action_logits = []
        for action in legal_actions:
            action_logits.append((action, float(logits[action.card_id])))

        if self.config.policy_temperature <= 1e-8:
            best_action = max(action_logits, key=lambda item: item[1])[0]
            return {action.card_id: (1.0 if action is best_action else 0.0) for action, _ in action_logits}

        scaled = [value / self.config.policy_temperature for _, value in action_logits]
        max_logit = max(scaled)
        exp_values = [math.exp(value - max_logit) for value in scaled]
        total = sum(exp_values)
        if total <= 0.0:
            prob = 1.0 / len(legal_actions)
            return {action.card_id: prob for action in legal_actions}

        priors: dict[int, float] = {}
        for (action, _), exp_value in zip(action_logits, exp_values):
            priors[action.card_id] = exp_value / total
        return priors

    def _leaf_value(self, state: GameState) -> float:
        """在 leaf_threshold 处使用 MLP 估值，并换算到队伍 0 视角。"""
        if self._is_terminal(state):
            return self._terminal_value(state)

        if self.model is None:
            # 没有模型时退化成轻量启发式：用当前已赢墩差近似。
            team0_tricks = state.tricks_won[0] + state.tricks_won[2]
            team1_tricks = state.tricks_won[1] + state.tricks_won[3]
            return float(team0_tricks - team1_tricks)

        feature = self.encoder.encode(state, state.turn)
        pred_value_view_scaled = float(self.model.predict(feature))
        pred_value_view = pred_value_view_scaled * self.config.value_scale
        return pred_value_view if self._current_team(state) == 0 else -pred_value_view

    def _terminal_value(self, state: GameState) -> float:
        """终局价值（队伍 0 视角）。"""
        return float(self.rules.score(state)[0])

    def _is_terminal(self, state: GameState) -> bool:
        """判断是否已经进入终局。"""
        return self.rules.end_trickgame(state)

    def _current_team(self, state: GameState) -> int:
        """返回当前行动方所属队伍。"""
        return state.teams[state.turn]

    def _remaining_cards(self, state: GameState) -> int:
        """返回当前局面的剩余牌数。"""
        return sum(len(hand) for hand in state.hands)

    def _legal_actions(self, state: GameState) -> list[Card]:
        """返回当前状态下的合法动作，并按 card_id 排序。"""
        hand = state.hands[state.turn]
        legal_actions = self.rules.playable(state, hand, state.turn)
        return sorted(legal_actions, key=lambda card: card.card_id)

    def _apply_action(self, state: GameState, action: Card) -> GameState:
        """返回应用动作后的新状态。"""
        return self.exact_solver._apply_action(state, action, state.turn)

    def _apply_action_in_place(self, state: GameState, action: Card) -> None:
        """在原状态上执行出牌推进。"""
        player_id = state.turn
        state.play_card_to_table(player_id, action)
        if action.suit == Suit.SPADES:
            state.spades_broken = True
            state.trump_broken = True
        state.turn = (player_id + 1) % state.num_players
        if state.trick_complete:
            winner = self.rules.winner_trick(state)
            state.complete_trick(winner)
            state.trick_leader = winner
            state.turn = winner


def _build_demo_state(seed: int) -> GameState:
    """构造一个从 52 张牌开始的可出牌测试局面。

    输入:
    - seed: int，发牌和叫牌随机种子

    输出:
    - GameState，已经完成叫牌并进入出牌阶段，52 张牌都还在各自手中
    """
    from data.training_data import build_state_with_remaining_cards

    return build_state_with_remaining_cards(target_remaining=52, seed=seed)


def main() -> None:
    """命令行入口：从 52 张牌开始跑完整出牌序列。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False, default=None, help="MLP 权重路径")
    parser.add_argument("--seed", type=int, default=0, help="牌局随机种子")
    parser.add_argument("--exact_threshold", type=int, default=30, help="剩余牌数 <= 该值时直接精确求解")
    parser.add_argument("--leaf_threshold", type=int, default=24, help="MCTS 搜索到该剩余牌数时接入 MLP")
    parser.add_argument("--simulations_per_action", type=int, default=5, help="每个根动作模拟次数")
    parser.add_argument("--exploration_constant", type=float, default=1.5, help="PUCT 探索系数")
    parser.add_argument("--policy_temperature", type=float, default=1.0, help="policy 先验温度")
    args = parser.parse_args()

    config = TruncatedMCTSConfig(
        exact_threshold=args.exact_threshold,
        leaf_threshold=args.leaf_threshold,
        simulations_per_action=args.simulations_per_action,
        exploration_constant=args.exploration_constant,
        policy_temperature=args.policy_temperature,
        checkpoint_path=args.checkpoint,
    )

    strategy = TruncatedMCTSStrategy(config)
    state = _build_demo_state(args.seed)

    print("初始局面：")
    print(f"  当前玩家: {state.turn}")
    print(f"  剩余牌数: {sum(len(hand) for hand in state.hands)}")
    print(f"  策略配置: exact_threshold={config.exact_threshold}, leaf_threshold={config.leaf_threshold}, simulations_per_action={config.simulations_per_action}")
    print()

    action_sequence = strategy.play_full_game(state)

    print("完整出牌序列：")
    print(" -> ".join(str(card) for card in action_sequence))
    print()
    print("终局统计：")
    print(f"  总动作数: {len(action_sequence)}")
    print(f"  最终得分: {strategy.rules.score(state)}")


if __name__ == "__main__":
    main()
