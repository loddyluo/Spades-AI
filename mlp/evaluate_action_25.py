"""基于 value-head 的动作一致性评估（x=25）。

文件作用：
- 读取一个 x=25 数据文件（默认 1000 条），利用每条样本的 seed 重建局面；
- 用精确求解器得到根节点最优动作；
- 对每个合法动作做一步前瞻，用 value-head 评估子状态价值并选动作；
- 统计模型选动作与精确最优动作的一致率。

函数输入/输出说明：
- load_samples(dataset_path: Path, max_samples: int | None) -> list[dict]
  输入: 数据文件路径、最多评估条数
  输出: 样本列表

- choose_action_by_value_head(...)
  输入: 模型、编码器、求解器、规则、根状态、合法动作
  输出: (chosen_action, chosen_team0_value)

- evaluate_action_match(...) -> dict[str, float]
  输入: checkpoint 路径、数据文件路径、最多评估条数
  输出: 评估指标字典（exact_match_rate, tie_match_rate, avg_legal_actions, evaluated）

- main() -> None
  输入: 命令行参数
  输出: 打印评估结果
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

from data.training_data import DEFAULT_OUTPUT_PREFIX, build_state_with_remaining_cards, load_bucket_dataset
from mlp.mlp_model import DoubleDummyMLP
from trick_taking.card import Card
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver
from trick_taking.utils.feature_encoder import SpadesFeatureEncoder


def load_samples(dataset_path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    """读取样本，按 max_samples 截断。"""
    loaded = load_bucket_dataset(dataset_path)
    samples = loaded["samples"]
    if max_samples is not None:
        return samples[:max_samples]
    return samples


def _next_state_after_action(
    state,
    action: Card,
    player_id: int,
    solver: ExactDoubleDummySolver,
):
    """对根状态应用一个动作，返回子状态。"""
    # 复用 solver 内部的状态推进逻辑，避免逻辑分叉。
    return solver._apply_action(state, action, player_id)


def _team0_value_from_view(pred_value_view: float, current_player_team: int) -> float:
    """把当前行动方视角 value 转回 team0 视角。"""
    return pred_value_view if current_player_team == 0 else -pred_value_view


def choose_action_by_value_head(
    model: DoubleDummyMLP,
    encoder: SpadesFeatureEncoder,
    solver: ExactDoubleDummySolver,
    root_state,
    legal_actions: list[Card],
) -> tuple[Card, float]:
    """使用 value-head 对每个合法动作做一步前瞻并选动作。

    关键逻辑：
    - 模型输出是“当前行动方视角 value_view/25”；
    - 先把每个子状态预测值统一换算为 team0 视角；
    - 根玩家若属于 team0，选 team0 值最大的动作；否则选最小的动作。

    这比“按奇偶轮数选最小/最大”更稳妥，因为黑桃中每墩赢家会改变出牌顺序。
    """
    root_team = root_state.teams[root_state.turn]

    best_action = legal_actions[0]
    best_team0_value = None

    for action in legal_actions:
        next_state = _next_state_after_action(root_state, action, root_state.turn, solver)
        feature = encoder.encode(next_state, next_state.turn)
        pred_value_view_scaled = float(model.predict(feature))
        pred_value_view = pred_value_view_scaled * 25.0
        next_team = next_state.teams[next_state.turn]
        pred_team0_value = _team0_value_from_view(pred_value_view, next_team)

        if best_team0_value is None:
            best_team0_value = pred_team0_value
            best_action = action
            continue

        if root_team == 0:
            if pred_team0_value > best_team0_value:
                best_team0_value = pred_team0_value
                best_action = action
        else:
            if pred_team0_value < best_team0_value:
                best_team0_value = pred_team0_value
                best_action = action

    return best_action, float(best_team0_value)


def evaluate_action_match(
    checkpoint: Path,
    dataset_path: Path,
    max_samples: int | None = 1000,
) -> dict[str, float]:
    """评估 value-head 一步前瞻动作与精确最优动作的一致率。"""
    encoder = SpadesFeatureEncoder()
    model = DoubleDummyMLP(input_dim=encoder.total_dim)
    model.load(str(checkpoint))

    solver = ExactDoubleDummySolver()
    rules = SpadesRules()

    samples = load_samples(dataset_path, max_samples=max_samples)

    exact_match = 0
    tie_match = 0
    evaluated = 0
    total_legal_actions = 0

    for sample in samples:
        x = int(sample["x"])
        seed = int(sample["seed"])
        state = build_state_with_remaining_cards(target_remaining=x, seed=seed)

        exact_result = solver.solve_with_q(state)
        exact_best = exact_result["best_action"]
        action_q_values = exact_result["action_q_values"]

        legal_actions = rules.playable(state, state.hands[state.turn], state.turn)
        if not legal_actions or exact_best is None:
            continue

        chosen_action, _ = choose_action_by_value_head(
            model=model,
            encoder=encoder,
            solver=solver,
            root_state=state,
            legal_actions=legal_actions,
        )

        total_legal_actions += len(legal_actions)
        evaluated += 1

        if chosen_action.card_id == exact_best.card_id:
            exact_match += 1

        root_team = state.teams[state.turn]
        if root_team == 0:
            best_q = max(float(v) for v in action_q_values.values())
        else:
            best_q = min(float(v) for v in action_q_values.values())

        chosen_q = float(action_q_values[chosen_action])
        if abs(chosen_q - best_q) <= 1e-9:
            tie_match += 1

    if evaluated == 0:
        raise RuntimeError("没有可评估样本，请检查数据文件。")

    return {
        "evaluated": float(evaluated),
        "exact_match_rate": exact_match / evaluated,
        "tie_match_rate": tie_match / evaluated,
        "avg_legal_actions": total_legal_actions / evaluated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="value-head 模型权重路径")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path("data") / f"{DEFAULT_OUTPUT_PREFIX}_x25_n1000.pt"),
        help="x=25 数据文件路径",
    )
    parser.add_argument("--max_samples", type=int, default=1000, help="最多评估多少条样本")
    args = parser.parse_args()

    metrics = evaluate_action_match(
        checkpoint=Path(args.checkpoint),
        dataset_path=Path(args.dataset),
        max_samples=args.max_samples,
    )

    print("=" * 100)
    print("x=25 value-head 一步前瞻动作评估")
    print("=" * 100)
    print(f"评估样本数: {int(metrics['evaluated'])}")
    print(f"精确最佳动作一致率 exact_match_rate: {metrics['exact_match_rate']:.6f}")
    print(f"并列最优一致率 tie_match_rate: {metrics['tie_match_rate']:.6f}")
    print(f"平均合法动作数 avg_legal_actions: {metrics['avg_legal_actions']:.4f}")


if __name__ == "__main__":
    main()
