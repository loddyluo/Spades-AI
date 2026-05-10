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
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from data.training_data import DEFAULT_OUTPUT_PREFIX, build_state_with_remaining_cards, load_bucket_dataset
from mlp.mlp_model import DoubleDummyMLP
from trick_taking.card import Card
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver
from trick_taking.utils.feature_encoder import SpadesFeatureEncoder


_WORKER_MODEL: DoubleDummyMLP | None = None
_WORKER_ENCODER: SpadesFeatureEncoder | None = None
_WORKER_SOLVER: ExactDoubleDummySolver | None = None
_WORKER_RULES: SpadesRules | None = None


def _init_worker(checkpoint: str, device: str, torch_num_threads: int) -> None:
    """Initialize per-process runtime objects once."""
    global _WORKER_MODEL, _WORKER_ENCODER, _WORKER_SOLVER, _WORKER_RULES

    if torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)

    _WORKER_ENCODER = SpadesFeatureEncoder()
    _WORKER_MODEL = DoubleDummyMLP(input_dim=_WORKER_ENCODER.total_dim)
    _WORKER_MODEL.load(checkpoint, device=device)
    _WORKER_MODEL.eval()
    _WORKER_SOLVER = ExactDoubleDummySolver()
    _WORKER_RULES = SpadesRules()


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


def _evaluate_one_sample(sample: dict[str, Any]) -> tuple[int, int, int, int]:
    """Worker function: evaluate one sample and return aggregate counters.

    Returns:
        (evaluated, exact_match, tie_match, legal_actions_count)
    """
    if _WORKER_MODEL is None or _WORKER_ENCODER is None or _WORKER_SOLVER is None or _WORKER_RULES is None:
        raise RuntimeError("worker is not initialized")

    x = int(sample["x"])
    seed = int(sample["seed"])
    state = build_state_with_remaining_cards(target_remaining=x, seed=seed)

    exact_result = _WORKER_SOLVER.solve_with_q(state)
    exact_best = exact_result["best_action"]
    action_q_values = exact_result["action_q_values"]

    legal_actions = _WORKER_RULES.playable(state, state.hands[state.turn], state.turn)
    if not legal_actions or exact_best is None:
        return 0, 0, 0, 0

    root_team = state.teams[state.turn]

    # Batch all one-step successor features into one model forward pass.
    next_states = [_next_state_after_action(state, action, state.turn, _WORKER_SOLVER) for action in legal_actions]
    features = np.asarray([_WORKER_ENCODER.encode(ns, ns.turn) for ns in next_states], dtype=np.float32)
    pred_value_view_scaled = _WORKER_MODEL.predict(features)
    if isinstance(pred_value_view_scaled, (float, int)):
        pred_value_view_scaled = np.asarray([float(pred_value_view_scaled)], dtype=np.float32)
    pred_value_view = np.asarray(pred_value_view_scaled, dtype=np.float32) * 25.0

    team0_values: list[float] = []
    for ns, pv in zip(next_states, pred_value_view):
        next_team = ns.teams[ns.turn]
        team0_values.append(_team0_value_from_view(float(pv), next_team))

    if root_team == 0:
        best_idx = int(np.argmax(team0_values))
    else:
        best_idx = int(np.argmin(team0_values))
    chosen_action = legal_actions[best_idx]

    exact_match = 1 if chosen_action.card_id == exact_best.card_id else 0

    if root_team == 0:
        best_q = max(float(v) for v in action_q_values.values())
    else:
        best_q = min(float(v) for v in action_q_values.values())

    chosen_q = float(action_q_values[chosen_action])
    tie_match = 1 if abs(chosen_q - best_q) <= 1e-9 else 0

    return 1, exact_match, tie_match, len(legal_actions)


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
    num_workers: int = 1,
    chunk_size: int = 8,
    device: str = "cpu",
    mp_start_method: str = "fork",
    torch_num_threads: int = 1,
) -> dict[str, float]:
    """评估 value-head 一步前瞻动作与精确最优动作的一致率。"""
    samples = load_samples(dataset_path, max_samples=max_samples)

    exact_match = 0
    tie_match = 0
    evaluated = 0
    total_legal_actions = 0

    if num_workers <= 1:
        _init_worker(str(checkpoint), device=device, torch_num_threads=torch_num_threads)
        for sample in samples:
            e, em, tm, la = _evaluate_one_sample(sample)
            evaluated += e
            exact_match += em
            tie_match += tm
            total_legal_actions += la
    else:
        # Avoid oversubscription; this workload is process-level parallel.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        ctx = get_context(mp_start_method)
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(str(checkpoint), device, torch_num_threads),
        ) as executor:
            for e, em, tm, la in executor.map(_evaluate_one_sample, samples, chunksize=max(1, chunk_size)):
                evaluated += e
                exact_match += em
                tie_match += tm
                total_legal_actions += la

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
    parser.add_argument("--num-workers", type=int, default=1, help="并行 worker 数（1 表示串行）")
    parser.add_argument("--chunk-size", type=int, default=8, help="并行 map 的样本分块大小")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="模型推理设备")
    parser.add_argument(
        "--mp-start-method",
        type=str,
        default="fork",
        choices=["fork", "spawn", "forkserver"],
        help="多进程启动方式",
    )
    parser.add_argument("--torch-num-threads", type=int, default=1, help="每个 worker 内部 Torch 线程数")
    args = parser.parse_args()

    if args.device == "cuda" and args.num_workers > 1:
        print("警告: 单卡下多进程 + CUDA 通常会更慢且更占显存，已自动回退为 num_workers=1。")
        args.num_workers = 1

    metrics = evaluate_action_match(
        checkpoint=Path(args.checkpoint),
        dataset_path=Path(args.dataset),
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        device=args.device,
        mp_start_method=args.mp_start_method,
        torch_num_threads=args.torch_num_threads,
    )

    print("=" * 100)
    print("x=25 value-head 一步前瞻动作评估")
    print("=" * 100)
    print(f"并行配置: workers={args.num_workers}, chunk_size={args.chunk_size}, device={args.device}")
    print(f"评估样本数: {int(metrics['evaluated'])}")
    print(f"精确最佳动作一致率 exact_match_rate: {metrics['exact_match_rate']:.6f}")
    print(f"并列最优一致率 tie_match_rate: {metrics['tie_match_rate']:.6f}")
    print(f"平均合法动作数 avg_legal_actions: {metrics['avg_legal_actions']:.4f}")


if __name__ == "__main__":
    main()
