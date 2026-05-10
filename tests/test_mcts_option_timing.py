"""Benchmark one-step MCTS timing for each legal action.

File purpose:
- Measure how long the local MCTS strategy takes to simulate each legal root
  action a fixed number of times on a reproducible Spades state.
- Provide a smaller 100-simulation timing and extrapolate it to a 10,000
  simulation estimate, as requested, instead of running 10,000 simulations
  directly.
- Offer a device-aware benchmark so the model can be loaded on CPU or CUDA.

Function input/output summary:
- build_strategy(...) -> TruncatedMCTSStrategy
    Input: model checkpoint path, device, and search hyperparameters.
    Output: a ready-to-use strategy instance.
- run_action_timing(...) -> dict[str, Any]
    Input: a strategy, a full-information GameState, and a simulation count.
    Output: per-action timing statistics and aggregate totals.
- main() -> None
    Input: command-line arguments.
    Output: prints the benchmark summary; raises on failure.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.training_data import build_state_with_remaining_cards
from strategy.truncated_mcts_strategy import TruncatedMCTSConfig, TruncatedMCTSStrategy


def build_strategy(
    checkpoint_path: str | None,
    device: str,
    exact_threshold: int,
    leaf_threshold: int,
    simulations_per_action: int,
    exploration_constant: float,
    policy_temperature: float,
) -> TruncatedMCTSStrategy:
    """Construct the benchmarked strategy.

    Input:
    - checkpoint_path: optional MLP checkpoint path.
    - device: torch device string such as "cpu" or "cuda".
    - exact_threshold / leaf_threshold / simulations_per_action: MCTS settings.
    - exploration_constant / policy_temperature: root search hyperparameters.

    Output:
    - A configured `TruncatedMCTSStrategy` instance.
    """
    config = TruncatedMCTSConfig(
        exact_threshold=exact_threshold,
        leaf_threshold=leaf_threshold,
        simulations_per_action=simulations_per_action,
        exploration_constant=exploration_constant,
        policy_temperature=policy_temperature,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return TruncatedMCTSStrategy(config)


def run_action_timing(
    strategy: TruncatedMCTSStrategy,
    state,
    simulations_per_action: int,
) -> dict[str, Any]:
    """Time one root search per legal action.

    Input:
    - strategy: configured MCTS strategy.
    - state: a full-information Spades GameState ready for trick play.
    - simulations_per_action: how many simulations to run for each root action.

    Output:
    - A dictionary with per-action elapsed time, legal-action count, and totals.
    """
    legal_actions = strategy._legal_actions(state)
    per_action: list[dict[str, Any]] = []
    total_elapsed = 0.0

    for action in legal_actions:
        strategy.exact_solver.tt.clear()
        strategy._leaf_value_cache.clear()
        strategy._policy_priors_cache.clear()

        child_state = strategy._apply_action(state, action)
        child_node = strategy._build_root_child(child_state, action)

        t0 = time.perf_counter()
        for _ in range(simulations_per_action):
            strategy._run_simulation(child_node)
        elapsed = time.perf_counter() - t0

        total_elapsed += elapsed
        per_action.append(
            {
                "action": action,
                "simulations": simulations_per_action,
                "elapsed_sec": elapsed,
                "per_simulation_sec": elapsed / max(simulations_per_action, 1),
            }
        )

    return {
        "legal_action_count": len(legal_actions),
        "per_action": per_action,
        "total_elapsed_sec": total_elapsed,
        "avg_elapsed_sec_per_action": total_elapsed / max(len(legal_actions), 1),
    }


def _print_report(title: str, report: dict[str, Any], scale_to_10000: bool = False) -> None:
    """Print a formatted benchmark report.

    Input:
    - title: heading for the report block.
    - report: dictionary returned by `run_action_timing`.
    - scale_to_10000: whether to print a 10,000-simulation extrapolation.

    Output:
    - None.
    """
    print()
    print(f"=== {title} ===")
    print(f"Legal actions: {report['legal_action_count']}")
    for index, item in enumerate(report["per_action"]):
        suffix = ""
        if scale_to_10000:
            estimated_10000 = item["elapsed_sec"] * 100.0
            suffix = f" | est_10000_sec={estimated_10000:.4f}"
        print(
            f"  [{index}] action={item['action']} | sims={item['simulations']} | "
            f"elapsed_sec={item['elapsed_sec']:.4f} | per_sim_sec={item['per_simulation_sec']:.6f}{suffix}"
        )
    print(f"Total elapsed_sec: {report['total_elapsed_sec']:.4f}")
    print(f"Avg elapsed_sec/action: {report['avg_elapsed_sec_per_action']:.4f}")


def main() -> None:
    """Run the benchmark from the command line.

    Input:
    - Command-line arguments.

    Output:
    - Prints actual 50-simulation timing and 100-simulation extrapolated timing.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark one-step MCTS timing for each legal action.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used to build the benchmark state")
    parser.add_argument(
        "--target-remaining",
        type=int,
        default=52,
        help="How many cards should remain in the benchmark state",
    )
    parser.add_argument("--checkpoint", type=str, default="", help="Optional MLP checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for the MLP")
    parser.add_argument("--exact-threshold", type=int, default=30, help="Exact-solver threshold")
    parser.add_argument("--leaf-threshold", type=int, default=24, help="Leaf evaluation threshold")
    parser.add_argument(
        "--simulations-per-action",
        type=int,
        default=50,
        help="Simulations to run for the main benchmark pass",
    )
    parser.add_argument(
        "--estimate-simulations",
        type=int,
        default=100,
        help="Smaller sample used to estimate the 10,000-simulation cost",
    )
    parser.add_argument(
        "--estimate-multiplier",
        type=float,
        default=100.0,
        help="Multiply the estimate-simulation timing by this factor to approximate 10,000 sims",
    )
    parser.add_argument("--exploration-constant", type=float, default=1.5, help="PUCT exploration constant")
    parser.add_argument("--policy-temperature", type=float, default=1.0, help="Policy prior temperature")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or None
    strategy = build_strategy(
        checkpoint_path=checkpoint_path,
        device=args.device,
        exact_threshold=args.exact_threshold,
        leaf_threshold=args.leaf_threshold,
        simulations_per_action=args.simulations_per_action,
        exploration_constant=args.exploration_constant,
        policy_temperature=args.policy_temperature,
    )

    state = build_state_with_remaining_cards(target_remaining=args.target_remaining, seed=args.seed)

    print("Benchmark state:")
    print(f"  seed={args.seed}")
    print(f"  target_remaining={args.target_remaining}")
    print(f"  current_player={state.turn}")
    print(f"  legal_actions={len(strategy._legal_actions(state))}")
    print(f"  device={args.device}")
    print(f"  checkpoint={'<none>' if checkpoint_path is None else checkpoint_path}")

    main_report = run_action_timing(strategy, state, args.simulations_per_action)
    _print_report(f"Per-action timing with {args.simulations_per_action} simulations", main_report)

    strategy.exact_solver.tt.clear()
    strategy._leaf_value_cache.clear()
    strategy._policy_priors_cache.clear()
    estimate_report = run_action_timing(strategy, state, args.estimate_simulations)
    print()
    print(f"=== 10,000-simulation estimate from {args.estimate_simulations} samples ===")
    print(f"Scale multiplier: {args.estimate_multiplier}")
    for index, item in enumerate(estimate_report["per_action"]):
        estimated_10000 = item["elapsed_sec"] * args.estimate_multiplier
        print(
            f"  [{index}] action={item['action']} | sims={item['simulations']} | "
            f"elapsed_sec={item['elapsed_sec']:.4f} | est_10000_sec={estimated_10000:.4f}"
        )
    print(f"Estimated total_10000_sec={estimate_report['total_elapsed_sec'] * args.estimate_multiplier:.4f}")


if __name__ == "__main__":
    main()
