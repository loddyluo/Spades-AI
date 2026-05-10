"""Function-level timing breakdown for one matchup game.

File purpose:
- Run one full game under the same player setup as `evaluate_model_matchups`.
- Instrument key `TruncatedMCTSStrategy` and exact-solver methods to collect
  call counts and wall-clock time per function.
- Help diagnose why runs are slow even when simulation/exact sample counts are
  set to small values.

Function input/output summary:
- _time_wrapper(name: str, fn: Callable, stats: dict[str, dict[str, float]]) -> Callable
    Input: function name, original callable, mutable stats dictionary.
    Output: wrapped callable that accumulates calls and elapsed seconds.
- _attach_timing_hooks(strategy: Any, stats: dict[str, dict[str, float]]) -> None
    Input: one strategy instance and a shared stats dictionary.
    Output: monkey-patches target methods on that strategy instance.
- _attach_hooks_to_players(players: list[Any], stats: dict[str, dict[str, float]]) -> int
    Input: player list and shared stats dictionary.
    Output: number of players successfully instrumented.
- run_breakdown(args: argparse.Namespace) -> dict[str, Any]
    Input: parsed CLI arguments.
    Output: per-game result with function-level timing stats.
- parse_args() -> argparse.Namespace
    Input: command-line arguments from `sys.argv`.
    Output: parsed options controlling one-game run and timing report.
- main() -> None
    Input: command-line arguments.
    Output: prints a sorted timing report and game summary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.evaluate_model_matchups import build_players, build_runtime
from strategy.spades_match_runner import SpadesMatchRunner
from trick_taking.games.spades import SpadesRules


def _time_wrapper(name: str, fn: Callable, stats: dict[str, dict[str, float]]) -> Callable:
    """Wrap one callable and accumulate timing stats.

    Input:
    - name: label used in report output.
    - fn: original function/method to call.
    - stats: mutable dictionary keyed by function name.

    Output:
    - wrapped callable with identical call signature.
    """

    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            record = stats.setdefault(name, {"calls": 0.0, "elapsed_sec": 0.0})
            record["calls"] += 1.0
            record["elapsed_sec"] += dt

    return wrapped


def _attach_timing_hooks(strategy: Any, stats: dict[str, dict[str, float]]) -> None:
    """Attach timers to one `TruncatedMCTSStrategy` instance.

    Input:
    - strategy: strategy instance from one local MCTS player.
    - stats: shared stats dictionary.

    Output:
    - None; strategy methods are monkey-patched in-place.
    """
    method_names = [
        "_run_simulation",
        "_determinize_state",
        "_apply_action",
        "_policy_priors",
        "_leaf_value",
        "_solve_with_determinization",
        "choose_action_with_info",
    ]

    for name in method_names:
        if hasattr(strategy, name):
            original = getattr(strategy, name)
            setattr(strategy, name, _time_wrapper(f"strategy.{name}", original, stats))

    if hasattr(strategy, "exact_solver") and hasattr(strategy.exact_solver, "solve_with_q"):
        original_solve_with_q = strategy.exact_solver.solve_with_q
        strategy.exact_solver.solve_with_q = _time_wrapper(
            "exact_solver.solve_with_q", original_solve_with_q, stats
        )


def _attach_hooks_to_players(players: list[Any], stats: dict[str, dict[str, float]]) -> int:
    """Attach timers to all local MCTS players in the list.

    Input:
    - players: list of seat players.
    - stats: shared stats dictionary.

    Output:
    - Number of players instrumented.
    """
    instrumented = 0
    for player in players:
        strategy = getattr(player, "strategy", None)
        if strategy is not None:
            _attach_timing_hooks(strategy, stats)
            instrumented += 1
    return instrumented


def run_breakdown(args: argparse.Namespace) -> dict[str, Any]:
    """Run one game and collect function-level timing stats.

    Input:
    - args: parsed CLI options.

    Output:
    - Dictionary with game summary and function timing table.
    """
    runtime = build_runtime(args)
    players = build_players(args, runtime, args.seed)

    stats: dict[str, dict[str, float]] = {}
    instrumented = _attach_hooks_to_players(players, stats)

    rules = SpadesRules(enable_nil=not args.disable_nil, enable_blind_nil=False)
    runner = SpadesMatchRunner(players=players, seed=args.seed, verbose=False, rules=rules)

    t0 = time.perf_counter()
    result = runner.play_game()
    wall = time.perf_counter() - t0

    return {
        "seed": int(args.seed),
        "instrumented_players": int(instrumented),
        "scores": [float(x) for x in result.scores],
        "winner": int(result.winner),
        "game_wall_sec": float(wall),
        "function_stats": stats,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this timing breakdown script.

    Input:
    - `sys.argv`.

    Output:
    - Parsed options for one-game setup and MCTS budget.
    """
    parser = argparse.ArgumentParser(
        description="Function-level timing breakdown for one local matchup game.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=37865, help="Game seed")
    parser.add_argument("--disable-nil", action="store_true", help="Disable nil bidding")
    parser.add_argument("--disable-blind-nil", action="store_true", help="Disable blind nil bidding")
    parser.add_argument("--p0", type=str, default="our_mcts", help="Seat 0 model spec")
    parser.add_argument("--p1", type=str, default="go_rule", help="Seat 1 model spec")
    parser.add_argument("--p2", type=str, default="our_mcts", help="Seat 2 model spec")
    parser.add_argument("--p3", type=str, default="go_rule", help="Seat 3 model spec")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--our-checkpoint", type=str, default="result/mlp_test_4.pth", help="Local checkpoint path")
    parser.add_argument("--our-exact-threshold", type=int, default=24, help="Exact threshold")
    parser.add_argument("--our-leaf-threshold", type=int, default=24, help="Leaf threshold")
    parser.add_argument("--our-simulations-per-action", type=int, default=1, help="MCTS samples per legal action")
    parser.add_argument("--our-number-of-exact-solvers", type=int, default=1, help="Determinized exact samples per exact step")
    parser.add_argument("--our-exploration-constant", type=float, default=1.5, help="PUCT exploration constant")
    parser.add_argument("--our-policy-temperature", type=float, default=1.0, help="Policy temperature")
    parser.add_argument("--our-value-scale", type=float, default=25.0, help="Value scale")
    parser.add_argument("--symmetric-seat-swap", type=int, default=0, help="Unused by this script; kept for namespace compatibility")
    parser.add_argument("--num-workers", type=int, default=1, help="Unused by this script; kept for namespace compatibility")
    parser.add_argument("--torch-num-threads", type=int, default=1, help="Torch intra-op threads")
    parser.add_argument("--torch-num-interop-threads", type=int, default=1, help="Torch inter-op threads")
    parser.add_argument("--mp-start-method", type=str, default="fork", help="Unused by this script")
    parser.add_argument("--go-pv-checkpoint", type=str, default="", help="Unused unless go_* seats require model")
    parser.add_argument("--go-bid-checkpoint", type=str, default="", help="Unused unless go_mlp_bid seat is used")
    parser.add_argument("--go-mcts-runs", type=int, default=100, help="GO-MCTS runs")
    parser.add_argument("--go-mcts-steps", type=int, default=5, help="GO-MCTS max steps")
    parser.add_argument("--go-mcts-c", type=float, default=0.3, help="GO-MCTS exploration C")
    parser.add_argument("--go-mcts-mu", type=float, default=0.01, help="GO-MCTS mu")
    parser.add_argument("--go-mcts-threshold", type=float, default=0.05, help="GO-MCTS threshold")
    parser.add_argument("--go-argmax-threshold", type=float, default=0.05, help="GO argmax threshold")
    parser.add_argument("--trace-log-dir", type=str, default="", help="Unused by this script")
    parser.add_argument("--output", type=str, default="", help="Unused by this script")
    return parser.parse_args()


def main() -> None:
    """Run the timing breakdown and print a sorted report.

    Input:
    - command-line arguments.

    Output:
    - Prints game summary and function timing table sorted by elapsed time.
    """
    args = parse_args()
    summary = run_breakdown(args)

    print("=== Function-level timing breakdown ===")
    print(f"seed={summary['seed']} instrumented_players={summary['instrumented_players']}")
    print(f"game_wall_sec={summary['game_wall_sec']:.6f} winner={summary['winner']} scores={summary['scores']}")

    rows = []
    for name, record in summary["function_stats"].items():
        calls = int(record.get("calls", 0.0))
        elapsed = float(record.get("elapsed_sec", 0.0))
        avg = elapsed / calls if calls > 0 else 0.0
        rows.append((name, calls, elapsed, avg))
    rows.sort(key=lambda item: item[2], reverse=True)

    for name, calls, elapsed, avg in rows:
        print(f"{name:34s} calls={calls:6d} total={elapsed:10.6f}s avg={avg:10.6f}s")


if __name__ == "__main__":
    main()
