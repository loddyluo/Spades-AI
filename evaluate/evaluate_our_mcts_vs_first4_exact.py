"""Evaluate our_mcts vs RuleExactFirst4Player in Spades matchups.

The our_mcts player uses MCTS with exact double-dummy solver for later tricks.
The RuleExactFirst4Player uses rule-based play for the first 4 tricks (16 cards)
and the exact double-dummy solver for the remaining cards.

Usage:
    python evaluate/evaluate_our_mcts_vs_first4_exact.py \
        --p0 our_mcts --p1 rule_first4_exact --p2 our_mcts --p3 rule_first4_exact \
        --num-games 40 --seed 8880000 --num-workers 20

Defaults: p0=our_mcts, p1=rule_first4_exact, p2=our_mcts, p3=rule_first4_exact
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GO_MCTS_DIR = REPO_ROOT / "evaluate" / "GO-MCTS"
if str(GO_MCTS_DIR) not in sys.path:
    sys.path.insert(0, str(GO_MCTS_DIR))

# Import base evaluation infrastructure from the our_mcts vs rule_v2 script
from evaluate.evaluate_our_mcts_vs_rule_v2 import (
    Runtime,
    run_evaluation,
    _build_trace_context,
    _render_game_trace,
    _init_trace_log,
    _append_game_trace,
    _print_summary,
    _resolve_checkpoint_path,
    build_runtime as _base_build_runtime,
    _init_parallel_worker,
    parse_args as _base_parse_args,
    GoPlayerAdapter,
    _MLPBidWithV2Play,
)

# RuleExactFirst4Player — rule-based first 4 tricks, exact solver for rest
from strategy.rule_exact_first4_player import RuleExactFirst4Player
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)


# ────────────────────────────────────────────────────────────────────────
# Worker globals (persist across games within a single worker process)
# ────────────────────────────────────────────────────────────────────────
_worker_exact_solver: ExactDoubleDummyCppFastestSolver | None = None


# ── Patch _init_parallel_worker ────────────────────────────────────────
import evaluate.evaluate_our_mcts_vs_rule_v2 as _eval_module

_original_init_worker = _eval_module._init_parallel_worker


def _patched_init_worker(args) -> None:
    """Initialize worker: call original, then load exact solver."""
    global _worker_exact_solver

    _original_init_worker(args)

    _worker_exact_solver = ExactDoubleDummyCppFastestSolver()


_eval_module._init_parallel_worker = _patched_init_worker

# ── Patch build_players ────────────────────────────────────────────────
_original_build_players = _eval_module.build_players


def _patched_build_players(args, runtime, game_seed, seat_specs=None):
    """Extended build_players with RuleExactFirst4Player support."""
    seat_specs_resolved = seat_specs or [args.p0, args.p1, args.p2, args.p3]

    has_rule_first4 = "rule_first4_exact" in seat_specs_resolved

    # If no rule_first4_exact in specs, fall back to original (handles our_mcts etc.)
    if not has_rule_first4:
        return _original_build_players(args, runtime, game_seed, seat_specs)

    players = []
    for seat_index, spec in enumerate(seat_specs_resolved):
        if spec == "rule_first4_exact":
            global _worker_exact_solver

            solver = _worker_exact_solver if _worker_exact_solver is not None \
                else ExactDoubleDummyCppFastestSolver()

            player = RuleExactFirst4Player(
                exact_solver=solver,
                exact_threshold=args.our_exact_threshold,
                bid_model=runtime.bid_model,
                bid_device=runtime.device,
            )
            players.append(player)
            continue

        # Fallback to original for other specs (our_mcts, go_rule_2, etc.)
        single_result = _original_build_players(
            args, runtime, game_seed, [spec, spec, spec, spec],
        )
        players.append(single_result[seat_index])
        continue

    return players


_eval_module.build_players = _patched_build_players


# ── CLI ────────────────────────────────────────────────────────────────
def parse_args():
    """Parse args with both our_mcts and rule_first4_exact support."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate our_mcts vs rule_first4_exact in Spades.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--num-games", type=int, default=10,
                        help="Number of games to play")
    parser.add_argument("--output", type=str, default="",
                        help="Optional JSON output path")
    parser.add_argument("--disable-nil", action="store_true",
                        help="Disable nil bidding")
    parser.add_argument("--disable-blind-nil", action="store_true",
                        help="Disable blind nil")
    parser.add_argument("--p0", type=str, default="our_mcts",
                        help="Seat 0 model spec")
    parser.add_argument("--p1", type=str, default="rule_first4_exact",
                        help="Seat 1 model spec")
    parser.add_argument("--p2", type=str, default="our_mcts",
                        help="Seat 2 model spec")
    parser.add_argument("--p3", type=str, default="rule_first4_exact",
                        help="Seat 3 model spec")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device")
    parser.add_argument("--our-checkpoint", type=str, default=None)
    parser.add_argument("--our-exact-threshold", type=int, default=36)
    parser.add_argument("--our-leaf-threshold", type=int, default=28)
    parser.add_argument("--our-simulations-per-action", type=int, default=40)
    parser.add_argument("--our-number-of-exact-solvers", type=int, default=32)
    parser.add_argument("--symmetric-seat-swap", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--torch-num-interop-threads", type=int, default=1)
    parser.add_argument("--mp-start-method", type=str, default="fork",
                        choices=["fork", "spawn", "forkserver"])

    # MCTS-related (for our_mcts player)
    parser.add_argument("--our-exploration-constant", type=float, default=25.0)
    parser.add_argument("--our-policy-temperature", type=float, default=1.0)
    parser.add_argument("--our-mcts-determinization-count", type=int, default=5)
    parser.add_argument("--our-value-scale", type=float, default=25.0)
    parser.add_argument("--go-pv-checkpoint", type=str, default="")
    parser.add_argument("--go-bid-checkpoint", type=str, default="")

    # Bid model (shared across all player types)
    parser.add_argument("--bid-checkpoint", type=str,
                        default="./Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt")

    # Go MCTS args
    parser.add_argument("--go-mcts-runs", type=int, default=100)
    parser.add_argument("--go-mcts-steps", type=int, default=5)
    parser.add_argument("--go-mcts-c", type=float, default=0.3)
    parser.add_argument("--go-mcts-mu", type=float, default=0.01)
    parser.add_argument("--go-mcts-threshold", type=float, default=0.05)
    parser.add_argument("--go-argmax-threshold", type=float, default=0.05)

    # Trace / profile
    parser.add_argument("--trace-log-dir", type=str, default="logs")
    parser.add_argument("--profile-breakdown", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    import json
    import time

    args = parse_args()
    print(f"=== our_mcts vs rule_first4_exact Evaluation ===")
    print(f"Seats: [{args.p0}, {args.p1}, {args.p2}, {args.p3}]")
    print(f"Exact threshold: {args.our_exact_threshold} "
          f"(first {52 - args.our_exact_threshold} cards use rule-based/MCTS)")
    print(f"Games: {args.num_games}, Seed: {args.seed}, "
          f"Symmetric: {args.symmetric_seat_swap}")
    print(f"Workers: {args.num_workers}")
    print()

    # ── For single-process mode, load solver here (fork inherits it) ──
    if args.num_workers <= 1:
        global _worker_exact_solver
        _worker_exact_solver = ExactDoubleDummyCppFastestSolver()

    result = run_evaluation(args)
    _print_summary(result)

    if result.get("trace_log_path"):
        print(f"Trace log: {result['trace_log_path']}")
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Saved JSON results to: {output_path}")


if __name__ == "__main__":
    main()
