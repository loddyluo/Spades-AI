"""Evaluate DDS (perfect-info cheating player) vs our_mcts in Spades matchups.

The DDS player sees ALL cards and uses the dds-bridge/dds double-dummy solver
(modified with spades_broken support) to make optimal plays.

This measures how close our MCTS player is to theoretical perfect play.

Usage:
    python evaluate/evaluate_dds_vs_mcts.py \
        --p0 dds --p1 our_mcts --p2 dds --p3 our_mcts \
        --num-games 50 --seed 42 --symmetric-seat-swap 1

    python evaluate/evaluate_dds_vs_mcts.py \
        --p0 dds --p1 go_rule_2 --p2 dds --p3 go_rule_2 \
        --num-games 50 --seed 42 --symmetric-seat-swap 1
"""

# This script reuses the infrastructure from evaluate_our_mcts_vs_rule_v2.py
# and adds the "dds" player spec.

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GO_MCTS_DIR = REPO_ROOT / "evaluate" / "GO-MCTS"
if str(GO_MCTS_DIR) not in sys.path:
    sys.path.insert(0, str(GO_MCTS_DIR))

# Import the existing evaluation infrastructure
from evaluate.evaluate_our_mcts_vs_rule_v2 import (
    Runtime,
    TracePlayerProxy,
    ProfilePlayerProxy,
    SpadesMatchRunner,
    SpadesRules,
    TruncatedMCTSConfig,
    GOMCTSConfig,
    build_runtime,
    _build_trace_context,
    _render_game_trace,
    _init_trace_log,
    _append_game_trace,
    _print_summary,
    _resolve_checkpoint_path,
    run_evaluation,
    parse_args as _base_parse_args,
)

# Monkey-patch: add DDS support to build_players
import evaluate.evaluate_our_mcts_vs_rule_v2 as _eval_module


_original_build_players = _eval_module.build_players


def _patched_build_players(args, runtime, game_seed, seat_specs=None):
    """Extended build_players with DDS player support."""
    seat_specs_resolved = seat_specs or [args.p0, args.p1, args.p2, args.p3]

    # Check if any spec is "dds" - if not, use original
    if "dds" not in seat_specs_resolved:
        return _original_build_players(args, runtime, game_seed, seat_specs)

    # Build players, handling "dds" spec
    from evaluate.dds_player import DDSPlayer

    players = []
    for seat_index, spec in enumerate(seat_specs_resolved):
        if spec == "dds":
            players.append(DDSPlayer(
                bid_model=runtime.bid_model,
                bid_device=runtime.device,
            ))
            continue
        # For non-dds specs, use a mini build
        # We need to handle the common specs manually
        if spec == "our_mcts":
            from evaluate.evaluate_our_mcts_vs_rule_v2 import (
                _OurMCTSWithMLPBid,
                OurHandStrengthMCTSPlayer,
            )
            if runtime.bid_model is not None:
                players.append(_OurMCTSWithMLPBid(
                    config=runtime.local_mcts_config,
                    bid_model=runtime.bid_model,
                    device=runtime.device,
                ))
            else:
                players.append(OurHandStrengthMCTSPlayer(config=runtime.local_mcts_config))
            continue
        if spec == "go_rule_2":
            from evaluate.evaluate_our_mcts_vs_rule_v2 import (
                GoPlayerAdapter,
                _MLPBidWithV2Play,
                RuleBasedPlayerV2,
            )
            if runtime.bid_model is not None:
                players.append(GoPlayerAdapter(_MLPBidWithV2Play(runtime.bid_model, runtime.device)))
            else:
                players.append(GoPlayerAdapter(RuleBasedPlayerV2()))
            continue
        if spec == "go_rule":
            from models import RuleBasedPlayer
            from adapters import GoPlayerAdapter as _GoPA
            players.append(_GoPA(RuleBasedPlayer()))
            continue
        # Fallback to original for other specs
        single_result = _original_build_players(args, runtime, game_seed, [spec, spec, spec, spec])
        players.append(single_result[seat_index])
        continue

    return players


# Apply the patch
_eval_module.build_players = _patched_build_players


def parse_args():
    """Parse args with DDS-specific defaults."""
    import argparse

    # Modify sys.argv description if needed
    parser = argparse.ArgumentParser(
        description="Evaluate DDS (perfect-info) vs our_mcts / go_rule_2 in Spades.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path")
    parser.add_argument("--disable-nil", action="store_true", help="Disable nil bidding")
    parser.add_argument("--disable-blind-nil", action="store_true", help="Disable blind nil")
    parser.add_argument("--p0", type=str, default="dds", help="Seat 0 model spec")
    parser.add_argument("--p1", type=str, default="our_mcts", help="Seat 1 model spec")
    parser.add_argument("--p2", type=str, default="dds", help="Seat 2 model spec")
    parser.add_argument("--p3", type=str, default="our_mcts", help="Seat 3 model spec")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--our-checkpoint", type=str, default=None)
    parser.add_argument("--our-exact-threshold", type=int, default=24)
    parser.add_argument("--our-leaf-threshold", type=int, default=0)
    parser.add_argument("--our-simulations-per-action", type=int, default=256)
    parser.add_argument("--our-number-of-exact-solvers", type=int, default=32)
    parser.add_argument("--symmetric-seat-swap", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--torch-num-interop-threads", type=int, default=1)
    parser.add_argument("--mp-start-method", type=str, default="fork",
                        choices=["fork", "spawn", "forkserver"])
    parser.add_argument("--our-exploration-constant", type=float, default=25.0)
    parser.add_argument("--our-policy-temperature", type=float, default=1.0)
    parser.add_argument("--our-mcts-determinization-count", type=int, default=10)
    parser.add_argument("--our-value-scale", type=float, default=25.0)
    parser.add_argument("--go-pv-checkpoint", type=str, default="")
    parser.add_argument("--go-bid-checkpoint", type=str, default="")
    parser.add_argument("--bid-checkpoint", type=str,
                        default="./Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt")
    parser.add_argument("--go-mcts-runs", type=int, default=100)
    parser.add_argument("--go-mcts-steps", type=int, default=5)
    parser.add_argument("--go-mcts-c", type=float, default=0.3)
    parser.add_argument("--go-mcts-mu", type=float, default=0.01)
    parser.add_argument("--go-mcts-threshold", type=float, default=0.05)
    parser.add_argument("--go-argmax-threshold", type=float, default=0.05)
    parser.add_argument("--trace-log-dir", type=str, default="logs")
    parser.add_argument("--profile-breakdown", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    import json
    import time

    args = parse_args()
    print(f"=== DDS vs {args.p1} Evaluation ===")
    print(f"Seats: [{args.p0}, {args.p1}, {args.p2}, {args.p3}]")
    print(f"Games: {args.num_games}, Seed: {args.seed}, Symmetric: {args.symmetric_seat_swap}")
    print()

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
