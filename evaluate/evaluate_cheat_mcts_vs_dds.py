"""Evaluate cheating MCTS (sees all cards) vs cheating DDS (sees all cards).

Both players have perfect information:
- DDS: uses double-dummy solver to maximize tricks
- MCTS: uses MCTS+exact solver with use_determinization=False (true hands visible)

This measures the gap between our MCTS search quality and theoretical optimal
when information asymmetry is removed.

Usage:
    python evaluate/evaluate_cheat_mcts_vs_dds.py \
        --num-games 10 --seed 42 \
        --our-simulations-per-action 64 \
        --our-exact-threshold 24
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GO_MCTS_DIR = REPO_ROOT / "evaluate" / "GO-MCTS"
if str(GO_MCTS_DIR) not in sys.path:
    sys.path.insert(0, str(GO_MCTS_DIR))

# Import evaluation infrastructure
import evaluate.evaluate_our_mcts_vs_rule_v2 as _eval_module
from evaluate.evaluate_our_mcts_vs_rule_v2 import (
    Runtime,
    build_runtime,
    run_evaluation,
    _print_summary,
    _resolve_checkpoint_path,
)
from strategy.truncated_mcts_strategy import TruncatedMCTSConfig


_original_build_players = _eval_module.build_players


def _patched_build_players(args, runtime, game_seed, seat_specs=None):
    """Build players with 'cheat_mcts' and 'dds' specs."""
    seat_specs_resolved = seat_specs or [args.p0, args.p1, args.p2, args.p3]

    has_special = any(s in ("dds", "cheat_mcts") for s in seat_specs_resolved)
    if not has_special:
        return _original_build_players(args, runtime, game_seed, seat_specs)

    from evaluate.dds_player import DDSPlayer
    from evaluate.evaluate_our_mcts_vs_rule_v2 import (
        _OurMCTSWithMLPBid,
        OurHandStrengthMCTSPlayer,
        GoPlayerAdapter,
        _MLPBidWithV2Play,
        RuleBasedPlayerV2,
    )

    players = []
    for seat_index, spec in enumerate(seat_specs_resolved):
        if spec == "dds":
            players.append(DDSPlayer(
                bid_model=runtime.bid_model,
                bid_device=runtime.device,
            ))
        elif spec == "cheat_mcts":
            # Build a cheating MCTS config: use_determinization=False
            cheat_config = TruncatedMCTSConfig(
                exact_threshold=args.our_exact_threshold,
                leaf_threshold=args.our_leaf_threshold,
                simulations_per_action=args.our_simulations_per_action,
                determinization_count=args.our_number_of_exact_solvers,
                mcts_determinization_count=getattr(args, 'our_mcts_determinization_count', 10),
                exploration_constant=args.our_exploration_constant,
                policy_temperature=getattr(args, 'our_policy_temperature', 1.0),
                value_scale=getattr(args, 'our_value_scale', 25.0),
                checkpoint_path=_resolve_checkpoint_path(args.our_checkpoint) if args.our_checkpoint else None,
                prior_oracle_spec=getattr(args, 'prior_oracle_spec', 'go_rule_2'),
                bid_checkpoint_path=_resolve_checkpoint_path(args.bid_checkpoint) if args.bid_checkpoint else "",
                # KEY: disable determinization = sees true hands
                use_determinization=False,
            )
            if runtime.bid_model is not None:
                players.append(_OurMCTSWithMLPBid(
                    config=cheat_config,
                    bid_model=runtime.bid_model,
                    device=runtime.device,
                ))
            else:
                players.append(OurHandStrengthMCTSPlayer(config=cheat_config))
        elif spec == "our_mcts":
            if runtime.bid_model is not None:
                from evaluate.evaluate_our_mcts_vs_rule_v2 import _OurMCTSWithMLPBid
                players.append(_OurMCTSWithMLPBid(
                    config=runtime.local_mcts_config,
                    bid_model=runtime.bid_model,
                    device=runtime.device,
                ))
            else:
                players.append(OurHandStrengthMCTSPlayer(config=runtime.local_mcts_config))
        elif spec == "go_rule_2":
            if runtime.bid_model is not None:
                players.append(GoPlayerAdapter(_MLPBidWithV2Play(runtime.bid_model, runtime.device)))
            else:
                players.append(GoPlayerAdapter(RuleBasedPlayerV2()))
        else:
            single = _original_build_players(args, runtime, game_seed, [spec]*4)
            players.append(single[seat_index])

    return players


_eval_module.build_players = _patched_build_players


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate cheating MCTS vs DDS (both see all cards).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-games", type=int, default=10)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--disable-nil", action="store_true")
    parser.add_argument("--disable-blind-nil", action="store_true")
    parser.add_argument("--p0", type=str, default="dds",
                        help="Seat 0 spec (dds, cheat_mcts, our_mcts, go_rule_2)")
    parser.add_argument("--p1", type=str, default="cheat_mcts",
                        help="Seat 1 spec")
    parser.add_argument("--p2", type=str, default="dds",
                        help="Seat 2 spec")
    parser.add_argument("--p3", type=str, default="cheat_mcts",
                        help="Seat 3 spec")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--our-checkpoint", type=str, default=None)
    parser.add_argument("--our-exact-threshold", type=int, default=28,
                        help="Exact solve threshold (cheating MCTS can use higher since it sees true hands)")
    parser.add_argument("--our-leaf-threshold", type=int, default=0)
    parser.add_argument("--our-simulations-per-action", type=int, default=64,
                        help="MCTS sims per action (fewer needed since no sampling noise)")
    parser.add_argument("--our-number-of-exact-solvers", type=int, default=1,
                        help="Only 1 needed (no need to resample, true hands are known)")
    parser.add_argument("--symmetric-seat-swap", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--torch-num-interop-threads", type=int, default=1)
    parser.add_argument("--mp-start-method", type=str, default="fork",
                        choices=["fork", "spawn", "forkserver"])
    parser.add_argument("--our-exploration-constant", type=float, default=25.0)
    parser.add_argument("--our-policy-temperature", type=float, default=1.0)
    parser.add_argument("--our-mcts-determinization-count", type=int, default=1,
                        help="1 = no resampling (true hands used directly)")
    parser.add_argument("--our-value-scale", type=float, default=25.0)
    parser.add_argument("--prior-oracle-spec", type=str, default="go_rule_2")
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
    import json

    args = parse_args()
    print("=" * 72)
    print("Cheating MCTS vs DDS — Both See All Cards")
    print("=" * 72)
    print(f"Seats: [{args.p0}, {args.p1}, {args.p2}, {args.p3}]")
    print(f"Games: {args.num_games}, Seed: {args.seed}, Symmetric: {args.symmetric_seat_swap}")
    print(f"MCTS config: exact_threshold={args.our_exact_threshold}, "
          f"sims/action={args.our_simulations_per_action}, "
          f"determinization=OFF (cheating)")
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
