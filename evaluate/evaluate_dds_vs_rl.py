"""
Evaluate DDS (perfect-info) vs RL-Exact in Spades matchups.

The RL-Exact player uses a policy network (MLP) for the first 4 tricks
(16 cards) and the exact double-dummy solver for the remaining cards.
Checkpoint selection depends on nil bids:
- Someone bids nil → 55_2nil.pt
- No one bids nil → 55_2.pt

Usage:
    python evaluate/evaluate_dds_vs_rl.py \
        --p0 dds --p1 rl_exact --p2 dds --p3 rl_exact \
        --num-games 200 --seed 8880000 --num-workers 20
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

# Import base evaluation infrastructure
from evaluate.evaluate_our_mcts_vs_rule_v2 import (
    Runtime,
    run_evaluation,
    _build_trace_context,
    _render_game_trace,
    _init_trace_log,
    _append_game_trace,
    _print_summary,
    _resolve_checkpoint_path,
    build_runtime,
    _init_parallel_worker,
    parse_args as _base_parse_args,
)
from evaluate.dds_player import DDSPlayer

# RL imports
from rl.policy_network import PolicyMLP
from rl.rl_exact_player import RLExactPlayer
from rl.rl_feature_encoder import RLFeatureEncoder
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)

MODEL_OUTPUT_DIM = 55

# ────────────────────────────────────────────────────────────────────────
# BothPlayer — selects checkpoint based on nil bids
# ────────────────────────────────────────────────────────────────────────
class BothPlayer(RLExactPlayer):
    """RLExactPlayer that switches policy network based on nil bids.

    After bidding (via set_teams callback):
    - Someone bids nil/blind_nil → use nil-specific policy (55_2nil.pt)
    - No one bids nil           → use non-nil policy (55_2.pt)
    """

    def __init__(
        self,
        policy_nets_nonil: list[PolicyMLP],
        policy_nets_nil: list[PolicyMLP],
        **kwargs,
    ) -> None:
        super().__init__(policy_nets=policy_nets_nonil, **kwargs)
        self.policy_nets_nonil = policy_nets_nonil
        self.policy_nets_nil = policy_nets_nil

    def set_teams(self, teams: list[int], bid_values: list[Any]) -> None:
        nil_bid = any(
            isinstance(bv, str) and bv in ("nil", "blind_nil")
            for bv in bid_values
        )
        if nil_bid:
            self.policy_nets = self.policy_nets_nil
        else:
            self.policy_nets = self.policy_nets_nonil


# ────────────────────────────────────────────────────────────────────────
# Worker globals (persist across games within a single worker process)
# ────────────────────────────────────────────────────────────────────────
_worker_rl_policy_nonil: PolicyMLP | None = None
_worker_rl_policy_nil: PolicyMLP | None = None
_worker_exact_solver: ExactDoubleDummyCppFastestSolver | None = None
_worker_encoder: RLFeatureEncoder | None = None


def _load_policy_in_worker(path: str, hidden_dims: list[int],
                           device: str) -> PolicyMLP | None:
    """Load a single policy network checkpoint in a worker process."""
    cp = Path(path)
    if not cp.exists():
        return None
    try:
        net = PolicyMLP(input_dim=264, hidden_dims=hidden_dims,
                        output_dim=MODEL_OUTPUT_DIM).to(device)
        net.eval()
        net.load(str(cp.resolve()), device=device)
        net.eval()
        return net
    except Exception as e:
        print(f"  [WARN] Failed to load policy {cp}: {e}", flush=True)
        return None


# ── Patch _init_parallel_worker ────────────────────────────────────────
import evaluate.evaluate_our_mcts_vs_rule_v2 as _eval_module

_original_init_worker = _eval_module._init_parallel_worker


def _patched_init_worker(args) -> None:
    """Initialize worker: call original, then load RL policy networks."""
    global _worker_rl_policy_nonil, _worker_rl_policy_nil
    global _worker_exact_solver, _worker_encoder

    _original_init_worker(args)

    device = args.device
    hidden_dims = getattr(args, 'rl_hidden_dims', [1024, 512, 512])

    _worker_rl_policy_nonil = _load_policy_in_worker(
        getattr(args, 'checkpoint_nonil', './55_2.pt'),
        hidden_dims, device,
    )
    _worker_rl_policy_nil = _load_policy_in_worker(
        getattr(args, 'checkpoint_nil', './55_2nil.pt'),
        hidden_dims, device,
    )

    _worker_exact_solver = ExactDoubleDummyCppFastestSolver()
    _worker_encoder = RLFeatureEncoder()


_eval_module._init_parallel_worker = _patched_init_worker

# ── Patch build_players ────────────────────────────────────────────────
_original_build_players = _eval_module.build_players


def _patched_build_players(args, runtime, game_seed, seat_specs=None):
    """Extended build_players with DDS and RL-Exact player support."""
    seat_specs_resolved = seat_specs or [args.p0, args.p1, args.p2, args.p3]

    has_dds = "dds" in seat_specs_resolved
    has_rl = "rl_exact" in seat_specs_resolved

    if not has_dds and not has_rl:
        return _original_build_players(args, runtime, game_seed, seat_specs)

    players = []
    for seat_index, spec in enumerate(seat_specs_resolved):
        if spec == "dds":
            players.append(DDSPlayer(
                bid_model=runtime.bid_model,
                bid_device=runtime.device,
            ))
            continue

        if spec == "rl_exact":
            global _worker_rl_policy_nonil, _worker_rl_policy_nil
            global _worker_exact_solver, _worker_encoder

            p_nonil = _worker_rl_policy_nonil
            p_nil = _worker_rl_policy_nil

            # Fallback: create random networks if loading failed
            if p_nonil is None:
                p_nonil = PolicyMLP(264, [1024, 512, 512],
                                    MODEL_OUTPUT_DIM).to(args.device)
                p_nonil.eval()
            if p_nil is None:
                p_nil = PolicyMLP(264, [1024, 512, 512],
                                  MODEL_OUTPUT_DIM).to(args.device)
                p_nil.eval()

            solver = _worker_exact_solver if _worker_exact_solver is not None \
                else ExactDoubleDummyCppFastestSolver()
            encoder = _worker_encoder if _worker_encoder is not None \
                else RLFeatureEncoder()

            player = BothPlayer(
                policy_nets_nonil=[p_nonil],
                policy_nets_nil=[p_nil],
                exact_solver=solver,
                encoder=encoder,
                exact_threshold=args.our_exact_threshold,
                is_training=False,
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
    """Parse args with RL-Exact specific defaults."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate DDS (perfect-info) vs RL-Exact in Spades.",
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
    parser.add_argument("--p0", type=str, default="dds",
                        help="Seat 0 model spec")
    parser.add_argument("--p1", type=str, default="rl_exact",
                        help="Seat 1 model spec")
    parser.add_argument("--p2", type=str, default="dds",
                        help="Seat 2 model spec")
    parser.add_argument("--p3", type=str, default="rl_exact",
                        help="Seat 3 model spec")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device")
    parser.add_argument("--our-checkpoint", type=str, default=None)
    parser.add_argument("--our-exact-threshold", type=int, default=36)
    parser.add_argument("--our-leaf-threshold", type=int, default=36)
    parser.add_argument("--our-simulations-per-action", type=int, default=40)
    parser.add_argument("--our-number-of-exact-solvers", type=int, default=64)
    parser.add_argument("--symmetric-seat-swap", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--torch-num-interop-threads", type=int, default=1)
    parser.add_argument("--mp-start-method", type=str, default="fork",
                        choices=["fork", "spawn", "forkserver"])

    # MCTS-related (kept for compatibility with base infra, unused by rl_exact)
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

    # RL-Exact checkpoint paths
    parser.add_argument("--checkpoint-nonil", type=str, default="./55_2.pt",
                        help="RL policy for games where no one bids nil")
    parser.add_argument("--checkpoint-nil", type=str, default="./55_2nil.pt",
                        help="RL policy for games where someone bids nil")
    parser.add_argument("--rl-hidden-dims", type=int, nargs="+",
                        default=[1024, 512, 512],
                        help="Policy network hidden layer sizes")

    # Trace / profile
    parser.add_argument("--trace-log-dir", type=str, default="logs")
    parser.add_argument("--profile-breakdown", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    import json
    import time

    args = parse_args()
    print(f"=== DDS vs RL-Exact Evaluation ===")
    print(f"Seats: [{args.p0}, {args.p1}, {args.p2}, {args.p3}]")
    print(f"RL checkpoint (nonil): {args.checkpoint_nonil}")
    print(f"RL checkpoint (nil):   {args.checkpoint_nil}")
    print(f"Exact threshold: {args.our_exact_threshold} "
          f"(first {52 - args.our_exact_threshold} cards use RL argmax)")
    print(f"Games: {args.num_games}, Seed: {args.seed}, "
          f"Symmetric: {args.symmetric_seat_swap}")
    print(f"Workers: {args.num_workers}")
    print()

    # ── For single-process mode, load policies here (fork inherits them) ──
    if args.num_workers <= 1:
        global _worker_rl_policy_nonil, _worker_rl_policy_nil
        global _worker_exact_solver, _worker_encoder

        _worker_rl_policy_nonil = _load_policy_in_worker(
            args.checkpoint_nonil, args.rl_hidden_dims, args.device,
        )
        _worker_rl_policy_nil = _load_policy_in_worker(
            args.checkpoint_nil, args.rl_hidden_dims, args.device,
        )
        _worker_exact_solver = ExactDoubleDummyCppFastestSolver()
        _worker_encoder = RLFeatureEncoder()

        if _worker_rl_policy_nonil is None:
            print("  [WARN] Nonil policy not loaded; using random weights",
                  flush=True)
        if _worker_rl_policy_nil is None:
            print("  [WARN] Nil policy not loaded; using random weights",
                  flush=True)

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
