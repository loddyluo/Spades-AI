"""Evaluate RL nil-policy vs Rule nil first-4 in Spades matchups.

仿 evaluate/evaluate_rl_exact_vs_rule_first4_exact.py 的 monkey-patch 风格:
- 两端**后 36 张都用 IS pool**(继承自 RLExactPlayer / RuleExactFirst4Player)
- 两端**前 4 墩按 bid 切换**:
    nil 局:    RL 用 55_2nil.pt MLP / Rule 用我们的 RuleBasedFirst4NilPlayer
    非 nil 局: RL 用 55_2.pt MLP    / Rule 用 RuleBasedFirst4Player (规则非 nil)
- **不筛局**,nil 和非 nil 局都打;统计时关心 nil 局的相对优势

用法:
    python evaluate/evaluate_rule_nil_vs_rl_nil.py \
        --p0 rl_both --p1 rule_nil_first4 --p2 rl_both --p3 rule_nil_first4 \
        --num-games 100 --seed 8880000 --num-workers 20

默认: [p0,p1,p2,p3] = [rl_both, rule_nil_first4, rl_both, rule_nil_first4]
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

# 复用对照脚本的全部 BothPlayer 定义 + worker 加载 + 大部分 patch
from evaluate.evaluate_rl_exact_vs_rule_first4_exact import (
    BothPlayer,
    _load_policy_in_worker,
    parse_args as _base_parse_args,
)
from evaluate.evaluate_our_mcts_vs_rule_v2 import (
    run_evaluation,
    _print_summary,
)
import evaluate.evaluate_our_mcts_vs_rule_v2 as _eval_module
import evaluate.evaluate_rl_exact_vs_rule_first4_exact as _peer_module

from rl.policy_network import PolicyMLP
from rl.rl_feature_encoder import RLFeatureEncoder
from strategy.rule_exact_first4_nil_player import RuleExactFirst4NilPlayer
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)

MODEL_OUTPUT_DIM = 55


# ────────────────────────────────────────────────────────────────────────
# Worker 全局(沿用对照脚本里的命名,但放本模块里独立)
# ────────────────────────────────────────────────────────────────────────
_worker_rl_policy_nonil: PolicyMLP | None = None
_worker_rl_policy_nil: PolicyMLP | None = None
_worker_exact_solver: ExactDoubleDummyCppFastestSolver | None = None
_worker_encoder: RLFeatureEncoder | None = None


# ── Patch _init_parallel_worker ────────────────────────────────────────
_original_init_worker = _eval_module._init_parallel_worker


def _patched_init_worker(args) -> None:
    """Worker 初始化:加载两个 RL net + 精确求解器 + encoder。"""
    global _worker_rl_policy_nonil, _worker_rl_policy_nil
    global _worker_exact_solver, _worker_encoder

    _original_init_worker(args)

    device = args.device
    hidden_dims = getattr(args, "rl_hidden_dims", [1024, 512, 512])

    _worker_rl_policy_nonil = _load_policy_in_worker(
        getattr(args, "checkpoint_nonil", "./55_2.pt"),
        hidden_dims, device,
    )
    _worker_rl_policy_nil = _load_policy_in_worker(
        getattr(args, "checkpoint_nil", "./55_2nil.pt"),
        hidden_dims, device,
    )

    _worker_exact_solver = ExactDoubleDummyCppFastestSolver()
    _worker_encoder = RLFeatureEncoder()


_eval_module._init_parallel_worker = _patched_init_worker


# ── Patch build_players:加 rl_both 和 rule_nil_first4 两个 spec ────────
_original_build_players = _eval_module.build_players


def _patched_build_players(args, runtime, game_seed, seat_specs=None):
    """扩展 build_players,支持 rl_both 和 rule_nil_first4。"""
    global _worker_rl_policy_nonil, _worker_rl_policy_nil
    global _worker_exact_solver, _worker_encoder

    seat_specs_resolved = seat_specs or [args.p0, args.p1, args.p2, args.p3]

    has_rl_both = "rl_both" in seat_specs_resolved
    has_rule_nil = "rule_nil_first4" in seat_specs_resolved

    if not has_rl_both and not has_rule_nil:
        return _original_build_players(args, runtime, game_seed, seat_specs)

    players = []
    for seat_index, spec in enumerate(seat_specs_resolved):
        if spec == "rule_nil_first4":
            solver = (
                _worker_exact_solver
                if _worker_exact_solver is not None
                else ExactDoubleDummyCppFastestSolver()
            )
            player = RuleExactFirst4NilPlayer(
                exact_solver=solver,
                exact_threshold=args.our_exact_threshold,
                bid_model=runtime.bid_model,
                bid_device=runtime.device,
            )
            players.append(player)
            continue

        if spec == "rl_both":
            p_nonil = _worker_rl_policy_nonil
            p_nil = _worker_rl_policy_nil

            # Fallback: 没加载到 → 随机权重(尽量别在真评测里发生)
            if p_nonil is None:
                p_nonil = PolicyMLP(
                    264, [1024, 512, 512], MODEL_OUTPUT_DIM,
                ).to(args.device)
                p_nonil.eval()
            if p_nil is None:
                p_nil = PolicyMLP(
                    264, [1024, 512, 512], MODEL_OUTPUT_DIM,
                ).to(args.device)
                p_nil.eval()

            solver = (
                _worker_exact_solver
                if _worker_exact_solver is not None
                else ExactDoubleDummyCppFastestSolver()
            )
            encoder = (
                _worker_encoder
                if _worker_encoder is not None
                else RLFeatureEncoder()
            )

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

        # 其它 spec 退回原版(支持 our_mcts / go_rule_2 / rl_exact 等)
        single_result = _original_build_players(
            args, runtime, game_seed, [spec, spec, spec, spec],
        )
        players.append(single_result[seat_index])
        continue

    return players


_eval_module.build_players = _patched_build_players
# 也要把 peer 模块里被它自己 patch 过的 build_players 顺序考虑一下:
# 我们的 patch 必须最后生效。peer 模块 import 时已经 patch 过一次,这里再覆盖。
_peer_module._eval_module = _eval_module  # 保持引用一致


# ── CLI ────────────────────────────────────────────────────────────────
def parse_args():
    """命令行参数,默认 [rl_both, rule_nil_first4, rl_both, rule_nil_first4]。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate RL nil-policy vs Rule nil first-4 in Spades.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--num-games", type=int, default=10,
                        help="Number of games to play")
    parser.add_argument("--output", type=str, default="",
                        help="Optional JSON output path")
    parser.add_argument("--disable-nil", action="store_true",
                        help="Disable nil bidding (评测 nil 局策略时不应启用)")
    parser.add_argument("--disable-blind-nil", action="store_true",
                        help="Disable blind nil")
    parser.add_argument("--p0", type=str, default="rl_both",
                        help="Seat 0 model spec")
    parser.add_argument("--p1", type=str, default="rule_nil_first4",
                        help="Seat 1 model spec")
    parser.add_argument("--p2", type=str, default="rl_both",
                        help="Seat 2 model spec")
    parser.add_argument("--p3", type=str, default="rule_nil_first4",
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

    # MCTS-related (兼容其它 spec)
    parser.add_argument("--our-exploration-constant", type=float, default=25.0)
    parser.add_argument("--our-policy-temperature", type=float, default=1.0)
    parser.add_argument("--our-mcts-determinization-count", type=int, default=5)
    parser.add_argument("--our-value-scale", type=float, default=25.0)
    parser.add_argument("--go-pv-checkpoint", type=str, default="")
    parser.add_argument("--go-bid-checkpoint", type=str, default="")

    # Bid model
    parser.add_argument("--bid-checkpoint", type=str,
                        default="./Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt")

    # Go MCTS args
    parser.add_argument("--go-mcts-runs", type=int, default=100)
    parser.add_argument("--go-mcts-steps", type=int, default=5)
    parser.add_argument("--go-mcts-c", type=float, default=0.3)
    parser.add_argument("--go-mcts-mu", type=float, default=0.01)
    parser.add_argument("--go-mcts-threshold", type=float, default=0.05)
    parser.add_argument("--go-argmax-threshold", type=float, default=0.05)

    # RL checkpoint paths
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

    args = parse_args()
    print(f"=== RL nil-policy vs Rule nil first-4 Evaluation ===")
    print(f"Seats: [{args.p0}, {args.p1}, {args.p2}, {args.p3}]")
    print(f"RL checkpoint (nonil): {args.checkpoint_nonil}")
    print(f"RL checkpoint (nil):   {args.checkpoint_nil}")
    print(f"Exact threshold: {args.our_exact_threshold} "
          f"(前 {52 - args.our_exact_threshold} 张走各自的前 4 墩策略,"
          f"后 {args.our_exact_threshold} 张统一 IS pool)")
    print(f"Games: {args.num_games}, Seed: {args.seed}, "
          f"Symmetric: {args.symmetric_seat_swap}")
    print(f"Workers: {args.num_workers}")
    print()

    # 单进程时(fork 时主进程也需要这些资源,因为 fork 会继承)
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
            print("  [WARN] Nonil policy 未加载,将用随机权重", flush=True)
        if _worker_rl_policy_nil is None:
            print("  [WARN] Nil policy 未加载,将用随机权重", flush=True)

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
