"""Configurable evaluation runner for local and collaborator Spades models.

File purpose:
- Evaluate the local MCTS player and the collaborator GO-MCTS players in the
  same local `SpadesMatchRunner` environment.
- Expose command-line switches for seat assignment, MCTS simulation depth,
  number of games, random seed, and checkpoint paths.

Function input/output summary:
- parse_args() -> argparse.Namespace
    Input: command-line arguments from `sys.argv`.
    Output: parsed evaluation configuration.
- build_runtime(args: argparse.Namespace) -> Runtime
    Input: parsed command-line arguments.
    Output: shared models and reusable MCTS configuration.
- build_players(args: argparse.Namespace, runtime: Runtime, game_seed: int) -> list[AIPlayer]
    Input: parsed command-line arguments, shared runtime, and the current game seed.
    Output: four local player adapters, one per seat.
- run_evaluation(args: argparse.Namespace) -> dict[str, Any]
    Input: parsed command-line arguments.
    Output: aggregate and per-game statistics for the requested matchup.
- main() -> None
    Input: command-line arguments.
    Output: prints a summary and optionally saves JSON results.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
GO_MCTS_DIR = REPO_ROOT / "evaluate" / "GO-MCTS"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GO_MCTS_DIR) not in sys.path:
    sys.path.insert(0, str(GO_MCTS_DIR))

from strategy.spades_match_runner import SpadesMatchRunner
from strategy.truncated_mcts_strategy import TruncatedMCTSConfig
from trick_taking.games.spades import SpadesRules

from adapters import GoPlayerAdapter, OurHandStrengthMCTSPlayer
from models import (
    ArgmaxPlayer,
    BidMLP,
    GOMCTSConfig,
    GOMCTSPlayer,
    MLPBidPlayer,
    RandomPlayer,
    RuleBasedPlayer,
    load_bid_mlp_model,
    load_gpt2_policy_value_model,
)


def _resolve_checkpoint_path(checkpoint_path: str) -> str:
        """Resolve a checkpoint path against common repository locations.

        Input:
        - checkpoint_path: user-supplied path, which may be relative to the
            current working directory or to common repo folders.

        Output:
        - A path string that exists if a matching file is found, otherwise the
            original input string.
        """
        if not checkpoint_path:
                return checkpoint_path

        candidate = Path(checkpoint_path)
        if candidate.is_file():
                return str(candidate)

        search_roots = [REPO_ROOT, REPO_ROOT / "result", REPO_ROOT / "evaluate", GO_MCTS_DIR]
        for root in search_roots:
                resolved = root / candidate
                if resolved.is_file():
                        return str(resolved)

        return checkpoint_path


@dataclass(frozen=True)
class Runtime:
    """Shared objects reused across games.

    Input:
    - Loaded checkpoints and reusable hyperparameters.

    Output:
    - Models and configs that seat factories can reuse without reloading.
    """

    device: str
    local_mcts_config: TruncatedMCTSConfig
    go_mcts_config: GOMCTSConfig
    go_pv_model: Any
    go_bid_model: BidMLP | None
    go_argmax_threshold: float
    go_pv_checkpoint: str
    go_bid_checkpoint: str


def parse_args() -> argparse.Namespace:
    """Parse the evaluation command line.

    Input:
    - `sys.argv`.

    Output:
    - Parsed arguments with seat specs and model hyperparameters.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate local and collaborator Spades models in one runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path")
    parser.add_argument("--disable-nil", action="store_true", help="Disable nil bidding in the local rules")
    parser.add_argument(
        "--disable-blind-nil",
        action="store_true",
        help="Disable blind nil bidding in the local rules",
    )
    parser.add_argument("--p0", type=str, default="our_mcts", help="Seat 0 model spec")
    parser.add_argument("--p1", type=str, default="go_rule", help="Seat 1 model spec")
    parser.add_argument("--p2", type=str, default="go_rule", help="Seat 2 model spec")
    parser.add_argument("--p3", type=str, default="go_random", help="Seat 3 model spec")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for loaded models")
    parser.add_argument("--our-checkpoint", type=str, default="", help="Optional local MLP checkpoint")
    parser.add_argument("--our-exact-threshold", type=int, default=30, help="Exact solve threshold for our MCTS")
    parser.add_argument("--our-leaf-threshold", type=int, default=24, help="Leaf threshold for our MCTS")
    parser.add_argument(
        "--our-simulations-per-action",
        type=int,
        default=50,
        help="Root simulations per legal action for our MCTS",
    )
    parser.add_argument(
        "--our-exploration-constant",
        type=float,
        default=1.5,
        help="PUCT exploration constant for our MCTS",
    )
    parser.add_argument(
        "--our-policy-temperature",
        type=float,
        default=1.0,
        help="Policy temperature for our MCTS leaf prior",
    )
    parser.add_argument(
        "--our-value-scale",
        type=float,
        default=25.0,
        help="Value scaling factor for our MCTS leaf value",
    )
    # 合作者仓库模型与 GOMCTS 参数：用于接入 Spades_AI_GO-MCTS 的推理模型和搜索配置。
    parser.add_argument("--go-pv-checkpoint", type=str, default="", help="合作者仓库 GPT-2 策略/价值模型的 checkpoint 路径")
    parser.add_argument("--go-bid-checkpoint", type=str, default="", help="合作者仓库叫牌 MLP 模型的 checkpoint 路径")
    parser.add_argument("--go-mcts-runs", type=int, default=100, help="合作者 GOMCTS 每次决策的模拟/rollout 次数")
    parser.add_argument("--go-mcts-steps", type=int, default=5, help="合作者 GOMCTS 每次 rollout 的最大步数/深度")
    parser.add_argument("--go-mcts-c", type=float, default=0.3, help="合作者 GOMCTS 的探索常数，越大越偏向探索")
    parser.add_argument("--go-mcts-mu", type=float, default=0.01, help="合作者 GOMCTS 的非法牌惩罚系数")
    parser.add_argument(
        "--go-mcts-threshold",
        type=float,
        default=0.05,
        help="合作者 GOMCTS 的剪枝阈值，越大越激进地剪枝",
    )
    parser.add_argument(
        "--go-argmax-threshold",
        type=float,
        default=0.05,
        help="合作者 ArgmaxPlayer 的阈值，影响 argmax/采样边界",
    )
    return parser.parse_args()


def build_runtime(args: argparse.Namespace) -> Runtime:
    """Load shared checkpoints and construct reusable model objects.

    Input:
    - args: parsed command-line arguments.

    Output:
    - A `Runtime` with optional loaded collaborator models.
    """
    local_mcts_config = TruncatedMCTSConfig(
        exact_threshold=args.our_exact_threshold,
        leaf_threshold=args.our_leaf_threshold,
        simulations_per_action=args.our_simulations_per_action,
        exploration_constant=args.our_exploration_constant,
        policy_temperature=args.our_policy_temperature,
        value_scale=args.our_value_scale,
        checkpoint_path=_resolve_checkpoint_path(args.our_checkpoint) if args.our_checkpoint else None,
    )
    go_mcts_config = GOMCTSConfig(
        n_runs=args.go_mcts_runs,
        n_steps=args.go_mcts_steps,
        C=args.go_mcts_c,
        mu=args.go_mcts_mu,
        threshold=args.go_mcts_threshold,
    )
    seat_specs = [args.p0, args.p1, args.p2, args.p3]
    need_go_pv_model = any(spec in {"go_argmax", "go_gomcts"} for spec in seat_specs)
    need_go_bid_model = any(spec == "go_mlp_bid" for spec in seat_specs)
    go_pv_model = (
        load_gpt2_policy_value_model(_resolve_checkpoint_path(args.go_pv_checkpoint), args.device)
        if need_go_pv_model and args.go_pv_checkpoint
        else None
    )
    go_bid_model = (
        load_bid_mlp_model(_resolve_checkpoint_path(args.go_bid_checkpoint), args.device)
        if need_go_bid_model and args.go_bid_checkpoint
        else None
    )
    return Runtime(
        device=args.device,
        local_mcts_config=local_mcts_config,
        go_mcts_config=go_mcts_config,
        go_pv_model=go_pv_model,
        go_bid_model=go_bid_model,
        go_argmax_threshold=args.go_argmax_threshold,
        go_pv_checkpoint=args.go_pv_checkpoint,
        go_bid_checkpoint=args.go_bid_checkpoint,
    )


def build_players(args: argparse.Namespace, runtime: Runtime, game_seed: int) -> list:
    """Build the four seat-specific players requested by the CLI.

    Input:
    - args: parsed command-line arguments with seat model specs.
    - runtime: shared models and configs loaded once for the whole run.
    - game_seed: the seed for the current game, used for random baselines.

    Output:
    - A list of 4 local `AIPlayer` adapters, one per seat.
    """
    seat_specs = [args.p0, args.p1, args.p2, args.p3]
    players = []
    for seat_index, spec in enumerate(seat_specs):
        if spec == "our_mcts":
            players.append(OurHandStrengthMCTSPlayer(config=runtime.local_mcts_config))
            continue
        if spec == "go_random":
            players.append(GoPlayerAdapter(RandomPlayer(seed=game_seed + seat_index)))
            continue
        if spec == "go_rule":
            players.append(GoPlayerAdapter(RuleBasedPlayer()))
            continue
        if spec == "go_argmax":
            if runtime.go_pv_model is None:
                if not runtime.go_pv_checkpoint:
                    raise SystemExit("--go-pv-checkpoint is required for spec=go_argmax")
                runtime = Runtime(
                    device=runtime.device,
                    local_mcts_config=runtime.local_mcts_config,
                    go_mcts_config=runtime.go_mcts_config,
                    go_pv_model=load_gpt2_policy_value_model(runtime.go_pv_checkpoint, runtime.device),
                    go_bid_model=runtime.go_bid_model,
                    go_argmax_threshold=runtime.go_argmax_threshold,
                    go_pv_checkpoint=runtime.go_pv_checkpoint,
                    go_bid_checkpoint=runtime.go_bid_checkpoint,
                )
            players.append(
                GoPlayerAdapter(
                    ArgmaxPlayer(
                        runtime.go_pv_model,
                        threshold=runtime.go_argmax_threshold,
                        device=runtime.device,
                    )
                )
            )
            continue
        if spec == "go_gomcts":
            if runtime.go_pv_model is None:
                if not runtime.go_pv_checkpoint:
                    raise SystemExit("--go-pv-checkpoint is required for spec=go_gomcts")
                runtime = Runtime(
                    device=runtime.device,
                    local_mcts_config=runtime.local_mcts_config,
                    go_mcts_config=runtime.go_mcts_config,
                    go_pv_model=load_gpt2_policy_value_model(runtime.go_pv_checkpoint, runtime.device),
                    go_bid_model=runtime.go_bid_model,
                    go_argmax_threshold=runtime.go_argmax_threshold,
                    go_pv_checkpoint=runtime.go_pv_checkpoint,
                    go_bid_checkpoint=runtime.go_bid_checkpoint,
                )
            players.append(
                GoPlayerAdapter(
                    GOMCTSPlayer(
                        runtime.go_pv_model,
                        config=runtime.go_mcts_config,
                        device=runtime.device,
                    )
                )
            )
            continue
        if spec == "go_mlp_bid":
            if runtime.go_bid_model is None:
                if not runtime.go_bid_checkpoint:
                    raise SystemExit("--go-bid-checkpoint is required for spec=go_mlp_bid")
                runtime = Runtime(
                    device=runtime.device,
                    local_mcts_config=runtime.local_mcts_config,
                    go_mcts_config=runtime.go_mcts_config,
                    go_pv_model=runtime.go_pv_model,
                    go_bid_model=load_bid_mlp_model(runtime.go_bid_checkpoint, runtime.device),
                    go_argmax_threshold=runtime.go_argmax_threshold,
                    go_pv_checkpoint=runtime.go_pv_checkpoint,
                    go_bid_checkpoint=runtime.go_bid_checkpoint,
                )
            players.append(GoPlayerAdapter(MLPBidPlayer(runtime.go_bid_model, device=runtime.device)))
            continue
        raise SystemExit(f"Unknown seat model spec: {spec!r}")

    return players


def _play_single_game(args: argparse.Namespace, runtime: Runtime, seed: int) -> dict[str, Any]:
    """Play one seeded game and collect the per-seat and per-team scores.

    Input:
    - args: parsed command-line arguments.
    - runtime: shared models and configs loaded once for the whole run.
    - seed: game seed used for deal generation.

    Output:
    - A dictionary containing the game seed, seat scores, team scores, and
      winner index.
    """
    rules = SpadesRules(enable_nil=not args.disable_nil, enable_blind_nil=not args.disable_blind_nil)
    runner = SpadesMatchRunner(
        players=build_players(args, runtime, seed),
        seed=seed,
        verbose=False,
        rules=rules,
    )
    result = runner.play_game()
    seat_scores = [float(score) for score in result.scores]
    return {
        "seed": seed,
        "seat_scores": seat_scores,
        "team0_score": float((seat_scores[0] + seat_scores[2]) / 2.0),
        "team1_score": float((seat_scores[1] + seat_scores[3]) / 2.0),
        "winner": int(result.winner),
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Run the requested matchup and compute aggregate statistics.

    Input:
    - args: parsed command-line arguments.

    Output:
    - A result dictionary containing per-game records and aggregate seat/team
      averages.
    """
    runtime = build_runtime(args)
    games: list[dict[str, Any]] = []
    seat_totals = [0.0, 0.0, 0.0, 0.0]
    for offset in range(args.num_games):
        seed = args.seed + offset
        game = _play_single_game(args, runtime, seed)
        games.append(game)
        for index, score in enumerate(game["seat_scores"]):
            seat_totals[index] += float(score)

    n = max(args.num_games, 1)
    seat_avgs = [total / n for total in seat_totals]
    team0_avg = (seat_avgs[0] + seat_avgs[2]) / 2.0
    team1_avg = (seat_avgs[1] + seat_avgs[3]) / 2.0
    return {
        "seat_specs": [args.p0, args.p1, args.p2, args.p3],
        "num_games": args.num_games,
        "seed": args.seed,
        "seat_avg_scores": seat_avgs,
        "team_avg_scores": {"team0": team0_avg, "team1": team1_avg},
        "games": games,
    }


def _print_summary(result: dict[str, Any]) -> None:
    """Print a compact human-readable summary of the matchup.

    Input:
    - result: dictionary returned by `run_evaluation`.

    Output:
    - None.
    """
    print("=" * 72)
    print("Spades matchup evaluation")
    print("=" * 72)
    print(f"Games: {result['num_games']} | Base seed: {result['seed']}")
    for seat_index, (spec, avg) in enumerate(zip(result["seat_specs"], result["seat_avg_scores"])):
        print(f"Seat {seat_index}: {spec:<18} avg score = {avg:+.2f}")
    team_scores = result["team_avg_scores"]
    print(f"Team 0 avg score: {team_scores['team0']:+.2f}")
    print(f"Team 1 avg score: {team_scores['team1']:+.2f}")
    print("=" * 72)


def main() -> None:
    """Command-line entry point for the matchup evaluation script.

    Input:
    - `sys.argv`.

    Output:
    - Prints a summary and optionally writes a JSON file.
    """
    args = parse_args()
    result = run_evaluation(args)
    _print_summary(result)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON results to: {output_path}")


if __name__ == "__main__":
    main()
