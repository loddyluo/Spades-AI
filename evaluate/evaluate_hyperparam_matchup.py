"""
超参数选拔赛: 两个具有不同超参数的 RuleExactFirst4NilPlayer 队式对打。

用法:
    # 用默认超参数 vs 自定义超参数，打 100 局
    python evaluate/evaluate_hyperparam_matchup.py \
        --config-a configs/4.yaml \
        --config-b configs/3.yaml \
        --num-games 100 --seed 8880000 --num-workers 20

    python evaluate/evaluate_hyperparam_matchup.py \
        --config-a configs/1.yaml \
        --config-b configs/7.yaml \
        --num-games 100 --seed 8880000 --num-workers 20

    python evaluate/evaluate_hyperparam_matchup.py \
        --config-a configs/8.yaml \
        --config-b configs/seed2.yaml \
        --num-games 72 --seed 8880014 --num-workers 20


    python evaluate/evaluate_hyperparam_matchup.py \
        --config-a configs/5.yaml \
        --config-b configs/6.yaml \
        --num-games 100 --seed 8880000 --num-workers 20

    【半决赛】
    python evaluate/evaluate_hyperparam_matchup.py \
        --config-a configs/7.yaml \
        --config-b configs/5.yaml \
        --num-games 100 --seed 8880000 --num-workers 20

    python evaluate/evaluate_hyperparam_matchup.py \
        --config-a configs/4.yaml \
        --config-b configs/8.yaml \
        --num-games 100 --seed 8880000 --num-workers 20

    【决赛】
    python evaluate/evaluate_hyperparam_matchup.py \
        --config-a configs/5.yaml \
        --config-b configs/4.yaml \
        --num-games 100 --seed 8880000 --num-workers 20

    【铜牌赛】
    python evaluate/evaluate_hyperparam_matchup.py \
        --config-a configs/7.yaml \
        --config-b configs/8.yaml \
        --num-games 100 --seed 8880000 --num-workers 20

    # 只指定一个 config：另一侧用默认值
    python evaluate/evaluate_hyperparam_matchup.py \
        --config-a configs/hyperparams_A.yaml \
        --num-games 200

    # 生成默认配置文件模板
    python evaluate/evaluate_hyperparam_matchup.py --write-default-config
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GO_MCTS_DIR = REPO_ROOT / "evaluate" / "GO-MCTS"
for p in (str(REPO_ROOT), str(GO_MCTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluate.evaluate_our_mcts_vs_rule_v2 import (
    TracePlayerProxy,
    _build_trace_context,
    _init_trace_log,
    _append_game_trace,
)
from evaluate.evaluate_rl_first4_vs_rule_first4 import (
    _build_exact_solver,
    _compute_team_scores,
)
from residual_bidder.actions import to_local_bid
from residual_bidder.deployment import (
    DEFAULT_CHECKPOINT_PATH as DEFAULT_ACTING_BID_CKPT,
    DEFAULT_CONFIG_PATH as DEFAULT_RESIDUAL_BIDDER_CONFIG,
    load_deployed_acting_bidder,
)
from strategy.hyperparam_config import HyperparamConfig
from strategy.rule_exact_first4_nil_player import RuleExactFirst4NilPlayer
from strategy.spades_match_runner import SpadesMatchRunner
from trick_taking.game_state import Phase, Bid
from trick_taking.games.spades import SpadesRules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter tournament: two RuleExactFirst4NilPlayer configs vs each other",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-a", type=str, default="",
                        help="YAML config for player A (team 0). Empty = defaults.")
    parser.add_argument("--config-b", type=str, default="",
                        help="YAML config for player B (team 1). Empty = defaults.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Total games (each episode = 2 games with swapped seats)")
    parser.add_argument("--exact-threshold", type=int, default=36,
                        help="Remaining cards threshold for exact solver")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--acting-bid-checkpoint", type=str,
                        default=str(DEFAULT_ACTING_BID_CKPT),
                        help="Path to the selected residual-Q acting checkpoint")
    parser.add_argument("--residual-bidder-config", type=str,
                        default=str(DEFAULT_RESIDUAL_BIDDER_CONFIG),
                        help="Frozen residual bidder provenance config")
    parser.add_argument("--bid-policy-seed", type=int, default=None,
                        help="Override the frozen acting-policy seed")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Parallel workers (1 = single process)")
    parser.add_argument("--max-redeals", type=int, default=64,
                        help="Max redeals per game when encountering nil bids")
    parser.add_argument("--print-every", type=int, default=50,
                        help="Progress print interval (0 = silent)")
    parser.add_argument("--trace-log-dir", type=str, default="SCL",
                        help="Directory for per-game trace logs; set to empty string to disable")
    parser.add_argument("--output", type=str, default="",
                        help="Optional JSON output path")
    parser.add_argument("--write-default-config", action="store_true",
                        help="Write default YAML config templates and exit")
    return parser.parse_args()


def _load_config(path: str) -> HyperparamConfig:
    if not path:
        return HyperparamConfig.default()
    return HyperparamConfig.from_yaml(path)


def _build_player(
    config: HyperparamConfig,
    exact_solver,
    exact_threshold: int,
) -> RuleExactFirst4NilPlayer:
    return RuleExactFirst4NilPlayer(
        exact_solver=exact_solver,
        exact_threshold=exact_threshold,
        bid_model=None,
        bid_device="cpu",
        hyperparam_config=config,
    )


def _build_team_players(
    config_a: HyperparamConfig,
    config_b: HyperparamConfig,
    exact_solver,
    exact_threshold: int,
    swap_seats: bool,
) -> list:
    """Team match: config A on team 0 (seats 0,2), config B on team 1 (seats 1,3)."""
    build_a = lambda: _build_player(config_a, exact_solver, exact_threshold)
    build_b = lambda: _build_player(config_b, exact_solver, exact_threshold)
    if not swap_seats:
        return [build_a(), build_b(), build_a(), build_b()]
    return [build_b(), build_a(), build_b(), build_a()]


_SEAT_SPECS_BASE = ["hp_A", "hp_B", "hp_A", "hp_B"]
_SEAT_SPECS_SWAPPED = ["hp_B", "hp_A", "hp_B", "hp_A"]


def _seat_specs_for_game(swap: bool) -> list[str]:
    return _SEAT_SPECS_SWAPPED if swap else _SEAT_SPECS_BASE


def play_one_game(
    players: list,
    seed: int,
    rules: SpadesRules,
    acting_bidder,
    game_trace: dict[str, Any] | None = None,
    show_card_progress: bool = False,
) -> Any:
    """Play one seeded game (no nil-skip, matching evaluate_rl_exact_vs_rule_first4_exact.py).

    Bidding uses the residual-Q acting bidder (``acting_bidder.choose()``) instead
    of the player's ``place_bid()``, matching ``game_server.py`` behavior.

    If game_trace is provided, players should already be wrapped with TracePlayerProxy.
    If show_card_progress is True, display a per-game tqdm progress bar.
    """
    pbar = None
    if show_card_progress:
        pbar = tqdm(total=52, desc=f"Seed {seed}", unit="card", leave=False)

    runner = SpadesMatchRunner(
        players=players,
        seed=seed,
        verbose=False,
        rules=rules,
        on_card_played=(lambda _c, _t: pbar.update(1)) if pbar else None,
        on_bidding_finished=(lambda: pbar.update(0) if pbar else None) if show_card_progress else None,
    )
    runner._start_game()

    # Manual bidding phase using residual-Q acting bidder (same as game_server.py)
    state = runner.state
    state.phase = Phase.BIDDING
    while not rules.end_bidding(state):
        bidder = state.current_bidder
        legal_bids = rules.legal_bids(state, bidder)

        decision = acting_bidder.choose(
            state, legal_bids,
            logical_seat=bidder,
            deal_id=f"eval:seed:{seed}",
            room_id="hyperparam_matchup",
        )
        bid = to_local_bid(decision.action)

        if legal_bids and bid not in legal_bids:
            raise ValueError(f"玩家{bidder}叫牌非法: {bid!r}, 合法叫牌: {legal_bids}")

        state.bids.append(Bid(player_id=bidder, value=bid, is_pass=(bid == "pass")))
        if bid != "pass":
            state.max_bid[bidder] = bid

        for player in players:
            player.bid_placed(bidder, bid)

        # 记录叫牌到 trace（手动叫牌未经过 TracePlayerProxy.place_bid）
        if game_trace is not None:
            game_trace["bids"].append({
                "seat": bidder,
                "spec": game_trace["players"][bidder]["spec"],
                "legal_bids": [str(b) for b in legal_bids],
                "chosen_bid": str(bid),
                "raw_strength": None,
            })

        state.current_bidder = rules.next_bid_turn(state)
        runner._refresh_all_player_features()

    runner._set_teams()
    runner._play_phase()
    result = runner._score_game()

    if pbar is not None:
        pbar.close()

    # Enrich game_trace with tricks/scores if available
    if game_trace is not None:
        game_trace["tricks_won"] = [int(t) for t in result.tricks_won]
        game_trace["scores"] = [float(s) for s in result.scores]

    return result


def play_episode(
    config_a: HyperparamConfig,
    config_b: HyperparamConfig,
    exact_solver,
    exact_threshold: int,
    episode_seed: int,
    rules: SpadesRules,
    acting_bidder,
    trace_enabled: bool = False,
    show_card_progress: bool = False,
) -> dict[str, Any] | None:
    """One team match episode = 2 games (seats swapped). Returns episode stats
    and optionally collected traces."""
    team_a_score = 0.0
    team_b_score = 0.0
    traces: list[dict[str, Any]] = []

    for game_idx in range(2):
        swap = (game_idx == 1)
        players = _build_team_players(
            config_a, config_b, exact_solver, exact_threshold,
            swap_seats=swap,
        )
        game_seed = episode_seed  # 队式赛：两局同一副牌，只交换座位
        seat_specs = _seat_specs_for_game(swap)

        game_trace: dict[str, Any] | None = None
        if trace_enabled:
            game_trace = _build_trace_context(game_seed, seat_specs)
            players = [
                TracePlayerProxy(p, seat, seat_specs[seat], game_trace)
                for seat, p in enumerate(players)
            ]

        result = play_one_game(
            players=players,
            seed=game_seed,
            rules=rules,
            acting_bidder=acting_bidder,
            game_trace=game_trace,
            show_card_progress=show_card_progress,
        )

        team0_score, team1_score = _compute_team_scores(result)
        if not swap:
            a_score, b_score = team0_score, team1_score
        else:
            a_score, b_score = team1_score, team0_score
        team_a_score += a_score
        team_b_score += b_score

        if game_trace is not None:
            traces.append(game_trace)

    return {
        "episode_game_reward": (team_a_score - team_b_score) / 2.0,
        "episode_reward": team_a_score - team_b_score,
        "team_a_score": team_a_score,
        "team_b_score": team_b_score,
        "traces": traces,
    }


# ── Worker globals (preloaded once per process, matching evaluate_rl_exact_vs_rule_first4_exact.py) ──
_WORKER_SOLVER = None
_WORKER_ACTING_BIDDER = None
_WORKER_RULES = None
_WORKER_CONFIG_A = None
_WORKER_CONFIG_B = None
_WORKER_EXACT_THRESHOLD = None


def _init_worker(init_args: tuple) -> None:
    """Preload solver/acting bidder once per worker process."""
    global _WORKER_SOLVER, _WORKER_ACTING_BIDDER, _WORKER_RULES
    global _WORKER_CONFIG_A, _WORKER_CONFIG_B
    global _WORKER_EXACT_THRESHOLD

    (config_a, config_b, exact_threshold,
     acting_bid_ckpt, residual_bidder_config, device, bid_policy_seed) = init_args
    _WORKER_CONFIG_A = config_a
    _WORKER_CONFIG_B = config_b
    # 并行模式下，每局内部 solver 只用单线程；多局之间靠外层 ProcessPoolExecutor 并行
    _WORKER_CONFIG_A.num_workers = 1
    _WORKER_CONFIG_B.num_workers = 1
    _WORKER_EXACT_THRESHOLD = exact_threshold
    _WORKER_SOLVER = _build_exact_solver()
    _WORKER_ACTING_BIDDER = load_deployed_acting_bidder(
        checkpoint_path=Path(acting_bid_ckpt),
        config_path=Path(residual_bidder_config),
        repo_root=REPO_ROOT,
        device=device,
        policy_seed=bid_policy_seed,
    )
    _WORKER_RULES = SpadesRules(enable_nil=True, enable_blind_nil=False)


def _worker_eval_single(job: tuple[int, int]) -> dict[str, Any] | None:
    """Play one episode in a worker process (reads cached globals)."""
    ep_idx, base_seed = job
    return play_episode(
        config_a=_WORKER_CONFIG_A,
        config_b=_WORKER_CONFIG_B,
        exact_solver=_WORKER_SOLVER,
        exact_threshold=_WORKER_EXACT_THRESHOLD,
        episode_seed=base_seed + ep_idx,
        rules=_WORKER_RULES,
        acting_bidder=_WORKER_ACTING_BIDDER,
        trace_enabled=True,
        show_card_progress=True,  # each worker shows its own per-card progress
    )




def main() -> None:
    args = parse_args()

    # Optional: write default config template
    if args.write_default_config:
        cfg_dir = Path("configs")
        cfg_dir.mkdir(exist_ok=True)
        HyperparamConfig.default().to_yaml(cfg_dir / "hyperparams_default.yaml")
        HyperparamConfig(
            num_proposals=500,
            num_proposals_limit=3000,
            min_pool_size=50,
            bad_action_weight="0.0",
            trick_num_threshold=6,
            swap_is_fill=True,
            multiplier_clip=20.0,
            multiplier_clip_factor=2.0,
        ).to_yaml(cfg_dir / "hyperparams_experimental.yaml")
        print("Default config templates written to configs/")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)

    config_a = _load_config(args.config_a)
    config_b = _load_config(args.config_b)
    num_episodes = args.num_games // 2
    trace_enabled = bool(args.trace_log_dir)

    print("=" * 72, flush=True)
    print("超参数选拔赛 — RuleExactFirst4NilPlayer 队式对打", flush=True)
    print(f"总对局数: {args.num_games} ({num_episodes} episodes × 2 games)", flush=True)
    print(f"Config A: {args.config_a or '(defaults)'}", flush=True)
    print(f"Config B: {args.config_b or '(defaults)'}", flush=True)
    print(f"精确阈值: {args.exact_threshold}", flush=True)
    print(f"叫牌: residual-Q acting bidder ({args.acting_bid_checkpoint})", flush=True)
    print(f"Workers: {args.num_workers}", flush=True)
    print(f"Trace log: {args.trace_log_dir if trace_enabled else 'disabled'}", flush=True)
    print(f"JSON output: {args.output or 'none'}", flush=True)
    print("=" * 72, flush=True)
    print("Config A:", flush=True)
    _print_config_diff(config_a)
    print("Config B:", flush=True)
    _print_config_diff(config_b)
    print("=" * 72, flush=True)

    # ── Initialize trace log BEFORE processing (matching evaluate_rl_exact_vs_rule_first4_exact.py) ──
    trace_log_path: str | None = None
    if trace_enabled:
        trace_log_path = _init_trace_log(
            args.trace_log_dir, args.seed, args.num_games, _SEAT_SPECS_BASE)

    t_start = time.perf_counter()
    all_rewards: list[float] = []
    a_scores: list[float] = []
    b_scores: list[float] = []
    # redeals_list no longer needed (nil-skip removed)
    skipped_episodes = 0
    all_game_records: list[dict[str, Any]] = []

    if args.num_workers <= 1:
        exact_solver = _build_exact_solver()
        acting_bidder = load_deployed_acting_bidder(
            checkpoint_path=Path(args.acting_bid_checkpoint),
            config_path=Path(args.residual_bidder_config),
            repo_root=REPO_ROOT,
            device=args.device,
            policy_seed=args.bid_policy_seed,
        )
        rules = SpadesRules(enable_nil=True, enable_blind_nil=False)

        for ep_idx in tqdm(range(num_episodes), desc="Playing games", unit="episode"):
            ep_res = play_episode(
                config_a=config_a,
                config_b=config_b,
                exact_solver=exact_solver,
                exact_threshold=args.exact_threshold,
                episode_seed=args.seed + ep_idx,
                rules=rules,
                acting_bidder=acting_bidder,
                trace_enabled=trace_enabled,
                show_card_progress=(args.num_workers <= 1),
            )
            if ep_res is None:
                skipped_episodes += 1
                continue

            all_rewards.append(ep_res["episode_game_reward"])
            a_scores.append(ep_res["team_a_score"])
            b_scores.append(ep_res["team_b_score"])
            # redeals no longer tracked (nil-skip removed)

            # Write trace log immediately after each game
            for game_trace in ep_res.get("traces", []):
                if trace_log_path is not None:
                    _append_game_trace(trace_log_path, {"trace": game_trace})
                seat_scores = game_trace.get("scores", [0, 0, 0, 0])
                all_game_records.append({
                    "seed": game_trace["seed"],
                    "seat_specs": game_trace["players"],
                    "seat_scores": seat_scores,
                    "team0_score": (seat_scores[0] + seat_scores[2]) / 2.0,
                    "team1_score": (seat_scores[1] + seat_scores[3]) / 2.0,
                })
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        init_args = (config_a, config_b, args.exact_threshold,
                     args.acting_bid_checkpoint, args.residual_bidder_config,
                     args.device, args.bid_policy_seed)
        jobs = [(ep_idx, args.seed) for ep_idx in range(num_episodes)]
        n_workers = min(args.num_workers, num_episodes)

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(init_args,),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            for ep_res in tqdm(
                executor.map(_worker_eval_single, jobs),
                total=num_episodes, desc="Playing games", unit="episode",
            ):
                if ep_res is None:
                    skipped_episodes += 1
                    continue
                all_rewards.append(ep_res["episode_game_reward"])
                a_scores.append(ep_res["team_a_score"])
                b_scores.append(ep_res["team_b_score"])
                # redeals no longer tracked (nil-skip removed)
                # Write trace log immediately after each game
                for game_trace in ep_res.get("traces", []):
                    if trace_log_path is not None:
                        _append_game_trace(trace_log_path, {"trace": game_trace})
                    seat_scores = game_trace.get("scores", [0, 0, 0, 0])
                    all_game_records.append({
                        "seed": game_trace["seed"],
                        "seat_specs": game_trace["players"],
                        "seat_scores": seat_scores,
                        "team0_score": (seat_scores[0] + seat_scores[2]) / 2.0,
                        "team1_score": (seat_scores[1] + seat_scores[3]) / 2.0,
                    })
        skipped_episodes = num_episodes - len(all_rewards)

    t_elapsed = time.perf_counter() - t_start
    n_ep = len(all_rewards)
    n_games = n_ep * 2

    print(flush=True)
    print("=" * 72, flush=True)
    print("评估完成", flush=True)
    print(f"有效 episode: {n_ep}/{num_episodes} (跳过 {skipped_episodes})", flush=True)
    print(f"总耗时: {t_elapsed:.0f}s "
          f"(平均 {t_elapsed / max(n_ep, 1):.2f}s/episode)", flush=True)
    print(f"重发已移除（nil-skip 已删除，与 evaluate_rl_exact_vs_rule_first4_exact.py 一致）", flush=True)
    print(flush=True)

    if n_ep == 0:
        print("无有效对局（可能 max-redeals 过小或叫牌模型几乎总叫 nil）", flush=True)
        print("=" * 72, flush=True)
        return

    print(f"{'统计项':<35} {'数值':>10}", flush=True)
    print("-" * 47, flush=True)
    print(f"{'Config A 平均 game 奖励':<35} {np.mean(all_rewards):>+10.2f}", flush=True)
    print(f"{'Config A 方平均总分':<35} {np.mean(a_scores):>+10.2f}", flush=True)
    print(f"{'Config B 方平均总分':<35} {np.mean(b_scores):>+10.2f}", flush=True)
    print(f"{'总对局数':<35} {n_games:>10d}", flush=True)
    print(f"{'A - B 均值':<35} {np.mean(all_rewards):>+10.2f}", flush=True)

    if n_ep >= 20:
        print(flush=True)
        print(f"{'局段':<20} {'A game 奖励':>15}", flush=True)
        print("-" * 38, flush=True)
        for start in range(0, n_ep, max(1, n_ep // 5)):
            end = min(start + max(1, n_ep // 5), n_ep)
            seg = all_rewards[start:end]
            if seg:
                print(f"{start * 2 + 1:>4}~{end * 2:>4}  {np.mean(seg):>+15.2f}", flush=True)

    print("=" * 72, flush=True)

    # ── Trace log path (written during processing) ──
    if trace_log_path is not None:
        print(f"Trace log: {trace_log_path}", flush=True)

    # ── JSON output ──
    if args.output:
        result_json = {
            "seed": args.seed,
            "config_a": args.config_a or "(defaults)",
            "config_b": args.config_b or "(defaults)",
            "num_games": n_games,
            "num_episodes": n_ep,
            "skipped_episodes": skipped_episodes,
            "avg_game_reward": float(np.mean(all_rewards)),
            "avg_team_a_score": float(np.mean(a_scores)),
            "avg_team_b_score": float(np.mean(b_scores)),
            "games": all_game_records,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result_json, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"JSON saved to: {output_path}", flush=True)


def _print_config_diff(cfg: HyperparamConfig) -> None:
    """Print config values, skipping ones at default."""
    defaults = HyperparamConfig.default()
    fields = [
        ("num_proposals", cfg.num_proposals, defaults.num_proposals),
        ("num_proposals_limit", cfg.num_proposals_limit, defaults.num_proposals_limit),
        ("min_pool_size", cfg.min_pool_size, defaults.min_pool_size),
        ("bad_action_weight", cfg.bad_action_weight, defaults.bad_action_weight),
        ("bad_action_penalty_factor", cfg.bad_action_penalty_factor, defaults.bad_action_penalty_factor),
        ("gamma", cfg.gamma, defaults.gamma),
        ("trick_num_threshold", cfg.trick_num_threshold, defaults.trick_num_threshold),
        ("swap_is_fill", cfg.swap_is_fill, defaults.swap_is_fill),
        ("multiplier_clip", cfg.multiplier_clip, defaults.multiplier_clip),
        ("multiplier_clip_factor", cfg.multiplier_clip_factor, defaults.multiplier_clip_factor),
        ("budget", str(cfg.budget.thresholds), str(defaults.budget.thresholds)),
    ]
    for name, val, default in fields:
        if val != default:
            print(f"  {name}: {val} (default: {default})", flush=True)
        else:
            print(f"  {name}: {val}", flush=True)


if __name__ == "__main__":
    main()
