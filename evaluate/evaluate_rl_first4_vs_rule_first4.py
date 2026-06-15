"""
前 4 墩对比评估: RL MLP (rl_exact) vs 规则式 (rule_exact_first4)。

流程 (每局):
1. 发牌并完成叫牌 (启用 nil, 禁用 blind_nil)
2. 若四家任一人叫 nil → 重新发牌 (换 seed), 直到无人叫 nil
3. 前 4 墩由各自策略出牌, 后 9 墩 (剩余 36 张) 双方均使用同一精确求解器

队式赛: 每 episode 打 2 局并交换座位 (与 evaluate/eval_rule_first4_multicpu.py 一致)。

用法:
    python evaluate/evaluate_rl_first4_vs_rule_first4.py --num-games 200 --seed 42
    python evaluate/evaluate_rl_first4_vs_rule_first4.py --num-games 400 --num-workers 8
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GO_MCTS_DIR = REPO_ROOT / "evaluate" / "GO-MCTS"
for p in (str(REPO_ROOT), str(GO_MCTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rl.policy_network import PolicyMLP
from rl.rl_exact_player import RLExactPlayer
from rl.rl_feature_encoder import RLFeatureEncoder
from strategy.rule_exact_first4_player import RuleExactFirst4Player
from strategy.spades_match_runner import SpadesMatchRunner
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)

MODEL_INPUT_DIM = 264
MODEL_HIDDEN_DIMS = [1024, 512, 512]
MODEL_OUTPUT_DIM = 55


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RL first-4 (MLP) vs Rule first-4, team match, skip nil deals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--num-games", type=int, default=200,
        help="评估总对局数 (每 episode 2 局, 共 num_games/2 次队式赛)",
    )
    parser.add_argument("--exact-threshold", type=int, default=36,
                        help="剩余牌数 <= 该值时使用精确求解器")
    parser.add_argument("--device", type=str, default="cpu", help="Torch 设备")
    parser.add_argument("--bid-checkpoint", type=str,
                        default="./Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt",
                        help="叫牌 MLP checkpoint")
    parser.add_argument("--checkpoint-nonil", type=str, default="./55_2.pt",
                        help="前 4 墩 RL 策略 (仅非 nil 局使用)")
    parser.add_argument("--rl-hidden-dims", type=int, nargs="+",
                        default=MODEL_HIDDEN_DIMS,
                        help="RL 策略网络隐藏层")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="并行 worker 数 (1=单进程)")
    parser.add_argument("--max-redeals", type=int, default=64,
                        help="单局最多重发次数 (仍遇 nil 则跳过该局)")
    parser.add_argument("--print-every", type=int, default=50,
                        help="单进程模式下每 N episode 打印进度 (0=不打印)")
    return parser.parse_args()


def _has_nil_bid(max_bid: list[Any]) -> bool:
    return any(
        isinstance(b, str) and b in ("nil", "blind_nil")
        for b in max_bid
    )


def _compute_team_scores(result: Any) -> tuple[float, float]:
    """与 eval_rule_first4_multicpu / eval_rl_multicpu 一致。"""
    scores = result.scores
    return scores[0] / 2.0, scores[1] / 2.0


def _build_exact_solver() -> ExactDoubleDummyCppFastestSolver:
    solver = ExactDoubleDummyCppFastestSolver()
    if not solver.native_available:
        from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver

        return ExactDoubleDummySolver()  # type: ignore[return-value]
    return solver


def _load_bid_model(bid_checkpoint: str, device: str):
    try:
        from models import load_bid_mlp_model

        cp = Path(bid_checkpoint)
        if cp.exists():
            return load_bid_mlp_model(str(cp.resolve()), device)
    except Exception:
        pass
    return None


def _load_policy(path: str, hidden_dims: list[int], device: str) -> PolicyMLP:
    cp = Path(path)
    net = PolicyMLP(MODEL_INPUT_DIM, hidden_dims, MODEL_OUTPUT_DIM).to(device)
    net.eval()
    if cp.exists():
        net.load(str(cp.resolve()), device=device)
        net.eval()
    return net


def _build_rl_player(
    policy_net: PolicyMLP,
    exact_solver: ExactDoubleDummyCppFastestSolver,
    encoder: RLFeatureEncoder,
    exact_threshold: int,
    bid_model,
    bid_device: str,
) -> RLExactPlayer:
    return RLExactPlayer(
        policy_nets=[policy_net],
        exact_solver=exact_solver,
        encoder=encoder,
        exact_threshold=exact_threshold,
        is_training=False,
        bid_model=bid_model,
        bid_device=bid_device,
    )


def _build_rule_player(
    exact_solver: ExactDoubleDummyCppFastestSolver,
    exact_threshold: int,
    bid_model,
    bid_device: str,
) -> RuleExactFirst4Player:
    return RuleExactFirst4Player(
        exact_solver=exact_solver,
        exact_threshold=exact_threshold,
        bid_model=bid_model,
        bid_device=bid_device,
    )


def _build_team_players(
    policy_net: PolicyMLP,
    exact_solver: ExactDoubleDummyCppFastestSolver,
    encoder: RLFeatureEncoder,
    exact_threshold: int,
    bid_model,
    bid_device: str,
    swap_seats: bool,
) -> list:
    """队式座位: RL 在 0/2, Rule 在 1/3 (或交换后相反)。"""
    rl_kw = dict(
        policy_net=policy_net,
        exact_solver=exact_solver,
        encoder=encoder,
        exact_threshold=exact_threshold,
        bid_model=bid_model,
        bid_device=bid_device,
    )
    rule_kw = dict(
        exact_solver=exact_solver,
        exact_threshold=exact_threshold,
        bid_model=bid_model,
        bid_device=bid_device,
    )
    if not swap_seats:
        return [
            _build_rl_player(**rl_kw),
            _build_rule_player(**rule_kw),
            _build_rl_player(**rl_kw),
            _build_rule_player(**rule_kw),
        ]
    return [
        _build_rule_player(**rule_kw),
        _build_rl_player(**rl_kw),
        _build_rule_player(**rule_kw),
        _build_rl_player(**rl_kw),
    ]


def play_one_game_skip_nil(
    players: list,
    seed: int,
    rules: SpadesRules,
    max_redeals: int,
) -> tuple[Any | None, int]:
    """叫牌后若无 nil 则打完; 遇 nil 则重发。返回 (result, redeals)。"""
    for attempt in range(max_redeals + 1):
        game_seed = seed + attempt
        runner = SpadesMatchRunner(
            players=players,
            seed=game_seed,
            verbose=False,
            rules=rules,
        )
        runner._start_game()
        runner._bidding_phase()

        if _has_nil_bid(runner.state.max_bid):
            continue

        runner._set_teams()
        runner._play_phase()
        return runner._score_game(), attempt

    return None, max_redeals + 1


def play_episode(
    policy_net: PolicyMLP,
    exact_solver: ExactDoubleDummyCppFastestSolver,
    encoder: RLFeatureEncoder,
    exact_threshold: int,
    episode_seed: int,
    rules: SpadesRules,
    bid_model,
    bid_device: str,
    max_redeals: int,
) -> dict[str, Any] | None:
    """一局队式赛 (2 盘), 返回 episode 统计; 两盘均无法避开 nil 时返回 None。"""
    episode_rl_score = 0.0
    episode_rule_score = 0.0
    total_redeals = 0

    for game_idx in range(2):
        players = _build_team_players(
            policy_net, exact_solver, encoder, exact_threshold,
            bid_model, bid_device, swap_seats=(game_idx == 1),
        )
        result, redeals = play_one_game_skip_nil(
            players=players,
            seed=episode_seed + game_idx * 1000,
            rules=rules,
            max_redeals=max_redeals,
        )
        total_redeals += redeals
        if result is None:
            return None

        team0_score, team1_score = _compute_team_scores(result)
        if game_idx == 0:
            rl_score, rule_score = team0_score, team1_score
        else:
            rl_score, rule_score = team1_score, team0_score

        episode_rl_score += rl_score
        episode_rule_score += rule_score

    episode_reward = episode_rl_score - episode_rule_score
    return {
        "episode_game_reward": episode_reward / 2.0,
        "episode_reward": episode_reward,
        "episode_rl_score": episode_rl_score,
        "episode_rule_score": episode_rule_score,
        "redeals": total_redeals,
    }


def _worker_eval_batch(args_tuple: tuple) -> list[dict[str, Any]]:
    (
        episode_indices,
        base_seed,
        exact_threshold,
        bid_checkpoint,
        checkpoint_nonil,
        hidden_dims,
        device,
        max_redeals,
    ) = args_tuple

    exact_solver = _build_exact_solver()
    encoder = RLFeatureEncoder()
    bid_model = _load_bid_model(bid_checkpoint, device)
    policy_net = _load_policy(checkpoint_nonil, hidden_dims, device)
    rules = SpadesRules(enable_nil=True, enable_blind_nil=False)

    results: list[dict[str, Any]] = []
    for ep_idx in episode_indices:
        ep_res = play_episode(
            policy_net=policy_net,
            exact_solver=exact_solver,
            encoder=encoder,
            exact_threshold=exact_threshold,
            episode_seed=base_seed + ep_idx * 2,
            rules=rules,
            bid_model=bid_model,
            bid_device=device,
            max_redeals=max_redeals,
        )
        if ep_res is not None:
            results.append(ep_res)
    return results


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    num_episodes = args.num_games // 2
    bid_cp = Path(args.bid_checkpoint)
    policy_cp = Path(args.checkpoint_nonil)

    print("=" * 72, flush=True)
    print("RL first-4 (MLP) vs Rule first-4 — 队式赛 (跳过 nil 局)", flush=True)
    print(f"总对局数: {args.num_games} ({num_episodes} episodes × 2 games)", flush=True)
    print(f"精确阈值: {args.exact_threshold} (前 {52 - args.exact_threshold} 张各用自有策略)",
          flush=True)
    print(f"叫牌: {bid_cp} (nil 启用, 遇 nil 重发)", flush=True)
    print(f"RL 策略: {policy_cp}", flush=True)
    print(f"Workers: {args.num_workers}", flush=True)
    print("=" * 72, flush=True)

    t_start = time.perf_counter()
    all_game_rewards: list[float] = []
    rl_scores: list[float] = []
    rule_scores: list[float] = []
    redeals_list: list[int] = []
    skipped_episodes = 0

    if args.num_workers <= 1:
        exact_solver = _build_exact_solver()
        encoder = RLFeatureEncoder()
        bid_model = _load_bid_model(args.bid_checkpoint, args.device)
        policy_net = _load_policy(
            args.checkpoint_nonil, args.rl_hidden_dims, args.device,
        )
        rules = SpadesRules(enable_nil=True, enable_blind_nil=False)

        for ep_idx in range(num_episodes):
            ep_res = play_episode(
                policy_net=policy_net,
                exact_solver=exact_solver,
                encoder=encoder,
                exact_threshold=args.exact_threshold,
                episode_seed=args.seed + ep_idx * 2,
                rules=rules,
                bid_model=bid_model,
                bid_device=args.device,
                max_redeals=args.max_redeals,
            )
            if ep_res is None:
                skipped_episodes += 1
                continue

            all_game_rewards.append(ep_res["episode_game_reward"])
            rl_scores.append(ep_res["episode_rl_score"])
            rule_scores.append(ep_res["episode_rule_score"])
            redeals_list.append(ep_res["redeals"])

            if args.print_every > 0 and (ep_idx + 1) % args.print_every == 0:
                elapsed = time.perf_counter() - t_start
                print(
                    f"  [{len(all_game_rewards) * 2:>4d} 局] "
                    f"RL 平均 game 奖励: {np.mean(all_game_rewards):>+8.1f}  "
                    f"重发累计: {sum(redeals_list)}  "
                    f"跳过 episode: {skipped_episodes}  "
                    f"耗时: {elapsed:.0f}s",
                    flush=True,
                )
    else:
        import multiprocessing as mp

        episode_indices = list(range(num_episodes))
        chunks = np.array_split(episode_indices, min(args.num_workers, num_episodes))
        worker_args = [
            (
                chunk.tolist() if hasattr(chunk, "tolist") else list(chunk),
                args.seed,
                args.exact_threshold,
                args.bid_checkpoint,
                args.checkpoint_nonil,
                args.rl_hidden_dims,
                args.device,
                args.max_redeals,
            )
            for chunk in chunks
            if len(chunk) > 0
        ]

        ctx = mp.get_context("spawn")
        with ctx.Pool(len(worker_args)) as pool:
            for batch in pool.map(_worker_eval_batch, worker_args):
                for ep_res in batch:
                    all_game_rewards.append(ep_res["episode_game_reward"])
                    rl_scores.append(ep_res["episode_rl_score"])
                    rule_scores.append(ep_res["episode_rule_score"])
                    redeals_list.append(ep_res["redeals"])

        skipped_episodes = num_episodes - len(all_game_rewards)

    t_elapsed = time.perf_counter() - t_start
    n_ep = len(all_game_rewards)
    n_games = n_ep * 2

    print(flush=True)
    print("=" * 72, flush=True)
    print("评估完成", flush=True)
    print(f"有效 episode: {n_ep}/{num_episodes}  (跳过 {skipped_episodes})", flush=True)
    print(f"总耗时: {t_elapsed:.0f}s (平均 {t_elapsed / max(n_ep, 1):.1f}s/episode)",
          flush=True)
    print(f"nil 重发总次数: {sum(redeals_list)} "
          f"(平均 {np.mean(redeals_list) if redeals_list else 0:.2f}/episode)",
          flush=True)
    print(flush=True)

    if n_ep == 0:
        print("无有效对局 (可能 max-redeals 过小或叫牌模型几乎总叫 nil)", flush=True)
        print("=" * 72, flush=True)
        return

    print(f"{'统计项':<30} {'数值':>10}", flush=True)
    print("-" * 42, flush=True)
    print(f"{'RL 平均 game 奖励':<30} {np.mean(all_game_rewards):>+10.1f}", flush=True)
    print(f"{'RL 方平均总分':<30} {np.mean(rl_scores):>+10.1f}", flush=True)
    print(f"{'Rule 方平均总分':<30} {np.mean(rule_scores):>+10.1f}", flush=True)
    print(f"{'总对局数':<30} {n_games:>10d}", flush=True)
    print(flush=True)

    if n_ep >= 20:
        print(f"{'局段 (game)':<20} {'RL game 奖励':>15}", flush=True)
        print("-" * 38, flush=True)
        for start in range(0, n_ep, max(1, n_ep // 5)):
            end = min(start + max(1, n_ep // 5), n_ep)
            seg = all_game_rewards[start:end]
            if seg:
                print(
                    f"{start * 2 + 1:>4}~{end * 2:>4}  "
                    f"{np.mean(seg):>+15.1f}",
                    flush=True,
                )

    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
