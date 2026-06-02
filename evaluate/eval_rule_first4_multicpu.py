"""
规则式策略评估脚本 (多核版本): rule_exact_first4 vs DDS。

设计与 rl/eval_rl_multicpu.py 一一对应:
- 玩家结构: 我方 = RuleExactFirst4Player (前 4 墩规则式 + 后 36 张精确求解);
            对方 = DDSPlayer (作弊全开手). 队式赛对换座位。
- 奖励计算: 与 _compute_team_scores 完全一致 (达成叫墩=0, 否则 -100; payoff = 我方 - 对方)。
- 统计输出: 队式赛累积平均 game 奖励、分段平均、奖励分布。

用法:
    python evaluate/eval_rule_first4_multicpu.py --num-games 1500 --seed 42 --num-workers 30

不需要 checkpoint、不需要 hidden-dims —— 规则式玩家无可学参数。
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.dds_player import DDSPlayer
from strategy.rule_exact_first4_player import RuleExactFirst4Player
from strategy.spades_match_runner import SpadesMatchRunner
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rule-based first-4-tricks evaluation (multi-CPU): "
                    "rule_exact_first4 vs DDS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-games", type=int, default=1500,
                        help="评估总对局数 (每 episode 2 局, 共 num_games/2 次队式赛)")
    parser.add_argument("--exact-threshold", type=int, default=36,
                        help="剩余牌数 <= 该值时使用精确求解器")
    parser.add_argument("--disable-blind-nil", action="store_true", default=True,
                        help="禁用 blind nil 叫牌")
    parser.add_argument("--bid-checkpoint", type=str,
                        default="./Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt",
                        help="叫牌 MLP 模型 checkpoint 路径")
    parser.add_argument("--num-workers", type=int, default=30,
                        help="并行打牌的进程数")
    return parser.parse_args()


def _compute_team_scores(result: Any) -> tuple[float, float]:
    """与 rl/eval_rl_multicpu.py 完全一致的奖励计算。

    叫墩达成 → 0 分; 未达成 → -100 分。返回 (team0 payoff, team1 payoff)。
    """
    ## Mode 1
    scores = result.scores
    t0 = scores[0]
    t1 = scores[1]

    return t0/2.0, t1/2.0

    bids = result.bids
    tricks = result.tricks_won
    teams = [0, 1, 0, 1]

    def numeric_bid(bid) -> int:
        if bid is None:
            return 0
        if isinstance(bid, str):
            if bid in ("nil", "blind_nil"):
                return 0
            if bid.startswith("bid_"):
                return int(bid.split("_")[1])
        return 0

    team0_bid = sum(numeric_bid(bids[i]) for i in range(4) if teams[i] == 0)
    team1_bid = sum(numeric_bid(bids[i]) for i in range(4) if teams[i] == 1)
    team0_tricks = sum(tricks[i] for i in range(4) if teams[i] == 0)
    team1_tricks = sum(tricks[i] for i in range(4) if teams[i] == 1)

    t0 = -100.0 if team0_tricks < team0_bid else 0.0
    t1 = -100.0 if team1_tricks < team1_bid else 0.0
    return t0 - t1, t1 - t0


def _build_dds_player(bid_model=None) -> DDSPlayer:
    return DDSPlayer(bid_model=bid_model)


def _build_rule_player(
    exact_solver: ExactDoubleDummyCppFastestSolver,
    exact_threshold: int,
    bid_model=None,
    bid_device: str = "cpu",
) -> RuleExactFirst4Player:
    return RuleExactFirst4Player(
        exact_solver=exact_solver,
        exact_threshold=exact_threshold,
        bid_model=bid_model,
        bid_device=bid_device,
    )


def play_one_game(
    exact_solver: ExactDoubleDummyCppFastestSolver,
    exact_threshold: int,
    seed: int,
    rules: SpadesRules,
    bid_model=None,
    bid_device: str = "cpu",
    swap_seats: bool = False,
) -> Any:
    """打一局: rule_exact_first4 vs DDS。"""
    if not swap_seats:
        players = [
            _build_rule_player(exact_solver, exact_threshold,
                               bid_model=bid_model, bid_device=bid_device),
            _build_dds_player(bid_model=bid_model),
            _build_rule_player(exact_solver, exact_threshold,
                               bid_model=bid_model, bid_device=bid_device),
            _build_dds_player(bid_model=bid_model),
        ]
    else:
        players = [
            _build_dds_player(bid_model=bid_model),
            _build_rule_player(exact_solver, exact_threshold,
                               bid_model=bid_model, bid_device=bid_device),
            _build_dds_player(bid_model=bid_model),
            _build_rule_player(exact_solver, exact_threshold,
                               bid_model=bid_model, bid_device=bid_device),
        ]

    runner = SpadesMatchRunner(
        players=players,
        seed=seed,
        verbose=False,
        rules=rules,
    )
    return runner.play_game()


def _load_bid_model_worker(bid_checkpoint: str, device: str):
    """worker 中加载叫牌模型 (照抄 rl/eval_rl_multicpu.py 的实现)。"""
    try:
        go_mcts_dir = REPO_ROOT / "evaluate" / "GO-MCTS"
        if str(go_mcts_dir) not in sys.path:
            sys.path.insert(0, str(go_mcts_dir))
        from models import load_bid_mlp_model
        cp = Path(bid_checkpoint)
        if cp.exists():
            return load_bid_mlp_model(str(cp.resolve()), device)
    except Exception:
        pass
    return None


def worker_eval_batch(args_tuple: tuple) -> list[dict]:
    """worker 进程: 跑一批 episode (评估模式)。

    args_tuple: (episode_indices, base_seed, exact_threshold, rules_args,
                 bid_checkpoint, device)
    """
    (episode_indices, base_seed, exact_threshold, rules_args,
     bid_checkpoint, device) = args_tuple

    # 每个 worker 独立创建自己的资源
    exact_solver = ExactDoubleDummyCppFastestSolver()
    if not exact_solver.native_available:
        from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver
        exact_solver = ExactDoubleDummySolver()

    rules = SpadesRules(*rules_args)
    bid_model = _load_bid_model_worker(bid_checkpoint, device)

    results = []
    for ep_idx in episode_indices:
        game_seed = base_seed + ep_idx * 2

        episode_our_score = 0.0
        episode_opp_score = 0.0

        for game_idx in range(2):
            result = play_one_game(
                exact_solver=exact_solver,
                exact_threshold=exact_threshold,
                seed=game_seed,
                rules=rules,
                bid_model=bid_model,
                bid_device=device,
                swap_seats=(game_idx == 1),
            )

            team0_score, team1_score = _compute_team_scores(result)

            if game_idx == 0:
                our_score = team0_score
                opp_score = team1_score
            else:
                our_score = team1_score
                opp_score = team0_score

            episode_our_score += our_score
            episode_opp_score += opp_score

        episode_reward = episode_our_score - episode_opp_score
        episode_game_reward = episode_reward / 2.0

        results.append({
            "episode_game_reward": episode_game_reward,
            "episode_reward": episode_reward,
            "episode_our_score": episode_our_score,
            "episode_opp_score": episode_opp_score,
        })

    return results


def main() -> None:
    import multiprocessing as mp

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── 叫牌模型 ──────────────────────────────────────────────────
    bid_checkpoint = args.bid_checkpoint
    if bid_checkpoint:
        cp = Path(bid_checkpoint)
        if cp.exists():
            print(f"叫牌模型: {cp} (worker 进程会自行加载)", flush=True)
        else:
            print(f"警告: 叫牌模型 checkpoint 不存在: {cp}", flush=True)

    rules_args = (False, not args.disable_blind_nil)

    num_episodes = args.num_games // 2

    print("=" * 72, flush=True)
    print("Rule-Based First-4-Tricks Evaluation (MULTI-CPU): "
          "rule_exact_first4 vs DDS", flush=True)
    print(f"总对局数: {args.num_games} ({num_episodes} episodes × 2 games)",
          flush=True)
    print(f"精确阈值: {args.exact_threshold} (前 {52 - args.exact_threshold} 张用 rule)",
          flush=True)
    print(f"Workers: {args.num_workers}", flush=True)
    print("=" * 72, flush=True)

    t_start = time.perf_counter()
    print_interval_episodes = 150  # 每 300 局 (=150 episodes) 输出一次运行平均

    all_game_rewards: list[float] = []
    episode_rewards: list[float] = []
    our_team_scores: list[float] = []
    opp_team_scores: list[float] = []

    eval_report_ep: int = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.num_workers) as pool:
        for batch_start in range(0, num_episodes, print_interval_episodes):
            batch_end = min(batch_start + print_interval_episodes, num_episodes)
            batch_episodes = list(range(batch_start, batch_end))
            if not batch_episodes:
                continue

            n_workers = min(args.num_workers, len(batch_episodes))
            chunks = np.array_split(batch_episodes, n_workers)

            worker_args = [
                (chunk.tolist() if hasattr(chunk, 'tolist') else chunk,
                 args.seed,
                 args.exact_threshold, rules_args, bid_checkpoint,
                 "cpu")
                for chunk in chunks
            ]

            batch_results = pool.map(worker_eval_batch, worker_args)
            eval_report_ep += batch_end - batch_start
            games_done = eval_report_ep * 2

            for worker_results in batch_results:
                for ep_res in worker_results:
                    all_game_rewards.append(ep_res["episode_game_reward"])
                    episode_rewards.append(ep_res["episode_reward"])
                    our_team_scores.append(ep_res["episode_our_score"])
                    opp_team_scores.append(ep_res["episode_opp_score"])

            running_mean = np.mean(all_game_rewards)
            elapsed = time.perf_counter() - t_start
            print(
                f"  [{games_done:>4d} 局] "
                f"累积平均 game 奖励: {running_mean:>+8.1f}  "
                f"我方: {np.mean(our_team_scores):>+7.1f}  "
                f"对方: {np.mean(opp_team_scores):>+7.1f}  "
                f"耗时: {elapsed:.0f}s",
                flush=True,
            )

    t_elapsed = time.perf_counter() - t_start

    # ── 输出统计 ──────────────────────────────────────────────────
    n_ep = len(all_game_rewards)
    n_games_total = n_ep * 2

    print(flush=True)
    print("=" * 72, flush=True)
    print(f"评估完成!", flush=True)
    print(f"总耗时: {t_elapsed:.0f}s (平均 {t_elapsed/max(n_ep,1):.1f}s/episode)",
          flush=True)
    print(f"总 episode 数: {n_ep} ({n_games_total} 局)", flush=True)
    print(flush=True)
    print(f"{'统计项':<30} {'数值':>10}", flush=True)
    print("-" * 42, flush=True)
    print(f"{'平均 episode 奖励':<30} {np.mean(episode_rewards):>+10.1f}",
          flush=True)
    print(f"{'平均 game 奖励':<30} {np.mean(all_game_rewards):>+10.1f}",
          flush=True)
    print(f"{'我方平均总分':<30} {np.mean(our_team_scores):>+10.1f}",
          flush=True)
    print(f"{'对方平均总分':<30} {np.mean(opp_team_scores):>+10.1f}",
          flush=True)
    print(flush=True)

    # 分段统计
    if n_ep >= 200:
        intervals = []
        for i in range(0, n_ep, 100):
            end = min(i + 100, n_ep)
            intervals.append((i, end))
            if end == n_ep:
                break

        print(f"{'局段 (game)':<20} {'平均 game 奖励':>15}", flush=True)
        print("-" * 38, flush=True)
        for start_ep, end_ep in intervals:
            seg = all_game_rewards[start_ep:end_ep]
            if seg:
                game_start = start_ep * 2 + 1
                game_end = end_ep * 2
                label = f"{game_start:>4}~{game_end:>4}"
                print(f"{label:<20} {np.mean(seg):>+15.1f}", flush=True)

    # 末尾奖励分布信息
    if all_game_rewards:
        mean_r = np.mean(all_game_rewards)
        std_r = np.std(all_game_rewards)
        print(flush=True)
        print(f"Game 奖励分布: mean={mean_r:+.1f}, std={std_r:.1f}", flush=True)
        print(f"最大: {np.max(all_game_rewards):+.1f}, 最小: {np.min(all_game_rewards):+.1f}",
              flush=True)
        positive_ratio = np.mean([r >= 0 for r in all_game_rewards])
        print(f"非负 game 奖励占比: {positive_ratio*100:.1f}%", flush=True)

    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
