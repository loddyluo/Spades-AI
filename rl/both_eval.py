"""
Both 评估脚本：根据实际叫牌结果选择 55_2.pt 或 55_2nil.pt。

流程：
1. 所有玩家叫牌结束后，根据四个人的叫牌判断是否有人叫 0 (nil / blind_nil)
2. 有人叫 0 → rl_exact 使用 55_2nil.pt
3. 没有人叫 0 → rl_exact 使用 55_2.pt

用法:
    python rl/both_eval.py --num-games 1500 --seed 42 --num-workers 30
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.dds_player import DDSPlayer
from rl.policy_network import PolicyMLP
from rl.rl_exact_player import RLExactPlayer
from rl.rl_feature_encoder import RLFeatureEncoder
from strategy.spades_match_runner import SpadesMatchRunner
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)

MODEL_OUTPUT_DIM = 55


class BothPlayer(RLExactPlayer):
    """RLExactPlayer，根据叫牌结果自动切换 checkpoint。

    叫牌结束后 (set_teams 回调)，检查是否有玩家叫了 nil/blind_nil:
    - 有人叫 0 → 使用 nil 专用策略网络 (55_2nil.pt)
    - 无人叫 0 → 使用非 nil 策略网络 (55_2.pt)
    """

    def __init__(
        self,
        policy_nets_nonil: list[PolicyMLP],
        policy_nets_nil: list[PolicyMLP],
        **kwargs,
    ) -> None:
        # 先以 nonil 策略初始化父类
        super().__init__(policy_nets=policy_nets_nonil, **kwargs)
        self.policy_nets_nonil = policy_nets_nonil
        self.policy_nets_nil = policy_nets_nil

    def set_teams(self, teams: list[int], bid_values: list[Any]) -> None:
        """叫牌结束后，根据四个人的叫牌选择使用的策略网络。"""
        nil_bid = any(
            isinstance(bv, str) and bv in ("nil", "blind_nil")
            for bv in bid_values
        )
        if nil_bid:
            self.policy_nets = self.policy_nets_nil
        else:
            self.policy_nets = self.policy_nets_nonil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Both evaluation (multi-CPU): 55_2.pt or 55_2nil.pt based on bids",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-games", type=int, default=1500,
                        help="评估总对局数（每 episode 2 局，共 num_games/2 次队式赛）")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[1024, 512, 512],
                        help="策略网络隐藏层维度")
    parser.add_argument("--exact-threshold", type=int, default=36,
                        help="剩余牌数 <= 该值时使用精确求解器")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    parser.add_argument("--disable-blind-nil", action="store_true", default=True,
                        help="禁用 blind nil 叫牌")
    parser.add_argument("--bid-checkpoint", type=str,
                        default="./Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt",
                        help="叫牌 MLP 模型 checkpoint 路径")
    parser.add_argument("--num-workers", type=int, default=30,
                        help="并行打牌的进程数")
    parser.add_argument("--checkpoint-nonil", type=str, default="./55_2.pt",
                        help="无人叫 0 时使用的 checkpoint")
    parser.add_argument("--checkpoint-nil", type=str, default="./55_2nil.pt",
                        help="有人叫 0 时使用的 checkpoint")
    return parser.parse_args()


def _compute_team_scores(result: Any) -> tuple[float, float]:
    """从游戏结果计算队伍实际得分（含 nil ±50/±100, 超墩 -9 等全规则）。"""
    bids = result.bids
    tricks = result.tricks_won
    teams = [0, 1, 0, 1]

    team_scores = [0.0, 0.0]

    for team_id in (0, 1):
        members = [i for i in range(4) if teams[i] == team_id]
        team_bid = 0
        score = 0.0

        for pid in members:
            bid = bids[pid]
            if bid in ("nil", "blind_nil"):
                if tricks[pid] == 0:
                    score += 100.0 if bid == "blind_nil" else 50.0
                else:
                    score += -100.0 if bid == "blind_nil" else -50.0
            elif bid is not None and isinstance(bid, str) and bid.startswith("bid_"):
                team_bid += int(bid.split("_")[1])

        # 所有队员的得墩都计入（含叫 nil 的玩家），与 SpadesRules.score() 一致
        team_tricks = sum(tricks[pid] for pid in members)

        if team_bid > 0:
            if team_tricks >= team_bid:
                overtricks = team_tricks - team_bid
                score += team_bid * 10 - overtricks * 9
            else:
                score -= team_bid * 10

        team_scores[team_id] = score
    #print(team_scores[0], team_scores[1], flush=True)
    return team_scores[0], team_scores[1]


def _build_dds_player(bid_model=None) -> DDSPlayer:
    return DDSPlayer(bid_model=bid_model)


def _build_both_player(
    policy_nets_nonil: list[PolicyMLP],
    policy_nets_nil: list[PolicyMLP],
    exact_solver: ExactDoubleDummyCppFastestSolver,
    encoder: RLFeatureEncoder,
    exact_threshold: int,
    bid_model=None,
    bid_device: str = "cpu",
) -> BothPlayer:
    return BothPlayer(
        policy_nets_nonil=policy_nets_nonil,
        policy_nets_nil=policy_nets_nil,
        exact_solver=exact_solver,
        encoder=encoder,
        exact_threshold=exact_threshold,
        is_training=False,
        bid_model=bid_model,
        bid_device=bid_device,
    )


def play_one_game(
    policy_nets_nonil: list[PolicyMLP],
    policy_nets_nil: list[PolicyMLP],
    exact_solver: ExactDoubleDummyCppFastestSolver,
    encoder: RLFeatureEncoder,
    exact_threshold: int,
    seed: int,
    rules: SpadesRules,
    bid_model=None,
    bid_device: str = "cpu",
    swap_seats: bool = False,
) -> Any:
    """打一局：BothPlayer vs DDS，返回结果。"""
    if not swap_seats:
        players = [
            _build_both_player(policy_nets_nonil, policy_nets_nil, exact_solver,
                               encoder, exact_threshold, bid_model=bid_model,
                               bid_device=bid_device),
            _build_dds_player(bid_model=bid_model),
            _build_both_player(policy_nets_nonil, policy_nets_nil, exact_solver,
                               encoder, exact_threshold, bid_model=bid_model,
                               bid_device=bid_device),
            _build_dds_player(bid_model=bid_model),
        ]
    else:
        players = [
            _build_dds_player(bid_model=bid_model),
            _build_both_player(policy_nets_nonil, policy_nets_nil, exact_solver,
                               encoder, exact_threshold, bid_model=bid_model,
                               bid_device=bid_device),
            _build_dds_player(bid_model=bid_model),
            _build_both_player(policy_nets_nonil, policy_nets_nil, exact_solver,
                               encoder, exact_threshold, bid_model=bid_model,
                               bid_device=bid_device),
        ]

    runner = SpadesMatchRunner(
        players=players,
        seed=seed,
        verbose=False,
        rules=rules,
    )
    return runner.play_game()


def _load_bid_model_worker(bid_checkpoint: str, device: str):
    """worker 中加载叫牌模型。"""
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


def _load_policy_worker(state_dict, hidden_dims, device) -> PolicyMLP:
    """在 worker 中加载策略网络。"""
    net = PolicyMLP(input_dim=264, hidden_dims=hidden_dims, output_dim=MODEL_OUTPUT_DIM)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    return net


def worker_eval_batch(args_tuple: tuple) -> list[dict]:
    """在 worker 进程中打一批 episode（评估模式，argmax，不收集轨迹，不计算梯度）。"""
    (episode_indices, base_seed,
     policy_state_nonil, policy_state_nil,
     hidden_dims, exact_threshold, rules_args,
     bid_checkpoint, device) = args_tuple

    # 加载两个策略网络
    policy_net_nonil = _load_policy_worker(policy_state_nonil, hidden_dims, device)
    policy_net_nil = _load_policy_worker(policy_state_nil, hidden_dims, device)

    exact_solver = ExactDoubleDummyCppFastestSolver()
    if not exact_solver.native_available:
        from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver
        exact_solver = ExactDoubleDummySolver()

    encoder = RLFeatureEncoder()
    rules = SpadesRules(*rules_args)
    bid_model = _load_bid_model_worker(bid_checkpoint, device)

    policy_nets_nonil = [policy_net_nonil]
    policy_nets_nil = [policy_net_nil]

    results = []
    for ep_idx in episode_indices:
        game_seed = base_seed + ep_idx * 2

        episode_our_score = 0.0
        episode_opp_score = 0.0

        for game_idx in range(2):
            result = play_one_game(
                policy_nets_nonil=policy_nets_nonil,
                policy_nets_nil=policy_nets_nil,
                exact_solver=exact_solver,
                encoder=encoder,
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
        print(episode_reward, flush=True)
        episode_game_reward = episode_reward / 2.0
        # print(episode_game_reward, flush=True)
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
    device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── 加载两个策略网络 ────────────────────────────────────
    cp_nonil_path = Path(args.checkpoint_nonil)
    cp_nil_path = Path(args.checkpoint_nil)

    if cp_nonil_path.exists():
        policy_host_nonil = PolicyMLP(input_dim=264, hidden_dims=args.hidden_dims,
                                      output_dim=MODEL_OUTPUT_DIM).to(device)
        policy_host_nonil.eval()
        policy_host_nonil.load(str(cp_nonil_path.resolve()), device=args.device)
        policy_host_nonil.eval()
        print(f"已加载非 nil checkpoint: {cp_nonil_path.resolve()}", flush=True)
    else:
        print(f"警告: 非 nil checkpoint 不存在: {cp_nonil_path}，使用随机权重", flush=True)
        policy_host_nonil = PolicyMLP(input_dim=264, hidden_dims=args.hidden_dims,
                                      output_dim=MODEL_OUTPUT_DIM).to(device)
        policy_host_nonil.eval()

    if cp_nil_path.exists():
        policy_host_nil = PolicyMLP(input_dim=264, hidden_dims=args.hidden_dims,
                                    output_dim=MODEL_OUTPUT_DIM).to(device)
        policy_host_nil.eval()
        policy_host_nil.load(str(cp_nil_path.resolve()), device=args.device)
        policy_host_nil.eval()
        print(f"已加载 nil checkpoint: {cp_nil_path.resolve()}", flush=True)
    else:
        print(f"警告: nil checkpoint 不存在: {cp_nil_path}，使用随机权重", flush=True)
        policy_host_nil = PolicyMLP(input_dim=264, hidden_dims=args.hidden_dims,
                                    output_dim=MODEL_OUTPUT_DIM).to(device)
        policy_host_nil.eval()

    # ── 叫牌模型 ────────────────────────────────────────────
    bid_checkpoint = args.bid_checkpoint
    if bid_checkpoint:
        cp = Path(bid_checkpoint)
        if cp.exists():
            print(f"叫牌模型: {cp} (worker 进程会自行加载)", flush=True)
        else:
            print(f"警告: 叫牌模型 checkpoint 不存在: {cp}", flush=True)

    rules_args = (True, not args.disable_blind_nil)

    num_episodes = args.num_games // 2

    print("=" * 72, flush=True)
    print("Both Policy Evaluation (MULTI-CPU, ARGMAX): rl_exact vs DDS", flush=True)
    print(f"  根据叫牌选择: 有人叫 0 → 55_2nil.pt, 无人叫 0 → 55_2.pt", flush=True)
    print(f"总对局数: {args.num_games} ({num_episodes} episodes × 2 games)", flush=True)
    print(f"隐藏层: {args.hidden_dims}", flush=True)
    print(f"精确阈值: {args.exact_threshold} (前 {52 - args.exact_threshold} 张用 RL argmax)", flush=True)
    print(f"Workers: {args.num_workers}", flush=True)
    print("=" * 72, flush=True)

    t_start = time.perf_counter()
    print_interval_episodes = 150

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

            # 发送两个 state_dict
            policy_state_nonil = policy_host_nonil.state_dict()
            policy_state_nil = policy_host_nil.state_dict()

            worker_args = [
                (chunk.tolist() if hasattr(chunk, 'tolist') else chunk,
                 args.seed, policy_state_nonil, policy_state_nil,
                 args.hidden_dims, args.exact_threshold, rules_args,
                 bid_checkpoint, args.device)
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
            print(all_game_rewards[-40:], flush=True)
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

    # ── 输出统计 ────────────────────────────────────────────
    n_ep = len(all_game_rewards)
    n_games_total = n_ep * 2

    print(flush=True)
    print("=" * 72, flush=True)
    print(f"评估完成！", flush=True)
    print(f"总耗时: {t_elapsed:.0f}s (平均 {t_elapsed/max(n_ep,1):.1f}s/episode)", flush=True)
    print(f"总 episode 数: {n_ep} ({n_games_total} 局)", flush=True)
    print(flush=True)
    print(f"{'统计项':<30} {'数值':>10}", flush=True)
    print("-" * 42, flush=True)
    print(f"{'平均 episode 奖励':<30} {np.mean(episode_rewards):>+10.1f}", flush=True)
    print(f"{'平均 game 奖励':<30} {np.mean(all_game_rewards):>+10.1f}", flush=True)
    print(f"{'我方平均总分':<30} {np.mean(our_team_scores):>+10.1f}", flush=True)
    print(f"{'对方平均总分':<30} {np.mean(opp_team_scores):>+10.1f}", flush=True)
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

    if all_game_rewards:
        mean_r = np.mean(all_game_rewards)
        std_r = np.std(all_game_rewards)
        print(flush=True)
        print(f"Game 奖励分布: mean={mean_r:+.1f}, std={std_r:.1f}", flush=True)
        print(f"最大: {np.max(all_game_rewards):+.1f}, 最小: {np.min(all_game_rewards):+.1f}", flush=True)
        positive_ratio = np.mean([r >= 0 for r in all_game_rewards])
        print(f"非负 game 奖励占比: {positive_ratio*100:.1f}%", flush=True)

    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
