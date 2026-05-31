"""
RL policy gradient 训练脚本：rl_exact vs DDS。

训练方法：
- 每 episode 打 2 局，rl_exact（座位 0&2）vs DDS（座位 1&3）
- 前 16 张牌（剩余 > 36）：rl_exact 使用 PolicyMLP 采样出牌
- 后 36 张牌（剩余 <= 36）：rl_exact 使用精确双明手求解器
- DDS 全程使用外部双明手求解器
- 奖励 = (我方 2 局得分之和) - (对方 2 局得分之和)
- 使用 REINFORCE 算法更新策略网络

用法:
    python rl/train_rl.py --num-games 5000 --seed 123 --lr 0.001
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.dds_player import DDSPlayer
from rl.policy_network import PolicyMLP
from rl.rl_exact_player import RLExactPlayer
from strategy.spades_match_runner import SpadesMatchRunner
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)
from trick_taking.utils.feature_encoder import SpadesFeatureEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RL policy gradient training: rl_exact vs DDS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-games", type=int, default=10000,
                        help="训练总对局数（每 episode 2 局，共 num_games/2 次更新）")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256],
                        help="策略网络隐藏层维度")
    parser.add_argument("--exact-threshold", type=int, default=36,
                        help="剩余牌数 <= 该值时使用精确求解器（默认 36 = 前16张用RL）")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子（当前未使用，完整 REINFORCE 可加）")
    parser.add_argument("--update-interval", type=int, default=40,
                        help="每多少 episode 做一次梯度更新（累积多episode数据后更新，减小方差）")
    parser.add_argument("--save-dir", type=str, default="rl_checkpoints",
                        help="模型保存目录")
    parser.add_argument("--save-interval", type=int, default=1000,
                        help="每多少 episode 保存一次 checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="训练设备")
    parser.add_argument("--disable-blind-nil", action="store_true", default=True,
                        help="禁用 blind nil 叫牌")
    parser.add_argument("--bid-checkpoint", type=str,
                        default="./Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt",
                        help="叫牌 MLP 模型 checkpoint 路径")
    parser.add_argument("--entropy-coef", type=float, default=0.15,
                        help="熵奖励系数")
    parser.add_argument("--max-grad-norm", type=float, default=15.0,
                        help="梯度裁剪最大范数")
    parser.add_argument("--baseline-decay", type=float, default=0.95,
                        help="全局 EMA 基线衰减系数")
    parser.add_argument("--tensorboard", action="store_true", default=True,
                        help="启用 TensorBoard 日志")
    parser.add_argument("--log-dir", type=str, default="runs/rl_train",
                        help="TensorBoard 日志目录")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="从指定路径加载之前训练过的 checkpoint（.pt 文件），在此基础上继续训练")
    return parser.parse_args()


def _compute_team_scores(result: Any) -> tuple[float, float]:
    """从游戏结果计算队伍得分（仅看是否完成叫牌）。

    新算法：得墩 ≥ 叫墩总和 → 0分，否则 → -100分。
    返回 (队伍0的payoff, 队伍1的payoff)，其中 payoff = 本方得分 - 对方得分。
    """
    bids = result.bids        # list[str|None], length=4
    tricks = result.tricks_won  # list[int], length=4
    # 队伍固定的黑桃王：0&2 vs 1&3
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

    diff0 = team0_tricks - team0_bid
    diff1 = team1_tricks - team1_bid
    t0 = -100.0 + diff0 * 3.0 if team0_tricks < team0_bid else diff0 * 3.0
    t1 = -100.0 + diff1 * 3.0 if team1_tricks < team1_bid else diff1 * 3.0
    return t0 - t1, t1 - t0


def _build_dds_player(bid_model=None) -> DDSPlayer:
    """构造一个 DDS 玩家（使用给定的叫牌模型）。"""
    return DDSPlayer(bid_model=bid_model)


def _build_rl_exact_player(
    policy_net: PolicyMLP,
    exact_solver: ExactDoubleDummyCppFastestSolver,
    encoder: SpadesFeatureEncoder,
    exact_threshold: int,
    is_training: bool,
    bid_model=None,
    bid_device: str = "cpu",
) -> RLExactPlayer:
    """构造一个 RL + Exact 混合玩家。"""
    return RLExactPlayer(
        policy_net=policy_net,
        exact_solver=exact_solver,
        encoder=encoder,
        exact_threshold=exact_threshold,
        is_training=is_training,
        bid_model=bid_model,
        bid_device=bid_device,
    )


def play_one_game(
    policy_net: PolicyMLP,
    exact_solver: ExactDoubleDummyCppFastestSolver,
    encoder: SpadesFeatureEncoder,
    exact_threshold: int,
    seed: int,
    is_training: bool,
    rules: SpadesRules,
    bid_model=None,
    bid_device: str = "cpu",
    swap_seats: bool = False,
) -> tuple[Any, list[dict[str, Any]]]:
    """打一局：rl_exact vs DDS，返回结果和 RL 轨迹。

    swap_seats=False (默认): rl_exact 在座位 0&2, DDS 在 1&3
    swap_seats=True:         DDS 在座位 0&2, rl_exact 在 1&3
    """
    if not swap_seats:
        players = [
            _build_rl_exact_player(policy_net, exact_solver, encoder, exact_threshold, is_training,
                                    bid_model=bid_model, bid_device=bid_device),
            _build_dds_player(bid_model=bid_model),
            _build_rl_exact_player(policy_net, exact_solver, encoder, exact_threshold, is_training,
                                    bid_model=bid_model, bid_device=bid_device),
            _build_dds_player(bid_model=bid_model),
        ]
    else:
        players = [
            _build_dds_player(bid_model=bid_model),
            _build_rl_exact_player(policy_net, exact_solver, encoder, exact_threshold, is_training,
                                    bid_model=bid_model, bid_device=bid_device),
            _build_dds_player(bid_model=bid_model),
            _build_rl_exact_player(policy_net, exact_solver, encoder, exact_threshold, is_training,
                                    bid_model=bid_model, bid_device=bid_device),
        ]

    runner = SpadesMatchRunner(
        players=players,
        seed=seed,
        verbose=False,
        rules=rules,
    )
    result = runner.play_game()
    all_trajectories: list[dict[str, Any]] = []
    for player in players:
        if isinstance(player, RLExactPlayer):
            all_trajectories.extend(player.trajectory)

    return result, all_trajectories


def compute_reinforce_loss(
    trajectories: list[dict[str, Any]],
    reward: float,
    baseline: float,
) -> torch.Tensor:
    """计算 REINFORCE 损失。

    对于每个轨迹点，loss = -log_prob * (reward - baseline)。

    输入:
        trajectories: 每个决策点的轨迹列表
        reward: 整个 episode 的累积奖励
        baseline: 奖励基线（用于减小方差）

    输出:
        loss: 标量损失值
    """
    if not trajectories:
        return torch.tensor(0.0, requires_grad=True)

    advantage = reward - baseline
    loss = torch.tensor(0.0)
    for traj in trajectories:
        loss = loss - traj["log_prob"] * advantage

    return loss / len(trajectories)


def train(args: argparse.Namespace) -> None:
    """主训练循环。"""
    device = torch.device(args.device)

    # 设置所有随机种子以保证可复现性
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── 初始化 ──────────────────────────────────────────────────────────────
    policy_net = PolicyMLP(input_dim=1229, hidden_dims=args.hidden_dims).to(device)

    # 加载 checkpoint（如果指定）
    if args.load_checkpoint:
        cp_path = Path(args.load_checkpoint)
        if cp_path.exists():
            policy_net.load(str(cp_path.resolve()), device=args.device)
            policy_net.train()  # 加载后切回训练模式
            print(f"从 checkpoint 加载模型: {cp_path.resolve()}")
        else:
            print(f"警告: checkpoint 不存在: {cp_path}，将使用随机初始化")

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)

    exact_solver = ExactDoubleDummyCppFastestSolver()
    if not exact_solver.native_available:
        print("警告: C++ 精确求解器不可用，尝试使用 Python 版本...")
        from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver
        exact_solver = ExactDoubleDummySolver()

    encoder = SpadesFeatureEncoder()

    rules = SpadesRules(
        enable_nil=False,
        enable_blind_nil=not args.disable_blind_nil,
    )

    # ── 保存目录 ────────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 叫牌模型 ────────────────────────────────────────────────────────────
    bid_model = None
    if args.bid_checkpoint:
        try:
            go_mcts_dir = REPO_ROOT / "evaluate" / "GO-MCTS"
            if str(go_mcts_dir) not in sys.path:
                sys.path.insert(0, str(go_mcts_dir))
            from models import load_bid_mlp_model
            _checkpoint = Path(args.bid_checkpoint)
            if _checkpoint.exists():
                bid_model = load_bid_mlp_model(str(_checkpoint.resolve()), args.device)
                print(f"加载叫牌模型: {_checkpoint}")
            else:
                print(f"警告: 叫牌模型 checkpoint 不存在: {_checkpoint}，将使用启发式叫牌")
        except Exception as e:
            print(f"警告: 加载叫牌模型失败 ({e})，将使用启发式叫牌")

    # ── 训练参数 ────────────────────────────────────────────────────────────
    num_episodes = args.num_games // 2  # 每 episode 2 局

    # 统计
    episode_rewards: list[float] = []
    game_rewards_list: list[float] = []  # 所有单局 reward
    our_team_scores: list[float] = []
    opp_team_scores: list[float] = []

    print("=" * 72)
    print("RL Policy Gradient Training: rl_exact vs DDS")
    print(f"总对局数: {args.num_games} ({num_episodes} episodes × 2 games)")
    print(f"学习率: {args.lr}")
    print(f"隐藏层: {args.hidden_dims}")
    print(f"精确阈值: {args.exact_threshold} (前 {52 - args.exact_threshold} 张用 RL)")
    print(f"熵系数: {args.entropy_coef}")
    print(f"最大梯度范数: {args.max_grad_norm}")
    print(f"设备: {args.device}")
    if args.load_checkpoint:
        print(f"加载 checkpoint: {args.load_checkpoint}")
    print("=" * 72)

    # ── TensorBoard ────────────────────────────────────────────────────────
    writer = None
    if args.tensorboard:
        from pathlib import Path as _Path
        log_path = _Path(args.log_dir) / f"seed{args.seed}_lr{args.lr}_hid{'_'.join(str(h) for h in args.hidden_dims)}"
        writer = SummaryWriter(log_dir=str(log_path))
        print(f"TensorBoard 日志: {log_path}")
        # 记录超参数
        writer.add_text("hyperparams/lr", str(args.lr), 0)
        writer.add_text("hyperparams/hidden_dims", str(args.hidden_dims), 0)
        writer.add_text("hyperparams/entropy_coef", str(args.entropy_coef), 0)
        writer.add_text("hyperparams/update_interval", str(args.update_interval), 0)

    t_start = time.perf_counter()

    accumulated_trajectories: list[dict[str, Any]] = []
    batch_game_rewards: list[float] = []

    # 全局 EMA 基线（比批次内平均更稳定）
    global_baseline = 0.0
    global_baseline_init = False

    for episode in range(num_episodes):
        # ── 打 2 局 ──────────────────────────────────────────────────────
        episode_our_score = 0.0
        episode_opp_score = 0.0
        episode_trajs: list[dict[str, Any]] = []

        for game_idx in range(2):
            # 队式赛：两局使用完全相同的牌面（相同 seed），只互换座位
            seed = args.seed + episode * 2
            result, trajectories = play_one_game(
                policy_net=policy_net,
                exact_solver=exact_solver,
                encoder=encoder,
                exact_threshold=args.exact_threshold,
                seed=seed,
                is_training=True,
                rules=rules,
                bid_model=bid_model,
                bid_device=args.device,
                swap_seats=(game_idx == 1),  # 队式赛：第二局互换座位
            )

            team0_score, team1_score = _compute_team_scores(result)

            # 从 rl_exact 视角计算该局的我们得分和对方得分
            if game_idx == 0:
                our_score = team0_score   # rl_exact 在 team0（座位 0&2）
                opp_score = team1_score   # DDS 在 team1（座位 1&3）
            else:
                our_score = team1_score   # rl_exact 在 team1（座位 1&3）
                opp_score = team0_score   # DDS 在 team0（座位 0&2）

            episode_our_score += our_score
            episode_opp_score += opp_score

            # 暂存轨迹，等整组队式赛结束后再统一绑定同一个 episode-level reward。
            episode_trajs.extend(trajectories)

        # ── 计算 episode 奖励 ────────────────────────────────────────────
        episode_reward = episode_our_score - episode_opp_score
        episode_game_reward = episode_reward / 2.0

        # 队式赛里，两局轨迹都应该共享同一个训练信号：A-B。
        # 这里的 episode_reward 实际上是 2*(A-B)，所以要除以 2。
        for traj in episode_trajs:
            traj["_game_reward"] = episode_game_reward

        # 收集 episode-level reward 用于后续 baseline
        batch_game_rewards.append(episode_game_reward)

        episode_rewards.append(episode_reward)
        our_team_scores.append(episode_our_score)
        opp_team_scores.append(episode_opp_score)

        # ── 累积到缓冲区 ─────────────────────────────────────────────────
        accumulated_trajectories.extend(episode_trajs)

        # ── REINFORCE 更新（每 update_interval 个 episode 做一次） ─────
        if (episode + 1) % args.update_interval == 0 and accumulated_trajectories:
            optimizer.zero_grad()

            # 收集所有 loss 项
            reinforce_terms = []
            raw_advantages = []
            has_entropy = "entropy" in accumulated_trajectories[0]
            entropy_terms = []

            # 使用批次均值做基线
            rewards_batch = np.array(batch_game_rewards)
            baseline = float(np.mean(rewards_batch)) if len(rewards_batch) > 0 else 0.0

            for traj in accumulated_trajectories:
                raw_adv = traj["_game_reward"] - baseline
                raw_advantages.append(raw_adv)
                reinforce_terms.append(-traj["log_prob"] * raw_adv)
                if has_entropy:
                    entropy_terms.append(-args.entropy_coef * traj["entropy"])

            # 合并 loss
            reinforce_loss = torch.stack(reinforce_terms).mean()
            loss = reinforce_loss
            if entropy_terms:
                entropy_loss = torch.stack(entropy_terms).mean()
                loss = reinforce_loss + entropy_loss

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()

            # 记录 update 信息
            if (episode + 1) % (args.update_interval * 5) == 0:
                n_pos_adv = sum(1 for a in raw_advantages if a > 0)
                print(f"  [Update] episode={episode+1}, reinforce_loss={reinforce_loss.item():.3f}, "
                      f"grad_norm={grad_norm:.3f}, "
                      f"baseline={baseline:.1f}, "
                      f"pos_adv_ratio={n_pos_adv}/{len(accumulated_trajectories)}")
            # TensorBoard: 每次 update 都记录
            if writer is not None:
                writer.add_scalar("update/reinforce_loss", reinforce_loss.item(), episode + 1)
                writer.add_scalar("update/grad_norm", grad_norm, episode + 1)
                writer.add_scalar("update/baseline", baseline, episode + 1)
                writer.add_scalar("update/batch_avg", np.mean(rewards_batch), episode + 1)

            accumulated_trajectories = []
            batch_game_rewards = []

        # ── 日志 ──────────────────────────────────────────────────────────
        if (episode + 1) % 40 == 0:
            recent = episode_rewards[-40:]
            avg_reward = np.mean(recent)
            avg_our = np.mean(our_team_scores[-40:])
            avg_opp = np.mean(opp_team_scores[-40:])
            elapsed = time.perf_counter() - t_start

            print(
                f"Episode {episode + 1:5d}/{num_episodes} | "
                f"AvgEpReward={avg_reward:+7.1f} | "
                f"AvgGameReward={avg_reward/2:+7.1f} | "
                f"AvgOur={avg_our:+7.1f} | AvgOpp={avg_opp:+7.1f} | "
                f"Trajs={len(accumulated_trajectories):4d} | "
                f"Time={elapsed:.0f}s"
            )

        # ── TensorBoard 日志（每20个episode记录一次平均reward） ────────
        if writer is not None and (episode + 1) % 20 == 0:
            recent20 = episode_rewards[-20:]
            avg_ep_reward = np.mean(recent20)
            avg_game_reward = avg_ep_reward / 2.0
            avg_our_20 = np.mean(our_team_scores[-20:])
            avg_opp_20 = np.mean(opp_team_scores[-20:])

            writer.add_scalar("train/avg_episode_reward", avg_ep_reward, episode + 1)
            writer.add_scalar("train/avg_game_reward", avg_game_reward, episode + 1)
            writer.add_scalar("train/avg_our_score", avg_our_20, episode + 1)
            writer.add_scalar("train/avg_opp_score", avg_opp_20, episode + 1)
            writer.add_scalar("train/batch_avg", np.mean(batch_game_rewards) if batch_game_rewards else 0.0, episode + 1)

        # ── 保存 checkpoint ──────────────────────────────────────────────
        if (episode + 1) % args.save_interval == 0:
            checkpoint_path = save_dir / f"policy_ep{episode + 1}.pt"
            policy_net.save(str(checkpoint_path))
            print(f"  -> 保存 checkpoint: {checkpoint_path}")

    # ── 训练结束 ──────────────────────────────────────────────────────────
    t_elapsed = time.perf_counter() - t_start
    final_path = save_dir / "policy_final.pt"
    policy_net.save(str(final_path))

    avg_reward_all = np.mean(episode_rewards)
    avg_our_all = np.mean(our_team_scores)
    avg_opp_all = np.mean(opp_team_scores)

    # 按游戏局数区间统计
    all_game_rewards = []
    for ep_rew in episode_rewards:
        all_game_rewards.append(ep_rew / 2)

    n_games = len(all_game_rewards)

    print()
    print("=" * 72)
    print("训练完成！")
    print(f"总耗时: {t_elapsed:.0f}s (平均 {(t_elapsed / num_episodes):.1f}s / episode)")
    print(f"最终模型: {final_path}")
    print()
    print(f"全场平均 game 奖励:     {np.mean(all_game_rewards):+7.1f}")
    print(f"我方平均得分:  {avg_our_all:+7.1f}")
    print(f"对方平均得分:  {avg_opp_all:+7.1f}")

    # 【修改目标验证】第1~200局 vs 第801~1000局
    if n_games >= 500:
        first_200 = np.mean(all_game_rewards[:100])    # 第1~200局 (episodes 0-99)
        last_200_total = np.mean(all_game_rewards[400:500]) if n_games >= 500 else 0
        print()
        print(f"第1~200局平均 game reward: {first_200:+7.1f}")
        print(f"第801~1000局平均 game reward: {last_200_total:+7.1f}")
        improvement = last_200_total - first_200
        print(f"改进: {improvement:+7.1f}")
        if improvement >= 12:
            print("✅ 【修改目标达成】改进 >= 12分！")
        else:
            print(f"❌ 【修改目标未达成】改进 < 12分，还需要 {12 - improvement:.1f} 分")
    elif n_games >= 100:
        first_100 = np.mean(all_game_rewards[:50])
        last_100 = np.mean(all_game_rewards[-50:])
        print()
        print(f"前100局平均 game reward: {first_100:+7.1f}")
        print(f"最后100局平均 game reward: {last_100:+7.1f}")
        print(f"改进: {last_100 - first_100:+7.1f}")

    print(f"最后 50 episode 平均奖励: {np.mean(episode_rewards[-50:]):+7.1f}")
    print(f"最后 50 episode 我方得分: {np.mean(our_team_scores[-50:]):+7.1f}")
    print(f"最后 50 episode 对方得分: {np.mean(opp_team_scores[-50:]):+7.1f}")
    print("=" * 72)

    # ── 关闭 TensorBoard ────────────────────────────────────────────
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    train(parse_args())
