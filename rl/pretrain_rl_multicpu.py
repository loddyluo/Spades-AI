"""
预训练 RL policy gradient 脚本（多核版本）：前4墩逐牌奖励。

只打前4墩（16张牌），每张出牌计算逐牌奖励：
  - 赢墩（成为下一墩首攻）：+5
  - 输墩（未赢）：按点数扣分 A=18, K=12, Q=8, J=3, T=1, 其余0

用法:
    python rl/pretrain_rl_multicpu.py --num-games 10000 --seed 42 --lr 0.001 --num-workers 8
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
from rl.rl_feature_encoder import RLFeatureEncoder
from strategy.spades_match_runner import SpadesMatchRunner
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy_cpp_fastest import (
    ExactDoubleDummyCppFastestSolver,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain RL policy gradient (multi-CPU): 前4墩逐牌奖励",
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
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子（当前未使用）")
    parser.add_argument("--update-interval", type=int, default=200,
                        help="每多少 episode 做一次梯度更新")
    parser.add_argument("--num-epochs", type=int, default=10,
                        help="每个 batch 的轨迹被复用的 epoch 数")
    parser.add_argument("--save-dir", type=str, default="rl_checkpoints/pretrain",
                        help="模型保存目录")
    parser.add_argument("--save-interval", type=int, default=5000,
                        help="每多少 episode 保存一次 checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="训练设备")
    parser.add_argument("--disable-blind-nil", action="store_true", default=True,
                        help="禁用 blind nil 叫牌")
    parser.add_argument("--bid-checkpoint", type=str,
                        default="./Spades_AI_GO-MCTS/checkpoints/bid_nsfp.pt",
                        help="叫牌 MLP 模型 checkpoint 路径")
    parser.add_argument("--entropy-coef", type=float, default=0.25,
                        help="熵奖励系数")
    parser.add_argument("--max-grad-norm", type=float, default=15.0,
                        help="梯度裁剪最大范数")
    parser.add_argument("--num-workers", type=int, default=30,
                        help="并行打牌的进程数")
    parser.add_argument("--tensorboard", action="store_true", default=True,
                        help="启用 TensorBoard 日志")
    parser.add_argument("--log-dir", type=str, default="runs/pretrain_rl",
                        help="TensorBoard 日志目录")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="从指定路径加载之前训练过的 checkpoint（.pt 文件），在此基础上继续训练")
    return parser.parse_args()


def _build_dds_player(bid_model=None) -> DDSPlayer:
    return DDSPlayer(bid_model=bid_model)


def _build_rl_exact_player(
    policy_net: PolicyMLP,
    exact_solver: ExactDoubleDummyCppFastestSolver,
    encoder: RLFeatureEncoder,
    exact_threshold: int,
    is_training: bool,
    bid_model=None,
    bid_device: str = "cpu",
) -> RLExactPlayer:
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
    encoder: RLFeatureEncoder,
    exact_threshold: int,
    seed: int,
    is_training: bool,
    rules: SpadesRules,
    bid_model=None,
    bid_device: str = "cpu",
    swap_seats: bool = False,
) -> tuple[Any, list[dict[str, Any]]]:
    """打一局：rl_exact vs DDS，只打前4墩，返回结果和 RL 轨迹。"""
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
        max_tricks=4,  # 只打前4墩，后9墩跳过
    )
    _ = runner.play_game()  # 我们不使用 result，只收集 trajectory
    all_trajectories: list[dict[str, Any]] = []
    for player in players:
        if isinstance(player, RLExactPlayer):
            all_trajectories.extend(player.trajectory)
    return None, all_trajectories


# ── 工作进程函数 ────────────────────────────────────────────────────
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


def worker_batch(args_tuple: tuple) -> list[dict]:
    """在 worker 进程中打一批 episode，返回纯数据（无 torch tensor 图）。

    每个 trajectory entry 包含 per-step reward_val 而非 _game_reward。
    """
    (episode_indices, base_seed, policy_state_dict, hidden_dims,
     exact_threshold, rules_args, bid_checkpoint, device, entropy_coef) = args_tuple
    # 每个 worker 独立创建自己的资源
    policy_net = PolicyMLP(input_dim=387, hidden_dims=hidden_dims)
    policy_net.load_state_dict(policy_state_dict)
    policy_net.to(device)

    # 标记 pretrain_mode（RLExactPlayer 通过 getattr 读取 policy_net 上的属性）
    policy_net.pretrain_mode = True

    exact_solver = ExactDoubleDummyCppFastestSolver()
    if not exact_solver.native_available:
        from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver
        exact_solver = ExactDoubleDummySolver()

    encoder = RLFeatureEncoder()
    rules = SpadesRules(*rules_args)
    bid_model = _load_bid_model_worker(bid_checkpoint, device)

    results = []
    for ep_idx in episode_indices:
        game_seed = base_seed + ep_idx * 2

        episode_trajs: list[dict] = []

        for game_idx in range(2):
            # 队式赛：两局使用完全相同的牌面（相同 seed），只互换座位
            result, trajectories = play_one_game(
                policy_net=policy_net,
                exact_solver=exact_solver,
                encoder=encoder,
                exact_threshold=exact_threshold,
                seed=game_seed,
                is_training=True,
                rules=rules,
                bid_model=bid_model,
                bid_device=device,
                swap_seats=(game_idx == 1),
            )

            # 提取纯数据（不要 torch tensor 图）
            for traj in trajectories:
                reward_val = traj.get("reward", 0.0)
                episode_trajs.append({
                    "feature": traj["feature"].copy(),
                    "action_id": traj["action"].card_id,
                    "legal_card_ids": traj["legal_card_ids"],
                    "log_prob_val": traj["log_prob"].item(),
                    "entropy_val": traj["entropy"].item() if "entropy" in traj else 0.0,
                    "reward_val": reward_val,
                })

        results.append({
            "trajectories": episode_trajs,
        })

    return results


def train(args: argparse.Namespace) -> None:
    import multiprocessing as mp

    device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── 主进程网络 ────────────────────────────────────────────────────
    policy_net = PolicyMLP(input_dim=387, hidden_dims=args.hidden_dims).to(device)

    if args.load_checkpoint:
        cp_path = Path(args.load_checkpoint)
        if cp_path.exists():
            policy_net.load(str(cp_path.resolve()), device=args.device)
            policy_net.train()
            print(f"从 checkpoint 加载模型: {cp_path.resolve()}")
        else:
            print(f"警告: checkpoint 不存在: {cp_path}，将使用随机初始化")

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)

    # ── 叫牌模型 ──────────────────────────────────────────────────────
    if args.bid_checkpoint:
        cp = Path(args.bid_checkpoint)
        if cp.exists():
            print(f"叫牌模型: {cp} (worker 进程会自行加载)")
        else:
            print(f"警告: 叫牌模型 checkpoint 不存在: {cp}")

    rules_args = (False, not args.disable_blind_nil)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    num_episodes = args.num_games // 2

    all_rewards_flat: list[float] = []

    print("=" * 72, flush=True)
    print("Pretrain RL Policy Gradient (MULTI-CPU): 前4墩逐牌奖励", flush=True)
    print(f"总对局数: {args.num_games} ({num_episodes} episodes × 2 games)", flush=True)
    print(f"学习率: {args.lr}", flush=True)
    print(f"隐藏层: {args.hidden_dims}", flush=True)
    print(f"精确阈值: {args.exact_threshold} (前 {52 - args.exact_threshold} 张用 RL)", flush=True)
    print(f"熵系数: {args.entropy_coef}", flush=True)
    print(f"Workers: {args.num_workers}", flush=True)
    print(f"保存目录: {save_dir}", flush=True)
    print(f"TensorBoard: {args.log_dir}", flush=True)
    print("=" * 72, flush=True)

    writer = None
    if args.tensorboard:
        log_path = Path(args.log_dir) / f"06011045_pretrain_seed{args.seed}_lr{args.lr}_hid{'_'.join(str(h) for h in args.hidden_dims)}_w{args.num_workers}"
        writer = SummaryWriter(log_dir=str(log_path))
        print(f"TensorBoard 日志: {log_path}", flush=True)

    t_start = time.perf_counter()

    accumulated_trajectories: list[dict] = []

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.num_workers) as pool:
        for batch_start in range(0, num_episodes, args.update_interval):
            batch_end = min(batch_start + args.update_interval, num_episodes)
            batch_episodes = list(range(batch_start, batch_end))

            if not batch_episodes:
                continue

            # 将 episode 列表分块给 workers
            n_workers = min(args.num_workers, len(batch_episodes))
            chunks = np.array_split(batch_episodes, n_workers)

            # 准备 worker 参数：发送当前 policy 权重
            policy_state_dict = policy_net.state_dict()
            worker_args = [
                (chunk.tolist() if hasattr(chunk, 'tolist') else chunk,
                 args.seed, policy_state_dict, args.hidden_dims,
                 args.exact_threshold, rules_args, args.bid_checkpoint,
                 args.device, args.entropy_coef)
                for chunk in chunks
            ]

            # 并行打牌
            results = pool.map(worker_batch, worker_args)

            # 汇总
            batch_rewards: list[float] = []
            for worker_results in results:
                for ep_res in worker_results:
                    for traj in ep_res["trajectories"]:
                        accumulated_trajectories.append(traj)
                        batch_rewards.append(traj["reward_val"])
                        all_rewards_flat.append(traj["reward_val"])

            # ── 梯度更新 ──────────────────────────────────────────
            if accumulated_trajectories:
                baseline = float(np.mean(batch_rewards)) if batch_rewards else 0.0
                total_trajs = len(accumulated_trajectories)

                last_reinforce_loss = 0.0
                last_grad_norm = 0.0
                last_n_pos_adv = 0
                last_mean_max_prob = 0.0
                last_mean_entropy = 0.0
                last_mean_norm_entropy = 0.0

                has_entropy = "entropy_val" in accumulated_trajectories[0]

                for epoch in range(args.num_epochs):
                    optimizer.zero_grad()

                    reinforce_terms = []
                    raw_advantages = []
                    entropy_terms = []
                    max_probs_legal: list[float] = []
                    entropies_legal: list[float] = []
                    norm_entropies_legal: list[float] = []

                    for traj in accumulated_trajectories:
                        feature = traj["feature"]
                        feat_tensor = torch.from_numpy(feature).float().unsqueeze(0)
                        logits = policy_net(feat_tensor).squeeze(0)

                        legal_ids = traj["legal_card_ids"]
                        mask = torch.full((52,), float("-inf"))
                        for cid in legal_ids:
                            mask[cid] = 0.0
                        masked_logits = logits + mask
                        log_probs = torch.log_softmax(masked_logits, dim=0)
                        log_prob = log_probs[traj["action_id"]]

                        # 逐牌奖励优势
                        raw_adv = traj["reward_val"] - baseline
                        raw_advantages.append(raw_adv)
                        reinforce_terms.append(-log_prob * raw_adv)

                        # 熵
                        probs = torch.exp(log_probs)
                        safe_log_probs = torch.where(probs > 0, log_probs, torch.zeros_like(log_probs))
                        entropy = -torch.sum(probs * safe_log_probs)
                        if has_entropy:
                            entropy_terms.append(-args.entropy_coef * entropy)

                        # 塌缩监控
                        with torch.no_grad():
                            n_legal = max(len(legal_ids), 1)
                            max_probs_legal.append(float(probs.max().item()))
                            ent_val = float(entropy.item())
                            entropies_legal.append(ent_val)
                            if n_legal > 1:
                                norm_entropies_legal.append(ent_val / float(np.log(n_legal)))
                            else:
                                norm_entropies_legal.append(0.0)

                    reinforce_loss = torch.stack(reinforce_terms).mean()
                    loss = reinforce_loss
                    if has_entropy and entropy_terms:
                        entropy_loss = torch.mean(torch.stack(entropy_terms))
                        loss = reinforce_loss + entropy_loss

                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=args.max_grad_norm)
                    optimizer.step()

                    last_reinforce_loss = reinforce_loss.item()
                    last_grad_norm = float(grad_norm)
                    last_n_pos_adv = sum(1 for a in raw_advantages if a > 0)
                    last_mean_max_prob = float(np.mean(max_probs_legal))
                    last_mean_entropy = float(np.mean(entropies_legal))
                    last_mean_norm_entropy = float(np.mean(norm_entropies_legal))

                    if writer is not None:
                        tb_step = batch_end * args.num_epochs + epoch
                        writer.add_scalar("update/reinforce_loss", last_reinforce_loss, tb_step)
                        writer.add_scalar("update/grad_norm", last_grad_norm, tb_step)
                        writer.add_scalar("update/baseline", baseline, tb_step)
                        writer.add_scalar("policy/mean_max_prob", last_mean_max_prob, tb_step)
                        writer.add_scalar("policy/mean_entropy", last_mean_entropy, tb_step)
                        writer.add_scalar("policy/mean_norm_entropy", last_mean_norm_entropy, tb_step)

                print(f"  [Update] ep {batch_start+1}-{batch_end} (×{args.num_epochs} epochs, "
                      f"trajs={total_trajs}), "
                      f"loss={last_reinforce_loss:.3f}, "
                      f"gn={last_grad_norm:.3f}, bl={baseline:.1f}, "
                      f"avg_rew={np.mean(batch_rewards):+.2f}, "
                      f"pos={last_n_pos_adv}/{total_trajs}, "
                      f"max_p={last_mean_max_prob:.3f}, "
                      f"H={last_mean_entropy:.3f}, "
                      f"H_norm={last_mean_norm_entropy:.3f}", flush=True)

                accumulated_trajectories = []

            # ── 日志 ──────────────────────────────────────────────
            if batch_end % 40 == 0 and all_rewards_flat:
                recent = all_rewards_flat[-40 * 8:]  # 约40步 × 8 traj/步
                avg_reward = np.mean(recent)
                elapsed = time.perf_counter() - t_start
                print(
                    f"Episode {batch_end:5d}/{num_episodes} | "
                    f"AvgReward={avg_reward:+7.2f} | "
                    f"Time={elapsed:.0f}s", flush=True
                )

            # ── TensorBoard ──────────────────────────────────────
            if writer is not None and batch_end % 300 == 0 and all_rewards_flat:
                recent = all_rewards_flat[-300 * 8:]
                writer.add_scalar("train/avg_reward", np.mean(recent), batch_end)

            # ── 保存 checkpoint ──────────────────────────────────
            if batch_end % args.save_interval == 0:
                cp_path = save_dir / f"pretrain_policy_ep{batch_end}.pt"
                policy_net.save(str(cp_path))
                print(f"  -> 保存: {cp_path}", flush=True)

    # ── 训练结束 ──────────────────────────────────────────────────────
    t_elapsed = time.perf_counter() - t_start
    final_path = save_dir / "pretrain_policy_final.pt"
    policy_net.save(str(final_path))

    n_rewards = len(all_rewards_flat)

    print(flush=True)
    print("=" * 72, flush=True)
    print("预训练完成！", flush=True)
    print(f"总耗时: {t_elapsed:.0f}s", flush=True)
    print(f"最终模型: {final_path}", flush=True)
    print(f"全场平均逐牌奖励: {np.mean(all_rewards_flat):+7.2f}", flush=True)
    print("=" * 72, flush=True)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    train(parse_args())
