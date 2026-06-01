"""
RL policy gradient 训练脚本（多核版本）：rl_exact vs DDS。

使用 multiprocessing.Pool 并行化打牌（数据收集），汇总到主进程做梯度更新。

用法:
    python rl/train_rl_multicpu.py --num-games 10000 --seed 42 --lr 0.001 --num-workers 8
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
        description="RL policy gradient training (multi-CPU): rl_exact vs DDS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-games", type=int, default=10000,
                        help="训练总对局数（每 episode 2 局，共 num_games/2 次更新）")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512,256],
                        help="策略网络隐藏层维度")
    parser.add_argument("--exact-threshold", type=int, default=36,
                        help="剩余牌数 <= 该值时使用精确求解器（默认 36 = 前16张用RL）")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子（当前未使用，完整 REINFORCE 可加）")
    parser.add_argument("--update-interval", type=int, default=200,
                        help="每多少 episode 做一次梯度更新（默认 200 ≈ 3200 trajectories ≥ 3000）")
    parser.add_argument("--num-epochs", type=int, default=10,
                        help="每个 batch 的轨迹被复用的 epoch 数（每条轨迹学习 num_epochs 次）")
    parser.add_argument("--save-dir", type=str, default="rl_checkpoints",
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
    parser.add_argument("--baseline-decay", type=float, default=0.95,
                        help="全局 EMA 基线衰减系数")
    parser.add_argument("--num-workers", type=int, default=30,
                        help="并行打牌的进程数")
    parser.add_argument("--tensorboard", action="store_true", default=True,
                        help="启用 TensorBoard 日志")
    parser.add_argument("--log-dir", type=str, default="runs/rl_train",
                        help="TensorBoard 日志目录")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="从指定路径加载之前训练过的 checkpoint（.pt 文件），在此基础上继续训练")
    return parser.parse_args()


def _compute_team_scores(result: Any) -> tuple[float, float]:
    """从游戏结果计算队伍得分（仅看是否完成叫牌）。

    得墩 ≥ 叫墩总和 → 0分，否则 → -100分。
    返回 (队伍0的payoff, 队伍1的payoff)。
    """
    ### Mode 1
    # scores = result.scores
    # t0 = scores[0]
    # t1 = scores[1]

    # return t0/2.0, t1/2.0

    ### Mode 2
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

    diff0 = team0_tricks - team0_bid
    diff1 = team1_tricks - team1_bid
    t0 = -100.0 if team0_tricks < team0_bid else 0.0
    t1 = -100.0 if team1_tricks < team1_bid else 0.0
    return t0 - t1, t1 - t0


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
    """打一局：rl_exact vs DDS，返回结果和 RL 轨迹。"""
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


# ── 工作进程函数 ─────────────────────────────────────────────────────
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

    args_tuple: (episode_indices, base_seed, policy_state_dict, hidden_dims,
                 exact_threshold, rules_args, bid_checkpoint, device, entropy_coef)
    """
    (episode_indices, base_seed, policy_state_dict, hidden_dims,
     exact_threshold, rules_args, bid_checkpoint, device, entropy_coef) = args_tuple
    # 每个 worker 独立创建自己的资源
    policy_net = PolicyMLP(input_dim=387, hidden_dims=hidden_dims)
    policy_net.load_state_dict(policy_state_dict)
    policy_net.to(device)

    exact_solver = ExactDoubleDummyCppFastestSolver()
    if not exact_solver.native_available:
        from trick_taking.solvers.exact_double_dummy import ExactDoubleDummySolver
        exact_solver = ExactDoubleDummySolver()

    encoder = RLFeatureEncoder()
    rules = SpadesRules(*rules_args)
    bid_model = _load_bid_model_worker(bid_checkpoint, device)

    results = []
    for ep_idx in episode_indices:
        # 和原始代码一致: seed = base_seed + ep_idx * 2
        game_seed = base_seed + ep_idx * 2

        episode_our_score = 0.0
        episode_opp_score = 0.0
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

            team0_score, team1_score = _compute_team_scores(result)

            if game_idx == 0:
                our_score = team0_score
                opp_score = team1_score
            else:
                our_score = team1_score
                opp_score = team0_score

            episode_our_score += our_score
            episode_opp_score += opp_score

            # 提取纯数据（不要 torch tensor 图）
            for traj in trajectories:
                episode_trajs.append({
                    "feature": traj["feature"].copy(),
                    "action_id": traj["action"].card_id,
                    "legal_card_ids": traj["legal_card_ids"],
                    "log_prob_val": traj["log_prob"].item(),
                    "entropy_val": traj["entropy"].item() if "entropy" in traj else 0.0,
                })

        episode_reward = episode_our_score - episode_opp_score
        episode_game_reward = episode_reward / 2.0

        for traj in episode_trajs:
            traj["_game_reward"] = episode_game_reward

        results.append({
            "episode_game_reward": episode_game_reward,
            "episode_reward": episode_reward,
            "episode_our_score": episode_our_score,
            "episode_opp_score": episode_opp_score,
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
            # 加载后切回训练模式
            policy_net.train()
            print(f"从 checkpoint 加载模型: {cp_path.resolve()}")
        else:
            print(f"警告: checkpoint 不存在: {cp_path}，将使用随机初始化")

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)

    # ── 叫牌模型（主进程加载一次用于验证存在性） ─────────────────────
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

    episode_rewards: list[float] = []
    our_team_scores: list[float] = []
    opp_team_scores: list[float] = []

    print("=" * 72, flush=True)
    print("RL Policy Gradient Training (MULTI-CPU): rl_exact vs DDS", flush=True)
    print(f"总对局数: {args.num_games} ({num_episodes} episodes × 2 games)", flush=True)
    print(f"学习率: {args.lr}", flush=True)
    print(f"隐藏层: {args.hidden_dims}", flush=True)
    print(f"精确阈值: {args.exact_threshold} (前 {52 - args.exact_threshold} 张用 RL)", flush=True)
    print(f"熵系数: {args.entropy_coef}", flush=True)
    print(f"Workers: {args.num_workers}", flush=True)
    print("=" * 72, flush=True)

    writer = None
    if args.tensorboard:
        log_path = Path(args.log_dir) / f"05312213_seed{args.seed}_lr{args.lr}_hid{'_'.join(str(h) for h in args.hidden_dims)}_w{args.num_workers}"
        writer = SummaryWriter(log_dir=str(log_path)+"[0531]")
        print(f"TensorBoard 日志: {log_path}", flush=True)

    t_start = time.perf_counter()
    all_episode_rewards_flat: list[float] = []

    accumulated_trajectories: list[dict] = []
    batch_game_rewards: list[float] = []

    # 全局 EMA 基线
    global_baseline = 0.0
    global_baseline_init = False

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
            print("worker args")
            # 并行打牌
            results = pool.map(worker_batch, worker_args)
            print("Summarizing\n")
            # 汇总
            for worker_results in results:
                for ep_res in worker_results:
                    batch_game_rewards.append(ep_res["episode_game_reward"])
                    episode_rewards.append(ep_res["episode_reward"])
                    all_episode_rewards_flat.append(ep_res["episode_reward"])
                    our_team_scores.append(ep_res["episode_our_score"])
                    opp_team_scores.append(ep_res["episode_opp_score"])

                    for traj in ep_res["trajectories"]:
                        accumulated_trajectories.append(traj)

            # ── 梯度更新 ──────────────────────────────────────────
            if accumulated_trajectories:
                rewards_batch = np.array(batch_game_rewards)
                baseline = float(np.mean(rewards_batch))
                total_trajs = len(accumulated_trajectories)

                # ── 多 epoch 复用同一批轨迹（每条轨迹学习 num_epochs 次）──
                # 注: epoch >= 2 时 logits 已被前一个 epoch 的更新改动，严格意义上
                # 已不是采样时的 on-policy 分布；这里按用户要求做朴素 REINFORCE 复用，
                # 不加 PPO-style importance ratio 修正。
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
                    # ── 监控分布塌缩：合法动作上的 max prob / 熵 / 归一化熵 ──
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

                        raw_adv = traj["_game_reward"] - baseline
                        raw_advantages.append(raw_adv)
                        reinforce_terms.append(-log_prob * raw_adv)

                        # 熵（用于熵正则 + 塌缩监控）
                        probs = torch.exp(log_probs)
                        safe_log_probs = torch.where(probs > 0, log_probs, torch.zeros_like(log_probs))
                        entropy = -torch.sum(probs * safe_log_probs)
                        if has_entropy:
                            entropy_terms.append(-args.entropy_coef * entropy)

                        # ── 塌缩监控（不参与反传，只取标量）──
                        with torch.no_grad():
                            n_legal = max(len(legal_ids), 1)
                            # 在合法动作上取 max prob（非法动作 prob=0，不影响 max）
                            max_probs_legal.append(float(probs.max().item()))
                            ent_val = float(entropy.item())
                            entropies_legal.append(ent_val)
                            # 归一化熵：除以 log(n_legal)，n_legal=1 时定义为 0
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

                    # 保存最后一个 epoch 的统计用于打印
                    last_reinforce_loss = reinforce_loss.item()
                    last_grad_norm = float(grad_norm)
                    last_n_pos_adv = sum(1 for a in raw_advantages if a > 0)
                    last_mean_max_prob = float(np.mean(max_probs_legal))
                    last_mean_entropy = float(np.mean(entropies_legal))
                    last_mean_norm_entropy = float(np.mean(norm_entropies_legal))

                    # 每个 epoch 都写 TensorBoard，step 用 batch_end * num_epochs + epoch 防止覆盖
                    if writer is not None:
                        tb_step = batch_end * args.num_epochs + epoch
                        writer.add_scalar("update/reinforce_loss", last_reinforce_loss, tb_step)
                        writer.add_scalar("update/grad_norm", last_grad_norm, tb_step)
                        writer.add_scalar("update/baseline", baseline, tb_step)
                        # ── 分布塌缩监控 ──
                        writer.add_scalar("policy/mean_max_prob", last_mean_max_prob, tb_step)
                        writer.add_scalar("policy/mean_entropy", last_mean_entropy, tb_step)
                        writer.add_scalar("policy/mean_norm_entropy", last_mean_norm_entropy, tb_step)

                print(f"  [Update] ep {batch_start+1}-{batch_end} (×{args.num_epochs} epochs, "
                      f"trajs={total_trajs}), "
                      f"loss={last_reinforce_loss:.3f}, "
                      f"gn={last_grad_norm:.3f}, bl={baseline:.1f}, "
                      f"avg_game_rew={np.mean(batch_game_rewards):+.1f}, "
                      f"pos={last_n_pos_adv}/{total_trajs}, "
                      f"max_p={last_mean_max_prob:.3f}, "
                      f"H={last_mean_entropy:.3f}, "
                      f"H_norm={last_mean_norm_entropy:.3f}", flush=True)

                accumulated_trajectories = []
                batch_game_rewards = []

            # ── 日志 ──────────────────────────────────────────────
            if batch_end % 40 == 0 and episode_rewards:
                recent = episode_rewards[-40:]
                avg_reward = np.mean(recent)
                print("[recent]" , recent, flush=True)
                avg_our = np.mean(our_team_scores[-40:])
                avg_opp = np.mean(opp_team_scores[-40:])
                elapsed = time.perf_counter() - t_start
                print(
                    f"Episode {batch_end:5d}/{num_episodes} | "
                    f"AvgEpReward={avg_reward:+7.1f} | "
                    f"AvgGameReward={avg_reward/2:+7.1f} | "
                    f"AvgOur={avg_our:+7.1f} | AvgOpp={avg_opp:+7.1f} | "
                    f"Time={elapsed:.0f}s", flush=True
                )

            # ── TensorBoard ──────────────────────────────────────
            if writer is not None and batch_end % 300 == 0 and episode_rewards:
                recent300 = episode_rewards[-300:]
                writer.add_scalar("train/avg_episode_reward", np.mean(recent300), batch_end)
                writer.add_scalar("train/avg_game_reward", np.mean(recent300) / 2.0, batch_end)

            # ── 保存 checkpoint ──────────────────────────────────
            if batch_end % args.save_interval == 0:
                cp_path = save_dir / f"312212policy_ep{batch_end}.pt"
                policy_net.save(str(cp_path))
                print(f"  -> 保存: {cp_path}", flush=True)

    # ── 训练结束 ──────────────────────────────────────────────────────
    t_elapsed = time.perf_counter() - t_start
    final_path = save_dir / "policy_final.pt"
    policy_net.save(str(final_path))

    all_game_rewards = [r / 2 for r in all_episode_rewards_flat]
    n_games = len(all_game_rewards)

    print(flush=True)
    print("=" * 72, flush=True)
    print("训练完成！", flush=True)
    print(f"总耗时: {t_elapsed:.0f}s (平均 {t_elapsed/max(num_episodes,1):.1f}s/episode)", flush=True)
    print(f"最终模型: {final_path}", flush=True)
    print(f"全场平均 game 奖励: {np.mean(all_game_rewards):+7.1f}", flush=True)

    if n_games >= 500:
        first_200 = np.mean(all_game_rewards[:100])
        last_200 = np.mean(all_game_rewards[400:500])
        print(f"第1~200局: {first_200:+7.1f}", flush=True)
        print(f"第801~1000局: {last_200:+7.1f}", flush=True)
        impr = last_200 - first_200
        print(f"改进: {impr:+7.1f}", flush=True)
        if impr >= 12:
            print("✅ 目标达成！", flush=True)
        else:
            print(f"❌ 还需 {12-impr:.1f} 分", flush=True)
    print("=" * 72, flush=True)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    train(parse_args())
