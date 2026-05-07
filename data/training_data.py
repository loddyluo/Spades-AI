"""训练数据生成与读写工具。

这一层负责把“局面生成 -> 求解器标签 -> PyTorch 可加载文件”串起来。
目前默认基于 `cpp_opt1`，并按剩余牌数 x=24/25/28/32 分桶。
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import torch

from trick_taking.card import Card, Suit
from trick_taking.deck import Deck, STANDARD_52
from trick_taking.game_state import Bid, GameState, Phase
from trick_taking.games.spades import SpadesRules
from trick_taking.solvers.exact_double_dummy_cpp_opt1 import ExactDoubleDummyCppOpt1Solver
from trick_taking.utils.feature_encoder import SpadesFeatureEncoder

SUPPORTED_BUCKETS: tuple[int, ...] = (24, 25, 28, 32)
DEFAULT_OUTPUT_PREFIX = "spades_dd"


def _apply_action(state: GameState, action: Card, player_id: int, rules: SpadesRules) -> None:
    """把一步合法出牌写回状态。"""
    state.play_card_to_table(player_id, action)
    if action.suit == Suit.SPADES:
        state.spades_broken = True
        state.trump_broken = True
    state.turn = (player_id + 1) % state.num_players
    if state.trick_complete:
        winner = rules.winner_trick(state)
        state.complete_trick(winner)
        state.trick_leader = winner
        state.turn = winner


def build_state_with_remaining_cards(target_remaining: int, seed: int) -> GameState:
    """生成一个满足剩余牌数要求的确定性局面。"""
    rng = random.Random(seed)
    deck = Deck(STANDARD_52, seed=seed)
    rules = SpadesRules()

    hands = [deck.deal(13) for _ in range(4)]

    state = GameState()
    state.init_for_deal(4, hands, [], deck.all_cards)

    # 叫牌要合法且可复现；这里保留随机性，但完全由 seed 控制。
    bids: list[Bid] = []
    max_bid: list[str] = []
    for pid in range(4):
        if rng.random() < 0.25:
            bid = "nil" if rng.random() < 0.85 else "blind_nil"
        else:
            bid = f"bid_{rng.randint(1, 13)}"
        bids.append(Bid(player_id=pid, value=bid))
        max_bid.append(bid)

    state.bids = bids
    state.max_bid = max_bid
    state.teams = [0, 1, 0, 1]
    state.phase = Phase.PLAYING
    state.turn = 0
    state.trick_leader = 0

    while sum(len(hand) for hand in state.hands) > target_remaining:
        pid = state.turn
        legal_actions = rules.playable(state, state.hands[pid], pid)
        if not legal_actions:
            break
        _apply_action(state, rng.choice(legal_actions), pid, rules)

    actual_remaining = sum(len(hand) for hand in state.hands)
    if actual_remaining != target_remaining:
        raise AssertionError(
            f"局面构造失败: target={target_remaining}, actual={actual_remaining}, seed={seed}"
        )

    return state


def _summarize_state(state: GameState) -> dict[str, Any]:
    """保存少量可读摘要，便于调试和回溯。"""
    return {
        "turn": int(state.turn),
        "trick_leader": int(state.trick_leader),
        "tricks_played": int(state.tricks_played),
        "spades_broken": bool(state.spades_broken),
        "table_cards": [(int(pid), str(card)) for pid, card in state.table_cards],
        "hand_sizes": [len(hand) for hand in state.hands],
        "tricks_won": [int(x) for x in state.tricks_won],
        "bids": [str(bid.value) for bid in state.bids],
    }


def generate_bucket_sample(
    target_remaining: int,
    seed: int,
    *,
    encoder: SpadesFeatureEncoder | None = None,
    solver: ExactDoubleDummyCppOpt1Solver | None = None,
) -> dict[str, Any]:
    """生成单条训练样本。

    返回值是 PyTorch 可直接 `torch.save` 的字典。
    其中 `value_team0` 是全局视角，`value_view` 是当前行动方视角。
    """
    if encoder is None:
        encoder = SpadesFeatureEncoder()
    if solver is None:
        solver = ExactDoubleDummyCppOpt1Solver()

    if not solver.native_available:
        raise RuntimeError("cpp_opt1 不可用，无法生成训练数据")

    state = build_state_with_remaining_cards(target_remaining, seed)
    player_id = state.turn

    feature = encoder.encode(state, player_id)
    result = solver.solve_with_q(state)

    action_items = sorted(
        result["action_q_values"].items(),
        key=lambda item: item[0].card_id,
    )
    action_ids = [action.card_id for action, _ in action_items]
    action_q_values = [float(q_value) for _, q_value in action_items]

    best_action = result["best_action"]
    best_action_id = int(best_action.card_id) if best_action is not None else -1

    value_team0 = float(result["value"])
    value_view = value_team0 if state.teams[player_id] == 0 else -value_team0

    return {
        "x": int(target_remaining),
        "seed": int(seed),
        "feature": torch.tensor(feature, dtype=torch.float32),
        "value_team0": torch.tensor(value_team0, dtype=torch.float32),
        "value_view": torch.tensor(value_view, dtype=torch.float32),
        "best_action_id": int(best_action_id),
        "current_player": int(result["current_player"]),
        "optimize_for_team": int(result["optimize_for_team"]),
        "action_ids": torch.tensor(action_ids, dtype=torch.int64),
        "action_q_values": torch.tensor(action_q_values, dtype=torch.float32),
        "state_summary": _summarize_state(state),
        "feature_dim": int(feature.shape[0]),
    }


def generate_bucket_dataset(
    target_remaining: int,
    num_samples: int,
    *,
    seed_start: int = 0,
    encoder: SpadesFeatureEncoder | None = None,
    solver: ExactDoubleDummyCppOpt1Solver | None = None,
) -> list[dict[str, Any]]:
    """批量生成一个剩余牌数桶的数据集。"""
    if target_remaining not in SUPPORTED_BUCKETS:
        raise ValueError(f"不支持的桶: x={target_remaining}, 仅支持 {SUPPORTED_BUCKETS}")

    if encoder is None:
        encoder = SpadesFeatureEncoder()
    if solver is None:
        solver = ExactDoubleDummyCppOpt1Solver()

    samples: list[dict[str, Any]] = []
    for index in range(num_samples):
        seed = seed_start + index
        samples.append(
            generate_bucket_sample(
                target_remaining,
                seed,
                encoder=encoder,
                solver=solver,
            )
        )
    return samples


def save_bucket_dataset(samples: list[dict[str, Any]], output_path: str | Path) -> None:
    """把样本列表保存成 PyTorch 文件。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generator": "cpp_opt1",
        "feature_dim": int(samples[0]["feature_dim"]) if samples else SpadesFeatureEncoder().total_dim,
        "supported_buckets": list(SUPPORTED_BUCKETS),
        "num_samples": len(samples),
    }
    torch.save({"meta": meta, "samples": samples}, output_path)


def load_bucket_dataset(dataset_path: str | Path) -> dict[str, Any]:
    """从 PyTorch 文件读取数据集。"""
    return torch.load(dataset_path, map_location="cpu")


def dataset_file_name(target_remaining: int, num_samples: int, prefix: str = DEFAULT_OUTPUT_PREFIX) -> str:
    """生成统一的文件名。"""
    return f"{prefix}_x{target_remaining}_n{num_samples}.pt"


def dataset_path(output_dir: str | Path, target_remaining: int, num_samples: int, prefix: str = DEFAULT_OUTPUT_PREFIX) -> Path:
    """生成统一的文件路径。"""
    return Path(output_dir) / dataset_file_name(target_remaining, num_samples, prefix=prefix)


def benchmark_generation(target_remaining: int, num_samples: int, seed_start: int = 0) -> dict[str, Any]:
    """统计生成耗时，便于写基准测试或打印日志。"""
    encoder = SpadesFeatureEncoder()
    solver = ExactDoubleDummyCppOpt1Solver()
    if not solver.native_available:
        raise RuntimeError("cpp_opt1 不可用，无法生成数据")

    t0 = time.perf_counter()
    samples = generate_bucket_dataset(
        target_remaining,
        num_samples,
        seed_start=seed_start,
        encoder=encoder,
        solver=solver,
    )
    elapsed = time.perf_counter() - t0
    return {
        "x": target_remaining,
        "num_samples": num_samples,
        "elapsed": elapsed,
        "avg": elapsed / max(num_samples, 1),
        "samples": samples,
    }
