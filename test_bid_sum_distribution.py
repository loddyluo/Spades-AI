"""Measure bidding distributions from the residual-Q bidder deployed by the GUI."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass

from residual_bidder.actions import BidAction, to_local_bid
from residual_bidder.deployment import load_deployed_acting_bidder
from strategy.spades_match_runner import build_random_state
from trick_taking.game_state import Bid
from trick_taking.games.spades import SpadesRules


RUNTIME_ROOM_ID = "http-local"


@dataclass(frozen=True)
class SimulationResult:
    n_games: int
    seed_start: int
    model: dict[str, object]
    checkpoint_path: str
    policy_seed: int
    total_sum_counter: Counter[int]
    team_sum_counter: Counter[int]
    bid_counter: Counter[int]
    bid_position_counters: tuple[Counter[int], ...]
    average_probability_mass: tuple[float, ...]
    residual_delta_counter: Counter[int]


def _bid_label(value: int) -> str:
    return "nil" if value == int(BidAction.NIL) else str(value)


def _value_at_index(counter: Counter[int], index: int) -> int:
    if not 0 <= index < sum(counter.values()):
        raise IndexError("counter index is outside the sample")
    cumulative = 0
    for value in sorted(counter):
        cumulative += counter[value]
        if index < cumulative:
            return value
    raise AssertionError("nonempty counter did not contain the requested index")


def _summary(counter: Counter[int]) -> dict[str, float | int]:
    sample_count = sum(counter.values())
    if sample_count <= 0:
        raise ValueError("cannot summarize an empty counter")
    mean = math.fsum(value * count for value, count in counter.items()) / sample_count
    variance = (
        math.fsum(count * (value - mean) ** 2 for value, count in counter.items())
        / sample_count
    )
    lower_middle = _value_at_index(counter, (sample_count - 1) // 2)
    upper_middle = _value_at_index(counter, sample_count // 2)
    mode, mode_count = counter.most_common(1)[0]
    return {
        "mean": mean,
        "stddev": math.sqrt(variance),
        "minimum": min(counter),
        "maximum": max(counter),
        "median": (lower_middle + upper_middle) / 2.0,
        "mode": mode,
        "mode_count": mode_count,
    }


def run_simulation(
    n_games: int = 10_000,
    seed_start: int = 42,
    *,
    device: str = "cpu",
    policy_seed: int | None = None,
) -> SimulationResult:
    """Run four production-path bids on each reproducibly shuffled deal."""

    if type(n_games) is not int or n_games <= 0:
        raise ValueError("n_games must be a positive integer")
    if type(seed_start) is not int:
        raise TypeError("seed_start must be an integer")

    bidder = load_deployed_acting_bidder(device=device, policy_seed=policy_seed)
    rules = SpadesRules(enable_nil=True, enable_blind_nil=False)

    total_sum_counter: Counter[int] = Counter()
    team_sum_counter: Counter[int] = Counter()
    bid_counter: Counter[int] = Counter()
    bid_position_counters = tuple(Counter() for _ in range(rules.num_players))
    probability_mass = [0.0] * len(BidAction)
    residual_delta_counter: Counter[int] = Counter()

    for game_offset in range(n_games):
        shuffle_seed = seed_start + game_offset
        state = build_random_state(shuffle_seed)
        hand_bids = [0] * rules.num_players

        for bid_index in range(rules.num_players):
            seat = state.current_bidder
            legal_bids = rules.legal_bids(state, seat)
            decision = bidder.choose(
                state,
                legal_bids,
                logical_seat=seat,
                deal_id=f"local:{shuffle_seed}",
                room_id=RUNTIME_ROOM_ID,
            )
            if decision.fallback_reason is not None:
                raise RuntimeError(
                    "deployed bidder triggered a forbidden fallback on "
                    f"seed={shuffle_seed}, bid_index={bid_index}: "
                    f"{decision.fallback_reason}"
                )

            local_bid = to_local_bid(decision.action)
            if local_bid not in legal_bids:
                raise AssertionError(f"deployed bidder returned illegal bid {local_bid!r}")
            numeric_bid = int(decision.action)
            hand_bids[seat] = numeric_bid
            bid_counter[numeric_bid] += 1
            bid_position_counters[bid_index][numeric_bid] += 1
            residual_delta_counter[
                numeric_bid - int(decision.distribution.center)
            ] += 1
            for action, probability in zip(
                BidAction,
                decision.distribution.probabilities,
                strict=True,
            ):
                probability_mass[int(action)] += probability

            state.bids.append(Bid(player_id=seat, value=local_bid, is_pass=False))
            state.max_bid[seat] = local_bid
            state.current_bidder = rules.next_bid_turn(state)
            state.turn = state.current_bidder

        if not rules.end_bidding(state) or any(value is None for value in state.max_bid):
            raise AssertionError("simulated auction did not produce four actual bids")
        total_sum_counter[sum(hand_bids)] += 1
        team_sum_counter[hand_bids[0] + hand_bids[2]] += 1
        team_sum_counter[hand_bids[1] + hand_bids[3]] += 1

    total_bids = rules.num_players * n_games
    return SimulationResult(
        n_games=n_games,
        seed_start=seed_start,
        model=bidder.describe(),
        checkpoint_path=str(bidder.checkpoint_path),
        policy_seed=bidder.policy_seed,
        total_sum_counter=total_sum_counter,
        team_sum_counter=team_sum_counter,
        bid_counter=bid_counter,
        bid_position_counters=bid_position_counters,
        average_probability_mass=tuple(
            mass / total_bids for mass in probability_mass
        ),
        residual_delta_counter=residual_delta_counter,
    )


def _print_numeric_distribution(
    title: str,
    counter: Counter[int],
    *,
    value_label: str,
) -> None:
    sample_count = sum(counter.values())
    print(f"\n=== {title}（{sample_count:,} 个样本）===\n")
    print(f"{value_label:>6}  {'次数':>9}  {'占比':>8}  {'累积':>8}")
    print("-" * 39)
    cumulative = 0
    for value in sorted(counter):
        count = counter[value]
        cumulative += count
        print(
            f"{value:6d}  {count:9,d}  "
            f"{100.0 * count / sample_count:7.3f}%  "
            f"{100.0 * cumulative / sample_count:7.3f}%"
        )

    stats = _summary(counter)
    print(
        "统计量: "
        f"均值={stats['mean']:.3f}, "
        f"标准差={stats['stddev']:.3f}, "
        f"中位数={stats['median']:.1f}, "
        f"范围=[{stats['minimum']}, {stats['maximum']}], "
        f"众数={stats['mode']} "
        f"({100.0 * int(stats['mode_count']) / sample_count:.3f}%)"
    )


def print_report(result: SimulationResult) -> None:
    model = result.model
    calibration = model["calibration"]
    assert isinstance(calibration, dict)
    total_bids = sum(result.bid_counter.values())

    print("=== GUI 当前部署 residual-Q 叫牌模型分布测试 ===")
    print(f"随机牌局: {result.n_games:,}")
    print(
        f"种子范围: [{result.seed_start}, "
        f"{result.seed_start + result.n_games - 1}]"
    )
    print(f"模型名称: {model['name']}")
    print(f"model_id: {model['model_id']}")
    print(f"policy_id: {model['policy_id']}")
    print(f"checkpoint: {result.checkpoint_path}")
    print(f"checkpoint_sha256: {model['checkpoint_sha256']}")
    print(f"policy_seed: {result.policy_seed}")
    print(
        "校准参数: "
        + ", ".join(f"{key}={value}" for key, value in calibration.items())
    )
    print("fallback: 0（脚本遇到任何 fallback 会立即报错退出）")

    print(f"\n=== 单家实际叫牌分布（{total_bids:,} 次叫牌）===\n")
    print(f"{'叫牌':>6}  {'次数':>9}  {'实际占比':>10}  {'平均策略概率':>12}")
    print("-" * 48)
    for action in BidAction:
        value = int(action)
        count = result.bid_counter[value]
        print(
            f"{_bid_label(value):>6}  {count:9,d}  "
            f"{100.0 * count / total_bids:9.3f}%  "
            f"{100.0 * result.average_probability_mass[value]:11.3f}%"
        )

    print("\n按叫牌次序:")
    for bid_index, counter in enumerate(result.bid_position_counters, start=1):
        stats = _summary(counter)
        nil_count = counter[int(BidAction.NIL)]
        print(
            f"  第 {bid_index} 叫: 均值={stats['mean']:.3f}, "
            f"nil={100.0 * nil_count / result.n_games:.3f}%, "
            f"范围=[{stats['minimum']}, {stats['maximum']}]"
        )

    print("\nResidual 动作相对旧 NSFP 中心的偏移:")
    for delta in sorted(result.residual_delta_counter):
        count = result.residual_delta_counter[delta]
        print(
            f"  {delta:+d}: {count:9,d} "
            f"({100.0 * count / total_bids:7.3f}%)"
        )

    _print_numeric_distribution(
        "四家叫牌总和分布",
        result.total_sum_counter,
        value_label="总和",
    )
    _print_numeric_distribution(
        "队伍叫牌和分布：0&2 / 1&3",
        result.team_sum_counter,
        value_label="队伍和",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测试 GUI 当前部署 residual-Q 叫牌模型的输出分布",
    )
    parser.add_argument("--games", type=int, default=10_000, help="随机牌局数")
    parser.add_argument("--seed-start", type=int, default=42, help="首个洗牌种子")
    parser.add_argument("--device", default="cpu", help="PyTorch 推理设备")
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=None,
        help="覆盖部署配置中的策略随机种子",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_simulation(
        n_games=args.games,
        seed_start=args.seed_start,
        device=args.device,
        policy_seed=args.policy_seed,
    )
    print_report(result)


if __name__ == "__main__":
    main()
