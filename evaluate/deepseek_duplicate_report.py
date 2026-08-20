"""Pair fixed-seat DeepSeek match records into a duplicate Spades report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from evaluate.deepseek_team_match import write_match_record


CURRENT_AI = "current_spades_ai"
DEEPSEEK = "deepseek-v4-flash"
TABLE_A_ASSIGNMENT = {
    "0": CURRENT_AI,
    "1": DEEPSEEK,
    "2": CURRENT_AI,
    "3": DEEPSEEK,
}
TABLE_B_ASSIGNMENT = {
    "0": DEEPSEEK,
    "1": CURRENT_AI,
    "2": DEEPSEEK,
    "3": CURRENT_AI,
}
STANDARD_DECK = {
    f"{rank}{suit}"
    for rank in "23456789TJQKA"
    for suit in "CDHS"
}


class DuplicateReportError(ValueError):
    """A source record cannot form a valid duplicate pair."""


def _load_single_game(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DuplicateReportError(f"无法读取记录 {path}: {exc}") from exc
    if report.get("format") != "spades-ai-deepseek-team-match":
        raise DuplicateReportError(f"记录格式不正确: {path}")
    games = report.get("games")
    if not isinstance(games, list) or len(games) != 1:
        raise DuplicateReportError(f"记录必须恰好包含一桌: {path}")
    return report, games[0]


def _team_scores(game: dict[str, Any]) -> tuple[float, float]:
    bids = game.get("bids")
    tricks_won = game.get("tricks_won")
    if not isinstance(bids, list) or len(bids) != 4:
        raise DuplicateReportError("叫牌记录必须包含四家")
    if not isinstance(tricks_won, list) or len(tricks_won) != 4:
        raise DuplicateReportError("赢墩记录必须包含四家")

    scores: list[float] = []
    for seats in ((0, 2), (1, 3)):
        score = 0.0
        bid_total = 0
        trick_total = sum(int(tricks_won[seat]) for seat in seats)
        for seat in seats:
            bid = bids[seat]
            if bid == "nil":
                score += 50.0 if int(tricks_won[seat]) == 0 else -50.0
            elif bid == "blind_nil":
                score += 100.0 if int(tricks_won[seat]) == 0 else -100.0
            elif isinstance(bid, str) and bid.startswith("bid_"):
                bid_total += int(bid.removeprefix("bid_"))
            else:
                raise DuplicateReportError(f"无效叫牌: {bid!r}")
        if bid_total:
            if trick_total >= bid_total:
                score += bid_total * 10 - (trick_total - bid_total) * 9
            else:
                score -= bid_total * 10
        scores.append(score)
    return scores[0], scores[1]


def _validate_game(
    game: dict[str, Any],
    *,
    seed: int,
    assignment: dict[str, str],
) -> tuple[float, float]:
    if game.get("seed") != seed:
        raise DuplicateReportError(
            f"seed 不一致：期望 {seed}，记录为 {game.get('seed')}"
        )
    if game.get("seat_assignment") != assignment:
        raise DuplicateReportError(f"seed {seed} 座位分配不正确")

    hands = game.get("initial_hands")
    if not isinstance(hands, dict) or set(hands) != {"0", "1", "2", "3"}:
        raise DuplicateReportError(f"seed {seed} 缺少四家初始手牌")
    cards = [card for seat in range(4) for card in hands[str(seat)]]
    if len(cards) != 52 or set(cards) != STANDARD_DECK:
        raise DuplicateReportError(f"seed {seed} 初始手牌不是标准 52 张牌")
    if len(game.get("plays", [])) != 52 or len(game.get("tricks", [])) != 13:
        raise DuplicateReportError(f"seed {seed} 不是完整 13 墩对局")

    ns_score, ew_score = _team_scores(game)
    expected_payoffs = [
        ns_score - ew_score,
        ew_score - ns_score,
        ns_score - ew_score,
        ew_score - ns_score,
    ]
    recorded_payoffs = game.get("scores")
    if (
        not isinstance(recorded_payoffs, list)
        or len(recorded_payoffs) != 4
        or any(
            abs(float(actual) - expected) > 1e-9
            for actual, expected in zip(recorded_payoffs, expected_payoffs)
        )
    ):
        raise DuplicateReportError(f"seed {seed} 计分与叫牌/墩数不一致")

    current_seat = next(
        int(seat) for seat, model in assignment.items() if model == CURRENT_AI
    )
    deepseek_seat = next(
        int(seat) for seat, model in assignment.items() if model == DEEPSEEK
    )
    if abs(float(game["current_ai_payoff"]) - expected_payoffs[current_seat]) > 1e-9:
        raise DuplicateReportError(f"seed {seed} 当前 AI payoff 归一化错误")
    if abs(float(game["deepseek_payoff"]) - expected_payoffs[deepseek_seat]) > 1e-9:
        raise DuplicateReportError(f"seed {seed} DeepSeek payoff 归一化错误")
    return ns_score, ew_score


def _portable_bid(bid: str) -> dict[str, Any]:
    if bid == "nil":
        return {"value": 0, "type": "nil"}
    if isinstance(bid, str) and bid.startswith("bid_"):
        return {"value": int(bid.removeprefix("bid_")), "type": "normal"}
    raise DuplicateReportError(f"GUI 复盘不支持叫牌 {bid!r}")


def _display_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _outcome(payoff: float) -> str:
    if payoff > 0:
        return f"当前 AI 胜 +{_display_number(payoff)}"
    if payoff < 0:
        return f"DeepSeek 胜 +{_display_number(-payoff)}"
    return "平局"


def _replay_record(
    game: dict[str, Any],
    *,
    table: str,
    ns_score: float,
    ew_score: float,
    duplicate_payoff: float,
) -> dict[str, Any]:
    seat_names = []
    compass = ("North", "East", "South", "West")
    for seat in range(4):
        model = game["seat_assignment"][str(seat)]
        name = "当前 AI" if model == CURRENT_AI else "DeepSeek"
        seat_names.append(f"{name} · {compass[seat]}")
    payoff = float(game["current_ai_payoff"])
    return {
        "format": "spades-ai-replay",
        "version": 1,
        "seed": game["seed"],
        "viewSeat": 0,
        "seats": seat_names,
        "bids": [_portable_bid(bid) for bid in game["bids"]],
        "initialHands": [game["initial_hands"][str(seat)] for seat in range(4)],
        "tricks": [
            {
                "trickNumber": index + 1,
                "leader": trick["leader"],
                "winner": trick["winner"],
                "plays": trick["cards"],
            }
            for index, trick in enumerate(game["tricks"])
        ],
        "tricksWon": game["tricks_won"],
        "score": {
            "northSouth": _display_number(ns_score),
            "eastWest": _display_number(ew_score),
        },
        "label": (
            f"副牌 {game['seed']} · {table} 桌 · {_outcome(payoff)}"
            f" · 双桌合计 {_display_number(duplicate_payoff):+}"
        ),
    }


def build_duplicate_report(
    *,
    table_a_dir: Path,
    table_b_dir: Path,
    seeds: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load, validate, pair, and score two tables for every seed."""

    boards: list[dict[str, Any]] = []
    replay_records: list[dict[str, Any]] = []
    source_pairs: list[dict[str, str]] = []
    current_table_wins = 0
    deepseek_table_wins = 0
    table_ties = 0
    total_usage: dict[str, int] = {}
    deepseek_api_attempts = 0
    valid_deepseek_responses = 0
    invalid_deepseek_responses = 0

    for seed in seeds:
        a_path = table_a_dir / f"seed_{seed}.json"
        b_path = table_b_dir / f"table_b_seed_{seed}.json"
        _, table_a = _load_single_game(a_path)
        _, table_b = _load_single_game(b_path)
        a_ns, a_ew = _validate_game(
            table_a, seed=seed, assignment=TABLE_A_ASSIGNMENT
        )
        b_ns, b_ew = _validate_game(
            table_b, seed=seed, assignment=TABLE_B_ASSIGNMENT
        )
        if table_a["initial_hands"] != table_b["initial_hands"]:
            raise DuplicateReportError(f"seed {seed} 两桌初始四手牌不同")

        a_payoff = float(table_a["current_ai_payoff"])
        b_payoff = float(table_b["current_ai_payoff"])
        duplicate_payoff = a_payoff + b_payoff
        winner = (
            CURRENT_AI
            if duplicate_payoff > 0
            else DEEPSEEK
            if duplicate_payoff < 0
            else "tie"
        )
        for payoff in (a_payoff, b_payoff):
            if payoff > 0:
                current_table_wins += 1
            elif payoff < 0:
                deepseek_table_wins += 1
            else:
                table_ties += 1
        for game in (table_a, table_b):
            for key, value in game.get("deepseek_usage", {}).items():
                total_usage[key] = total_usage.get(key, 0) + int(value)
            for attempt in game.get("deepseek_attempts", []):
                deepseek_api_attempts += 1
                if attempt.get("valid"):
                    valid_deepseek_responses += 1
                else:
                    invalid_deepseek_responses += 1
        boards.append(
            {
                "seed": seed,
                "winner": winner,
                "current_ai_duplicate_payoff": duplicate_payoff,
                "deepseek_duplicate_payoff": -duplicate_payoff,
                "table_a": {
                    "current_ai_seats": [0, 2],
                    "current_ai_payoff": a_payoff,
                    "deepseek_payoff": -a_payoff,
                    "north_south_score": a_ns,
                    "east_west_score": a_ew,
                },
                "table_b": {
                    "current_ai_seats": [1, 3],
                    "current_ai_payoff": b_payoff,
                    "deepseek_payoff": -b_payoff,
                    "north_south_score": b_ns,
                    "east_west_score": b_ew,
                },
            }
        )
        replay_records.extend(
            (
                _replay_record(
                    table_a,
                    table="A",
                    ns_score=a_ns,
                    ew_score=a_ew,
                    duplicate_payoff=duplicate_payoff,
                ),
                _replay_record(
                    table_b,
                    table="B",
                    ns_score=b_ns,
                    ew_score=b_ew,
                    duplicate_payoff=duplicate_payoff,
                ),
            )
        )
        source_pairs.append(
            {"seed": seed, "table_a": str(a_path.resolve()), "table_b": str(b_path.resolve())}
        )

    current_board_wins = sum(board["winner"] == CURRENT_AI for board in boards)
    deepseek_board_wins = sum(board["winner"] == DEEPSEEK for board in boards)
    board_ties = len(boards) - current_board_wins - deepseek_board_wins
    total_payoff = sum(
        float(board["current_ai_duplicate_payoff"]) for board in boards
    )
    created_at = datetime.now(timezone.utc).isoformat()
    report = {
        "format": "spades-deepseek-duplicate-match-summary",
        "version": 1,
        "created_at": created_at,
        "configuration": {
            "deals": len(seeds),
            "played_tables": len(seeds) * 2,
            "seeds": list(seeds),
            "table_a_current_ai_seats": [0, 2],
            "table_b_current_ai_seats": [1, 3],
            "scoring": "sum of current-AI-relative point differentials across both tables",
        },
        "standings": {
            "current_spades_ai_board_wins": current_board_wins,
            "deepseek_v4_flash_board_wins": deepseek_board_wins,
            "board_ties": board_ties,
            "current_spades_ai_duplicate_payoff": total_payoff,
            "deepseek_v4_flash_duplicate_payoff": -total_payoff,
            "current_spades_ai_table_wins": current_table_wins,
            "deepseek_v4_flash_table_wins": deepseek_table_wins,
            "table_ties": table_ties,
        },
        "deepseek_totals": {
            "api_attempts": deepseek_api_attempts,
            "valid_responses": valid_deepseek_responses,
            "invalid_responses": invalid_deepseek_responses,
            "usage": total_usage,
        },
        "validation": {
            "paired_deals": len(boards),
            "same_initial_hands_on_both_tables": True,
            "all_tables_have_52_plays_and_13_tricks": True,
            "all_scores_recomputed_from_bids_and_tricks": True,
            "all_payoffs_normalized_to_model_side": True,
        },
        "boards": boards,
        "source_pairs": source_pairs,
        "replay_records": replay_records,
    }
    bundle = {
        "format": "spades-ai-replay-bundle",
        "version": 1,
        "created_at": created_at,
        "records": replay_records,
    }
    return report, bundle


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将已完成的固定座位 A/B 桌记录合并为双桌队式赛"
    )
    parser.add_argument("--table-a-dir", type=Path, required=True)
    parser.add_argument("--table-b-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-bundle", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.games <= 0:
        raise DuplicateReportError("games 必须大于 0")
    seeds = list(range(args.seed, args.seed + args.games))
    report, bundle = build_duplicate_report(
        table_a_dir=args.table_a_dir,
        table_b_dir=args.table_b_dir,
        seeds=seeds,
    )
    summary_path = write_match_record(report, args.output, overwrite=args.overwrite)
    bundle_path = write_match_record(
        bundle, args.replay_bundle, overwrite=args.overwrite
    )
    print(f"双桌队式赛汇总已写入: {summary_path}")
    print(f"GUI 复盘包已写入: {bundle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
