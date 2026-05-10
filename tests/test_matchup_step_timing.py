"""Single-game timing benchmark for the matchup evaluation pipeline.

File purpose:
- Run exactly one full Spades game using the same seat-spec machinery as
  `evaluate/evaluate_model_matchups.py`.
- Print the elapsed time of every bid and every play step so the slowest
  action can be identified.
- Summarize the slowest step, the cumulative per-player time, and whether any
  single step exceeded a user-provided warning threshold.

Function input/output summary:
- build_timing_wrapped_players(...) -> list
    Input: players returned by the normal matchup builder and a shared recorder.
    Output: proxy players that time bid/play calls while preserving behavior.
- run_single_game_timing(...) -> dict[str, Any]
    Input: parsed CLI arguments.
    Output: a timing summary dictionary for one completed game.
- main() -> None
    Input: command-line arguments.
    Output: prints the per-step timing trace and a final summary.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.evaluate_model_matchups import build_players, build_runtime
from strategy.spades_match_runner import SpadesMatchRunner
from trick_taking.games.spades import SpadesRules


@dataclass
class TimingEvent:
    """A single timed bid or play step.

    Input fields:
    - phase: "bid" or "play".
    - player_id: seat index that executed the step.
    - elapsed_sec: wall-clock seconds spent in the step.
    - legal_count: number of legal actions presented to the player.
    - chosen: the action chosen by the player.

    Output:
    - Timing metadata stored for reporting.
    """

    phase: str
    player_id: int
    elapsed_sec: float
    legal_count: int
    chosen: Any
    extra: dict[str, Any] = field(default_factory=dict)


class TimingPlayerProxy:
    """Wrap a player and time its bid/play calls.

    Input:
    - inner: the actual player object from the matchup builder.
    - seat_index: the player's seat number.
    - label: the seat spec string used for the player.
    - recorder: list that receives `TimingEvent` entries.

    Output:
    - An object implementing the same player interface while recording timing.
    """

    def __init__(self, inner: Any, seat_index: int, label: str, recorder: list[TimingEvent]) -> None:
        self._inner = inner
        self._seat_index = seat_index
        self._label = label
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def start_game(self, position: int, hand: list, num_players: int) -> None:
        self._inner.start_game(position, hand, num_players)

    def place_bid(self, legal_bids: list[Any], state_view: dict) -> Any:
        t0 = time.perf_counter()
        bid = self._inner.place_bid(legal_bids, state_view)
        elapsed = time.perf_counter() - t0
        event = TimingEvent(
            phase="bid",
            player_id=self._seat_index,
            elapsed_sec=elapsed,
            legal_count=len(legal_bids),
            chosen=bid,
            extra={"label": self._label},
        )
        self._recorder.append(event)
        step_index = len(self._recorder)
        print(
            f"[{step_index:03d}] BID  seat={self._seat_index} spec={self._label:<18} "
            f"legal={len(legal_bids):2d} elapsed={elapsed:.6f}s bid={bid}"
        )
        return bid

    def play_card(self, legal_cards: list, state_view: dict) -> Any:
        t0 = time.perf_counter()
        card = self._inner.play_card(legal_cards, state_view)
        elapsed = time.perf_counter() - t0
        event = TimingEvent(
            phase="play",
            player_id=self._seat_index,
            elapsed_sec=elapsed,
            legal_count=len(legal_cards),
            chosen=card,
            extra={"label": self._label},
        )
        self._recorder.append(event)
        step_index = len(self._recorder)
        print(
            f"[{step_index:03d}] PLAY seat={self._seat_index} spec={self._label:<18} "
            f"legal={len(legal_cards):2d} elapsed={elapsed:.6f}s card={card}"
        )
        return card

    def bid_placed(self, bidder: int, bid: Any) -> None:
        self._inner.bid_placed(bidder, bid)

    def set_teams(self, teams: list[int], bid_values: list[Any]) -> None:
        self._inner.set_teams(teams, bid_values)

    def card_played(self, player_id: int, card: Any) -> None:
        self._inner.card_played(player_id, card)


def build_timing_wrapped_players(players: list[Any], seat_specs: list[str]) -> tuple[list[Any], list[TimingEvent]]:
    """Wrap players so each bid/play call is timed.

    Input:
    - players: four players created by the normal matchup builder.
    - seat_specs: list of four seat spec strings used for reporting.

    Output:
    - A tuple of (wrapped players, recorder list).
    """
    recorder: list[TimingEvent] = []
    wrapped = [TimingPlayerProxy(player, seat, seat_specs[seat], recorder) for seat, player in enumerate(players)]
    return wrapped, recorder


def _summarize_events(events: list[TimingEvent], slow_step_threshold_sec: float) -> dict[str, Any]:
    """Compute summary statistics from a timing trace.

    Input:
    - events: ordered list of `TimingEvent` objects.
    - slow_step_threshold_sec: threshold used to flag unusually slow steps.

    Output:
    - Dictionary with totals, per-player aggregates, and the slowest step.
    """
    total_elapsed = sum(event.elapsed_sec for event in events)
    per_player: dict[int, float] = {}
    per_phase: dict[str, float] = {"bid": 0.0, "play": 0.0}
    for event in events:
        per_player[event.player_id] = per_player.get(event.player_id, 0.0) + event.elapsed_sec
        per_phase[event.phase] = per_phase.get(event.phase, 0.0) + event.elapsed_sec

    slowest = max(events, key=lambda event: event.elapsed_sec, default=None)
    over_threshold = [event for event in events if event.elapsed_sec >= slow_step_threshold_sec]
    return {
        "total_elapsed_sec": total_elapsed,
        "per_player_elapsed_sec": per_player,
        "per_phase_elapsed_sec": per_phase,
        "slowest_event": slowest,
        "slow_count": len(over_threshold),
        "over_threshold_events": over_threshold,
        "event_count": len(events),
    }


def run_single_game_timing(args: argparse.Namespace) -> dict[str, Any]:
    """Run one matchup game and collect a full timing trace.

    Input:
    - args: parsed command-line arguments matching the matchup evaluation CLI.

    Output:
    - A dictionary containing the game result, timing events, and summary stats.
    """
    runtime = build_runtime(args)
    seat_specs = [args.p0, args.p1, args.p2, args.p3]
    players = build_players(args, runtime, args.seed)
    wrapped_players, recorder = build_timing_wrapped_players(players, seat_specs)

    rules = SpadesRules(enable_nil=not args.disable_nil, enable_blind_nil=not args.disable_blind_nil)
    runner = SpadesMatchRunner(players=wrapped_players, seed=args.seed, verbose=False, rules=rules)

    t0 = time.perf_counter()
    result = runner.play_game()
    total_wall = time.perf_counter() - t0

    # Collect per-player MCTS diagnostics when available
    per_player_diagnostics: dict[int, dict[str, int]] = {}
    for seat_index, player in enumerate(wrapped_players):
        inner = getattr(player, "_inner", None)
        diag = {"model_calls": 0, "policy_model_calls": 0, "exact_calls": 0}
        if inner is not None:
            strategy = getattr(inner, "strategy", None)
            if strategy is not None and hasattr(strategy, "get_diagnostics"):
                diag = strategy.get_diagnostics()
        per_player_diagnostics[seat_index] = diag

    summary = _summarize_events(recorder, args.slow_step_threshold_sec)
    summary.update(
        {
            "seed": args.seed,
            "seat_specs": seat_specs,
            "scores": [float(score) for score in result.scores],
            "winner": int(result.winner),
            "game_wall_sec": total_wall,
            "rules": {
                "nil_enabled": not args.disable_nil,
                "blind_nil_enabled": not args.disable_blind_nil,
            },
            "per_player_diagnostics": per_player_diagnostics,
        }
    )
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable summary of the one-game timing run.

    Input:
    - summary: dictionary returned by `run_single_game_timing`.

    Output:
    - None.
    """
    print()
    print("=== Single-game timing summary ===")
    print(f"Seed: {summary['seed']}")
    print(f"Wall time: {summary['game_wall_sec']:.3f}s")
    print(f"Event count: {summary['event_count']}")
    print(f"Scores: {summary['scores']}")
    print(f"Winner: player {summary['winner']}")
    print(f"Phase time: bid={summary['per_phase_elapsed_sec'].get('bid', 0.0):.3f}s play={summary['per_phase_elapsed_sec'].get('play', 0.0):.3f}s")

    print("Per-player time:")
    for seat in sorted(summary["per_player_elapsed_sec"]):
        print(f"  seat {seat}: {summary['per_player_elapsed_sec'][seat]:.3f}s ({summary['seat_specs'][seat]})")

    slowest = summary["slowest_event"]
    if slowest is not None:
        print(
            "Slowest step: "
            f"phase={slowest.phase} seat={slowest.player_id} label={slowest.extra.get('label')} "
            f"elapsed={slowest.elapsed_sec:.3f}s chosen={slowest.chosen}"
        )

    if summary["slow_count"] > 0:
        print(f"Steps over threshold: {summary['slow_count']}")
        for event in summary["over_threshold_events"]:
            print(
                f"  phase={event.phase} seat={event.player_id} label={event.extra.get('label')} "
                f"elapsed={event.elapsed_sec:.3f}s chosen={event.chosen}"
            )
    else:
        print("No step exceeded the slow-step threshold.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the single-game timing benchmark.

    Input:
    - `sys.argv`.

    Output:
    - Parsed matchup configuration with an additional slow-step threshold.
    """
    parser = argparse.ArgumentParser(
        description="Time every step of one matchup game.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--disable-nil", action="store_true", help="Disable nil bidding in the local rules")
    parser.add_argument(
        "--disable-blind-nil",
        action="store_true",
        help="Disable blind nil bidding in the local rules",
    )
    parser.add_argument("--p0", type=str, default="our_mcts", help="Seat 0 model spec")
    parser.add_argument("--p1", type=str, default="go_rule", help="Seat 1 model spec")
    parser.add_argument("--p2", type=str, default="our_mcts", help="Seat 2 model spec")
    parser.add_argument("--p3", type=str, default="go_rule", help="Seat 3 model spec")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for loaded models")
    parser.add_argument("--our-checkpoint", type=str, default="", help="Optional local MLP checkpoint")
    parser.add_argument("--our-exact-threshold", type=int, default=30, help="Exact solve threshold for our MCTS")
    parser.add_argument("--our-leaf-threshold", type=int, default=24, help="Leaf threshold for our MCTS")
    parser.add_argument(
        "--our-simulations-per-action",
        type=int,
        default=50,
        help="Root simulations per legal action for our MCTS",
    )
    parser.add_argument(
        "--our-exploration-constant",
        type=float,
        default=1.5,
        help="PUCT exploration constant for our MCTS",
    )
    parser.add_argument(
        "--our-policy-temperature",
        type=float,
        default=1.0,
        help="Policy temperature for our MCTS leaf prior",
    )
    parser.add_argument(
        "--our-value-scale",
        type=float,
        default=25.0,
        help="Value scaling factor for our MCTS leaf value",
    )
    parser.add_argument("--go-pv-checkpoint", type=str, default="", help="合作者仓库 GPT-2 策略/价值模型的 checkpoint 路径")
    parser.add_argument("--go-bid-checkpoint", type=str, default="", help="合作者仓库叫牌 MLP 模型的 checkpoint 路径")
    parser.add_argument("--go-mcts-runs", type=int, default=100, help="合作者 GOMCTS 每次决策的模拟/rollout 次数")
    parser.add_argument("--go-mcts-steps", type=int, default=5, help="合作者 GOMCTS 每次 rollout 的最大步数/深度")
    parser.add_argument("--go-mcts-c", type=float, default=0.3, help="合作者 GOMCTS 的探索常数，越大越偏向探索")
    parser.add_argument("--go-mcts-mu", type=float, default=0.01, help="合作者 GOMCTS 的非法牌惩罚系数")
    parser.add_argument(
        "--go-mcts-threshold",
        type=float,
        default=0.05,
        help="合作者 GOMCTS 的剪枝阈值，越大越激进地剪枝",
    )
    parser.add_argument(
        "--go-argmax-threshold",
        type=float,
        default=0.05,
        help="合作者 ArgmaxPlayer 的阈值，影响 argmax/采样边界",
    )
    parser.add_argument(
        "--slow-step-threshold-sec",
        type=float,
        default=120.0,
        help="超过该秒数的步骤会在总结里单独标记",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for the single-game timing benchmark.

    Input:
    - `sys.argv`.

    Output:
    - Prints the per-step timing trace and a final summary.
    """
    args = parse_args()
    summary = run_single_game_timing(args)
    _print_summary(summary)


if __name__ == "__main__":
    main()
