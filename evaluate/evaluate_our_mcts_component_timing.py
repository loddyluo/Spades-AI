"""Per-card timing breakdown for our_mcts.

This program runs one matchup game and prints, for each our_mcts card choice,
how much time was spent in:
- hidden-hand sampling / determinization helpers
- the MLP value model or exact solver

Any remaining wall time is reported as "other" overhead, which includes tree
bookkeeping, copying, and policy/PUCT logic.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
GO_MCTS_DIR = REPO_ROOT / "evaluate" / "GO-MCTS"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GO_MCTS_DIR) not in sys.path:
    sys.path.insert(0, str(GO_MCTS_DIR))

from evaluate.evaluate_our_mcts_vs_rule_v2 import build_players, build_runtime
from strategy.spades_match_runner import SpadesMatchRunner
from trick_taking.games.spades import SpadesRules

from adapters import OurHandStrengthMCTSPlayer


@dataclass
class DecisionTiming:
    seat: int
    remaining_cards: int
    mode: str = ""
    chosen_card: str = ""
    total_sec: float = 0.0
    sample_sec: float = 0.0
    solve_sec: float = 0.0
    sample_calls: int = 0
    solve_calls: int = 0
    model_calls: int = 0
    note: str = ""
    deepcopy_sec: float = 0.0
    apply_sec: float = 0.0
    select_sec: float = 0.0

    @property
    def other_sec(self) -> float:
        return max(self.total_sec - self.sample_sec - self.solve_sec - self.deepcopy_sec - self.apply_sec - self.select_sec, 0.0)


@dataclass
class StrategyTimingProbe:
    strategy: Any
    active: DecisionTiming | None = None
    records: list[DecisionTiming] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._patch_method("_build_is_pool", "sample")
        self._patch_method("_draw_is_sample", "sample")
        self._patch_method("_apply_proposal", "sample")
        self._patch_method("_determinize_state", "sample")
        self._patch_model_predict()
        self._patch_exact_solver()
        # additional probes: measure apply_action and select cost, and deepcopy
        try:
            self._patch_method("_apply_action", "apply")
        except Exception:
            pass
        try:
            self._patch_method("_select_child_puct", "select")
        except Exception:
            pass
        # patch copy.deepcopy globally to measure copy time
        try:
            import copy as _copy

            _orig_deepcopy = _copy.deepcopy

            def _wrapped_deepcopy(obj, memo=None):
                start = time.perf_counter()
                try:
                    return _orig_deepcopy(obj, memo) if memo is not None else _orig_deepcopy(obj)
                finally:
                    if self.active is not None:
                        self.active.deepcopy_sec += time.perf_counter() - start

            _copy.deepcopy = _wrapped_deepcopy
        except Exception:
            pass

    def begin(self, seat: int, remaining_cards: int) -> None:
        self.active = DecisionTiming(seat=seat, remaining_cards=remaining_cards)

    def finish(self, mode: str, chosen_card: Any) -> DecisionTiming:
        if self.active is None:
            raise RuntimeError("TimingProbe.finish called without an active decision")
        self.active.mode = mode
        self.active.chosen_card = str(chosen_card)
        self.records.append(self.active)
        finished = self.active
        self.active = None
        return finished

    def _patch_method(self, method_name: str, bucket: str) -> None:
        original = getattr(self.strategy, method_name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                if self.active is not None:
                    if bucket == "sample":
                        self.active.sample_sec += elapsed
                        self.active.sample_calls += 1
                    elif bucket == "solve":
                        self.active.solve_sec += elapsed
                        self.active.solve_calls += 1

        setattr(self.strategy, method_name, wrapped)

    def _patch_model_predict(self) -> None:
        model = getattr(self.strategy, "model", None)
        if model is None or not hasattr(model, "predict"):
            return

        original_predict = model.predict

        def wrapped_predict(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return original_predict(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                if self.active is not None:
                    self.active.solve_sec += elapsed
                    self.active.model_calls += 1

        model.predict = wrapped_predict

    def _patch_exact_solver(self) -> None:
        exact_solver = getattr(self.strategy, "exact_solver", None)
        if exact_solver is None or not hasattr(exact_solver, "solve_with_q"):
            return

        original_solve_with_q = exact_solver.solve_with_q

        def wrapped_solve_with_q(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return original_solve_with_q(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                if self.active is not None:
                    self.active.solve_sec += elapsed
                    self.active.solve_calls += 1

        exact_solver.solve_with_q = wrapped_solve_with_q


def _instrument_our_mcts_players(players: list[Any]) -> list[tuple[int, Any, StrategyTimingProbe]]:
    instrumented: list[tuple[int, Any, StrategyTimingProbe]] = []
    for seat_index, player in enumerate(players):
        if not isinstance(player, OurHandStrengthMCTSPlayer):
            continue
        probe = StrategyTimingProbe(player.strategy)
        original_play_card = player.play_card

        def wrapped_play_card(legal_cards: list[Any], state_view: dict[str, Any], *, _seat=seat_index, _player=player, _probe=probe, _original=original_play_card) -> Any:
            state = state_view.get("state")
            remaining_cards = sum(len(hand) for hand in state.hands) if state is not None else len(legal_cards)
            _probe.begin(_seat, remaining_cards)
            t0 = time.perf_counter()
            chosen = _original(legal_cards, state_view)
            elapsed = time.perf_counter() - t0
            info = getattr(_player, "last_play_info", None) or {}
            record = _probe.finish(str(info.get("mode", "unknown")), chosen)
            record.total_sec = elapsed
            print(
                f"[seat {_seat}] rem={record.remaining_cards:2d} mode={record.mode:<5} "
                f"card={record.chosen_card:<18} total={record.total_sec:.4f}s "
                f"sample={record.sample_sec:.4f}s solve={record.solve_sec:.4f}s "
                f"deepcopy={record.deepcopy_sec:.4f}s apply={record.apply_sec:.4f}s "
                f"select={record.select_sec:.4f}s other={record.other_sec:.4f}s "
                f"sample_calls={record.sample_calls} solve_calls={record.solve_calls} model_calls={record.model_calls}"
            )
            return chosen

        player.play_card = wrapped_play_card
        instrumented.append((seat_index, player, probe))
    return instrumented


def _print_summary(probes: list[tuple[int, Any, StrategyTimingProbe]]) -> None:
    print()
    print("=== our_mcts timing summary ===")
    for seat_index, player, probe in probes:
        if not probe.records:
            print(f"seat {seat_index}: no our_mcts card decisions recorded")
            continue
        total = sum(item.total_sec for item in probe.records)
        sample = sum(item.sample_sec for item in probe.records)
        solve = sum(item.solve_sec for item in probe.records)
        other = sum(item.other_sec for item in probe.records)
        count = len(probe.records)
        print(
            f"seat {seat_index}: decisions={count} total={total:.4f}s "
            f"avg_total={total / count:.4f}s avg_sample={sample / count:.4f}s "
            f"avg_solve={solve / count:.4f}s avg_other={other / count:.4f}s"
        )


def main() -> None:
    """Run a single matchup game and print per-card our_mcts timing."""
    parser = argparse.ArgumentParser(
        description="Print per-card our_mcts timing breakdown for one matchup game.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--p0", type=str, default="our_mcts", help="Seat 0 model spec")
    parser.add_argument("--p1", type=str, default="go_rule_2", help="Seat 1 model spec")
    parser.add_argument("--p2", type=str, default="our_mcts", help="Seat 2 model spec")
    parser.add_argument("--p3", type=str, default="go_rule_2", help="Seat 3 model spec")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for loaded models")
    parser.add_argument("--our-checkpoint", type=str, default="", help="Optional local MLP checkpoint")
    parser.add_argument("--our-exact-threshold", type=int, default=24, help="Exact solve threshold for our MCTS")
    parser.add_argument("--our-leaf-threshold", type=int, default=24, help="Leaf threshold for our MCTS")
    parser.add_argument(
        "--our-simulations-per-action",
        type=int,
        default=200,
        help="Total MCTS samples per legal action",
    )
    parser.add_argument(
        "--our-number-of-exact-solvers",
        type=int,
        default=100,
        help="Number of determinized exact solves per exact-decision step",
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
        "--our-mcts-determinization-count",
        type=int,
        default=10,
        help="Number of importance-sampled determinizations to draw for each MCTS decision",
    )
    parser.add_argument(
        "--our-value-scale",
        type=float,
        default=25.0,
        help="Value scaling factor for our MCTS leaf value",
    )
    parser.add_argument("--go-pv-checkpoint", type=str, default="", help="Collaborator GPT-2 checkpoint path")
    parser.add_argument("--go-bid-checkpoint", type=str, default="", help="Collaborator bid MLP checkpoint path")
    parser.add_argument("--go-mcts-runs", type=int, default=100, help="Collaborator GOMCTS rollout count")
    parser.add_argument("--go-mcts-steps", type=int, default=5, help="Collaborator GOMCTS rollout depth")
    parser.add_argument("--go-mcts-c", type=float, default=0.3, help="Collaborator GOMCTS exploration constant")
    parser.add_argument("--go-mcts-mu", type=float, default=0.01, help="Collaborator GOMCTS illegal-card penalty")
    parser.add_argument(
        "--go-mcts-threshold",
        type=float,
        default=0.05,
        help="Collaborator GOMCTS pruning threshold",
    )
    parser.add_argument(
        "--go-argmax-threshold",
        type=float,
        default=0.05,
        help="Collaborator ArgmaxPlayer threshold",
    )
    parser.add_argument(
        "--disable-nil",
        action="store_true",
        help="Disable nil bidding in the local rules",
    )
    parser.add_argument(
        "--disable-pbar",
        action="store_true",
        help="Disable tqdm progress bar updates (reduces 'other' overhead for benchmarking)",
    )
    args = parser.parse_args()
    args = parser.parse_args()

    # Optionally disable tqdm used by the MCTS implementation to reduce update overhead
    if getattr(args, "disable_pbar", False):
        try:
            import strategy.truncated_mcts_strategy as _tms

            class _DummyPbar:
                def __init__(self, *a, **k):
                    pass

                def update(self, n=1):
                    return None

                def close(self):
                    return None

            _tms.tqdm = lambda *a, **k: _DummyPbar()
        except Exception:
            pass

    runtime = build_runtime(args)
    players = build_players(args, runtime, args.seed)
    probes = _instrument_our_mcts_players(players)

    rules = SpadesRules(enable_nil=not args.disable_nil, enable_blind_nil=False)
    runner = SpadesMatchRunner(players=players, seed=args.seed, verbose=False, rules=rules)

    print("Seat specs:", [args.p0, args.p1, args.p2, args.p3])
    print("Running one game; printing only our_mcts card decisions.")
    result = runner.play_game()
    print(f"Game finished. winner={result.winner} scores={[float(score) for score in result.scores]}")
    _print_summary(probes)


if __name__ == "__main__":
    main()