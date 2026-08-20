"""Run resumable solver-leaf PPO league cycles until a practical plateau."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "output" / "solver-leaf-convergence-phase3-20260811"
DEFAULT_SOURCE_DIR = REPO_ROOT / "output" / "solver-leaf-selfplay-converge-20260811"
DEFAULT_SOURCE_ACTOR = DEFAULT_SOURCE_DIR / "actor_update_000035.pt"
DEFAULT_SOURCE_TRAINER = DEFAULT_SOURCE_DIR / "trainer_update_000035.pt"
DEFAULT_U75_ACTOR = (
    REPO_ROOT
    / "output"
    / "solver-leaf-ppo-400k-20260810"
    / "actor_update_000075.pt"
)


@dataclass(frozen=True, slots=True)
class Checkpoint:
    actor: Path
    trainer: Path
    cycle: int
    update: int

    def json(self) -> dict[str, Any]:
        return {
            "actor": str(self.actor),
            "trainer": str(self.trainer),
            "cycle": self.cycle,
            "update": self.update,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable league PPO with statistical champion promotion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-actor", default=str(DEFAULT_SOURCE_ACTOR))
    parser.add_argument("--source-trainer", default=str(DEFAULT_SOURCE_TRAINER))
    parser.add_argument("--u75-actor", default=str(DEFAULT_U75_ACTOR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--holdout-rule-json",
        default=str(DEFAULT_OUTPUT / "holdout-rule-10000.json"),
    )
    parser.add_argument(
        "--holdout-vs-u75-json",
        default=str(DEFAULT_OUTPUT / "holdout-vs-u75-5000.json"),
    )
    parser.add_argument("--history-checkpoints", nargs="*", default=[])
    parser.add_argument("--max-cycles", type=int, default=6)
    parser.add_argument("--max-hours", type=float, default=14.0)
    parser.add_argument("--cycle-games", type=int, default=100_000)
    parser.add_argument("--rollout-deals", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-every-updates", type=int, default=5)
    parser.add_argument("--screen-deals", type=int, default=1000)
    parser.add_argument("--formal-rule-deals", type=int, default=5000)
    parser.add_argument("--formal-actor-deals", type=int, default=5000)
    parser.add_argument("--equivalence-deals", type=int, default=10_000)
    parser.add_argument("--formal-candidates", type=int, default=2)
    parser.add_argument("--plateau-cycles", type=int, default=3)
    parser.add_argument("--equivalence-margin", type=float, default=0.5)
    parser.add_argument("--rule-noninferiority-margin", type=float, default=0.5)
    parser.add_argument("--rule-opponent-weight", type=float, default=0.45)
    parser.add_argument("--champion-opponent-weight", type=float, default=0.30)
    parser.add_argument("--history-opponent-weight", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--learning-rate-decay", type=float, default=0.80)
    parser.add_argument("--entropy-start", type=float, default=0.003)
    parser.add_argument("--entropy-final", type=float, default=0.0015)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1_353_142)
    parser.add_argument("--base-shuffle-seed", type=int, default=1_053_600_000)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    positive_ints = (
        "max_cycles",
        "cycle_games",
        "rollout_deals",
        "workers",
        "save_every_updates",
        "screen_deals",
        "formal_rule_deals",
        "formal_actor_deals",
        "equivalence_deals",
        "formal_candidates",
        "plateau_cycles",
    )
    for name in positive_ints:
        if type(getattr(args, name)) is not int or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.cycle_games % 2:
        raise ValueError("--cycle-games must be even")
    finite_positive = (
        "max_hours",
        "equivalence_margin",
        "rule_noninferiority_margin",
        "learning_rate",
        "minimum_learning_rate",
        "learning_rate_decay",
        "target_kl",
    )
    for name in finite_positive:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("entropy_start", "entropy_final"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    weights = (
        args.rule_opponent_weight,
        args.champion_opponent_weight,
        args.history_opponent_weight,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("opponent weights must be finite and nonnegative")
    if sum(weights) <= 0.0:
        raise ValueError("at least one opponent weight must be positive")
    for name in ("seed", "base_shuffle_seed"):
        if type(getattr(args, name)) is not int or getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_path(actor: Path) -> Path:
    return actor.with_name(f"{actor.name}.json")


def _validate_actor(actor: Path) -> dict[str, Any]:
    metadata_path = _metadata_path(actor)
    if not actor.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"actor and sidecar are required: {actor}")
    metadata = _load_json(metadata_path)
    if metadata.get("encoder_schema") != "first4-observation-v2-536":
        raise ValueError(f"unexpected encoder schema for {actor}")
    if metadata.get("input_dim") != 536 or metadata.get("output_dim") != 52:
        raise ValueError(f"unexpected actor dimensions for {actor}")
    if metadata.get("actor_sha256") != _sha256(actor):
        raise ValueError(f"actor SHA-256 mismatch: {actor}")
    return metadata


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _unique_actor_paths(paths: Sequence[Path], *, exclude: Path) -> list[Path]:
    unique: list[Path] = []
    seen = {exclude.resolve()}
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        _validate_actor(resolved)
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _ci(report: dict[str, Any]) -> tuple[float, float]:
    values = report.get("confidence_interval_95_points")
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError("evaluation report is missing its 95% confidence interval")
    return float(values[0]), float(values[1])


def _mean(report: dict[str, Any]) -> float:
    return float(report["mean_duplicate_margin_points"])


def _rule_noninferior(
    candidate: dict[str, Any], incumbent: dict[str, Any], margin: float
) -> bool:
    return _ci(candidate)[0] > 0.0 and _mean(candidate) >= _mean(incumbent) - margin


def _actor_superior(report: dict[str, Any]) -> bool:
    return _ci(report)[0] > 0.0


def _actor_equivalent(report: dict[str, Any], margin: float) -> bool:
    lower, upper = _ci(report)
    return lower >= -margin and upper <= margin


def _checkpoint_from_json(payload: dict[str, Any]) -> Checkpoint:
    return Checkpoint(
        actor=_resolved(payload["actor"]),
        trainer=_resolved(payload["trainer"]),
        cycle=int(payload.get("cycle", 0)),
        update=int(payload["update"]),
    )


def _complete_checkpoints(directory: Path, cycle: int) -> list[Checkpoint]:
    found: list[Checkpoint] = []
    pattern = re.compile(r"actor_update_(\d{6})\.pt$")
    if directory.is_dir():
        for actor in directory.glob("actor_update_*.pt"):
            match = pattern.fullmatch(actor.name)
            if match is None or not _metadata_path(actor).is_file():
                continue
            update = int(match.group(1))
            trainer = directory / f"trainer_update_{update:06d}.pt"
            if trainer.is_file():
                found.append(Checkpoint(actor.resolve(), trainer.resolve(), cycle, update))
        final_actor = directory / "actor_final.pt"
        final_trainer = directory / "trainer_final.pt"
        if final_actor.is_file() and _metadata_path(final_actor).is_file() and final_trainer.is_file():
            metadata = _load_json(_metadata_path(final_actor))
            found.append(
                Checkpoint(
                    final_actor.resolve(),
                    final_trainer.resolve(),
                    cycle,
                    int(metadata["training_update"]),
                )
            )
    unique = {item.update: item for item in found}
    return sorted(unique.values(), key=lambda item: (item.update, str(item.actor)))


def _latest_partial_trainer(directory: Path) -> Path | None:
    pattern = re.compile(r"trainer_update_(\d{6})\.pt$")
    candidates: list[tuple[int, Path]] = []
    if directory.is_dir():
        for path in directory.glob("trainer_update_*.pt"):
            match = pattern.fullmatch(path.name)
            if match is not None:
                candidates.append((int(match.group(1)), path.resolve()))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


class ConvergenceRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_dir = _resolved(args.output_dir)
        self.state_path = self.output_dir / "state.json"
        self.report_json = self.output_dir / "convergence-report.json"
        self.report_markdown = self.output_dir / "convergence-report.md"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.state_path.is_file():
            self.state = _load_json(self.state_path)
        else:
            started = datetime.now(timezone.utc)
            self.state = {
                "schema": "solver-leaf-convergence-v1",
                "status": "initializing",
                "started_at": started.isoformat(),
                "deadline": (started + timedelta(hours=args.max_hours)).isoformat(),
                "events": [],
                "cycles": [],
                "plateau_streak": 0,
            }
            self._save_state()

    def _save_state(self) -> None:
        _atomic_json(self.state_path, self.state)

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        self.state["status"] = event
        events = [*self.state.get("events", []), record]
        self.state["events"] = events[-200:]
        self._save_state()
        print(json.dumps({"convergence": record}, ensure_ascii=False), flush=True)

    def seconds_remaining(self) -> float:
        deadline = datetime.fromisoformat(self.state["deadline"])
        return (deadline - datetime.now(timezone.utc)).total_seconds()

    def run_command(self, name: str, command: Sequence[str], log_path: Path) -> None:
        if self.seconds_remaining() <= 0.0:
            raise TimeoutError("convergence wall-time budget exhausted")
        self.emit("command_started", name=name, command=list(command), log=str(log_path))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                list(command),
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            try:
                while process.poll() is None:
                    if self.seconds_remaining() <= 0.0:
                        self._terminate(process)
                        raise TimeoutError(f"wall-time budget exhausted during {name}")
                    time.sleep(2)
            except BaseException:
                self._terminate(process)
                raise
        if process.returncode != 0:
            raise RuntimeError(f"{name} failed with exit code {process.returncode}")
        self.emit("command_finished", name=name)

    @staticmethod
    def _terminate(process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)

    def eval_rule(
        self,
        checkpoint: Checkpoint,
        *,
        deals: int,
        seed: int,
        base_seed: int,
        destination: Path,
        label: str,
    ) -> dict[str, Any]:
        if destination.is_file():
            return _load_json(destination)
        command = [
            sys.executable,
            "-m",
            "evaluate.evaluate_solver_leaf_ppo_vs_rule",
            "--checkpoint",
            str(checkpoint.actor),
            "--deals",
            str(deals),
            "--workers",
            str(self.args.workers),
            "--seed",
            str(seed),
            "--base-shuffle-seed",
            str(base_seed),
            "--output-json",
            str(destination),
        ]
        self.run_command(f"rule-{label}", command, destination.with_suffix(".log"))
        return _load_json(destination)

    def eval_actor(
        self,
        checkpoint: Checkpoint,
        opponent: Checkpoint,
        *,
        deals: int,
        seed: int,
        base_seed: int,
        destination: Path,
        label: str,
    ) -> dict[str, Any]:
        if destination.is_file():
            return _load_json(destination)
        command = [
            sys.executable,
            "-m",
            "evaluate.evaluate_solver_leaf_ppo_vs_actor",
            "--checkpoint",
            str(checkpoint.actor),
            "--opponent-checkpoint",
            str(opponent.actor),
            "--deals",
            str(deals),
            "--workers",
            str(self.args.workers),
            "--seed",
            str(seed),
            "--base-shuffle-seed",
            str(base_seed),
            "--output-json",
            str(destination),
        ]
        self.run_command(f"actor-{label}", command, destination.with_suffix(".log"))
        return _load_json(destination)

    def train_cycle(
        self,
        cycle: int,
        incumbent: Checkpoint,
        history: Sequence[Path],
    ) -> tuple[list[Checkpoint], dict[str, Any]]:
        cycle_dir = self.output_dir / f"cycle-{cycle:03d}"
        checkpoints = _complete_checkpoints(cycle_dir, cycle)
        if any(item.actor.name == "actor_final.pt" for item in checkpoints):
            return checkpoints, {"reused": True}
        learning_rate = max(
            self.args.minimum_learning_rate,
            self.args.learning_rate * self.args.learning_rate_decay ** (cycle - 1),
        )
        if self.args.max_cycles == 1:
            entropy = self.args.entropy_final
        else:
            progress = (cycle - 1) / (self.args.max_cycles - 1)
            entropy = self.args.entropy_start + progress * (
                self.args.entropy_final - self.args.entropy_start
            )
        run_seed = self.args.seed + cycle * 10_000
        base_seed = self.args.base_shuffle_seed + cycle * 10_000_000
        resolved_history = _unique_actor_paths(history, exclude=incumbent.actor)
        config = {
            "cycle": cycle,
            "source": incumbent.json(),
            "history": [str(path) for path in resolved_history],
            "learning_rate": learning_rate,
            "entropy_coefficient": entropy,
            "seed": run_seed,
            "base_shuffle_seed": base_seed,
        }
        command = [
            sys.executable,
            "-m",
            "rl.train_solver_leaf_ppo_multicpu",
            "--total-games",
            str(self.args.cycle_games),
            "--rollout-deals",
            str(self.args.rollout_deals),
            "--workers",
            str(self.args.workers),
            "--seed",
            str(run_seed),
            "--base-shuffle-seed",
            str(base_seed),
            "--learning-rate",
            str(learning_rate),
            "--entropy-coefficient",
            str(entropy),
            "--update-epochs",
            "4",
            "--minibatch-size",
            "1024",
            "--clip-ratio",
            "0.2",
            "--value-coefficient",
            "0.5",
            "--max-grad-norm",
            "0.5",
            "--target-kl",
            str(self.args.target_kl),
            "--save-every-updates",
            str(self.args.save_every_updates),
            "--save-dir",
            str(cycle_dir),
            "--rule-opponent-weight",
            str(self.args.rule_opponent_weight),
            "--champion-opponent-weight",
            str(self.args.champion_opponent_weight),
            "--history-opponent-weight",
            str(self.args.history_opponent_weight),
            "--champion-checkpoint",
            str(incumbent.actor),
            "--history-checkpoints",
            *[str(path) for path in resolved_history],
        ]
        partial = _latest_partial_trainer(cycle_dir)
        if partial is None:
            command.extend(("--finetune-from", str(incumbent.trainer)))
        else:
            command.extend(("--resume", str(partial)))
            config["resume"] = str(partial)
        self.run_command(
            f"cycle-{cycle:03d}-training",
            command,
            self.output_dir / f"cycle-{cycle:03d}-training.jsonl",
        )
        checkpoints = _complete_checkpoints(cycle_dir, cycle)
        if not checkpoints or not any(
            item.actor.name == "actor_final.pt" for item in checkpoints
        ):
            raise RuntimeError(f"cycle {cycle} did not produce final artifacts")
        return checkpoints, config

    def screen(
        self,
        cycle: int,
        candidates: Sequence[Checkpoint],
        incumbent: Checkpoint,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        eval_dir = self.output_dir / f"cycle-{cycle:03d}" / "eval"
        for candidate in candidates:
            tag = f"u{candidate.update:06d}"
            rule = self.eval_rule(
                candidate,
                deals=self.args.screen_deals,
                seed=1_036_500 + cycle,
                base_seed=553_600_000 + cycle * 10_000_000,
                destination=eval_dir / f"screen-{tag}-rule.json",
                label=f"cycle-{cycle:03d}-{tag}-screen",
            )
            versus = self.eval_actor(
                candidate,
                incumbent,
                deals=self.args.screen_deals,
                seed=1_036_700 + cycle,
                base_seed=653_600_000 + cycle * 10_000_000,
                destination=eval_dir / f"screen-{tag}-vs-incumbent.json",
                label=f"cycle-{cycle:03d}-{tag}-screen-vs-incumbent",
            )
            result = {
                "checkpoint": candidate.json(),
                "rule_mean": _mean(rule),
                "rule_ci95": list(_ci(rule)),
                "versus_incumbent_mean": _mean(versus),
                "versus_incumbent_ci95": list(_ci(versus)),
            }
            result["ranking_score"] = (
                result["versus_incumbent_mean"] + 0.25 * result["rule_mean"]
            )
            results.append(result)
            self.emit("screen_result", cycle=cycle, **result)
        return sorted(results, key=lambda item: item["ranking_score"], reverse=True)

    def formal(
        self,
        cycle: int,
        screen_results: Sequence[dict[str, Any]],
        incumbent: Checkpoint,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        eval_dir = self.output_dir / f"cycle-{cycle:03d}" / "eval"
        rule_seed = 2_036_500 + cycle
        rule_base = 753_600_000 + cycle * 10_000_000
        actor_seed = 2_036_700 + cycle
        actor_base = 853_600_000 + cycle * 10_000_000
        incumbent_rule = self.eval_rule(
            incumbent,
            deals=self.args.formal_rule_deals,
            seed=rule_seed,
            base_seed=rule_base,
            destination=eval_dir / "formal-incumbent-rule.json",
            label=f"cycle-{cycle:03d}-incumbent-formal",
        )
        results: list[dict[str, Any]] = []
        for screened in screen_results[: self.args.formal_candidates]:
            candidate = _checkpoint_from_json(screened["checkpoint"])
            tag = f"u{candidate.update:06d}"
            rule = self.eval_rule(
                candidate,
                deals=self.args.formal_rule_deals,
                seed=rule_seed,
                base_seed=rule_base,
                destination=eval_dir / f"formal-{tag}-rule.json",
                label=f"cycle-{cycle:03d}-{tag}-formal",
            )
            versus = self.eval_actor(
                candidate,
                incumbent,
                deals=self.args.formal_actor_deals,
                seed=actor_seed,
                base_seed=actor_base,
                destination=eval_dir / f"formal-{tag}-vs-incumbent.json",
                label=f"cycle-{cycle:03d}-{tag}-formal-vs-incumbent",
            )
            result = {
                "checkpoint": candidate.json(),
                "rule": rule,
                "versus_incumbent": versus,
                "rule_noninferior": _rule_noninferior(
                    rule, incumbent_rule, self.args.rule_noninferiority_margin
                ),
                "actor_superior": _actor_superior(versus),
            }
            result["promotion_passed"] = bool(
                result["rule_noninferior"] and result["actor_superior"]
            )
            result["ranking_score"] = _mean(versus) + 0.25 * _mean(rule)
            results.append(result)
            self.emit(
                "formal_result",
                cycle=cycle,
                checkpoint=candidate.json(),
                rule_mean=_mean(rule),
                rule_ci95=list(_ci(rule)),
                versus_incumbent_mean=_mean(versus),
                versus_incumbent_ci95=list(_ci(versus)),
                promotion_passed=result["promotion_passed"],
            )
        return incumbent_rule, sorted(
            results, key=lambda item: item["ranking_score"], reverse=True
        )

    def resolve_cycle(
        self,
        cycle: int,
        incumbent: Checkpoint,
        incumbent_rule: dict[str, Any],
        formal_results: list[dict[str, Any]],
    ) -> tuple[Checkpoint, dict[str, Any], bool, bool, dict[str, Any]]:
        passing = [item for item in formal_results if item["promotion_passed"]]
        if passing:
            winner = passing[0]
            return (
                _checkpoint_from_json(winner["checkpoint"]),
                winner["rule"],
                True,
                False,
                winner,
            )
        challenger = formal_results[0]
        candidate = _checkpoint_from_json(challenger["checkpoint"])
        eval_dir = self.output_dir / f"cycle-{cycle:03d}" / "eval"
        equivalence = self.eval_actor(
            candidate,
            incumbent,
            deals=self.args.equivalence_deals,
            seed=3_036_700 + cycle,
            base_seed=953_600_000 + cycle * 10_000_000,
            destination=eval_dir / f"equivalence-u{candidate.update:06d}-vs-incumbent.json",
            label=f"cycle-{cycle:03d}-equivalence",
        )
        challenger["equivalence"] = equivalence
        challenger["equivalent"] = _actor_equivalent(
            equivalence, self.args.equivalence_margin
        )
        challenger["promotion_after_equivalence"] = bool(
            challenger["rule_noninferior"] and _actor_superior(equivalence)
        )
        if challenger["promotion_after_equivalence"]:
            return candidate, challenger["rule"], True, False, challenger
        plateau = bool(challenger["rule_noninferior"] and challenger["equivalent"])
        return incumbent, incumbent_rule, False, plateau, challenger

    def write_report(self, *, converged: bool, reason: str) -> None:
        report = {
            "schema": "solver-leaf-convergence-report-v1",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "converged": converged,
            "reason": reason,
            "incumbent": self.state.get("incumbent"),
            "incumbent_rule": self.state.get("incumbent_rule"),
            "plateau_streak": self.state.get("plateau_streak", 0),
            "cycles": self.state.get("cycles", []),
            "initial_holdout": self.state.get("initial_holdout"),
        }
        _atomic_json(self.report_json, report)
        incumbent_rule = report.get("incumbent_rule") or {}
        lines = [
            "# Solver-leaf PPO convergence report",
            "",
            f"- Finished: {report['finished_at']}",
            f"- Converged: {converged}",
            f"- Reason: {reason}",
            f"- Incumbent: `{(report.get('incumbent') or {}).get('actor')}`",
            f"- Rule mean: {incumbent_rule.get('mean_duplicate_margin_points')}",
            f"- Rule 95% CI: {incumbent_rule.get('confidence_interval_95_points')}",
            f"- Plateau streak: {report['plateau_streak']}",
            f"- Completed cycles: {len(report['cycles'])}",
        ]
        self.report_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.emit("convergence_complete", converged=converged, reason=reason)
        self.state["status"] = "complete"
        self.state["converged"] = converged
        self.state["completion_reason"] = reason
        self.state["report_json"] = str(self.report_json)
        self.state["report_markdown"] = str(self.report_markdown)
        self._save_state()

    def run(self) -> None:
        if self.state.get("status") == "complete" and self.report_json.is_file():
            print(json.dumps(_load_json(self.report_json), ensure_ascii=False), flush=True)
            return
        source_actor = _resolved(self.args.source_actor)
        source_trainer = _resolved(self.args.source_trainer)
        u75_actor = _resolved(self.args.u75_actor)
        source_metadata = _validate_actor(source_actor)
        _validate_actor(u75_actor)
        if not source_trainer.is_file():
            raise FileNotFoundError(source_trainer)
        holdout_rule_path = _resolved(self.args.holdout_rule_json)
        holdout_actor_path = _resolved(self.args.holdout_vs_u75_json)
        if not holdout_rule_path.is_file() or not holdout_actor_path.is_file():
            raise FileNotFoundError("independent holdout reports must exist before Phase3")
        holdout_rule = _load_json(holdout_rule_path)
        holdout_actor = _load_json(holdout_actor_path)
        if _resolved(holdout_rule["checkpoint"]) != source_actor:
            raise ValueError("rule holdout does not evaluate the configured source actor")
        if _resolved(holdout_actor["checkpoint"]) != source_actor:
            raise ValueError("actor holdout does not evaluate the configured source actor")
        if _resolved(holdout_actor["opponent_checkpoint"]) != u75_actor:
            raise ValueError("actor holdout does not use the configured Update75 opponent")
        holdout_passed = _ci(holdout_rule)[0] > 0.0 and _ci(holdout_actor)[0] > 0.0
        self.state["initial_holdout"] = {
            "rule": holdout_rule,
            "versus_u75": holdout_actor,
            "passed": holdout_passed,
        }
        if not holdout_passed:
            self.write_report(converged=False, reason="independent holdout gate failed")
            return

        if self.state.get("incumbent"):
            incumbent = _checkpoint_from_json(self.state["incumbent"])
            incumbent_rule = dict(self.state["incumbent_rule"])
        else:
            incumbent = Checkpoint(
                source_actor,
                source_trainer,
                0,
                int(source_metadata["training_update"]),
            )
            incumbent_rule = holdout_rule
            self.state["incumbent"] = incumbent.json()
            self.state["incumbent_rule"] = incumbent_rule
        history = [
            _resolved(path) for path in self.state.get("history", self.args.history_checkpoints)
        ]
        history = _unique_actor_paths(history, exclude=incumbent.actor)
        if self.args.history_opponent_weight > 0.0 and not history:
            raise ValueError("positive history weight requires at least one history actor")
        self.state["history"] = [str(path) for path in history]
        self.state["status"] = "running"
        self._save_state()
        self.emit(
            "holdout_gate_passed",
            rule_mean=_mean(holdout_rule),
            rule_ci95=list(_ci(holdout_rule)),
            versus_u75_mean=_mean(holdout_actor),
            versus_u75_ci95=list(_ci(holdout_actor)),
        )

        completed = len(self.state.get("cycles", []))
        for cycle in range(completed + 1, self.args.max_cycles + 1):
            if self.seconds_remaining() <= 0.0:
                self.write_report(converged=False, reason="wall-time budget exhausted")
                return
            old_incumbent = incumbent
            checkpoints, training_config = self.train_cycle(cycle, incumbent, history)
            screen_results = self.screen(cycle, checkpoints, incumbent)
            cycle_incumbent_rule, formal_results = self.formal(
                cycle, screen_results, incumbent
            )
            incumbent, incumbent_rule, promoted, plateau, decision = self.resolve_cycle(
                cycle,
                incumbent,
                cycle_incumbent_rule,
                formal_results,
            )
            if promoted:
                history = _unique_actor_paths(
                    [old_incumbent.actor, *history], exclude=incumbent.actor
                )[:12]
                self.state["plateau_streak"] = 0
            elif plateau:
                self.state["plateau_streak"] = int(
                    self.state.get("plateau_streak", 0)
                ) + 1
            else:
                self.state["plateau_streak"] = 0
            cycle_record = {
                "cycle": cycle,
                "training_config": training_config,
                "checkpoints": [item.json() for item in checkpoints],
                "screen": screen_results,
                "formal": formal_results,
                "promoted": promoted,
                "plateau": plateau,
                "decision": decision,
                "incumbent_after_cycle": incumbent.json(),
                "plateau_streak": self.state["plateau_streak"],
            }
            self.state.setdefault("cycles", []).append(cycle_record)
            self.state["incumbent"] = incumbent.json()
            self.state["incumbent_rule"] = incumbent_rule
            self.state["history"] = [str(path) for path in history]
            self._save_state()
            self.emit(
                "cycle_resolved",
                cycle=cycle,
                promoted=promoted,
                plateau=plateau,
                plateau_streak=self.state["plateau_streak"],
                incumbent=incumbent.json(),
            )
            if self.state["plateau_streak"] >= self.args.plateau_cycles:
                self.write_report(
                    converged=True,
                    reason=(
                        f"{self.args.plateau_cycles} consecutive statistically "
                        "equivalent challenger cycles"
                    ),
                )
                return
        self.write_report(converged=False, reason="maximum cycle budget reached")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    runner = ConvergenceRunner(args)
    try:
        runner.run()
    except BaseException as exc:
        failure = {
            "time": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        runner.state["status"] = "failed"
        runner.state["failure"] = failure
        runner._save_state()
        print(json.dumps({"convergence_failed": failure}, ensure_ascii=False), flush=True)
        raise


if __name__ == "__main__":
    main()
