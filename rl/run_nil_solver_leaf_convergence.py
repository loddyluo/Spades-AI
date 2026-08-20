"""Run resumable four-role Nil PPO cycles until a statistical plateau."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from rl.nil_solver_leaf_env import NIL_ROLES
from rl.nil_solver_leaf_ppo import (
    NIL_TRAINER_SCHEMA,
    load_nil_role_actor_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE1 = REPO_ROOT / "output" / "solver-leaf-nil-four-role-phase1-20260817"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "solver-leaf-nil-four-role-convergence-20260817"


@dataclass(frozen=True, slots=True)
class BundleCheckpoint:
    bundle: Path
    trainer: Path
    cycle: int
    update: int

    def json(self) -> dict[str, Any]:
        return {
            "bundle": str(self.bundle),
            "trainer": str(self.trainer),
            "cycle": self.cycle,
            "update": self.update,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable four-role Nil PPO convergence league",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase1-dir", default=str(DEFAULT_PHASE1))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
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
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--learning-rate-decay", type=float, default=0.8)
    parser.add_argument("--entropy-start", type=float, default=0.003)
    parser.add_argument("--entropy-final", type=float, default=0.0015)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--oversample-factor", type=float, default=6.5)
    parser.add_argument("--seed", type=int, default=1_636_142)
    parser.add_argument("--base-shuffle-seed", type=int, default=1_063_600_000)
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="zero means no cycle cap; stop only at statistical convergence",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
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
    ):
        value = getattr(args, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.cycle_games % 2:
        raise ValueError("--cycle-games must be even")
    if type(args.max_cycles) is not int or args.max_cycles < 0:
        raise ValueError("--max-cycles must be nonnegative")
    for name in (
        "equivalence_margin",
        "rule_noninferiority_margin",
        "learning_rate",
        "minimum_learning_rate",
        "learning_rate_decay",
        "target_kl",
        "oversample_factor",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("entropy_start", "entropy_final"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
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
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _validate_bundle(path: Path) -> dict[str, Any]:
    _, manifest, metadata = load_nil_role_actor_bundle(path, device="cpu")
    return {"manifest": manifest, "metadata": metadata, "sha256": _sha256(path)}


def _validate_trainer(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != NIL_TRAINER_SCHEMA:
        raise ValueError(f"unsupported Nil trainer: {path}")
    if tuple(payload.get("roles", ())) != NIL_ROLES:
        raise ValueError(f"Nil trainer roles mismatch: {path}")
    if payload.get("encoder_schema") != "first4-nil-observation-v1-536":
        raise ValueError(f"Nil trainer encoder mismatch: {path}")
    if payload.get("input_dim") != 536 or payload.get("output_dim") != 52:
        raise ValueError(f"Nil trainer dimensions mismatch: {path}")
    return {
        "sha256": _sha256(path),
        "update": int(payload["update"]),
        "deals_trained": int(payload["deals_trained"]),
    }


def _checkpoint_from_json(payload: dict[str, Any]) -> BundleCheckpoint:
    return BundleCheckpoint(
        bundle=_resolved(payload["bundle"]),
        trainer=_resolved(payload["trainer"]),
        cycle=int(payload["cycle"]),
        update=int(payload["update"]),
    )


def _complete_checkpoints(directory: Path, cycle: int) -> list[BundleCheckpoint]:
    found: dict[int, BundleCheckpoint] = {}
    pattern = re.compile(r"actors_update_(\d{6})\.json$")
    if directory.is_dir():
        for bundle in directory.glob("actors_update_*.json"):
            match = pattern.fullmatch(bundle.name)
            if match is None:
                continue
            update = int(match.group(1))
            trainer = directory / f"trainer_update_{update:06d}.pt"
            if trainer.is_file():
                found[update] = BundleCheckpoint(
                    bundle.resolve(), trainer.resolve(), cycle, update
                )
        final_bundle = directory / "actors_final.json"
        final_trainer = directory / "trainer_final.pt"
        if final_bundle.is_file() and final_trainer.is_file():
            manifest = _load_json(final_bundle)
            update = int(manifest["training_update"])
            found[update] = BundleCheckpoint(
                final_bundle.resolve(), final_trainer.resolve(), cycle, update
            )
    return sorted(found.values(), key=lambda item: item.update)


def _latest_partial_trainer(directory: Path) -> Path | None:
    pattern = re.compile(r"trainer_update_(\d{6})\.pt$")
    found: list[tuple[int, Path]] = []
    if directory.is_dir():
        for path in directory.glob("trainer_update_*.pt"):
            match = pattern.fullmatch(path.name)
            if match is not None:
                found.append((int(match.group(1)), path.resolve()))
    return max(found, default=(0, None), key=lambda item: item[0])[1]


def _ci(report: dict[str, Any]) -> tuple[float, float]:
    values = report.get("confidence_interval_95_points")
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError("evaluation report is missing its 95% confidence interval")
    return float(values[0]), float(values[1])


def _mean(report: dict[str, Any]) -> float:
    return float(report["mean_duplicate_margin_points"])


def _validate_evaluation_report(
    report: dict[str, Any],
    candidate: BundleCheckpoint,
    opponent: BundleCheckpoint | None,
    *,
    deals: int,
    workers: int,
    seed: int,
    base_seed: int,
    oversample_factor: float,
) -> dict[str, Any]:
    """Reject stale or mismatched JSON before a resumed run reuses it."""

    if report.get("schema") != "solver-leaf-nil-four-role-evaluation-v1":
        raise ValueError("evaluation report schema mismatch")
    if _resolved(report.get("bundle", "")) != candidate.bundle.resolve():
        raise ValueError("evaluation report candidate bundle mismatch")
    expected_opponent = None if opponent is None else opponent.bundle.resolve()
    reported_opponent = report.get("opponent_bundle")
    if expected_opponent is None:
        if reported_opponent is not None:
            raise ValueError("evaluation report unexpectedly names an opponent bundle")
    elif _resolved(reported_opponent or "") != expected_opponent:
        raise ValueError("evaluation report opponent bundle mismatch")
    if report.get("bundle_manifest") != _load_json(candidate.bundle):
        raise ValueError("evaluation report candidate manifest is stale")
    expected_opponent_manifest = (
        None if opponent is None else _load_json(opponent.bundle)
    )
    if report.get("opponent_bundle_manifest") != expected_opponent_manifest:
        raise ValueError("evaluation report opponent manifest is stale")
    expected_comparison = (
        "candidate-nil-bundle-vs-RuleBasedFirst4NilPlayer"
        if opponent is None
        else "candidate-nil-bundle-vs-frozen-nil-bundle"
    )
    if report.get("comparison") != expected_comparison:
        raise ValueError("evaluation report comparison mismatch")
    exact_values = {
        "duplicate_deals": deals,
        "games": deals * 2,
        "solver_calls": deals * 2,
        "workers": workers,
        "seed": seed,
        "base_shuffle_seed": base_seed,
    }
    for name, expected in exact_values.items():
        if report.get(name) != expected:
            raise ValueError(
                f"evaluation report {name} mismatch: "
                f"expected {expected}, got {report.get(name)!r}"
            )
    reported_oversample = float(report.get("oversample_factor", float("nan")))
    if not math.isclose(
        reported_oversample, oversample_factor, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("evaluation report oversample factor mismatch")
    mean = _mean(report)
    lower, upper = _ci(report)
    standard_error = float(report["standard_error_points"])
    if not all(math.isfinite(value) for value in (mean, lower, upper, standard_error)):
        raise ValueError("evaluation report contains non-finite statistics")
    if standard_error < 0.0 or lower > mean or mean > upper:
        raise ValueError("evaluation report confidence interval is invalid")
    expected_half_width = 1.96 * standard_error
    if not (
        math.isclose(mean - lower, expected_half_width, rel_tol=1e-9, abs_tol=1e-9)
        and math.isclose(upper - mean, expected_half_width, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise ValueError("evaluation report confidence interval is inconsistent")
    outcome_count = sum(int(report[name]) for name in ("wins", "ties", "losses"))
    if outcome_count != deals:
        raise ValueError("evaluation report outcome counts do not match deals")
    return report


class NilConvergenceRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.phase1_dir = _resolved(args.phase1_dir)
        self.output_dir = _resolved(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / "state.json"
        self.report_path = self.output_dir / "convergence-report.json"
        if self.state_path.is_file():
            self.state = _load_json(self.state_path)
        else:
            self.state = {
                "schema": "solver-leaf-nil-four-role-convergence-v1",
                "status": "initializing",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "plateau_streak": 0,
                "cycles": [],
                "events": [],
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
        self.state["events"] = [*self.state.get("events", []), record][-300:]
        self._save_state()
        print(json.dumps({"nil_convergence": record}, ensure_ascii=False), flush=True)

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

    def run_command(self, name: str, command: Sequence[str], log_path: Path) -> None:
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
                return_code = process.wait()
            except BaseException:
                self._terminate(process)
                raise
        if return_code != 0:
            raise RuntimeError(f"{name} failed with exit code {return_code}")
        self.emit("command_finished", name=name)

    def evaluate(
        self,
        candidate: BundleCheckpoint,
        opponent: BundleCheckpoint | None,
        *,
        deals: int,
        seed: int,
        base_seed: int,
        destination: Path,
        label: str,
    ) -> dict[str, Any]:
        if destination.is_file():
            _validate_bundle(candidate.bundle)
            if opponent is not None:
                _validate_bundle(opponent.bundle)
            return _validate_evaluation_report(
                _load_json(destination),
                candidate,
                opponent,
                deals=deals,
                workers=self.args.workers,
                seed=seed,
                base_seed=base_seed,
                oversample_factor=self.args.oversample_factor,
            )
        command = [
            sys.executable,
            "-m",
            "evaluate.evaluate_nil_solver_leaf_ppo",
            "--bundle",
            str(candidate.bundle),
            "--deals",
            str(deals),
            "--workers",
            str(self.args.workers),
            "--seed",
            str(seed),
            "--base-shuffle-seed",
            str(base_seed),
            "--oversample-factor",
            str(self.args.oversample_factor),
            "--output-json",
            str(destination),
        ]
        if opponent is not None:
            command.extend(("--opponent-bundle", str(opponent.bundle)))
        self.run_command(label, command, destination.with_suffix(".log"))
        return _validate_evaluation_report(
            _load_json(destination),
            candidate,
            opponent,
            deals=deals,
            workers=self.args.workers,
            seed=seed,
            base_seed=base_seed,
            oversample_factor=self.args.oversample_factor,
        )

    def train_cycle(
        self,
        cycle: int,
        incumbent: BundleCheckpoint,
        history: Sequence[Path],
    ) -> tuple[list[BundleCheckpoint], dict[str, Any]]:
        cycle_dir = self.output_dir / f"cycle-{cycle:03d}"
        checkpoints = _complete_checkpoints(cycle_dir, cycle)
        if (cycle_dir / "actors_final.json").is_file():
            return checkpoints, {"reused": True}
        learning_rate = max(
            self.args.minimum_learning_rate,
            self.args.learning_rate * self.args.learning_rate_decay ** max(0, cycle - 2),
        )
        entropy_progress = min(max(0, cycle - 2) / 5.0, 1.0)
        entropy = self.args.entropy_start + entropy_progress * (
            self.args.entropy_final - self.args.entropy_start
        )
        history_paths = []
        seen = {incumbent.bundle.resolve()}
        for path in history:
            resolved = path.resolve()
            if resolved not in seen:
                _validate_bundle(resolved)
                seen.add(resolved)
                history_paths.append(resolved)
        if history_paths:
            rule_weight, champion_weight, history_weight = 0.45, 0.30, 0.25
        else:
            rule_weight, champion_weight, history_weight = 0.70, 0.30, 0.0
        run_seed = self.args.seed + cycle * 10_000
        base_seed = self.args.base_shuffle_seed + cycle * 10_000_000
        config = {
            "cycle": cycle,
            "source": incumbent.json(),
            "history": [str(path) for path in history_paths],
            "learning_rate": learning_rate,
            "entropy_coefficient": entropy,
            "opponent_weights": [rule_weight, champion_weight, history_weight],
            "seed": run_seed,
            "base_shuffle_seed": base_seed,
        }
        command = [
            sys.executable,
            "-m",
            "rl.train_nil_solver_leaf_ppo_multicpu",
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
            "--oversample-factor",
            str(self.args.oversample_factor),
            "--save-dir",
            str(cycle_dir),
            "--rule-opponent-weight",
            str(rule_weight),
            "--champion-opponent-weight",
            str(champion_weight),
            "--history-opponent-weight",
            str(history_weight),
            "--champion-checkpoint",
            str(incumbent.bundle),
            "--history-checkpoints",
            *[str(path) for path in history_paths],
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
        if not checkpoints or not (cycle_dir / "actors_final.json").is_file():
            raise RuntimeError(f"cycle {cycle} did not produce final artifacts")
        return checkpoints, config

    def screen(
        self,
        cycle: int,
        candidates: Sequence[BundleCheckpoint],
        incumbent: BundleCheckpoint,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        eval_dir = self.output_dir / f"cycle-{cycle:03d}" / "eval"
        for candidate in candidates:
            tag = f"u{candidate.update:06d}"
            rule = self.evaluate(
                candidate,
                None,
                deals=self.args.screen_deals,
                seed=1_636_500 + cycle,
                base_seed=563_600_000 + cycle * 10_000_000,
                destination=eval_dir / f"screen-{tag}-rule.json",
                label=f"cycle-{cycle:03d}-{tag}-screen-rule",
            )
            versus = self.evaluate(
                candidate,
                incumbent,
                deals=self.args.screen_deals,
                seed=1_636_700 + cycle,
                base_seed=663_600_000 + cycle * 10_000_000,
                destination=eval_dir / f"screen-{tag}-vs-incumbent.json",
                label=f"cycle-{cycle:03d}-{tag}-screen-incumbent",
            )
            item = {
                "checkpoint": candidate.json(),
                "rule_mean": _mean(rule),
                "rule_ci95": list(_ci(rule)),
                "versus_incumbent_mean": _mean(versus),
                "versus_incumbent_ci95": list(_ci(versus)),
            }
            item["ranking_score"] = (
                item["versus_incumbent_mean"] + 0.25 * item["rule_mean"]
            )
            results.append(item)
            self.emit("screen_result", cycle=cycle, **item)
        return sorted(results, key=lambda item: item["ranking_score"], reverse=True)

    def formal(
        self,
        cycle: int,
        screen_results: Sequence[dict[str, Any]],
        incumbent: BundleCheckpoint,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        eval_dir = self.output_dir / f"cycle-{cycle:03d}" / "eval"
        rule_seed = 2_636_500 + cycle
        rule_base = 763_600_000 + cycle * 10_000_000
        actor_seed = 2_636_700 + cycle
        actor_base = 863_600_000 + cycle * 10_000_000
        incumbent_rule = self.evaluate(
            incumbent,
            None,
            deals=self.args.formal_rule_deals,
            seed=rule_seed,
            base_seed=rule_base,
            destination=eval_dir / "formal-incumbent-rule.json",
            label=f"cycle-{cycle:03d}-formal-incumbent-rule",
        )
        results: list[dict[str, Any]] = []
        for screened in screen_results[: self.args.formal_candidates]:
            candidate = _checkpoint_from_json(screened["checkpoint"])
            tag = f"u{candidate.update:06d}"
            rule = self.evaluate(
                candidate,
                None,
                deals=self.args.formal_rule_deals,
                seed=rule_seed,
                base_seed=rule_base,
                destination=eval_dir / f"formal-{tag}-rule.json",
                label=f"cycle-{cycle:03d}-{tag}-formal-rule",
            )
            versus = self.evaluate(
                candidate,
                incumbent,
                deals=self.args.formal_actor_deals,
                seed=actor_seed,
                base_seed=actor_base,
                destination=eval_dir / f"formal-{tag}-vs-incumbent.json",
                label=f"cycle-{cycle:03d}-{tag}-formal-incumbent",
            )
            rule_noninferior = bool(
                _ci(rule)[0] > 0.0
                and _mean(rule) >= _mean(incumbent_rule) - self.args.rule_noninferiority_margin
            )
            actor_superior = _ci(versus)[0] > 0.0
            item = {
                "checkpoint": candidate.json(),
                "rule": rule,
                "versus_incumbent": versus,
                "rule_noninferior": rule_noninferior,
                "actor_superior": actor_superior,
                "promotion_passed": bool(rule_noninferior and actor_superior),
                "ranking_score": _mean(versus) + 0.25 * _mean(rule),
            }
            results.append(item)
            self.emit(
                "formal_result",
                cycle=cycle,
                checkpoint=candidate.json(),
                rule_mean=_mean(rule),
                rule_ci95=list(_ci(rule)),
                versus_incumbent_mean=_mean(versus),
                versus_incumbent_ci95=list(_ci(versus)),
                promotion_passed=item["promotion_passed"],
            )
        return incumbent_rule, sorted(
            results, key=lambda item: item["ranking_score"], reverse=True
        )

    def resolve(
        self,
        cycle: int,
        incumbent: BundleCheckpoint,
        incumbent_rule: dict[str, Any],
        formal_results: list[dict[str, Any]],
    ) -> tuple[BundleCheckpoint, dict[str, Any], bool, bool, dict[str, Any]]:
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
        equivalence = self.evaluate(
            candidate,
            incumbent,
            deals=self.args.equivalence_deals,
            seed=3_636_700 + cycle,
            base_seed=963_600_000 + cycle * 10_000_000,
            destination=eval_dir / f"equivalence-u{candidate.update:06d}.json",
            label=f"cycle-{cycle:03d}-equivalence",
        )
        challenger["equivalence"] = equivalence
        lower, upper = _ci(equivalence)
        challenger["equivalent"] = bool(
            lower >= -self.args.equivalence_margin
            and upper <= self.args.equivalence_margin
        )
        challenger["promotion_after_equivalence"] = bool(
            challenger["rule_noninferior"] and lower > 0.0
        )
        if challenger["promotion_after_equivalence"]:
            return candidate, challenger["rule"], True, False, challenger
        plateau = bool(challenger["rule_noninferior"] and challenger["equivalent"])
        return incumbent, incumbent_rule, False, plateau, challenger

    def write_report(self, *, reason: str) -> None:
        incumbent = _checkpoint_from_json(self.state["incumbent"])
        bundle_validation = _validate_bundle(incumbent.bundle)
        trainer_validation = _validate_trainer(incumbent.trainer)
        report = {
            "schema": "solver-leaf-nil-four-role-convergence-report-v1",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "converged": True,
            "reason": reason,
            "incumbent": incumbent.json(),
            "incumbent_bundle_validation": bundle_validation,
            "incumbent_trainer_validation": trainer_validation,
            "incumbent_rule": self.state["incumbent_rule"],
            "plateau_streak": self.state["plateau_streak"],
            "cycles": self.state["cycles"],
        }
        _atomic_json(self.report_path, report)
        self.state["status"] = "complete"
        self.state["converged"] = True
        self.state["completion_reason"] = reason
        self.state["report"] = str(self.report_path)
        self._save_state()
        print(json.dumps({"nil_convergence_complete": report}, ensure_ascii=False), flush=True)

    def run(self) -> None:
        if self.state.get("status") == "complete" and self.report_path.is_file():
            print(json.dumps(_load_json(self.report_path), ensure_ascii=False), flush=True)
            return
        if self.state.get("incumbent"):
            incumbent = _checkpoint_from_json(self.state["incumbent"])
            incumbent_rule = dict(self.state["incumbent_rule"])
        else:
            phase1 = _complete_checkpoints(self.phase1_dir, 1)
            if not phase1 or not (self.phase1_dir / "actors_final.json").is_file():
                raise FileNotFoundError("Phase1 final Nil bundle/trainer are not complete")
            incumbent = phase1[-1]
            _validate_bundle(incumbent.bundle)
            _validate_trainer(incumbent.trainer)
            incumbent_rule = self.evaluate(
                incumbent,
                None,
                deals=self.args.formal_rule_deals,
                seed=2_636_501,
                base_seed=773_600_000,
                destination=self.output_dir / "phase1-formal-rule.json",
                label="phase1-formal-rule",
            )
            self.state["incumbent"] = incumbent.json()
            self.state["incumbent_rule"] = incumbent_rule
            self.state["phase1_rule_gate_passed"] = _ci(incumbent_rule)[0] > 0.0
            self.state["history"] = [
                str(item.bundle) for item in reversed(phase1[:-1])
            ][:12]
            self._save_state()
            self.emit(
                "phase1_initialized",
                incumbent=incumbent.json(),
                history=self.state["history"],
                rule_mean=_mean(incumbent_rule),
                rule_ci95=list(_ci(incumbent_rule)),
                rule_gate_passed=self.state["phase1_rule_gate_passed"],
            )

        history = [_resolved(path) for path in self.state.get("history", [])]
        cycle = int(self.state.get("next_cycle", 2))
        while True:
            if self.args.max_cycles and cycle > self.args.max_cycles:
                raise RuntimeError(
                    "maximum cycle guard reached before statistical convergence"
                )
            old_incumbent = incumbent
            checkpoints, training_config = self.train_cycle(cycle, incumbent, history)
            screen_results = self.screen(cycle, checkpoints, incumbent)
            cycle_incumbent_rule, formal_results = self.formal(
                cycle, screen_results, incumbent
            )
            incumbent, incumbent_rule, promoted, plateau, decision = self.resolve(
                cycle, incumbent, cycle_incumbent_rule, formal_results
            )
            if promoted:
                history = [
                    old_incumbent.bundle,
                    *[path for path in history if path != old_incumbent.bundle],
                ][:12]
                self.state["plateau_streak"] = 0
            elif plateau:
                self.state["plateau_streak"] = int(self.state["plateau_streak"]) + 1
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
            self.state["cycles"].append(cycle_record)
            self.state["incumbent"] = incumbent.json()
            self.state["incumbent_rule"] = incumbent_rule
            self.state["history"] = [str(path) for path in history]
            self.state["next_cycle"] = cycle + 1
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
                    reason=(
                        f"{self.args.plateau_cycles} consecutive statistically "
                        "equivalent non-inferior challenger cycles"
                    )
                )
                return
            cycle += 1


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    output_dir = _resolved(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".runner.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another Nil convergence runner already holds the lock") from error
        runner = NilConvergenceRunner(args)
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
            print(json.dumps({"nil_convergence_failed": failure}, ensure_ascii=False), flush=True)
            raise


if __name__ == "__main__":
    main()
