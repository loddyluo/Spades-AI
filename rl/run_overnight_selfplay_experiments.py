"""Bounded overnight league-training experiment ending before 09:00 local time."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output"
PHASE1_DIR = OUTPUT_ROOT / "solver-leaf-selfplay-mix-200k-20260810"
PHASE2_DIR = OUTPUT_ROOT / "solver-leaf-selfplay-converge-20260811"
WORK_DIR = OUTPUT_ROOT / "overnight-rl-20260811"
STATE_PATH = WORK_DIR / "state.json"
REPORT_JSON = WORK_DIR / "overnight-report.json"
REPORT_MARKDOWN = WORK_DIR / "overnight-report.md"

CURRENT_TRAINING_PID = 59693
TIMEZONE = ZoneInfo("Asia/Shanghai")
PHASE2_TRAINING_STOP = datetime(2026, 8, 11, 7, 10, tzinfo=TIMEZONE)
FINAL_DEADLINE = datetime(2026, 8, 11, 8, 45, tzinfo=TIMEZONE)

CHAMPION_ACTOR = OUTPUT_ROOT / "solver-leaf-ppo-400k-20260810" / "actor_update_000075.pt"
CHAMPION_TRAINER = OUTPUT_ROOT / "solver-leaf-ppo-400k-20260810" / "trainer_update_000075.pt"
HISTORY_ACTORS = (
    OUTPUT_ROOT / "solver-leaf-ppo-200k-20260810" / "actor_final.pt",
    OUTPUT_ROOT / "solver-leaf-ppo-400k-20260810" / "actor_update_000060.pt",
    OUTPUT_ROOT / "solver-leaf-ppo-400k-20260810" / "actor_update_000065.pt",
    OUTPUT_ROOT / "solver-leaf-ppo-400k-20260810" / "actor_update_000070.pt",
    OUTPUT_ROOT / "solver-leaf-ppo-400k-20260810" / "actor_update_000080.pt",
    OUTPUT_ROOT / "solver-leaf-ppo-400k-20260810" / "actor_update_000085.pt",
)

RULE_EVAL_SEED = 536_500
RULE_EVAL_BASE_SEED = 153_600_000
ACTOR_EVAL_SEED = 536_700
ACTOR_EVAL_BASE_SEED = 253_600_000


@dataclass(frozen=True, slots=True)
class Checkpoint:
    phase: str
    update: int
    actor: Path
    trainer: Path

    def json(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "update": self.update,
            "actor": str(self.actor),
            "trainer": str(self.trainer),
        }


def _now() -> datetime:
    return datetime.now(TIMEZONE)


def _atomic_json(destination: Path, payload: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


_STATE: dict[str, Any] = {
    "schema": "solver-leaf-overnight-experiment-v1",
    "started_at": _now().isoformat(),
    "phase2_training_stop": PHASE2_TRAINING_STOP.isoformat(),
    "final_deadline": FINAL_DEADLINE.isoformat(),
    "stage": "initializing",
    "events": [],
}


def _emit(event: str, **fields: Any) -> None:
    record = {"time": _now().isoformat(), "event": event, **fields}
    _STATE["stage"] = event
    _STATE["events"].append(record)
    _STATE["events"] = _STATE["events"][-100:]
    _atomic_json(STATE_PATH, _STATE)
    print(json.dumps({"overnight": record}, ensure_ascii=False), flush=True)


def _pid_is_current_training(pid: int) -> bool:
    try:
        completed = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return (
        completed.returncode == 0
        and "rl.train_solver_leaf_ppo_multicpu" in completed.stdout
        and "solver-leaf-selfplay-mix-200k-20260810" in completed.stdout
    )


def _wait_for_phase1() -> None:
    _emit("waiting_for_phase1", pid=CURRENT_TRAINING_PID)
    while _pid_is_current_training(CURRENT_TRAINING_PID):
        if _now() >= PHASE2_TRAINING_STOP:
            raise RuntimeError("phase 1 was still running at the phase-2 cutoff")
        time.sleep(15)
    _emit("phase1_finished")


def _checkpoint_metadata_exists(actor: Path) -> bool:
    return actor.is_file() and actor.with_name(f"{actor.name}.json").is_file()


def _complete_checkpoints(directory: Path, phase: str) -> list[Checkpoint]:
    found: list[Checkpoint] = []
    pattern = re.compile(r"actor_update_(\d{6})\.pt$")
    if directory.is_dir():
        for actor in directory.glob("actor_update_*.pt"):
            match = pattern.fullmatch(actor.name)
            if match is None or not _checkpoint_metadata_exists(actor):
                continue
            update = int(match.group(1))
            trainer = directory / f"trainer_update_{update:06d}.pt"
            if trainer.is_file():
                found.append(Checkpoint(phase, update, actor, trainer))
        final_actor = directory / "actor_final.pt"
        final_trainer = directory / "trainer_final.pt"
        if _checkpoint_metadata_exists(final_actor) and final_trainer.is_file():
            metadata = json.loads(
                final_actor.with_name(f"{final_actor.name}.json").read_text(
                    encoding="utf-8"
                )
            )
            found.append(
                Checkpoint(
                    phase,
                    int(metadata["training_update"]),
                    final_actor,
                    final_trainer,
                )
            )
    unique = {(item.update, str(item.actor)): item for item in found}
    return sorted(unique.values(), key=lambda item: (item.update, str(item.actor)))


def _sample_checkpoints(
    checkpoints: Sequence[Checkpoint], maximum: int
) -> list[Checkpoint]:
    if len(checkpoints) <= maximum:
        return list(checkpoints)
    indices = {
        round(position * (len(checkpoints) - 1) / (maximum - 1))
        for position in range(maximum)
    }
    return [checkpoints[index] for index in sorted(indices)]


def _terminate_group(process: subprocess.Popen[Any]) -> None:
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


def _run_command(
    name: str,
    command: Sequence[str],
    *,
    log_path: Path,
    timeout_seconds: float,
) -> int:
    if timeout_seconds <= 0:
        raise TimeoutError(f"no time remains for {name}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _emit(
        "command_started",
        name=name,
        command=list(command),
        timeout_seconds=round(timeout_seconds, 1),
        log=str(log_path),
    )
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        started = time.monotonic()
        try:
            while process.poll() is None:
                if time.monotonic() - started >= timeout_seconds:
                    _terminate_group(process)
                    _emit("command_timed_out", name=name, pid=process.pid)
                    return 124
                time.sleep(2)
        except BaseException:
            _terminate_group(process)
            raise
    elapsed = time.monotonic() - started
    _emit(
        "command_finished",
        name=name,
        returncode=int(process.returncode or 0),
        elapsed_seconds=round(elapsed, 2),
    )
    return int(process.returncode or 0)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _evaluation_timeout(reserve_seconds: float = 600.0) -> float:
    return min(1200.0, max(0.0, (FINAL_DEADLINE - _now()).total_seconds() - reserve_seconds))


def _eval_rule(
    actor: Path,
    *,
    deals: int,
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
        str(actor),
        "--deals",
        str(deals),
        "--workers",
        "8",
        "--seed",
        str(RULE_EVAL_SEED),
        "--base-shuffle-seed",
        str(RULE_EVAL_BASE_SEED),
        "--output-json",
        str(destination),
    ]
    returncode = _run_command(
        f"rule-eval-{label}-{deals}",
        command,
        log_path=destination.with_suffix(".log"),
        timeout_seconds=_evaluation_timeout(),
    )
    if returncode != 0 or not destination.is_file():
        raise RuntimeError(f"rule evaluation failed for {actor}")
    return _load_json(destination)


def _eval_actor(
    actor: Path,
    opponent: Path,
    *,
    deals: int,
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
        str(actor),
        "--opponent-checkpoint",
        str(opponent),
        "--deals",
        str(deals),
        "--workers",
        "8",
        "--seed",
        str(ACTOR_EVAL_SEED),
        "--base-shuffle-seed",
        str(ACTOR_EVAL_BASE_SEED),
        "--output-json",
        str(destination),
    ]
    returncode = _run_command(
        f"actor-eval-{label}-{deals}",
        command,
        log_path=destination.with_suffix(".log"),
        timeout_seconds=_evaluation_timeout(),
    )
    if returncode != 0 or not destination.is_file():
        raise RuntimeError(f"actor evaluation failed for {actor}")
    return _load_json(destination)


def _short_sweep(
    checkpoints: Sequence[Checkpoint],
    *,
    prefix: str,
    deals: int = 1000,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in checkpoints:
        destination = WORK_DIR / f"{prefix}-{item.phase}-u{item.update:06d}-rule-{deals}.json"
        report = _eval_rule(
            item.actor,
            deals=deals,
            destination=destination,
            label=f"{item.phase}-u{item.update:06d}",
        )
        result = {
            "checkpoint": item.json(),
            "mean": float(report["mean_duplicate_margin_points"]),
            "ci95": list(report["confidence_interval_95_points"]),
            "report": str(destination),
        }
        results.append(result)
        _emit("short_rule_result", **result)
    return results


def _checkpoint_from_result(result: dict[str, Any]) -> Checkpoint:
    data = result["checkpoint"]
    return Checkpoint(
        phase=str(data["phase"]),
        update=int(data["update"]),
        actor=Path(data["actor"]),
        trainer=Path(data["trainer"]),
    )


def _start_phase2(
    source: Checkpoint,
    baseline_short_mean: float,
    source_short_mean: float,
    source_vs_champion_mean: float,
) -> dict[str, Any]:
    promoted = (
        source.actor.resolve() != CHAMPION_ACTOR.resolve()
        and source_short_mean >= baseline_short_mean + 0.10
        and source_vs_champion_mean > 0.0
    )
    if promoted:
        rule_weight, champion_weight, history_weight = 0.55, 0.30, 0.15
        champion_actor = source.actor
        history = (CHAMPION_ACTOR, *HISTORY_ACTORS)
    else:
        source = Checkpoint("baseline", 75, CHAMPION_ACTOR, CHAMPION_TRAINER)
        rule_weight, champion_weight, history_weight = 0.75, 0.15, 0.10
        champion_actor = CHAMPION_ACTOR
        history = HISTORY_ACTORS

    remaining = (PHASE2_TRAINING_STOP - _now()).total_seconds()
    config = {
        "promoted_phase1": promoted,
        "source": source.json(),
        "source_short_rule_mean": source_short_mean,
        "source_short_vs_champion_mean": source_vs_champion_mean,
        "baseline_short_rule_mean": baseline_short_mean,
        "rule_weight": rule_weight,
        "champion_weight": champion_weight,
        "history_weight": history_weight,
        "champion_actor": str(champion_actor),
        "history": [str(path) for path in history],
        "learning_rate": 5e-5,
        "entropy_coefficient": 0.003,
        "timeout_seconds": max(0, int(remaining)),
    }
    _STATE["phase2_config"] = config
    _atomic_json(STATE_PATH, _STATE)
    if remaining < 1800:
        _emit("phase2_skipped", reason="less than 30 minutes remain", config=config)
        return config
    if PHASE2_DIR.exists() and _complete_checkpoints(PHASE2_DIR, "phase2"):
        _emit("phase2_reused", config=config)
        return config

    command = [
        sys.executable,
        "-m",
        "rl.train_solver_leaf_ppo_multicpu",
        "--total-games",
        "300000",
        "--rollout-deals",
        "2048",
        "--workers",
        "8",
        "--seed",
        "753142",
        "--base-shuffle-seed",
        "85360000",
        "--learning-rate",
        "5e-5",
        "--entropy-coefficient",
        "0.003",
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
        "0.03",
        "--save-every-updates",
        "5",
        "--save-dir",
        str(PHASE2_DIR),
        "--finetune-from",
        str(source.trainer),
        "--rule-opponent-weight",
        str(rule_weight),
        "--champion-opponent-weight",
        str(champion_weight),
        "--history-opponent-weight",
        str(history_weight),
        "--champion-checkpoint",
        str(champion_actor),
        "--history-checkpoints",
        *[str(path) for path in history],
    ]
    _run_command(
        "phase2-training",
        command,
        log_path=WORK_DIR / "phase2-training.jsonl",
        timeout_seconds=max(0.0, remaining),
    )
    return config


def _read_training_updates(path: Path) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    if not path.is_file():
        return updates
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = payload.get("training_update") if isinstance(payload, dict) else None
        if isinstance(item, dict):
            updates.append(item)
    return updates


def _convergence_summary(updates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tail = list(updates[-5:])
    if len(tail) < 3:
        return {"enough_updates": False, "converged": False}
    margins = [float(item["mean_duplicate_margin_points"]) for item in tail]
    entropy = [float(item["ppo"]["entropy"]) for item in tail]
    kl = [float(item["ppo"]["approximate_kl"]) for item in tail]
    x_mean = (len(margins) - 1) / 2.0
    y_mean = statistics.fmean(margins)
    denominator = sum((index - x_mean) ** 2 for index in range(len(margins)))
    slope = sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(margins)
    ) / denominator
    converged = (
        statistics.pstdev(margins) < 0.30
        and abs(slope) < 0.08
        and abs(entropy[-1] - entropy[0]) < 0.03
        and max(kl) < 0.02
    )
    return {
        "enough_updates": True,
        "tail_updates": [int(item["update"]) for item in tail],
        "margin_mean": y_mean,
        "margin_std": statistics.pstdev(margins),
        "margin_slope_per_update": slope,
        "entropy_change": entropy[-1] - entropy[0],
        "max_approximate_kl": max(kl),
        "converged": converged,
    }


def _write_report(report: dict[str, Any]) -> None:
    _atomic_json(REPORT_JSON, report)
    formal = report.get("best_formal_rule") or {}
    actor_eval = report.get("best_vs_champion") or {}
    lines = [
        "# Overnight solver-leaf PPO report",
        "",
        f"- Finished: {report['finished_at']}",
        f"- Best actor: `{report.get('best_actor')}`",
        f"- Best trainer: `{report.get('best_trainer')}`",
        f"- Formal rule mean: {formal.get('mean_duplicate_margin_points')}",
        f"- Formal rule 95% CI: {formal.get('confidence_interval_95_points')}",
        f"- Versus Update 75 mean: {actor_eval.get('mean_duplicate_margin_points')}",
        f"- Versus Update 75 95% CI: {actor_eval.get('confidence_interval_95_points')}",
        f"- Rule objective passed: {report.get('rule_objective_passed')}",
        f"- Mean above Update 75 rule baseline: {report.get('mean_above_update75_baseline')}",
        f"- Convergence indicators passed: {report.get('convergence', {}).get('converged')}",
        "",
        "The original Update 75 rule baseline was mean 0.7227 with 95% CI "
        "[0.10612511111016765, 1.3392748888898325].",
    ]
    REPORT_MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    required = (CHAMPION_ACTOR, CHAMPION_TRAINER, *HISTORY_ACTORS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required checkpoints missing: {missing}")

    _emit("orchestrator_started")
    _wait_for_phase1()
    phase1_all = _complete_checkpoints(PHASE1_DIR, "phase1")
    if not phase1_all:
        raise RuntimeError("phase 1 produced no complete checkpoint")
    phase1_sample = _sample_checkpoints(phase1_all, maximum=6)
    _emit(
        "phase1_checkpoints_selected",
        available=len(phase1_all),
        selected=[item.json() for item in phase1_sample],
    )

    baseline_checkpoint = Checkpoint("baseline", 75, CHAMPION_ACTOR, CHAMPION_TRAINER)
    baseline_report = _eval_rule(
        CHAMPION_ACTOR,
        deals=1000,
        destination=WORK_DIR / "baseline-u000075-rule-1000.json",
        label="baseline-u000075",
    )
    baseline_short_mean = float(baseline_report["mean_duplicate_margin_points"])
    phase1_results = _short_sweep(phase1_sample, prefix="sweep1", deals=1000)
    best_phase1 = max(phase1_results, key=lambda item: float(item["mean"]))
    best_phase1_checkpoint = _checkpoint_from_result(best_phase1)

    top_phase1 = sorted(
        phase1_results, key=lambda item: float(item["mean"]), reverse=True
    )[:2]
    actor_short_results: list[dict[str, Any]] = []
    for result in top_phase1:
        item = _checkpoint_from_result(result)
        report = _eval_actor(
            item.actor,
            CHAMPION_ACTOR,
            deals=1000,
            destination=WORK_DIR / f"{item.phase}-u{item.update:06d}-vs-u75-1000.json",
            label=f"{item.phase}-u{item.update:06d}-vs-u75",
        )
        actor_short_results.append(
            {
                "checkpoint": item.json(),
                "mean": float(report["mean_duplicate_margin_points"]),
                "ci95": list(report["confidence_interval_95_points"]),
            }
        )
    _STATE["phase1_short_rule"] = phase1_results
    _STATE["phase1_short_vs_champion"] = actor_short_results
    _atomic_json(STATE_PATH, _STATE)

    source = best_phase1_checkpoint
    source_short_mean = float(best_phase1["mean"])
    best_phase1_vs_champion = next(
        (
            float(item["mean"])
            for item in actor_short_results
            if int(item["checkpoint"]["update"]) == best_phase1_checkpoint.update
            and item["checkpoint"]["phase"] == best_phase1_checkpoint.phase
        ),
        -math.inf,
    )
    phase2_config = _start_phase2(
        source,
        baseline_short_mean,
        source_short_mean,
        best_phase1_vs_champion,
    )

    phase2_all = _complete_checkpoints(PHASE2_DIR, "phase2")
    phase2_sample = _sample_checkpoints(phase2_all, maximum=6)
    phase2_results = (
        _short_sweep(phase2_sample, prefix="sweep2", deals=1000)
        if phase2_sample and (FINAL_DEADLINE - _now()).total_seconds() > 1800
        else []
    )
    all_short = [
        {
            "checkpoint": baseline_checkpoint.json(),
            "mean": baseline_short_mean,
            "ci95": list(baseline_report["confidence_interval_95_points"]),
            "report": str(WORK_DIR / "baseline-u000075-rule-1000.json"),
        },
        *phase1_results,
        *phase2_results,
    ]
    candidates = sorted(
        all_short, key=lambda item: float(item["mean"]), reverse=True
    )
    formal_results: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates[:2], start=1):
        if (FINAL_DEADLINE - _now()).total_seconds() < 900:
            break
        item = _checkpoint_from_result(candidate)
        report_path = WORK_DIR / f"formal-rank{rank}-{item.phase}-u{item.update:06d}-rule-5000.json"
        report = _eval_rule(
            item.actor,
            deals=5000,
            destination=report_path,
            label=f"formal-rank{rank}-{item.phase}-u{item.update:06d}",
        )
        formal_results.append(
            {
                "checkpoint": item.json(),
                "mean": float(report["mean_duplicate_margin_points"]),
                "ci95": list(report["confidence_interval_95_points"]),
                "report": str(report_path),
                "payload": report,
            }
        )
    if not formal_results:
        raise RuntimeError("no formal rule evaluation completed before the deadline")
    best_formal = max(formal_results, key=lambda item: float(item["mean"]))
    best_checkpoint = _checkpoint_from_result(best_formal)

    champion_report: dict[str, Any] | None = None
    if (FINAL_DEADLINE - _now()).total_seconds() >= 480:
        champion_path = WORK_DIR / (
            f"formal-{best_checkpoint.phase}-u{best_checkpoint.update:06d}-vs-u75-2000.json"
        )
        champion_report = _eval_actor(
            best_checkpoint.actor,
            CHAMPION_ACTOR,
            deals=2000,
            destination=champion_path,
            label=f"formal-{best_checkpoint.phase}-u{best_checkpoint.update:06d}-vs-u75",
        )

    convergence = _convergence_summary(
        _read_training_updates(WORK_DIR / "phase2-training.jsonl")
    )
    formal_payload = best_formal["payload"]
    report = {
        "schema": "solver-leaf-overnight-report-v1",
        "finished_at": _now().isoformat(),
        "deadline": FINAL_DEADLINE.isoformat(),
        "phase1_complete_checkpoints": [item.json() for item in phase1_all],
        "phase1_short_rule": phase1_results,
        "phase1_short_vs_champion": actor_short_results,
        "phase2_config": phase2_config,
        "phase2_complete_checkpoints": [item.json() for item in phase2_all],
        "phase2_short_rule": phase2_results,
        "formal_rule_candidates": formal_results,
        "best_actor": str(best_checkpoint.actor),
        "best_trainer": str(best_checkpoint.trainer),
        "best_formal_rule": formal_payload,
        "best_vs_champion": champion_report,
        "rule_objective_passed": bool(
            float(formal_payload["mean_duplicate_margin_points"]) > 0.0
            and float(formal_payload["confidence_interval_95_points"][0]) > 0.0
        ),
        "mean_above_update75_baseline": bool(
            float(formal_payload["mean_duplicate_margin_points"]) > 0.7227
        ),
        "convergence": convergence,
    }
    _write_report(report)
    _emit(
        "overnight_complete",
        best_actor=str(best_checkpoint.actor),
        formal_rule_mean=float(formal_payload["mean_duplicate_margin_points"]),
        formal_rule_ci=list(formal_payload["confidence_interval_95_points"]),
        versus_champion_mean=(
            None
            if champion_report is None
            else float(champion_report["mean_duplicate_margin_points"])
        ),
        converged=bool(convergence.get("converged")),
    )
    _STATE["report_json"] = str(REPORT_JSON)
    _STATE["report_markdown"] = str(REPORT_MARKDOWN)
    _STATE["stage"] = "complete"
    _STATE["completed_at"] = _now().isoformat()
    _atomic_json(STATE_PATH, _STATE)


def main() -> None:
    try:
        run()
    except BaseException as exc:
        failure = {
            "schema": "solver-leaf-overnight-report-v1",
            "finished_at": _now().isoformat(),
            "deadline": FINAL_DEADLINE.isoformat(),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _write_report(failure)
        _STATE["stage"] = "failed"
        _STATE["failure"] = failure
        _atomic_json(STATE_PATH, _STATE)
        _emit("overnight_failed", error=failure["error"])
        raise


if __name__ == "__main__":
    main()
