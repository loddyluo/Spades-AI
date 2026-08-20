"""Wait for one exact Phase1 trainer, then exec the Nil convergence runner."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from rl.run_nil_solver_leaf_convergence import (
    DEFAULT_OUTPUT,
    DEFAULT_PHASE1,
    _complete_checkpoints,
    _resolved,
    _validate_bundle,
    _validate_trainer,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely hand one running Nil Phase1 job to the convergence runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase1-pid", type=int, required=True)
    parser.add_argument("--phase1-dir", default=str(DEFAULT_PHASE1))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="extra arguments passed to rl.run_nil_solver_leaf_convergence",
    )
    return parser.parse_args(argv)


def _emit(event: str, **fields: Any) -> None:
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    print(json.dumps({"nil_resume_supervisor": record}, ensure_ascii=False), flush=True)


def _command_for_pid(pid: int) -> str | None:
    result = subprocess.run(
        ("ps", "-p", str(pid), "-o", "command="),
        check=False,
        capture_output=True,
        text=True,
    )
    command = result.stdout.strip()
    return command or None


def _existing_runner_pids() -> list[int]:
    result = subprocess.run(
        ("ps", "-axo", "pid=,command="),
        check=True,
        capture_output=True,
        text=True,
    )
    marker = " -m rl.run_nil_solver_leaf_convergence"
    found: list[int] = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and marker in fields[1]:
            found.append(int(fields[0]))
    return found


def _start_power_assertion() -> subprocess.Popen[Any]:
    """Keep macOS awake for this PID, including after exec into the runner."""

    return subprocess.Popen(
        ("/usr/bin/caffeinate", "-dimsu", "-w", str(os.getpid())),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _training_complete(log_path: Path, phase1_dir: Path) -> dict[str, Any] | None:
    if not log_path.is_file():
        return None
    completion: dict[str, Any] | None = None
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = payload.get("training_complete") if isinstance(payload, dict) else None
            if isinstance(value, dict):
                completion = value
    if completion is None:
        return None
    if int(completion.get("games_trained", -1)) != 100_000:
        raise RuntimeError("Phase1 completion event does not contain 100000 games")
    reported = _resolved(completion.get("save_dir", ""))
    if reported != phase1_dir:
        raise RuntimeError("Phase1 completion event save directory mismatch")
    return completion


def run(args: argparse.Namespace) -> None:
    if type(args.phase1_pid) is not int or args.phase1_pid <= 1:
        raise ValueError("--phase1-pid must identify a non-system process")
    if not (0.1 <= args.poll_seconds <= 60.0):
        raise ValueError("--poll-seconds must be between 0.1 and 60")
    phase1_dir = _resolved(args.phase1_dir)
    output_dir = _resolved(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    power_assertion = _start_power_assertion()
    _emit("power_assertion_started", pid=power_assertion.pid)
    initial_command = _command_for_pid(args.phase1_pid)
    expected_tokens = (
        "rl.train_nil_solver_leaf_ppo_multicpu",
        phase1_dir.name,
    )
    if initial_command is not None and not all(
        token in initial_command for token in expected_tokens
    ):
        raise RuntimeError(
            f"PID {args.phase1_pid} is not the expected Phase1 trainer"
        )
    _emit(
        "waiting_for_phase1",
        phase1_pid=args.phase1_pid,
        phase1_dir=str(phase1_dir),
    )
    while True:
        command = _command_for_pid(args.phase1_pid)
        if command is None:
            break
        if not all(token in command for token in expected_tokens):
            raise RuntimeError(f"PID {args.phase1_pid} was reused by another process")
        time.sleep(args.poll_seconds)

    log_path = phase1_dir / "training.jsonl"
    completion = None
    artifact_deadline = time.monotonic() + 30.0
    while completion is None and time.monotonic() < artifact_deadline:
        completion = _training_complete(log_path, phase1_dir)
        if completion is None:
            time.sleep(0.25)
    if completion is None:
        raise RuntimeError("Phase1 trainer disappeared without a completion event")
    checkpoints = _complete_checkpoints(phase1_dir, 1)
    if not checkpoints or not (phase1_dir / "actors_final.json").is_file():
        raise RuntimeError("Phase1 completed without final bundle/trainer artifacts")
    final = checkpoints[-1]
    bundle_validation = _validate_bundle(final.bundle)
    trainer_validation = _validate_trainer(final.trainer)
    runners = _existing_runner_pids()
    if runners:
        _emit("existing_runner_detected", pids=runners)
        return
    _emit(
        "phase1_validated",
        completion=completion,
        checkpoint=final.json(),
        bundle_sha256=bundle_validation["sha256"],
        trainer_sha256=trainer_validation["sha256"],
    )
    command = [
        sys.executable,
        "-m",
        "rl.run_nil_solver_leaf_convergence",
        "--phase1-dir",
        str(phase1_dir),
        "--output-dir",
        str(output_dir),
        *args.runner_args,
    ]
    _emit("starting_convergence_runner", command=command)
    os.chdir(REPO_ROOT)
    os.execv(sys.executable, command)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        run(args)
    except BaseException as exc:
        _emit("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
