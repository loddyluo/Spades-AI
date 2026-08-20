from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rl.run_solver_leaf_convergence import (
    ConvergenceRunner,
    _actor_equivalent,
    _actor_superior,
    _complete_checkpoints,
    _rule_noninferior,
    parse_args as parse_convergence_args,
)
from rl.train_solver_leaf_ppo_multicpu import (
    _ppo_config,
    _validate_resume_settings,
    parse_args as parse_training_args,
)


def _report(mean: float, lower: float, upper: float) -> dict[str, object]:
    return {
        "mean_duplicate_margin_points": mean,
        "confidence_interval_95_points": [lower, upper],
    }


def test_statistical_promotion_and_equivalence_rules() -> None:
    incumbent = _report(1.8, 1.2, 2.4)
    noninferior = _report(1.4, 0.8, 2.0)
    regressed = _report(1.2, 0.7, 1.7)

    assert _rule_noninferior(noninferior, incumbent, 0.5)
    assert not _rule_noninferior(regressed, incumbent, 0.5)
    assert _actor_superior(_report(0.7, 0.1, 1.3))
    assert not _actor_superior(_report(0.2, -0.3, 0.7))
    assert _actor_equivalent(_report(0.0, -0.4, 0.4), 0.5)
    assert not _actor_equivalent(_report(0.0, -0.6, 0.4), 0.5)


def test_complete_checkpoints_prefers_final_for_same_update(tmp_path: Path) -> None:
    for name in (
        "actor_update_000005.pt",
        "actor_update_000005.pt.json",
        "trainer_update_000005.pt",
        "actor_final.pt",
        "actor_final.pt.json",
        "trainer_final.pt",
    ):
        path = tmp_path / name
        if name.endswith(".json"):
            path.write_text(json.dumps({"training_update": 5}), encoding="utf-8")
        else:
            path.touch()

    checkpoints = _complete_checkpoints(tmp_path, cycle=2)

    assert len(checkpoints) == 1
    assert checkpoints[0].update == 5
    assert checkpoints[0].actor.name == "actor_final.pt"


def test_finetune_checkpoint_can_be_resumed_without_repeating_source_flag() -> None:
    args = parse_training_args(
        [
            "--resume",
            "trainer_update_000005.pt",
            "--seed",
            "17",
            "--base-shuffle-seed",
            "23",
            "--rollout-deals",
            "64",
            "--learning-rate",
            "0.00002",
            "--entropy-coefficient",
            "0.003",
            "--target-kl",
            "0.01",
            "--rule-opponent-weight",
            "0.45",
            "--champion-opponent-weight",
            "0.30",
            "--history-opponent-weight",
            "0.25",
            "--champion-checkpoint",
            "champion.pt",
            "--history-checkpoints",
            "history.pt",
        ]
    )
    saved_run = vars(args).copy()
    saved_run["resume"] = None
    saved_run["finetune_from"] = "source-trainer.pt"
    saved = {
        "actor_hidden_dims": list(args.actor_hidden_dims),
        "critic_hidden_dims": list(args.critic_hidden_dims),
        "ppo_config": asdict(_ppo_config(args)),
        "run_config": saved_run,
    }

    _validate_resume_settings(args, _ppo_config(args), saved)


def test_completed_report_leaves_runner_in_complete_state(tmp_path: Path) -> None:
    args = parse_convergence_args(["--output-dir", str(tmp_path)])
    runner = ConvergenceRunner(args)

    runner.write_report(converged=True, reason="test plateau")

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["converged"] is True
