from __future__ import annotations

import json
from pathlib import Path

from residual_bidder.checkpoint import CalibrationTuple
from residual_bidder.cli import evaluate_fast_pair


def test_cli_builds_independent_candidate_and_opponent_calibrations(
    monkeypatch, capsys
) -> None:
    config = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(evaluate_fast_pair.BidderConfig, "load", lambda path: config)

    def fake_evaluate(received_config, **kwargs):
        captured["config"] = received_config
        captured.update(kwargs)
        return {"ok": True, "mean_duplicate_margin": 1.25}

    monkeypatch.setattr(evaluate_fast_pair, "evaluate_calibration_pair", fake_evaluate)

    result = evaluate_fast_pair.main(
        [
            "--checkpoint",
            "candidate.pt",
            "--start-seed",
            "123",
            "--deals",
            "64",
            "--candidate-lambda",
            "1.25",
            "--opponent-lambda",
            "0.75",
        ]
    )

    assert result == 0
    assert captured["config"] is config
    assert captured["checkpoint"] == Path("candidate.pt")
    assert captured["candidate_calibration"] == CalibrationTuple(1.25, 0.0, 0.0, 1.0)
    assert captured["opponent_calibration"] == CalibrationTuple(0.75, 0.0, 0.0, 1.0)
    assert json.loads(capsys.readouterr().out)["mean_duplicate_margin"] == 1.25
