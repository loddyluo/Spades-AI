"""Trace log regression test for matchup evaluation.

File purpose:
- Verify that the matchup runner can write a single combined trace file under
  `logs/`-style output and that the file contains the requested game details.
- Check that our local `our_mcts` decisions include legal actions and q values.

Function input/output summary:
- build_args(trace_dir: str) -> argparse.Namespace
    Input: output directory for trace files.
    Output: a fully populated argument namespace for one deterministic game.
- main() -> None
    Input: none.
    Output: runs one evaluation game, asserts the trace file exists, and checks
    the logged sections and action-count expectations.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.evaluate_model_matchups import run_evaluation


def build_args(trace_dir: str) -> argparse.Namespace:
    """Build a deterministic one-game evaluation configuration.

    Input:
    - trace_dir: directory where the log file should be written.

    Output:
    - argparse namespace with the fields expected by `run_evaluation`.
    """
    return argparse.Namespace(
        seed=3786,
        num_games=1,
        output="",
        disable_nil=False,
        disable_blind_nil=True,
        p0="our_mcts",
        p1="go_rule",
        p2="our_mcts",
        p3="go_rule",
        device="cpu",
        our_checkpoint="result/mlp_test_3.pth",
        our_exact_threshold=30,
        our_leaf_threshold=24,
        our_simulations_per_action=1,
        our_exploration_constant=1.5,
        our_policy_temperature=1.0,
        our_value_scale=25.0,
        num_workers=1,
        torch_num_threads=1,
        torch_num_interop_threads=1,
        mp_start_method="fork",
        go_pv_checkpoint="",
        go_bid_checkpoint="",
        go_mcts_runs=100,
        go_mcts_steps=5,
        go_mcts_c=0.3,
        go_mcts_mu=0.01,
        go_mcts_threshold=0.05,
        go_argmax_threshold=0.05,
        trace_log_dir=trace_dir,
    )


def main() -> None:
    """Run the trace-log regression check.

    Input:
    - none.

    Output:
    - Asserts that the generated trace file contains the expected sections and
      per-move details.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        args = build_args(tmpdir)
        result = run_evaluation(args)
        trace_path = Path(result.get("trace_log_path", ""))
        assert trace_path.is_file(), f"trace file was not created: {trace_path}"
        assert trace_path.parent == Path(tmpdir)

        content = trace_path.read_text(encoding="utf-8")
        assert "Players:" in content
        assert "Bidding:" in content
        assert "Play:" in content
        assert "seat 0: our_mcts" in content
        assert "seat 1: go_rule" in content
        assert "raw_strength=" in content
        assert "q=[" in content

        play_block = content.split("Play:\n", 1)[1]
        play_lines = [line for line in play_block.splitlines() if line.startswith("  [")]
        assert len(play_lines) == 52, f"expected 52 play lines, got {len(play_lines)}"
        assert any("q=[" in line for line in play_lines), "our_mcts q-values were not logged"
        assert any("mode=" in line for line in play_lines), "decision mode metadata was not logged"

        print(f"trace log written to {trace_path}")


if __name__ == "__main__":
    main()
