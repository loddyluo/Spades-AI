"""Wrapper entry point for step-by-step matchup timing.

File purpose:
- Provide a dedicated program entry that runs the existing one-game timing
  benchmark for `evaluate/evaluate_model_matchups.py`.
- Reuse the established per-step real-time printing logic from
  `tests/test_matchup_step_timing.py` so the same matchup configuration can be
  observed step by step without waiting for the full evaluation summary.

Function input/output summary:
- main() -> None
    Input: command-line arguments accepted by the underlying timing benchmark.
    Output: executes the benchmark script as `__main__`, which prints each bid
    and play step in real time and then prints the final timing summary.
"""

from __future__ import annotations

from pathlib import Path
import runpy

REPO_ROOT = Path(__file__).resolve().parents[1]
TIMING_SCRIPT = REPO_ROOT / "tests" / "test_matchup_step_timing.py"


def main() -> None:
    """Run the existing step-timing benchmark as a standalone program.

    Input:
    - command-line arguments passed through to the underlying timing script.

    Output:
    - The benchmark prints each bid/play step as it runs and then emits a
      summary of the game timing breakdown.
    """
    runpy.run_path(str(TIMING_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
