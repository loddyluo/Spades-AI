#!/usr/bin/env python3
"""
Parse a Spades matchup trace log and plot sorted our_mcts total scores.

For each game, 'Team our_mcts: +XX.X' is the average of the two
our_mcts players' individual scores. This script multiplies by 2 to
get their sum, sorts all values, and draws a line chart.

Usage:
    python plot_our_mcts_scores.py <logfile> [output.png]
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_our_mcts_sums(log_path: str) -> list[float]:
    """Parse log and return list of our_mcts total sums per game.

    Each game block in the trace log ends with something like::

        Team our_mcts: +87.0
        Team go_rule_2: -87.0

    where ``Team our_mcts`` is ``(seat_score_a + seat_score_b) / 2``.
    We return ``2 * team_avg`` = the real sum of both our_mcts players.
    """
    text = Path(log_path).read_text(encoding="utf-8")

    # Split on game boundaries
    game_blocks = re.split(r'^=== GAME seed=\d+ ===$', text, flags=re.MULTILINE)

    sums: list[float] = []
    for block in game_blocks:
        m = re.search(r'^Team our_mcts:\s*([+-]?\d+\.?\d*)', block, re.MULTILINE)
        if m:
            team_avg = float(m.group(1))
            sums.append(1.0 * team_avg)

    # Adjacent pairs of games share the same seed; sum each pair.
    if len(sums)%2 == 1:
        sums.append(-sums[-1])
    paired_sums = [sums[i] + sums[i + 1] for i in range(0, len(sums), 2)]
    print("sigma for one single game", (sum((x - (sum(paired_sums) / len(paired_sums))) ** 2 for x in paired_sums) / len(paired_sums)) ** 0.5 / (len(paired_sums) ** 0.5) / 2.0)
    return paired_sums


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <logfile> [output.png]")
        sys.exit(1)

    log_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    sums = parse_our_mcts_sums(log_path)
    if not sums:
        print("No 'Team our_mcts' entries found in the log.")
        sys.exit(1)

    n = len(sums)
    sums.sort()
    indices = list(range(n))

    # ── stats ──
    mean_val = sum(sums) / n
    median_val = sums[n // 2]
    min_val = sums[0]
    max_val = sums[-1]

    # ── plot ──
    plt.figure(figsize=(14, 6))
    plt.plot(indices, sums, linewidth=1.2, color="#2c7bb6")
    plt.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.fill_between(indices, sums, alpha=0.12, color="#2c7bb6")

    plt.xlabel("Game index (sorted)", fontsize=12)
    plt.ylabel("Our MCTS total score (sum of two players)", fontsize=12)
    title = (
        f"Our MCTS total scores — sorted by paired game\n"
        f"{Path(log_path).name}  ({n * 2} games → {n} pairs)"
    )
    plt.title(title, fontsize=13)
    plt.grid(True, alpha=0.3)

    stats_text = (
        f"Pairs: {n}\n"
        f"Min:   {min_val:+.1f}\n"
        f"Max:   {max_val:+.1f}\n"
        f"Mean:  {mean_val:+.1f}\n"
        f"Median: {median_val:+.1f}"
    )
    plt.text(
        0.98, 0.97, stats_text,
        transform=plt.gca().transAxes,
        fontsize=10, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    print(f"Parsed {n} pairs ({n * 2} games)")
    print(f"Range: {min_val:+.1f} ~ {max_val:+.1f}")
    print(f"Mean:  {mean_val:+.1f}")


if __name__ == "__main__":
    main()
