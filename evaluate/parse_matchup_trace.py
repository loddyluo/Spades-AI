#!/usr/bin/env python3
"""Parse a matchup trace log and compute per-AI team score aggregates.

Usage: python evaluate/parse_matchup_trace.py /path/to/matchup_trace.txt
"""
import argparse
import re
from collections import defaultdict


def parse_file(path):
    team_re = re.compile(r"^Team\s+(?P<spec>\S+):\s*(?P<score>[+-]?\d+(?:\.\d+)?)")
    sums = defaultdict(float)
    counts = defaultdict(int)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = team_re.match(line.strip())
            if not m:
                continue
            spec = m.group("spec")
            score = float(m.group("score"))
            sums[spec] += score
            counts[spec] += 1
    return sums, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    args = p.parse_args()
    sums, counts = parse_file(args.path)
    if not sums:
        print("No team score lines found in the file.")
        return
    print(f"Parsed file: {args.path}")
    total_games = max(counts.values()) if counts else 0
    print(f"Detected games (per-spec counts): {dict(counts)}")
    print()
    # Print per-spec totals and averages
    for spec in sorted(sums.keys()):
        total = sums[spec]
        cnt = counts[spec]
        avg = total / cnt if cnt else 0.0
        print(f"{spec}: total={total:+.1f} over {cnt} games, avg={avg:+.3f}")

    # If both our_mcts and go_rule present, compute difference
    if "our_mcts" in sums and "go_rule" in sums:
        avg_our = sums["our_mcts"] / counts["our_mcts"]
        avg_go = sums["go_rule"] / counts["go_rule"]
        diff = avg_our - avg_go
        print()
        print(f"our_mcts avg - go_rule avg = {diff:+.3f} points per game")


if __name__ == "__main__":
    main()
