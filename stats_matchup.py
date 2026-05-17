#!/usr/bin/env python3
"""stats_matchup.py: 统计 log 文件中 our_mcts vs go_rule_2 的得墩/叫牌分布。

Usage:
    python stats_matchup.py <log_file>

统计内容:
  1. 每个队伍的 (叫墩和, 得墩和) 二维数组 (总和 = 游戏数)
  2. 每个队伍的 x/0 (nil 定约) 统计分布
  3. 绘制热力图和 nil 分布图
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_log(path: str) -> dict:
    """解析 log 文件，返回每场比赛的定约统计。

    返回:
        our_contracts: list of (sum_bid, sum_tricks) for our_mcts team per game
        go_contracts:  list of (sum_bid, sum_tricks) for go_rule_2 team per game
        our_nils:      list of tricks_won for nil contracts (x/0) by our_mcts
        go_nils:       list of tricks_won for nil contracts (x/0) by go_rule_2
    """
    text = Path(path).read_text()

    # Read seat_specs from header
    seat_specs_line = re.search(r"# seat_specs=(.*)", text)
    if seat_specs_line:
        specs = eval(seat_specs_line.group(1))  # ['our_mcts', 'go_rule_2', ...]
    else:
        specs = ["our_mcts", "go_rule_2", "our_mcts", "go_rule_2"]

    # Determine which seats belong to our_mcts in game 0
    base_our_seats = {i for i, s in enumerate(specs) if s == "our_mcts"}
    base_go_seats = {i for i, s in enumerate(specs) if s == "go_rule_2"}

    # Seats alternate every game (偶数游戏: 按 base_our_seats; 奇数游戏: 交换)
    def get_our_seats(game_idx: int) -> set[int]:
        if game_idx % 2 == 0:
            return base_our_seats
        else:
            return base_go_seats

    our_contracts: list[tuple[int, int]] = []
    go_contracts: list[tuple[int, int]] = []
    our_nils: list[int] = []
    go_nils: list[int] = []

    # Find all Results sections
    # Pattern:
    #   Results (tricks/bid):
    #     seat 0: <spec>           <tricks>/<bid>
    #     seat 1: <spec>           <tricks>/<bid>
    #     seat 2: <spec>           <tricks>/<bid>
    #     seat 3: <spec>           <tricks>/<bid>
    results_pattern = re.compile(
        r"Results \(tricks/bid\):\n"
        r"  seat 0: (\S+)\s+(\d+)/(\d+)\n"
        r"  seat 1: (\S+)\s+(\d+)/(\d+)\n"
        r"  seat 2: (\S+)\s+(\d+)/(\d+)\n"
        r"  seat 3: (\S+)\s+(\d+)/(\d+)"
    )

    for game_idx, m in enumerate(results_pattern.finditer(text)):
        our_seats = get_our_seats(game_idx)
        seats_info = []
        for i in range(4):
            spec = m.group(1 + i * 3)
            tricks = int(m.group(2 + i * 3))
            bid = int(m.group(3 + i * 3))
            seats_info.append((spec, tricks, bid))

        # Sum for our_mcts team
        our_bid_sum = 0
        our_trick_sum = 0
        go_bid_sum = 0
        go_trick_sum = 0
        for i, (spec, tricks, bid) in enumerate(seats_info):
            if i in our_seats:
                our_bid_sum += bid
                our_trick_sum += tricks
                if bid == 0:
                    our_nils.append(tricks)
            else:
                go_bid_sum += bid
                go_trick_sum += tricks
                if bid == 0:
                    go_nils.append(tricks)

        our_contracts.append((our_bid_sum, our_trick_sum))
        go_contracts.append((go_bid_sum, go_trick_sum))

    return {
        "our_contracts": our_contracts,
        "go_contracts": go_contracts,
        "our_nils": our_nils,
        "go_nils": go_nils,
    }


def build_2d_array(contracts: list[tuple[int, int]]) -> np.ndarray:
    """构建 (叫墩和, 得墩和) 二维数组。

    行列范围: 0~13 (两个玩家合计叫墩/得墩最多 26, 但实际不会超过 13*2)
    """
    max_val = 26
    arr = np.zeros((max_val + 1, max_val + 1), dtype=int)
    for bid_sum, trick_sum in contracts:
        arr[bid_sum, trick_sum] += 1
    return arr


def print_2d_array(arr: np.ndarray, label: str, total: int) -> None:
    """打印二维数组 (非零行)."""
    print(f"\n=== {label} (叫墩和 × 得墩和) 总和={total} ===")
    print("       ", end="")
    for t in range(min(14, arr.shape[1])):
        print(f"得{t:2d}", end=" ")
    print()
    for b in range(min(14, arr.shape[0])):
        row = arr[b, :14]
        if row.sum() > 0:
            print(f"叫{b:2d}: ", end=" ")
            for v in row:
                print(f"{v:4d}", end=" ")
            print(f"  (共{row.sum()})")


def print_nil_dist(nils: list[int], label: str) -> None:
    """打印 nil 定约 (x/0) 的得墩分布."""
    print(f"\n=== {label} nil 定约 (x/0) 得墩分布 ===")
    if not nils:
        print("  (无)")
        return
    counter = Counter(nils)
    for tricks in sorted(counter):
        print(f"  得 {tricks} 墩: {counter[tricks]} 次")


def plot_results(
    our_arr: np.ndarray,
    go_arr: np.ndarray,
    our_nils: list[int],
    go_nils: list[int],
    save_path: str,
) -> None:
    """绘制热力图和 nil 分布图."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # --- our_mcts heatmap ---
    ax = axes[0, 0]
    max_bid = min(14, our_arr.shape[0])
    max_trick = min(14, our_arr.shape[1])
    im = ax.imshow(our_arr[:max_bid, :max_trick], cmap="YlOrRd", aspect="auto")
    ax.set_title("our_mcts (bid_sum x tricks_sum)", fontsize=13)
    ax.set_xlabel("tricks_sum")
    ax.set_ylabel("bid_sum")
    for b in range(max_bid):
        for t in range(max_trick):
            v = our_arr[b, t]
            if v > 0:
                ax.text(t, b, str(v), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)

    # --- go_rule_2 heatmap ---
    ax = axes[0, 1]
    im = ax.imshow(go_arr[:max_bid, :max_trick], cmap="YlOrRd", aspect="auto")
    ax.set_title("go_rule_2 (bid_sum x tricks_sum)", fontsize=13)
    ax.set_xlabel("tricks_sum")
    ax.set_ylabel("bid_sum")
    for b in range(max_bid):
        for t in range(max_trick):
            v = go_arr[b, t]
            if v > 0:
                ax.text(t, b, str(v), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)

    # --- our_mcts nil distribution ---
    ax = axes[1, 0]
    if our_nils:
        bins = range(max(our_nils) + 2)
        counts, _, patches = ax.hist(our_nils, bins=bins, alpha=0.7, edgecolor="black")
        for count, patch in zip(counts, patches):
            if count > 0:
                ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height(),
                        str(int(count)), ha="center", va="bottom", fontsize=10)
        ax.set_title(f"our_mcts nil tricks (n={len(our_nils)})", fontsize=13)
        ax.set_xlabel("tricks")
        ax.set_ylabel("count")
    else:
        ax.text(0.5, 0.5, "no nil contracts", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("our_mcts nil", fontsize=13)

    # --- go_rule_2 nil distribution ---
    ax = axes[1, 1]
    if go_nils:
        bins = range(max(go_nils) + 2)
        counts, _, patches = ax.hist(go_nils, bins=bins, alpha=0.7, edgecolor="black")
        for count, patch in zip(counts, patches):
            if count > 0:
                ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height(),
                        str(int(count)), ha="center", va="bottom", fontsize=10)
        ax.set_title(f"go_rule_2 nil tricks (n={len(go_nils)})", fontsize=13)
        ax.set_xlabel("tricks")
        ax.set_ylabel("count")
    else:
        ax.text(0.5, 0.5, "no nil contracts", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("go_rule_2 nil", fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n[INFO] 图片已保存到: {save_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python stats_matchup.py <log_file>")
        sys.exit(1)

    log_path = sys.argv[1]
    data = parse_log(log_path)

    our_arr = build_2d_array(data["our_contracts"])
    go_arr = build_2d_array(data["go_contracts"])

    # 校验总和
    our_total = our_arr.sum()
    go_total = go_arr.sum()
    A = len(data["our_contracts"])

    print(f"\n总游戏数: {A}")
    print(f"our_mcts 队伍记录数: {our_total}  (应为 {A})")
    print(f"go_rule_2 队伍记录数: {go_total}  (应为 {A})")

    print_2d_array(our_arr, "our_mcts", our_total)
    print_2d_array(go_arr, "go_rule_2", go_total)

    print_nil_dist(data["our_nils"], "our_mcts")
    print_nil_dist(data["go_nils"], "go_rule_2")

    # 绘制图片
    plot_results(
        our_arr, go_arr,
        data["our_nils"], data["go_nils"],
        save_path=Path(log_path).with_suffix(".png"),
    )


if __name__ == "__main__":
    main()
