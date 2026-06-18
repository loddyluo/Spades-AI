"""统计 is_proposal_stats.txt 中 w>0.1 的分布并绘制条形图。"""

import re
import matplotlib.pyplot as plt

with open("is_proposal_stats.txt") as f:
    lines = f.readlines()

values = []
for line in lines:
    m = re.search(r"w>0\.01=(\d+)", line)
    if m:
        values.append(int(m.group(1)))

print(f"总样本数: {len(values)}")
print(f"最小值: {min(values)}, 最大值: {max(values)}")

bins = [
    (0, 0, "0"),
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 4, "3~4"),
    (5, 8, "5~8"),
    (9, 16, "9~16"),
    (17, 32, "17~32"),
    (33, 64, "33~64"),
    (65, 128, "65~128"),
    (129, 256, "129~256"),
    (257, 512, "257~512"),
    (513, 1234, "513~1234"),
]

labels = []
counts = []
for lo, hi, label in bins:
    cnt = sum(1 for v in values if lo <= v <= hi)
    labels.append(label)
    counts.append(cnt)
    print(f"  {label}: {cnt}")

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(labels, counts, color="steelblue", edgecolor="black")
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
            str(cnt), ha="center", va="bottom", fontsize=9)

ax.set_xlabel("w > 0.1 (log scale)")
ax.set_ylabel("Frequency")
ax.set_title("w > 0.1 Distribution")
ax.set_yscale("log")
fig.tight_layout()
fig.savefig("wgt01_dist.png", dpi=150)
print("\nSaved: wgt01_dist.png")
