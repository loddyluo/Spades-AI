"""Plot A/B and A/C ratio distribution from is_proposal_stats.txt.

Buckets: [1,2), [2,4), [4,8), ..., clipped at <1 or >= 2^max_pow.
"""

import re
import matplotlib.pyplot as plt
import numpy as np

MAX_POW = 16  # covers up to 2^16 = 65536, adjust if needed

def parse_ratios(path):
    ab_vals, ac_vals = [], []
    pat = re.compile(r"A/B=([\d.]+)\s+A/C=([\d.]+)")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                ab_vals.append(float(m.group(1)))
                ac_vals.append(float(m.group(2)))
    return np.array(ab_vals), np.array(ac_vals)

def bucket_dist(values, max_pow):
    """Bin values into buckets: <1, [1,2), [2,4), ..., [2^{max_pow-1}, 2^{max_pow}), >=2^{max_pow}."""
    bins = [2**i for i in range(max_pow + 1)]
    # prepend -inf for <1, append +inf for >=2^max_pow
    bin_edges = [-np.inf] + bins + [np.inf]
    labels = ["<1"] + [f"[{2**i},{2**(i+1)})" for i in range(max_pow)] + [f">={2**max_pow}"]
    counts, _ = np.histogram(values, bins=bin_edges)
    return labels, counts

def plot(name, values, max_pow):
    labels, counts = bucket_dist(values, max_pow)
    # trim trailing zero bins for readability
    while len(counts) > 1 and counts[-1] == 0:
        counts = counts[:-1]
        labels = labels[:-1]

    x = np.arange(len(counts))
    fig, ax = plt.subplots(figsize=(max(8, len(counts) * 0.5), 5))
    ax.bar(x, counts, width=0.7, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {name} (N={len(values)})")
    for xi, c in zip(x, counts):
        if c > 0:
            ax.text(xi, c + 0.3, str(c), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{name.replace('/','_')}_dist.png", dpi=150)
    print(f"Saved {name.replace('/','_')}_dist.png")
    plt.close()

ab, ac = parse_ratios("is_proposal_stats.txt")
print(f"Parsed {len(ab)} lines")
plot("A/B", ab, MAX_POW)
plot("A/C", ac, MAX_POW)
