#!/usr/bin/env python3
"""Optimize linear bid-strength coefficients for Spades.

Finds coefficients w such that:
    bid(hand) = round(dot(w, features(hand)))

And when dealing 100K random hands:
    P(sum_of_4_bids == 11) > 50%
    P(sum_of_4_bids <= 9 or sum_of_4_bids >= 13) < 10%

Feature vector (44 dimensions):
    [0]     spade_length
    [1-7]   spade honour one-hot: AKQ, AK, AQ, KQ, A, K, Q
    [8]     bias (constant 1)
    [9-43]  side-suit honour x length-bucket (5 buckets x 7 honours)
            buckets: d=1 / d=2 / d=3-4 / d=5-7 / d=8+
            value = count of side suits in that bucket with that honour combo (0-3)
"""

import time
import numpy as np
from scipy.optimize import differential_evolution

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_FEATURES = 44
HONOUR_TYPES = ['AKQ', 'AK', 'AQ', 'KQ', 'A', 'K', 'Q']
BUCKET_NAMES = ['d1', 'd2', 'd34', 'd57', 'd8p']

FEATURE_NAMES = (
    ['spade_len']
    + [f'sp_{h}' for h in HONOUR_TYPES]
    + ['bias']
    + [f'{b}_{h}' for b in BUCKET_NAMES for h in HONOUR_TYPES]
)
assert len(FEATURE_NAMES) == N_FEATURES

# Honour hierarchy pairs: (higher, lower) — w[higher] should >= w[lower]
HONOUR_PAIRS = [(0, 1), (1, 2), (2, 4), (0, 3), (3, 5), (5, 6), (4, 5)]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def classify_honours(ranks):
    has_A = 'A' in ranks
    has_K = 'K' in ranks
    has_Q = 'Q' in ranks
    if has_A and has_K and has_Q: return 0
    if has_A and has_K: return 1
    if has_A and has_Q: return 2
    if has_K and has_Q: return 3
    if has_A: return 4
    if has_K: return 5
    if has_Q: return 6
    return -1


def length_to_bucket(d):
    if d == 1: return 0
    if d == 2: return 1
    if d <= 4: return 2
    if d <= 7: return 3
    return 4


def extract_features(hand):
    suits = {'S': set(), 'H': set(), 'D': set(), 'C': set()}
    for c in hand:
        suits[c[0]].add(c[1])

    f = np.zeros(N_FEATURES, dtype=np.float64)
    f[0] = len(suits['S'])

    sp_hon = classify_honours(suits['S'])
    if sp_hon >= 0:
        f[1 + sp_hon] = 1.0

    f[8] = 1.0

    for suit in ['H', 'D', 'C']:
        d = len(suits[suit])
        if d == 0:
            continue
        hon = classify_honours(suits[suit])
        if hon < 0:
            continue
        idx = 9 + length_to_bucket(d) * 7 + hon
        f[idx] += 1.0

    return f


# ---------------------------------------------------------------------------
# Deal generation
# ---------------------------------------------------------------------------
def generate_all_features(n_deals, seed):
    rng = np.random.RandomState(seed)
    deck = [s + r for s in ['S', 'H', 'D', 'C']
            for r in ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']]

    all_features = np.zeros((n_deals, 4, N_FEATURES), dtype=np.float64)
    d = list(deck)

    for i in range(n_deals):
        rng.shuffle(d)
        for p in range(4):
            all_features[i, p] = extract_features(d[p * 13: (p + 1) * 13])

    return all_features


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_coefficients(w, all_features):
    strengths = all_features @ w
    bids = np.clip(np.round(strengths), 0, 13)
    totals = bids.sum(axis=1)

    N = len(totals)
    p11 = np.sum(totals == 11) / N
    p_extreme = np.sum((totals <= 9) | (totals >= 13)) / N

    dist = {}
    for v in range(int(totals.min()), int(totals.max()) + 1):
        dist[v] = np.sum(totals == v) / N

    bid_dist = {}
    flat_bids = bids.flatten()
    for v in range(int(flat_bids.min()), int(flat_bids.max()) + 1):
        bid_dist[v] = np.sum(flat_bids == v) / len(flat_bids)

    return {
        'p11': p11, 'p_extreme': p_extreme,
        'mean': totals.mean(), 'std': totals.std(),
        'distribution': dist, 'bid_distribution': bid_dist,
    }


# ---------------------------------------------------------------------------
# Objective function (module-level for pickling)
# ---------------------------------------------------------------------------
_SHARED_FEATURES = None  # set before optimization


def objective(w):
    all_features = _SHARED_FEATURES
    strengths = all_features @ w
    bids = np.clip(np.round(strengths), 0, 13)
    totals = bids.sum(axis=1)

    N = len(totals)
    mean_t = totals.mean()
    var_t = totals.var()
    p11 = np.sum(totals == 11) / N
    p_extreme = np.sum((totals <= 9) | (totals >= 13)) / N

    loss = 0.0
    loss += 5.0 * (mean_t - 11.0) ** 2
    loss += 2.0 * var_t
    loss += 20.0 * max(0, 0.55 - p11)
    loss += 20.0 * max(0, p_extreme - 0.08)

    # Spade honour hierarchy
    sp = w[1:8]
    for a, b in HONOUR_PAIRS:
        loss += 3.0 * max(0, sp[b] - sp[a])

    # Spade length positive
    loss += 5.0 * max(0, -w[0])

    # Spade honours >= side-suit honours
    for hon_idx in range(7):
        sp_val = w[1 + hon_idx]
        for bucket in range(5):
            side_val = w[9 + bucket * 7 + hon_idx]
            loss += 1.0 * max(0, side_val - sp_val)

    # Side-suit hierarchy per bucket
    for bucket in range(5):
        base = 9 + bucket * 7
        s = w[base:base + 7]
        for a, b in HONOUR_PAIRS:
            loss += 1.0 * max(0, s[b] - s[a])

    return loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def optimize():
    global _SHARED_FEATURES

    # --- Phase 1: Fast search with fewer deals ---
    n_fast = 30_000
    print(f'Phase 1: Generating {n_fast} deals for fast search...')
    t0 = time.time()
    fast_features = generate_all_features(n_fast, seed=42)
    print(f'  Generated in {time.time()-t0:.1f}s')

    bounds = []
    bounds.append((0.05, 0.8))    # spade_length
    bounds += [(1.5, 3.5), (1.0, 2.8), (0.5, 2.2), (0.3, 2.0),
               (0.5, 1.8), (0.1, 1.2), (0.0, 0.8)]  # spade honours
    bounds.append((0.0, 3.0))     # bias
    for _ in range(5):            # side suit buckets
        bounds += [(-0.3, 2.5), (-0.3, 2.2), (-0.3, 2.0), (-0.3, 1.5),
                   (-0.3, 1.5), (-0.3, 1.0), (-0.3, 0.8)]
    assert len(bounds) == N_FEATURES

    _SHARED_FEATURES = fast_features
    print(f'Phase 1: Running DE (popsize=20, maxiter=500)...')
    t0 = time.time()
    result1 = differential_evolution(
        objective, bounds=bounds,
        seed=42, maxiter=500, tol=1e-6,
        popsize=20, mutation=(0.5, 1.5), recombination=0.9,
        disp=True, workers=-1,
    )
    print(f'  Phase 1 done in {time.time()-t0:.1f}s, loss={result1.fun:.6f}')

    # --- Phase 2: Refine with full dataset ---
    n_full = 100_000
    print(f'\nPhase 2: Generating {n_full} deals for refinement...')
    t0 = time.time()
    full_features = generate_all_features(n_full, seed=42)
    print(f'  Generated in {time.time()-t0:.1f}s')

    _SHARED_FEATURES = full_features

    # Use phase 1 result as seed: create initial population around it
    print(f'Phase 2: Running DE (popsize=15, maxiter=300) seeded from Phase 1...')
    t0 = time.time()
    result2 = differential_evolution(
        objective, bounds=bounds,
        seed=42, maxiter=300, tol=1e-7,
        popsize=15, mutation=(0.3, 1.0), recombination=0.9,
        x0=result1.x,
        disp=True, workers=-1,
    )
    print(f'  Phase 2 done in {time.time()-t0:.1f}s, loss={result2.fun:.6f}')

    w = result2.x

    # --- Results ---
    print('\n' + '=' * 60)
    print('  RESULTS ON TRAINING SET (seed=42, 100K deals)')
    print('=' * 60)
    show_results(w, full_features)

    print('\n' + '=' * 60)
    print('  CROSS-VALIDATION (seed=123, 100K deals)')
    print('=' * 60)
    cv_features = generate_all_features(100_000, seed=123)
    show_results(w, cv_features)

    # --- Coefficients ---
    print('\n' + '=' * 60)
    print('  OPTIMIZED COEFFICIENTS')
    print('=' * 60)
    for i, (name, val) in enumerate(zip(FEATURE_NAMES, w)):
        print(f'  [{i:2d}] {name:12s} = {val:+.4f}')

    print('\n# Copy-paste into code:')
    print('_BID_COEFFICIENTS = [')
    for i, val in enumerate(w):
        print(f'    {val:.6f},  # [{i}] {FEATURE_NAMES[i]}')
    print(']')

    return w


def show_results(w, all_features):
    stats = evaluate_coefficients(w, all_features)
    p11 = stats['p11']
    p_ext = stats['p_extreme']
    ok11 = 'OK' if p11 > 0.50 else 'FAIL'
    ok_ext = 'OK' if p_ext < 0.10 else 'FAIL'
    print(f'  P(sum==11) = {p11:.4f}  (target > 0.50) [{ok11}]')
    print(f'  P(extreme) = {p_ext:.4f}  (target < 0.10) [{ok_ext}]')
    print(f'  Mean total = {stats["mean"]:.2f}')
    print(f'  Std total  = {stats["std"]:.2f}')

    print(f'\n  {"Sum":>5s} {"Pct":>8s}  Bar')
    print(f'  {"---":>5s} {"---":>8s}  ---')
    for s in sorted(stats['distribution'].keys()):
        pct = stats['distribution'][s] * 100
        bar = '#' * int(pct)
        print(f'  {s:>5.0f} {pct:>7.2f}%  {bar}')

    print(f'\n  {"Bid":>5s} {"Pct":>8s}')
    print(f'  {"---":>5s} {"---":>8s}')
    for b in sorted(stats['bid_distribution'].keys()):
        pct = stats['bid_distribution'][b] * 100
        print(f'  {b:>5.0f} {pct:>7.2f}%')


if __name__ == '__main__':
    optimize()
