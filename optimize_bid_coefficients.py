#!/usr/bin/env python3
"""Optimize linear bid-strength coefficients for Spades (v4).

Uses Nil pre-check + mean-centered spade trick estimation + piecewise-linear
basis functions + fine-grained spade length features + side-suit coefficients.

Feature vector (48 dimensions):
    [0]     sp_tricks_centered — spade_trick_estimate() - MU_SP
    [1]     sp_tricks_high     — max(0, sp_tricks - 5)
    [2]     sp_tricks_extreme  — max(0, sp_tricks - 8)
    [3]     sp_0               — 1 if spade length == 0
    [4]     sp_1               — 1 if spade length == 1
    [5]     sp_2               — 1 if spade length == 2
    [6]     sp_5               — 1 if spade length == 5
    [7]     sp_6               — 1 if spade length == 6
    [8]     sp_7p              — 1 if spade length >= 7
    [9]     short_side_void    — 1 if shortest side suit == 0
    [10]    short_side_1       — 1 if shortest side suit == 1
    [11]    short_side_2       — 1 if shortest side suit == 2
    [12]    bias               — constant 1
    [13-47] side-suit honour x length-bucket (5 buckets x 7 honours)
"""

import time
import numpy as np
from scipy.optimize import differential_evolution

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_FEATURES = 48
MU_SP = 0.6151  # empirically computed over 2M hands (seed=42)

HONOUR_TYPES = ['AKQ', 'AK', 'AQ', 'KQ', 'A', 'K', 'Q']
BUCKET_NAMES = ['d1', 'd2', 'd34', 'd57', 'd8p']

SIDE_SUIT_BASE = 13  # first side-suit feature index

FEATURE_NAMES = (
    ['sp_tricks_c', 'sp_tricks_hi', 'sp_tricks_ex',
     'sp_0', 'sp_1', 'sp_2', 'sp_5', 'sp_6', 'sp_7p',
     'short_void', 'short_1', 'short_2', 'bias']
    + [f'{b}_{h}' for b in BUCKET_NAMES for h in HONOUR_TYPES]
)
assert len(FEATURE_NAMES) == N_FEATURES

RANK_VAL = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
            '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
ALL_SPADE_VALS = set(range(2, 15))

# Honour hierarchy pairs (index within 7-tuple): higher >= lower
HONOUR_PAIRS = [(0, 1), (1, 2), (2, 4), (0, 3), (3, 5), (5, 6), (4, 5)]

# All 43 labeled hands (batch 1 + batch 2 + batch 3)
LABELED_HANDS = [
    # Batch 1 (13 hands)
    (['SJ','S6','S3','HK','HQ','HJ','H8','H5','H2','D5','D3','CJ','C3'], 1),
    (['SA','SK','SJ','S6','S3','H5','DQ','DJ','D5','D3','CQ','C7','C5'], 4),
    (['ST','S6','S4','H5','H2','DA','DK','DT','D8','D7','CQ','C6','C5'], 2),
    (['ST','S8','HQ','HT','H9','H8','H7','H3','DQ','D4','CQ','C7','C5'], 1),
    (['SA','SK','SQ','SJ','S9','S8','S6','S4','S3','HA','H8','D3','C6'], 10),
    (['S7','S6','S4','HA','DA','DK','DJ','D7','D5','CA','CJ','C5','C4'], 5),
    (['SK','S8','S6','HA','H8','DA','D9','D8','D6','CK','CQ','C9','C2'], 4),
    (['ST','S5','S3','HT','H8','H5','H4','H3','D9','D5','D3','CT','C7'], 0),
    (['SK','S9','S3','S2','HK','H5','H2','DA','DJ','D4','D3','CQ','CT'], 3),
    (['SA','SJ','S7','HK','HT','H8','H6','H5','D8','D6','D3','D2','C4'], 2),
    (['SA','SK','S8','S6','HK','HJ','H7','H2','DA','D7','D4','C8','C3'], 4),
    (['SA','SK','S9','S8','HA','HK','H8','H5','H2','DA','D9','C8','C2'], 5),
    (['S9','S8','HT','H9','H6','DT','D7','D5','D4','CJ','C8','C6','C5'], 0),
    # Batch 2 (15 hands)
    (['HA','D5','C6','HT','DA','D6','D2','H5','D8','D9','DK','DQ','H3'], 0),
    (['H8','C5','C3','HK','C9','HQ','C8','D8','H6','H4','S4','DJ','D2'], 0),
    (['D4','S5','D7','DQ','D2','HJ','D6','D5','H8','H4','SA','HK','H3'], 2),
    (['HT','H6','SJ','CT','D4','H5','HA','SK','HK','D3','C4','D5','C8'], 3),
    (['SQ','SJ','H2','DT','C2','D4','C5','DA','D8','C9','CJ','HT','S5'], 2),
    (['SA','H3','SK','CQ','DQ','C6','H9','HA','DK','HQ','ST','C4','DA'], 5),
    (['H6','H2','DT','C3','D8','S6','SA','DQ','C6','HJ','D3','S4','ST'], 2),
    (['H2','S3','S7','D6','CA','S5','C5','D8','H5','D4','DK','H9','S6'], 2),
    (['HT','DA','D2','D6','S4','S7','H9','HK','D8','S5','S3','HQ','S2'], 4),
    (['C4','S8','S6','C9','D5','D4','SJ','H7','S5','SA','C3','HA','D8'], 3),
    (['H5','DK','SQ','C7','DQ','SA','S7','HK','S8','S6','H9','S4','S2'], 6),
    (['S4','HJ','C4','H2','DA','ST','S3','SQ','DK','S2','S7','CA','SA'], 7),
    (['CQ','D4','H4','CA','HA','C8','S3','D3','HJ','C9','S9','C6','S8'], 2),
    (['ST','C5','C8','H8','C3','S5','HJ','C9','D9','H4','S3','HT','C2'], 0),
    (['D4','DQ','CQ','D9','SQ','DK','HT','SA','C3','HK','C2','C9','SJ'], 4),
    # Batch 3 (15 hands)
    (['ST','S8','S5','S3','S2','HQ','H7','DQ','D5','CJ','C7','C6','C3'], 2),
    (['S5','S2','HJ','H6','H2','D9','D8','D6','D5','D3','D2','C8','C3'], 0),
    (['SJ','ST','S9','S3','HA','HT','H9','H7','H4','D3','D2','C3','C2'], 2),
    (['SK','SJ','S9','S8','S7','S2','H9','H6','H3','H2','D7','CK','C2'], 4),
    (['SK','SQ','ST','HT','H9','H8','H7','DK','D7','D6','D4','C7','C2'], 2),
    (['SQ','S5','HK','HJ','H7','H4','DA','DT','D8','D5','CK','C6','C5'], 2),
    (['S9','S8','S7','S2','HT','H8','H2','DQ','D8','D3','C8','C5','C4'], 1),
    (['SA','SK','SQ','ST','S6','DA','D6','D3','CT','C9','C6','C4','C3'], 6),
    (['SK','ST','S8','S4','HQ','H5','H4','DA','DK','D9','D6','D5','C8'], 4),
    (['SA','SK','ST','S6','S3','S2','H6','H5','DK','DQ','DT','C7','C2'], 5),
    (['SA','SK','S7','S4','H8','H4','H3','DA','DK','DQ','D9','D4','D2'], 5),
    (['S7','S5','S2','HK','HT','H7','H6','H2','DJ','D4','CK','C9','C6'], 1),
    (['SQ','SJ','ST','S6','S4','S3','S2','HJ','H9','H5','D4','CQ','C4'], 4),
    (['SA','SQ','SJ','S9','HA','H5','DA','DK','D8','CQ','C7','C5','C2'], 6),
    (['SA','SK','SQ','HK','HQ','HT','H4','D7','D4','CA','CQ','C7','C4'], 5),
    # Batch 4 (28 hands — 7 deals × 4 players, extreme-sum cases)
    # Deal #21 (user: 4 2 3 2)
    (['SQ','ST','S8','S2','HA','H4','DK','DJ','DT','D6','D4','C9','C4'], 4),
    (['SA','HT','H8','H3','H2','DA','D8','D5','CK','CQ','CJ','C6','C3'], 2),
    (['SK','S9','S7','S6','HJ','H9','H7','H5','DQ','D2','CA','CT','C8'], 3),
    (['SJ','S5','S4','S3','HK','HQ','H6','D9','D7','D3','C7','C5','C2'], 2),
    # Deal #13430 (user: 2 4 3 2)
    (['S9','S8','S5','HQ','H7','H5','H4','DJ','DT','D5','CA','CK','CT'], 2),
    (['SQ','ST','S7','S4','HK','HJ','H9','H3','H2','DA','D7','C6','C5'], 4),
    (['SA','S6','HA','HT','H6','DK','DQ','D8','D4','D3','CQ','C4','C3'], 3),
    (['SK','SJ','S3','S2','H8','D9','D6','D2','CJ','C9','C8','C7','C2'], 2),
    # Deal #25369 (user: 4 2 4 2)
    (['SQ','SJ','S5','S2','HK','DA','DQ','D9','D8','D5','D4','D2','C8'], 4),
    (['SK','HA','HJ','H9','H8','H7','H4','H2','DK','DT','D6','C4','C3'], 2),
    (['ST','S8','S7','S4','HQ','H5','H3','D3','CA','CK','C6','C5','C2'], 4),
    (['SA','S9','S6','S3','HT','H6','DJ','D7','CQ','CJ','CT','C9','C7'], 2),
    # Deal #37588 (user: 5 2 1 2)  — P2 was NIL
    (['SK','ST','S8','S4','S2','HK','H7','H4','DA','DK','D6','D3','C6'], 5),
    (['SJ','S3','HJ','H6','H3','DJ','D5','D4','CA','CK','CJ','C3','C2'], 2),
    (['SQ','S5','HA','H8','H5','D9','D7','D2','CQ','C9','C8','C7','C4'], 1),
    (['SA','S9','S7','S6','HQ','HT','H9','H2','DQ','DT','D8','CT','C5'], 2),
    # Deal #7 (user: 1 5 2 3)
    (['S9','S7','HJ','HT','H9','H7','H6','H4','DK','DJ','D8','CK','C3'], 1),
    (['SA','SK','SJ','ST','S5','S2','H8','D9','D5','D4','D3','CJ','C9'], 5),
    (['SQ','S8','S4','HQ','H5','H3','H2','DA','DQ','DT','CT','C5','C2'], 2),
    (['S6','S3','HA','HK','D7','D6','D2','CA','CQ','C8','C7','C6','C4'], 3),
    # Deal #37764 (user: 3 2 2 4)
    (['S6','HQ','HT','H6','DA','DK','DT','D9','D6','CA','CT','C8','C5'], 3),
    (['SA','ST','S9','HK','H8','H3','H2','DJ','D5','CQ','C7','C4','C2'], 2),
    (['S7','S4','S3','HA','H9','DQ','D8','D4','D3','D2','CK','CJ','C3'], 2),
    (['SK','SQ','SJ','S8','S5','S2','HJ','H7','H5','H4','D7','C9','C6'], 4),
    # Deal #31 (user: 2 9 1 0)  — P4 was NIL
    (['S8','S3','HA','H7','H4','DA','DJ','D9','D8','D7','D5','CQ','C5'], 2),
    (['SA','SK','SQ','S9','S7','S6','S4','H2','DQ','CA','CK','CJ','CT'], 9),
    (['ST','S2','HQ','HT','H6','H5','DK','D6','D2','C9','C8','C6','C4'], 1),
    (['SJ','S5','HK','HJ','H9','H8','H3','DT','D4','D3','C7','C3','C2'], 0),
]


# ---------------------------------------------------------------------------
# Nil pre-check (rule-based, before linear model)
# ---------------------------------------------------------------------------
def is_nil_hand(hand):
    """Return True if this hand should bid 0 by rule."""
    suits = {'S': set(), 'H': set(), 'D': set(), 'C': set()}
    for c in hand:
        suits[c[0]].add(c[1])

    # Rule 3: spades <= 3
    if len(suits['S']) > 3:
        return False

    # Rule 1a: no big spades (A/K/Q)
    if any(r in suits['S'] for r in ['A', 'K', 'Q']):
        return False

    # Rule 1c: spade high cards (9/T/J) at most 1
    sp_vals = [RANK_VAL[r] for r in suits['S']]
    if sum(1 for v in sp_vals if v >= 9) >= 2:
        return False

    # Check each side suit
    for suit in ['H', 'D', 'C']:
        vals = sorted([RANK_VAL[r] for r in suits[suit]])
        d = len(vals)
        if d == 0:
            continue
        if d <= 3:  # short suit
            # Rule 1b: short(<=2) no AKQ; short(=3) no AK
            if d <= 2:
                if any(v >= 12 for v in vals):  # Q=12
                    return False
            else:  # d == 3
                if any(v >= 13 for v in vals):  # K=13
                    return False
            # Rule 2a: short suit no card >= Q; no 2 cards >= 9
            if any(v >= 12 for v in vals):
                return False
            if sum(1 for v in vals if v >= 9) >= 2:
                return False
        else:  # long suit (>= 4)
            # Rule 2b: min <= 5, second-min <= 9
            if vals[0] > 5 or vals[1] > 9:
                return False

    return True


# ---------------------------------------------------------------------------
# Spade trick estimation (rule-based)
# ---------------------------------------------------------------------------
def spade_trick_estimate(spade_ranks):
    """Simulated confrontation: estimate spade tricks."""
    my_vals = sorted([RANK_VAL[r] for r in spade_ranks], reverse=True)
    opp_vals = sorted(ALL_SPADE_VALS - set(my_vals), reverse=True)

    wins = 0
    oi = 0
    for mv in my_vals:
        if oi < len(opp_vals):
            if mv > opp_vals[oi]:
                wins += 1
            oi += 1
        else:
            wins += 1
    return wins


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

    sp_tricks = spade_trick_estimate(suits['S'])

    # [0] mean-centered spade tricks
    f[0] = sp_tricks - MU_SP

    # [1] piecewise-linear: extra credit above 5 tricks
    f[1] = max(0, sp_tricks - 5)

    # [2] piecewise-linear: extra credit above 8 tricks
    f[2] = max(0, sp_tricks - 8)

    # [3-8] spade length one-hot (3-4 is baseline, no feature)
    sp_len = len(suits['S'])
    if sp_len == 0:
        f[3] = 1.0
    elif sp_len == 1:
        f[4] = 1.0
    elif sp_len == 2:
        f[5] = 1.0
    elif sp_len == 5:
        f[6] = 1.0
    elif sp_len == 6:
        f[7] = 1.0
    elif sp_len >= 7:
        f[8] = 1.0

    # [9-11] shortest side suit
    side_lengths = [len(suits[s]) for s in ['H', 'D', 'C']]
    shortest = min(side_lengths)
    if shortest == 0:
        f[9] = 1.0
    elif shortest == 1:
        f[10] = 1.0
    elif shortest == 2:
        f[11] = 1.0

    # [12] bias
    f[12] = 1.0

    # [13-47] side suit honours x length buckets
    for suit in ['H', 'D', 'C']:
        d = len(suits[suit])
        if d == 0:
            continue
        hon = classify_honours(suits[suit])
        if hon < 0:
            continue
        idx = SIDE_SUIT_BASE + length_to_bucket(d) * 7 + hon
        f[idx] += 1.0

    return f


NIL_RAW_THRESHOLD = 3  # If raw linear bid >= this, don't risk Nil


def bid_with_nil(hand, w):
    """Compute bid: Nil check first, then linear model."""
    f = extract_features(hand)
    raw = f @ w
    if is_nil_hand(hand) and raw < NIL_RAW_THRESHOLD:
        return 0
    return int(np.clip(round(raw), 0, 13))


# ---------------------------------------------------------------------------
# Deal generation (with Nil flags)
# ---------------------------------------------------------------------------
def generate_deals(n_deals, seed):
    """Return (features[n,4,44], nil_flags[n,4]) arrays."""
    rng = np.random.RandomState(seed)
    deck = [s + r for s in ['S', 'H', 'D', 'C']
            for r in ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']]

    all_features = np.zeros((n_deals, 4, N_FEATURES), dtype=np.float64)
    nil_flags = np.zeros((n_deals, 4), dtype=bool)
    d = list(deck)

    for i in range(n_deals):
        rng.shuffle(d)
        for p in range(4):
            hand = d[p * 13: (p + 1) * 13]
            all_features[i, p] = extract_features(hand)
            nil_flags[i, p] = is_nil_hand(hand)

    return all_features, nil_flags


# ---------------------------------------------------------------------------
# Precompute labeled hand data
# ---------------------------------------------------------------------------
_LABELED_FEATURES = None
_LABELED_BIDS = None
_LABELED_NIL = None  # which labeled hands are Nil


def _init_labeled():
    global _LABELED_FEATURES, _LABELED_BIDS, _LABELED_NIL
    feats = []
    bids = []
    nils = []
    for hand, human_bid in LABELED_HANDS:
        feats.append(extract_features(hand))
        bids.append(human_bid)
        nils.append(is_nil_hand(hand))
    _LABELED_FEATURES = np.array(feats)
    _LABELED_BIDS = np.array(bids, dtype=np.float64)
    _LABELED_NIL = np.array(nils, dtype=bool)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_coefficients(w, all_features, nil_flags):
    strengths = all_features @ w
    bids = np.clip(np.round(strengths), 0, 13)
    # Override Nil hands to bid 0 only if raw strength < threshold
    effective_nil = nil_flags & (strengths < NIL_RAW_THRESHOLD)
    bids[effective_nil] = 0.0
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
_SHARED_FEATURES = None
_SHARED_NIL = None


def objective(w):
    all_features = _SHARED_FEATURES
    nil_flags = _SHARED_NIL

    strengths = all_features @ w
    bids = np.clip(np.round(strengths), 0, 13)
    effective_nil = nil_flags & (strengths < NIL_RAW_THRESHOLD)
    bids[effective_nil] = 0.0
    totals = bids.sum(axis=1)

    N = len(totals)
    mean_t = totals.mean()
    var_t = totals.var()

    loss = 0.0
    # (A) Mean constraint — keep mean near 11
    loss += 10.0 * (mean_t - 11.0) ** 2
    # (B) Variance constraint — prefer tight distribution
    loss += 2.0 * var_t

    # (C) Soft penalty pulling sp_tricks_centered coeff toward 1.0
    loss += 2.0 * (w[0] - 1.0) ** 2

    # (D) Side-suit honour hierarchy within each bucket
    for bucket in range(5):
        base = SIDE_SUIT_BASE + bucket * 7
        s = w[base:base + 7]
        for a, b in HONOUR_PAIRS:
            loss += 1.0 * max(0, s[b] - s[a])

    # (E) Supervised loss on labeled hands — primary objective
    if _LABELED_FEATURES is not None:
        raw_labeled = _LABELED_FEATURES @ w
        # Effective Nil: rule says Nil AND raw < threshold
        eff_nil = _LABELED_NIL & (raw_labeled < NIL_RAW_THRESHOLD)
        pred = np.clip(np.round(raw_labeled), 0, 13)
        pred[eff_nil] = 0.0
        labeled_mse = np.mean((pred - _LABELED_BIDS) ** 2)
        loss += 5.0 * labeled_mse

    return loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def optimize():
    global _SHARED_FEATURES, _SHARED_NIL

    _init_labeled()

    # --- Phase 1: Fast search ---
    n_fast = 30_000
    print(f'Phase 1: Generating {n_fast} deals for fast search...')
    t0 = time.time()
    fast_features, fast_nil = generate_deals(n_fast, seed=42)
    nil_pct = fast_nil.mean() * 100
    print(f'  Generated in {time.time()-t0:.1f}s  (Nil rate: {nil_pct:.1f}%)')

    bounds = []
    bounds.append((0.7, 1.3))     # [0] sp_tricks_centered
    bounds.append((0.0, 1.0))     # [1] sp_tricks_high
    bounds.append((0.0, 1.5))     # [2] sp_tricks_extreme
    bounds.append((-2.0, 0.0))    # [3] sp_0  (short spade → reduce bid)
    bounds.append((-1.5, 0.0))    # [4] sp_1
    bounds.append((-1.0, 0.0))    # [5] sp_2
    bounds.append((0.0, 1.5))     # [6] sp_5  (long spade → increase bid)
    bounds.append((0.0, 2.0))     # [7] sp_6
    bounds.append((0.0, 3.0))     # [8] sp_7p
    bounds.append((-0.5, 2.0))    # [9] short_side_void
    bounds.append((-0.5, 1.5))    # [10] short_side_1
    bounds.append((-0.5, 1.0))    # [11] short_side_2
    bounds.append((0.5, 3.5))     # [12] bias
    for _ in range(5):            # [13-47] side suit buckets
        bounds += [(-0.3, 2.5), (-0.3, 2.2), (-0.3, 2.0), (-0.3, 1.5),
                   (-0.3, 1.5), (-0.3, 1.0), (-0.3, 0.8)]
    assert len(bounds) == N_FEATURES

    _SHARED_FEATURES = fast_features
    _SHARED_NIL = fast_nil
    print(f'Phase 1: Running DE (popsize=25, maxiter=800)...')
    t0 = time.time()
    result1 = differential_evolution(
        objective, bounds=bounds,
        seed=42, maxiter=800, tol=1e-6,
        popsize=25, mutation=(0.5, 1.5), recombination=0.9,
        disp=True, workers=-1,
    )
    print(f'  Phase 1 done in {time.time()-t0:.1f}s, loss={result1.fun:.6f}')

    # --- Phase 2: Refine with full dataset ---
    n_full = 100_000
    print(f'\nPhase 2: Generating {n_full} deals for refinement...')
    t0 = time.time()
    full_features, full_nil = generate_deals(n_full, seed=42)
    print(f'  Generated in {time.time()-t0:.1f}s')

    _SHARED_FEATURES = full_features
    _SHARED_NIL = full_nil
    print(f'Phase 2: Running DE (popsize=20, maxiter=500) seeded from Phase 1...')
    t0 = time.time()
    result2 = differential_evolution(
        objective, bounds=bounds,
        seed=42, maxiter=500, tol=1e-7,
        popsize=20, mutation=(0.3, 1.0), recombination=0.9,
        x0=result1.x,
        disp=True, workers=-1,
    )
    print(f'  Phase 2 done in {time.time()-t0:.1f}s, loss={result2.fun:.6f}')

    w = result2.x

    # --- Results ---
    print('\n' + '=' * 60)
    print('  RESULTS ON TRAINING SET (seed=42, 100K deals)')
    print('=' * 60)
    show_results(w, full_features, full_nil)

    print('\n' + '=' * 60)
    print('  CROSS-VALIDATION (seed=123, 100K deals)')
    print('=' * 60)
    cv_features, cv_nil = generate_deals(100_000, seed=123)
    show_results(w, cv_features, cv_nil)

    # --- Coefficients ---
    print('\n' + '=' * 60)
    print('  OPTIMIZED COEFFICIENTS')
    print('=' * 60)
    for i, (name, val) in enumerate(zip(FEATURE_NAMES, w)):
        print(f'  [{i:2d}] {name:14s} = {val:+.4f}')

    print(f'\nMU_SP = {MU_SP}')
    print('\n# Copy-paste into code:')
    print('_W = [')
    for i, val in enumerate(w):
        print(f'    {val:.6f},  # [{i}] {FEATURE_NAMES[i]}')
    print(']')

    # --- Verify on labeled hands ---
    print('\n' + '=' * 60)
    print('  VERIFICATION ON LABELED HANDS')
    print('=' * 60)
    print(f'  {"#":>3s}  {"Func":>5s}  {"Human":>5s}  {"Diff":>5s}  {"Nil?":>5s}')
    print(f'  {"---":>3s}  {"-----":>5s}  {"-----":>5s}  {"-----":>5s}  {"-----":>5s}')
    total_err = 0
    exact = 0
    for i, (hand, human) in enumerate(LABELED_HANDS, 1):
        func_bid = bid_with_nil(hand, w)
        nil_flag = is_nil_hand(hand)
        diff = func_bid - human
        total_err += abs(diff)
        if diff == 0:
            exact += 1
        marker = '' if diff == 0 else ' *' if abs(diff) == 1 else ' **'
        print(f'  {i:>3d}  {func_bid:>5d}  {human:>5d}  {diff:>+5d}{marker}  {"Y" if nil_flag else "":>5s}')
    n = len(LABELED_HANDS)
    print(f'\n  MAE = {total_err/n:.2f}, exact = {exact}/{n}')

    return w


def show_results(w, all_features, nil_flags):
    stats = evaluate_coefficients(w, all_features, nil_flags)
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
