
# ---------------------------------------------------------------------------
# Hand-strength estimator (module-level for performance — avoids re-creating
# the function object, weight tuple, and lookup dicts every game iteration)
# ---------------------------------------------------------------------------

_HS_RANK_VAL = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
_HS_ALL_SPADE_VALS = frozenset(range(2, 15))
_HS_MU_SP = 0.6151
_HS_NIL_RAW_THRESHOLD = 2

# v4.2 optimized coefficients (48-dim) — tuple for immutability & faster indexing
_HS_W = (
    0.701044,   # [0]  sp_tricks_c
    0.007511,   # [1]  sp_tricks_hi
    0.881980,   # [2]  sp_tricks_ex
    -1.999145,  # [3]  sp_0
    -1.273403,  # [4]  sp_1
    -0.883269,  # [5]  sp_2
    0.558984,   # [6]  sp_5
    0.429923,   # [7]  sp_6
    0.266541,   # [8]  sp_7p
    0.370726,   # [9]  short_void
    0.488241,   # [10] short_1
    0.220489,   # [11] short_2
    1.476624,   # [12] bias
    2.203615, 1.432062, 0.920113, 0.806798, 0.785587, 0.753077, -0.033648,  # d1
    1.882863, 1.416556, 1.157554, 0.515139, 0.888351, 0.441180, 0.298371,   # d2
    1.877064, 1.811177, 1.169807, 0.934469, 0.978589, 0.675725, 0.260065,   # d34
    1.793521, 1.739455, 1.214013, 0.806248, 0.831830, 0.556648, 0.019048,   # d57
    2.081140, 0.738415, 0.667451, 1.104983, 0.265381, -0.123201, -0.185711, # d8p
)


def _hand_strength(hand_cards):
    """Hand-strength estimator v4.2: 48-dim features with Nil pre-check + NIL_RAW_THRESHOLD.

    Returns (bid: int, raw_strength: float).
    bid is the final recommended bid (0 for Nil hands).
    raw_strength is the linear model output before Nil override.
    """
    suits = {'S': set(), 'H': set(), 'D': set(), 'C': set()}
    for c in hand_cards:
        suits[c[0]].add(c[1])

    # --- Nil pre-check ---
    _nil_ok = False
    sp_count = len(suits['S'])
    if sp_count <= 3:
        _nil_ok = True
        # Rule 1a: no big spades (A/K/Q)
        if any(r in suits['S'] for r in ('A', 'K', 'Q')):
            _nil_ok = False
        # Rule 1c: spade high cards (9/T/J) at most 1
        if _nil_ok:
            sp_vals = [_HS_RANK_VAL[r] for r in suits['S']]
            if sum(1 for v in sp_vals if v >= 9) >= 2:
                _nil_ok = False
        if _nil_ok:
            for suit in ('H', 'D', 'C'):
                vals = sorted([_HS_RANK_VAL[r] for r in suits[suit]])
                d = len(vals)
                if d == 0:
                    continue
                if d <= 3:
                    if d <= 2:
                        if any(v >= 12 for v in vals):
                            _nil_ok = False; break
                    else:
                        if any(v >= 13 for v in vals):
                            _nil_ok = False; break
                    if any(v >= 12 for v in vals):
                        _nil_ok = False; break
                    if sum(1 for v in vals if v >= 9) >= 2:
                        _nil_ok = False; break
                else:
                    if vals[0] > 5 or vals[1] > 9:
                        _nil_ok = False; break

    # --- Spade trick estimate ---
    my_vals = sorted([_HS_RANK_VAL[r] for r in suits['S']], reverse=True)
    opp_vals = sorted(_HS_ALL_SPADE_VALS - set(my_vals), reverse=True)
    sp_tricks = 0
    oi = 0
    for mv in my_vals:
        if oi < len(opp_vals):
            if mv > opp_vals[oi]:
                sp_tricks += 1
            oi += 1
        else:
            sp_tricks += 1

    # --- Build 48-dim feature vector and compute strength ---
    W = _HS_W
    strength = W[12]  # bias

    # [0] mean-centered spade tricks
    strength += W[0] * (sp_tricks - _HS_MU_SP)
    # [1] piecewise-linear above 5
    if sp_tricks > 5:
        strength += W[1] * (sp_tricks - 5)
    # [2] piecewise-linear above 8
    if sp_tricks > 8:
        strength += W[2] * (sp_tricks - 8)

    # [3-8] spade length one-hot (3-4 is baseline)
    sp_len = len(suits['S'])
    if sp_len == 0:
        strength += W[3]
    elif sp_len == 1:
        strength += W[4]
    elif sp_len == 2:
        strength += W[5]
    elif sp_len == 5:
        strength += W[6]
    elif sp_len == 6:
        strength += W[7]
    elif sp_len >= 7:
        strength += W[8]

    # [9-11] shortest side suit
    side_lengths = [len(suits[s]) for s in ('H', 'D', 'C')]
    shortest = min(side_lengths)
    if shortest == 0:
        strength += W[9]
    elif shortest == 1:
        strength += W[10]
    elif shortest == 2:
        strength += W[11]

    # [13-47] Side suits: honour x length-bucket
    _SIDE_BASE = 13
    for suit in ('H', 'D', 'C'):
        cards = suits[suit]
        d = len(cards)
        if d == 0:
            continue
        has_A, has_K, has_Q = 'A' in cards, 'K' in cards, 'Q' in cards
        if has_A and has_K and has_Q:   hon = 0
        elif has_A and has_K:           hon = 1
        elif has_A and has_Q:           hon = 2
        elif has_K and has_Q:           hon = 3
        elif has_A:                     hon = 4
        elif has_K:                     hon = 5
        elif has_Q:                     hon = 6
        else:
            continue
        if d == 1:     bucket = 0
        elif d == 2:   bucket = 1
        elif d <= 4:   bucket = 2
        elif d <= 7:   bucket = 3
        else:          bucket = 4
        strength += W[_SIDE_BASE + bucket * 7 + hon]

    raw = strength
    # Nil override: rule-based Nil AND raw < threshold
    if _nil_ok and raw < _HS_NIL_RAW_THRESHOLD:
        return 0, raw
    return int(max(0, min(13, round(raw)))), raw