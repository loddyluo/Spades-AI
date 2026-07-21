# Residual Bidder Training Design

> Superseded on 2026-07-21 by
> [Stochastic Hybrid Residual Bidder Training Design](2026-07-21-stochastic-hybrid-residual-bidder-design.md).
> Do not use this document as the current specification.

Date: 2026-07-17

## 1. Objective

Train one new acting bidder that is shared by both seats of a candidate partnership and is optimized for the repository's existing fixed card-play pipeline. The primary objective is expected duplicate/team-match score margin, not DDS trick prediction, bid imitation, contract success rate, or ordinary classification accuracy.

The deployed acting bidder is a deterministic composite policy:

1. frozen `bid_nsfp.pt` produces the incumbent bid;
2. a newly trained residual-Q ensemble may change that bid by at most one;
3. a fixed decision rule chooses the final legal action.

The same composite policy is used at every seat where the new bidder is selected. A training or evaluation opponent uses one frozen opponent bidder for both seats of its partnership.

## 2. Frozen and Out-of-Scope Components

The following components are frozen throughout bidder training and deployment:

- the current card-play models and their control flow;
- the exact/endgame search process;
- the existing `bid_nsfp.pt` checkpoint used by endgame importance-sampling belief weighting;
- the existing `bid_nsfp.pt` checkpoint and `BidEncoder` path used to produce the acting bidder's center action;
- game rules and scoring.

Only the acting bidder gains a residual correction. The belief bidder is not replaced by the residual model.

The following are out of scope for the first implementation:

- blind nil;
- normal bid zero as a deployable action;
- changing the play process to make training easier;
- changing or "correcting" the frozen NSFP encoder or its bridge inputs;
- using `bid_dds.pt` in the residual model;
- searching actions farther than one from the NSFP action;
- a global-action audit outside the local action neighborhood;
- direct PPO, REINFORCE, or actor-critic fine-tuning of NSFP.

## 3. Why NSFP Is Frozen Instead of Fine-Tuned

Activation measurements on 12,000 current-path auction states found that NSFP remains usable but is poorly suited to direct policy fine-tuning:

- mean softmax entropy is approximately 0.13;
- mean top-one probability is approximately 0.95;
- the third hidden layer's activation variation is dominated by one direction;
- the probability assigned to local alternative actions is very small for most states;
- policy-gradient signal for alternatives is therefore absent in many high-confidence states;
- forced supervised changes can instead create large gradient spikes.

NSFP is nevertheless a strong incumbent and its deterministic `BidEncoder` contains useful hand and auction features. The design therefore reuses the exact 149-dimensional encoder output but does not reuse or train NSFP hidden activations.

## 4. Legal Actions and Hard Residual Constraint

The legal acting action space is:

```text
Nil, bid_1, bid_2, ..., bid_13
```

For neighborhood construction only, Nil is assigned numeric value zero. If NSFP's normalized center action is `b0`, the legal candidates are:

| NSFP center | Candidate actions |
|---|---|
| Nil | Nil, 1 |
| 1 | Nil, 1, 2 |
| 2 through 12 | b0-1, b0, b0+1 |
| 13 | 12, 13 |

No training or inference path may select an action outside this table.

The hard neighborhood is a product requirement, not a provisional search heuristic. There is no full-action fallback or diagnostic audit.

## 5. NSFP Legal-Action Scores

NSFP has 16 raw logits even though the deployed action space has 14 actions. To preserve the current normalization behavior, construct legal-action scores as follows:

- legal Nil score: maximum of raw Nil and raw Blind-Nil logits;
- legal bid-1 score: maximum of raw normal-0 and normal-1 logits;
- legal bid-k score for k=2 through 13: raw normal-k logit.

The center action is the argmax of these 14 legal-action scores. This is equivalent to applying the current raw-argmax action normalization while also providing one unambiguous score for every legal action.

For each legal local alternative, provide a normalized NSFP margin:

```text
m_minus = (score(center) - score(lower)) / 13.47
m_plus  = (score(center) - score(upper)) / 13.47
```

The value 13.47 is the measured temperature that raises NSFP's average entropy to the DDS checkpoint's average entropy without changing NSFP argmax actions. An unavailable alternative receives margin zero and mask zero.

## 6. Residual-Q Input

The residual model receives exactly 167 values:

1. the exact 149-dimensional `BidEncoder` tensor seen by frozen NSFP;
2. a 14-dimensional one-hot encoding of the normalized NSFP center action;
3. two normalized local margins, `m_minus` and `m_plus`;
4. two legality masks for the lower and upper alternatives.

The existing 149-dimensional encoder and the current state-to-NSFP bridge are not changed. This preserves the checkpoint's learned input protocol, including any legacy feature semantics.

The residual model must not receive:

- another player's hand;
- the complete deal;
- future bids;
- completed-play information;
- DDS results;
- NSFP hidden-layer activations.

## 7. Residual-Q Outputs and Targets

The residual model predicts two conditional advantages:

```text
A_minus(o) = E[R(lower) - R(center) | visible observation o]
A_plus(o)  = E[R(upper) - R(center) | visible observation o]
```

The center action has advantage exactly zero by definition.

For one full deal `d`, one visible state `o`, and one legal alternative, the training target is a single realized score difference:

```text
y_delta = R(d, forced alternative, continuation policy)
        - R(d, forced center, continuation policy)
```

The target is not the full-deal argmax action. Regressing realized advantages from many deals lets the model estimate the conditional expectation without leaking hidden hands into inference. Hard-labeling each deal's best action would instead imitate a full-information oracle and is explicitly prohibited.

## 8. Reward and Duplicate Pairing

For one deal `d` and frozen league opponent `l`, play two rooms with partnership ownership swapped. The candidate's paired match margin is:

```text
M(policy, l, d)
  = ((candidate_score - opponent_score)_room_1
   + (candidate_score - opponent_score)_room_2) / 2
```

The score is the repository's existing raw team score difference. It is not transformed into a classification target or an alternative trick-based reward.

For a counterfactual at one state, the other duplicate room is constant across the local alternatives and therefore cancels. The label uses the affected room's score difference. Both swapped rooms still generate their own candidate-team states for the dataset.

## 9. One Rollout per Deal and Action

Formal training uses one rollout for each legal local action in each encountered state. It does not repeat the same full deal to estimate an average.

Each different full deal is one Monte Carlo sample of hidden information and play-process randomness. Budget is spent on more unique deals, not repeated simulations of the same deal.

For reproducibility and variance reduction, all action branches from one state use deterministic corresponding seed bundles. A branch must be recreated from the saved game state and seed bundle; mutable player or RNG instances must not leak from one branch into another.

Repeated execution of the same deal is allowed only for deterministic test/debug checks and is not part of the training dataset.

## 10. Efficient Data Generation per Duplicate Deal

One duplicate deal produces at most four candidate-team bidding states: two candidate calls in each of two swapped rooms.

Each room's on-policy center rollout is shared as `R(center)` for both candidate states on that baseline auction path. Each state then branches only for its legal lower and upper alternatives.

The usual upper bound is therefore:

- two center games, one per room;
- up to eight alternative games, two per candidate state;
- at most ten complete games per duplicate deal.

Nil and bid-13 boundaries require fewer branches.

Every state, room, and action branch from one deal must remain in the same train/development/promotion/test partition.

## 11. Iterative Fitted Policy Improvement

Let `pi_0` be frozen NSFP with no residual correction. Training proceeds in accepted policy iterations.

For iteration `i`:

1. freeze the full incumbent acting policy `pi_i`;
2. generate on-policy duplicate deals against the opponent league;
3. at every candidate-team bidding state, force all legal local actions;
4. after the forced action, use frozen `pi_i` for every later candidate-team bid;
5. use the selected frozen league bidder for every opponent-team bid;
6. train a candidate residual ensemble to estimate advantages under this fixed continuation;
7. tune the conservative decision parameter only on development deals;
8. compare candidate `pi_(i+1)` with `pi_i` on fresh promotion deals;
9. accept and add the candidate to the league only if promotion criteria pass.

The acting policy must not change while one iteration's labels are being generated.

Old states may be retained, but labels generated under an older continuation policy may not be mixed unmarked with current-policy targets. Reusing an old state requires rerunning its branches with the current frozen continuation policy.

## 12. Opponent League

Every opponent league member is immutable once added. Both opponent seats in one partnership use the same member.

League sampling is:

- before any residual snapshot is accepted: 100% NSFP;
- with exactly one accepted residual snapshot: 50% NSFP and 50% latest snapshot;
- with at least two accepted snapshots:
  - 50% NSFP;
  - 25% latest accepted snapshot;
  - 25% uniformly sampled from older accepted snapshots.

The same opponent member is used for every local action branch of a state and for both rooms of the duplicate pair.

## 13. Natural and Stratified Data Pools

Maintain two pools:

### Natural pool

Keep every candidate state produced by unfiltered on-policy duplicate deals. This pool represents the deployment distribution and supplies the primary training and development objective.

### Stratified reservoir

Scan additional valid random deals and retain naturally occurring underrepresented states. Priority strata are:

1. NSFP center action: Nil, 1, 2, 3, 4, 5, 6, and 7+;
2. bidding position: first through fourth;
3. whether Nil has already appeared in the auction;
4. whether the partner bid is visible;
5. opponent league member.

Do not synthesize or edit hands to fill strata. Do not select states based on which forced action happened to win.

Every stratified sample records its natural-frequency estimate, sampling probability, and importance weight. Training may oversample the reservoir, but the weighted loss must target the natural on-policy distribution. Promotion and final testing use only unfiltered duplicate deals.

## 14. Residual Ensemble Architecture

The deployed residual estimator contains five independently initialized MLP members. They share an input schema but not a trunk.

Each member uses:

```text
Linear(167, 256)
LayerNorm(256)
SiLU
ResidualBlock(256)
ResidualBlock(256)
Linear(256, 128)
LayerNorm(128)
SiLU
Linear(128, 2)
```

Each residual block is:

```text
h -> Linear(256, 256) -> LayerNorm -> SiLU
  -> Linear(256, 256) -> add h -> SiLU
```

There is no dropout. Ensemble diversity comes from independent initialization and deal-level bootstrap resampling. A deterministic bootstrap multiplicity is derived from `(ensemble_member, deal_id)` so training can resume reproducibly.

All five members are stored in one residual checkpoint and together constitute one acting bidder.

## 15. Training Loss

Targets are divided by 100 for numerical scaling but are not clipped. For member `j`:

```text
loss_j = weighted mean over legal alternatives of
         (prediction_j - target / 100)^2
```

The sample weight combines:

- natural-distribution importance correction;
- the member's deal-level bootstrap multiplicity;
- the legality mask.

Mean-squared error is required because the desired quantity is the conditional mean score advantage. Huber loss or return clipping would downweight meaningful contract and Nil tails and would change the optimized statistic.

Use gradient clipping rather than target clipping for numerical stability. Train/development splitting and bootstrap sampling operate at the duplicate-deal level.

## 16. Deterministic Decision Rule

For each legal alternative, compute ensemble mean and standard deviation in scaled-score units:

```text
mu_delta    = mean_j(prediction_j_delta)
sigma_delta = std_j(prediction_j_delta)
V_delta     = mu_delta - lambda * sigma_delta
```

`lambda` is selected on development duplicate deals to maximize paired match margin. It is fixed in the promoted checkpoint metadata.

Choose the maximum among:

```text
center: 0
lower:  V_minus, if legal
upper:  V_plus,  if legal
```

Exact ties use the fixed priority `center`, then `lower`, then `upper`. Inference is in evaluation mode and contains no stochastic sampling, so identical visible state and checkpoint produce identical final bids.

Any NaN, infinity, checkpoint mismatch, or unavailable residual output causes a logged fallback to the frozen NSFP center action at runtime. The same condition is a hard failure during training or evaluation.

## 17. Data Growth and Iteration Stopping

Do not choose a fixed total deal count in advance. Generate unique duplicate deals in throughput-sized blocks while the continuation policy remains frozen.

After each block, assess on fixed development deals:

- weighted advantage MSE;
- advantage sign accuracy;
- calibration by predicted-advantage bucket;
- offline action regret;
- action stability on fixed probe states;
- candidate duplicate margin against the incumbent;
- correction rates for lower, center, and upper actions;
- separate behavior for Nil/1, ordinary bids, high bids, and all four positions.

Collect at least three blocks. Stop adding data to the current iteration when all three most recent block additions improve development duplicate margin by no more than one bootstrap standard error and fewer than 0.5% of fixed probe states change selected action between consecutive block-trained candidates. Block size is an operational throughput setting, not a semantic sample target.

## 18. Promotion Evaluation

For each promotion deal and league opponent, evaluate four paired games:

1. new candidate in room one;
2. new candidate with partnerships swapped in room two;
3. incumbent in room one;
4. incumbent with partnerships swapped in room two.

All four use the same deal, opponent, seat/dealer arrangement, and corresponding deterministic seed bundle.

For deal `d` and opponent `l`:

```text
Z(l, d) = M(new, l, d) - M(incumbent, l, d)
```

Use the fixed league weights to aggregate `Z`. Estimate uncertainty with a deal-level, opponent-stratified bootstrap that keeps all four games from a deal together.

Promotion requires all of the following:

1. the one-sided 95% lower confidence bound of the league-weighted mean `Z` is greater than zero;
2. the one-sided 95% upper confidence bound of `Z` against the NSFP anchor is at least zero, so the evaluation does not establish a regression against the anchor;
3. for every predeclared opponent or behavior stratum with at least 5% promotion weight or estimated natural frequency, the one-sided 95% upper confidence bound of stratum `Z` is at least zero.

Contract success, Nil success, overtricks, and correction rates remain diagnostics rather than automatic vetoes because they are intermediate outcomes rather than the optimized payoff. Any additional veto must be declared numerically in the promotion manifest before promotion games begin.

Promotion sample size is fixed before viewing promotion results, using variance measured on development data. Promotion deals are not reused for tuning. A failed candidate is not added to the league.

## 19. Dataset Partitions

Use disjoint deal-seed namespaces for:

- training;
- development and hyperparameter selection;
- per-candidate promotion;
- final one-time test.

The final test set is never read during iterative training or promotion. All child states and action branches inherit their parent deal's partition.

## 20. Required Records

Each generated counterfactual record stores enough information to reproduce and audit the label:

- schema version;
- deal ID and deal seed;
- room and candidate partnership;
- acting policy version;
- opponent league member and league weight;
- play/belief checkpoint identifiers;
- current bidder and chronological auction;
- exact 149-dimensional NSFP encoder input;
- NSFP legal-action scores and center action;
- legal residual masks;
- forced action;
- continuation auction;
- seed bundle;
- tricks won;
- both team scores and candidate score margin;
- center-relative target;
- stratum and sampling/importance metadata.

Data shards are written atomically and validated before being admitted to training. Duplicate deal IDs within one partition are rejected.

## 21. Testing Strategy

### Unit tests

- residual input's first 149 values exactly equal the tensor sent to frozen NSFP;
- 16 raw NSFP logits map correctly to 14 legal-action scores;
- local candidate sets match the Nil/1/13 boundary table;
- illegal alternatives are masked from loss and inference;
- advantage targets equal alternative margin minus center margin;
- target scaling does not clip values;
- tie-breaking always prefers center, then lower, then upper;
- runtime fallback returns the NSFP center for invalid residual output;
- no other player's hand enters the residual input;
- all records from one deal share one dataset partition.

### Determinism tests

- the same checkpoint and visible state produce the same bid across repeated calls;
- data generation from the same deal and seed bundle reproduces the same branch record;
- ensemble bootstrap multiplicities reproduce after restart;
- training resume preserves dataset and policy-version identity.

### Integration tests

- residual-disabled composite policy exactly matches existing NSFP acting behavior;
- the frozen belief bidder still loads the original NSFP checkpoint;
- play-model checkpoint identities and play flow remain unchanged;
- both candidate seats share one composite checkpoint;
- both opponent seats share the selected frozen league checkpoint;
- changing the forced current action leaves deal, opponent, and continuation-policy version fixed;
- one duplicate deal produces no more than the expected center and local branch games.

### Statistical pipeline tests

- synthetic known-return data recovers the correct conditional means;
- importance weighting reconstructs a known natural distribution from a stratified sample;
- bootstrap resamples whole deals rather than rooms or action branches;
- promotion rejects equal policies except at the configured false-positive rate;
- a synthetic superior policy passes promotion with adequate power.

## 22. Operational Safety and Observability

Training logs and evaluation artifacts report:

- policy, opponent, play, and belief checkpoint hashes;
- deal ranges and dataset schema version;
- natural and stratified state counts;
- effective sample weights;
- ensemble disagreement;
- predicted and realized advantage calibration;
- lower/center/upper correction rates;
- Nil rate and Nil success rate;
- contract success and average overtricks;
- paired margin by opponent and position;
- promotion confidence interval and decision.

The implementation must fail fast on training/evaluation schema or checkpoint mismatches. Runtime play remains available through the explicit NSFP fallback, with the fallback reason logged.

## 23. Implementation Boundary

This specification defines one coherent implementation project:

1. residual input and composite acting-bidder inference;
2. counterfactual duplicate data generation;
3. residual ensemble training;
4. iterative league management;
5. promotion and final evaluation.

No card-play retraining, belief-bidder replacement, DDS training, or rules changes are included.
