# Stochastic Hybrid Residual Bidder Training Design

> **Superseded for implementation.** This production-scale design is retained
> only as historical context. It must not drive further work. The active scope
> is the minimal experimental loop in
> `docs/superpowers/plans/2026-07-21-stochastic-hybrid-residual-bidder-minimal.md`.
> In particular, the old belief bidder remains frozen; Sections 12--21 are not
> authorized for the first experiment.

Date: 2026-07-21

Status: approved in the design discussion and awaiting written-spec review.

This document supersedes the 2026-07-17 residual-bidder design and its
implementation plan.

## 1. Objective

Train one new bidder that is shared by both seats of a partnership and is
optimized for duplicate/team-match score margin with the repository's current
deployed card-play pipeline.

The new bidder is a stochastic composite policy:

1. frozen bid_nsfp.pt and the current BidEncoder define a strong center bid;
2. a five-member residual-Q ensemble learns the values of the legal local
   actions centered on that bid;
3. a calibrated local softmax supplies the main policy mass;
4. a calibrated geometrically decaying tail can give every legal bid nonzero
   support;
5. inference samples from the resulting 14-action distribution using a
   reproducible random tape.

The optimized payoff is raw duplicate/team-match score margin. Bid imitation,
DDS trick count, contract success, Nil success, entropy, and ordinary
classification accuracy are diagnostics rather than primary objectives.
The calibration search deliberately retains the deterministic boundary, so the
final policy is stochastic only if evaluation shows that randomization helps.

## 2. Deliberate Training/Deployment Split

Deployment keeps the existing card-play models, rules, and control flow. It
does not retrain or replace the current play policy.

Training labels intentionally use a much faster surrogate continuation:

1. finish the auction;
2. play exactly the first four tricks with the existing deterministic first-four
   rule players, including the dedicated Nil rule player;
3. at the four-trick boundary, with 36 cards remaining, call
   ExactDoubleDummyCppFastestSolver.solve exactly once;
4. use its terminal team-score margin without playing the final nine tricks
   card by card.

The training generator never invokes the deployed endgame belief sampler or
the repeated imperfect-information exact-play path. Real deployed card play is
used only for development selection, promotion, and final testing. It never
produces gradient targets or replay data.

This split is intentional: fast full-information continuation supplies abundant
low-variance local action comparisons, while fresh duplicate evaluation with
the real play pipeline remains the authority on whether the bidder is actually
better for deployment.

## 3. Evidence for the Route Change

The previous design generated labels by replaying complete games through the
deployed belief-sampled exact-play path. On the current 8-CPU cloud pod this
cost roughly 150 to 200 seconds per complete branch game and made useful
dataset sizes impractical.

A read-only local benchmark of 20 four-trick-boundary states measured a single
native double-dummy terminal solve at:

- median: 0.023 seconds;
- mean: 0.055 seconds;
- maximum: 0.477 seconds.

The implementation must repeat an end-to-end benchmark on the target cloud
runtime before a long generation job, but the measured difference is large
enough to justify replacing training continuation rather than merely adding
CPU workers.

## 4. Frozen Components and Scope

The following remain frozen:

- bid_nsfp.pt parameters;
- the exact current state-to-BidEncoder bridge;
- the 149-dimensional BidEncoder semantics;
- all card-play model parameters;
- the first-four non-Nil and Nil rule semantics;
- Spades legality and scoring rules.

The new code may add:

- residual-Q training and inference;
- stochastic acting-bid sampling;
- a policy-probability interface;
- per-seat bid-likelihood adapters in the deployed belief sampler;
- the fast four-trick-plus-double-dummy training evaluator;
- data, league, calibration, and promotion infrastructure.

Blind Nil and deployable normal bid zero remain out of scope. The legal action
space is:

    Nil, bid_1, bid_2, ..., bid_13

No direct REINFORCE, PPO, actor-critic update, or fine-tuning of NSFP is part of
this design.

## 5. NSFP Center and Local Learned Actions

NSFP has 16 raw outputs but deployment exposes 14 legal actions. Construct the
14 legal scores exactly as follows:

- Nil score is the maximum of the raw Nil and Blind-Nil logits;
- bid-1 score is the maximum of raw normal-0 and normal-1 logits;
- bid-k score for k from 2 through 13 is raw normal-k.

The center c is the argmax of these 14 legal scores under the existing stable
normalization behavior.

For neighborhood construction only, assign Nil index 0 and bid-k index k.
The learned core neighborhood is:

| Center | Learned legal actions |
|---|---|
| Nil | Nil, 1 |
| 1 | Nil, 1, 2 |
| 2 through 12 | c-1, c, c+1 |
| 13 | 12, 13 |

Only these two or three actions receive learned Q values. Actions farther from
the center may still be sampled through the calibrated tail, but they never
become residual-Q targets and never enter the local core softmax directly.

For the legal lower and upper alternatives retain the normalized NSFP margins:

    m_minus = (score(center) - score(lower)) / 13.47
    m_plus  = (score(center) - score(upper)) / 13.47

An unavailable alternative has margin zero and mask zero.

## 6. Residual Input Contract

The residual model receives exactly 167 values:

1. the exact 149-dimensional tensor sent to frozen NSFP;
2. a 14-dimensional one-hot encoding of the normalized NSFP center;
3. m_minus and m_plus;
4. lower and upper legality masks.

The model must not receive:

- another player's hand;
- the complete deal;
- future bids;
- played-card information;
- double-dummy results;
- NSFP hidden-layer activations.

Keeping the encoder unchanged is a compatibility constraint, including any
legacy position, auction, or derived-feature semantics.

## 7. Hybrid Counterfactual Label Generation

For one frozen policy iteration and one duplicate deal:

1. select one immutable opponent-league member for both opponent seats;
2. create independent shuffle and policy random tapes;
3. run one baseline auction in each duplicate room;
4. save the candidate partnership's two encountered bidding observations in
   each room;
5. at each observation reconstruct every legal local branch by forcing lower,
   center, or upper;
6. continue all later bids with the same frozen seat policies and the same
   corresponding later random numbers;
7. play exactly four complete tricks with RuleBasedFirst4Player or
   RuleBasedFirst4NilPlayer as required by the auction;
8. assert that the table is empty, tricks_played is four, and 36 cards remain;
9. call the native double-dummy terminal solver once;
10. record the terminal raw score margin from the acting partnership's
    perspective.

If the baseline sampled action is local, its terminal value is reused for that
state. If the baseline sampled a tail action, all local branches are evaluated
separately. A duplicate therefore normally needs at most ten hybrid games and
has a hard upper bound of fourteen when all four recorded baseline actions
come from the nonlocal tail.

No branch is repeated to estimate an average. Every unique full deal is one
sample of hidden information. Budget is spent on additional deals.

The branch evaluator must be a dedicated training component. It must not call
RuleExactFirst4Player._exact_play, construct an importance-sampling pool, or
step through the last nine tricks.

## 8. Targets and Reward Semantics

For visible observation o, full deal d, and local action a, define the raw
realized target:

    y(d, o, a) = R(d, forced a, frozen continuation)
               - R(d, forced center, frozen continuation)

The center target is exactly zero. The model estimates:

    A(a | o) = E[y(d, o, a) | visible observation o]

The other room of the duplicate is constant across a local counterfactual and
therefore cancels. Training stores the affected room's raw score-margin
difference. Both swapped rooms still provide candidate-partnership states and
remain grouped under one duplicate deal.

The native solver returns team-0 score minus team-1 score. Negate that value
before target arithmetic whenever the acting partnership is team 1.

Store both the raw target and the scaled training target:

    y_scaled = y_raw / 100

The divisor is fixed at 100 for every run and iteration. Targets are never
clipped. Gradient clipping, not return clipping or Huber loss, handles extreme
contract and Nil outcomes.

The target is a realized advantage, not a per-deal best-action class. The full
deal and solver output are label-only information and cannot enter inference.

## 9. Five-Member Residual-Q Ensemble

The deployed residual estimator contains five wholly independent MLP members.
They are not five players and inference never randomly selects one member.

Each member uses:

    Linear(167, 256)
    LayerNorm(256)
    SiLU
    ResidualBlock(256)
    ResidualBlock(256)
    Linear(256, 128)
    LayerNorm(128)
    SiLU
    Linear(128, 2)

Output zero is the lower-action advantage and output one is the upper-action
advantage. A missing boundary action is masked from both loss and inference.

Each residual block is:

    h -> Linear(256, 256) -> LayerNorm -> SiLU
      -> Linear(256, 256) -> add h -> SiLU

There is no dropout. Members use different initialization seeds and
deterministic deal-level Poisson(1) bootstrap multiplicities. All rows, rooms,
and branches from one deal share one bootstrap draw for a member.

Member j minimizes weighted masked MSE in scaled-score units:

    loss_j = weighted_mean((prediction_j - y_raw / 100)^2)

Weights combine the legality mask, natural-distribution importance correction,
and member bootstrap multiplicity. The desired statistic is the conditional
mean, so Huber loss is not permitted.

At inference, compute population mean and standard deviation across members:

    mu_a    = mean_j(prediction_j,a)
    sigma_a = std_j(prediction_j,a)
    V_a     = mu_a - lambda * sigma_a

The center has V_center = 0. Lambda controls action-specific conservatism;
ensemble disagreement is not used as a source of random sampling.

## 10. Stochastic Policy Distribution

The calibration domains are:

    lambda >= 0
    T >= 0
    0 <= epsilon <= 1
    0 < rho <= 1

rho has no behavioral effect when epsilon is zero.

First form the learned local core. For T greater than zero:

    pi_local,T(a | o) = softmax(V_a / T), for legal local actions
    pi_local,T(a | o) = 0, otherwise

T is expressed in the scaled Q units. At T = 0 use a stable local argmax with
tie order center, lower, upper.

Define the ordered action index:

    index(Nil) = 0
    index(bid_k) = k

For center c and 0 < rho <= 1, define the full-support geometric tail:

    d(a, c) = abs(index(a) - index(c))
    r_rho(a | c) = rho^d(a,c) / sum_b rho^d(b,c)

The final 14-action policy is:

    pi(a | o) = (1 - epsilon) * pi_local,T(a | o)
                + epsilon * r_rho(a | c)

Properties:

- every action is nonnegative and the vector sums to one;
- every legal action has positive probability exactly when epsilon > 0;
- epsilon controls the tail mixture weight;
- rho controls how quickly tail probability decays with bid distance;
- rho = 1 is the uniform-tail boundary;
- T controls randomization only within the learned local core;
- T = 0 and epsilon = 0 recover deterministic local argmax.

The promoted checkpoint stores lambda, T, epsilon, and rho. They are not
learned by MSE and are not entropy coefficients.

## 11. Reproducible Random Tape

Bid sampling uses a policy_seed that is independent of the shuffle seed. Derive
one uniform variate per bidding decision with a domain-separated cryptographic
hash of:

    policy_seed
    deal_id
    room_id
    logical_seat
    bid_index

logical_seat means the canonical physical seat before either candidate or
incumbent is assigned to a partnership. room_id creates independent but
reproducible random tapes for the two duplicate rooms.

The checkpoint hash is deliberately not part of this derivation. Candidate and
incumbent therefore consume the same uniform variates on corresponding
decisions, providing common random numbers for lower-variance comparisons.
Every run manifest still records all policy and parameter hashes.

The same deal, room, policy, parameters, and policy_seed reproduce the same
auction. Different deals may select different actions even when their visible
observations happen to be identical. Counterfactual branches reuse the
corresponding later-seat variates after the forced action.

Sampling is implemented by a stable inverse CDF over the canonical action
order. No mutable global RNG state may affect a bid.

## 12. Acting and Belief Probability Consistency

Every bidder policy exposes two operations over the same implementation:

- probabilities(observation): return the legal 14-action distribution;
- sample(observation, u): sample that distribution with a supplied uniform
  variate.

For a hypothetical initial deal, the deployed endgame belief sampler weights
the observed auction by:

    w_bid = product over seats p of
            pi_policy[p](observed_bid_p |
                         hypothetical_hand_p, previous_observed_bids)

The sampler must use the actual policy adapter assigned to each seat:

- a promoted stochastic residual bidder uses its exact distribution and stored
  calibration parameters;
- a prior promoted residual snapshot uses that snapshot's distribution;
- a legacy NSFP seat uses the existing legacy softened-likelihood adapter;
- an explicitly unknown external or human seat uses a declared fallback
  likelihood adapter.

The current single global NSFP likelihood cannot impersonate all four seat
policies. Policy provenance must be available to the sampler.

This supersedes the old permanent-freeze decision for the belief bidder.
bid_nsfp.pt remains available for legacy seats and runtime fallback, but a new
acting bidder's observed actions are explained by its own actual probabilities.

## 13. Iterative Fitted Policy Improvement and League

Let pi_0 be frozen NSFP acting behavior. For accepted iteration i:

1. freeze incumbent pi_i and every opponent-league member;
2. generate unique on-policy hybrid counterfactual deals;
3. fit a new five-member local-advantage ensemble;
4. search calibration parameters only on development data;
5. compare the resulting candidate with pi_i on fresh promotion deals using
   the real deployed card-play pipeline;
6. accept and add the immutable candidate to the league only if promotion
   passes.

Both seats of one partnership always use the same bidder checkpoint and
calibration parameters. Both opponent seats use the same selected league
member.

League sampling retains the established schedule:

- before an accepted residual snapshot: 100% NSFP;
- with one accepted snapshot: 50% NSFP and 50% latest;
- with at least two accepted snapshots: 50% NSFP, 25% latest, and 25% uniform
  over older accepted snapshots.

Labels generated under a different continuation-policy version are stored with
that version and cannot be mixed as if on-policy. Reusing a state requires
rerunning its hybrid local branches under the current frozen continuation.

## 14. Natural and Stratified Data

Keep an unfiltered natural pool from on-policy duplicate deals. It defines the
primary training and development distribution.

Maintain a separate outcome-blind stratified reservoir for underrepresented:

- center bid: Nil, 1, 2, 3, 4, 5, 6, and 7+;
- bidding position one through four;
- whether Nil has already appeared;
- whether the partner bid is visible;
- opponent league member;
- whether the baseline action came from the geometric tail.

Do not synthesize hands and do not select states because a forced action won.
Every retained state records its estimated natural frequency, sampling
probability, and importance correction. Promotion and final test deals are
always unfiltered.

## 15. Calibration, Development, and Promotion

Parameter calibration has two stages.

First, use the fast hybrid development simulator to evaluate a predeclared
grid over lambda, T, epsilon, and rho. The grid must include:

- lambda = 0;
- T = 0;
- epsilon = 0;
- for every tested nonzero epsilon, at least one rho = 1 uniform-tail control;
- the fully deterministic point T = 0, epsilon = 0.

Use successive halving or an equivalent predeclared rule to retain only a small
shortlist. Freeze the shortlist before any complete deployed-play development
games.

Second, evaluate that shortlist on fixed development duplicate deals with the
real card-play pipeline and per-seat belief likelihoods. All parameter tuples
use the same deals and policy random tapes. Select maximum mean paired margin;
exact ties prefer lower epsilon, then lower T, then larger lambda, then smaller
rho. Record that rule in the run manifest.

Real deployed-play development games are evaluations only. They are never
converted into Q labels and never enter the optimizer.

For promotion, evaluate four games per deal/opponent:

1. candidate in duplicate room one;
2. candidate with partnerships swapped in room two;
3. incumbent in room one;
4. incumbent with partnerships swapped in room two.

Define:

    Z(opponent, deal) =
        duplicate_margin(candidate) - duplicate_margin(incumbent)

Bootstrap whole deals within opponent strata while keeping all four games
together. Promotion requires all of the following:

1. the one-sided 95% lower confidence bound of the league-weighted mean Z is
   greater than zero;
2. in the NSFP-opponent stratum, the one-sided 95% upper confidence bound of Z
   is at least zero;
3. for every predeclared opponent or behavior stratum with at least 5% of
   promotion weight or estimated natural frequency, the one-sided 95% upper
   confidence bound of stratum Z is at least zero.

The latter two checks reject a candidate only when the evaluation establishes
a regression in a protected stratum; they do not require every stratum to show
a statistically significant gain.

Fix promotion sample size before viewing promotion outcomes, using development
variance, one-sided alpha 0.05, power 0.80, a one-point minimum detectable
improvement, rounding to a multiple of 256, and a minimum of 4096 unique
duplicate deals. Promotion deals never tune parameters.

## 16. Data Growth and Stopping

Do not set one irrevocable total-deal target. Generate unique deals in
throughput-sized blocks under one frozen continuation.

After each block report:

- weighted MSE in scaled units;
- raw-score calibration by predicted-advantage bucket;
- lower/upper sign accuracy;
- offline local regret;
- fixed-probe Q and probability stability;
- ensemble disagreement;
- center/lower/upper and tail action rates;
- fast hybrid duplicate margin.

Collect at least three independent blocks. Compare every block-trained
candidate with its immediate predecessor on the same fixed deals and random
tapes. Stop growing one iteration only when each of the three latest additions
improves mean fast-hybrid duplicate margin by no more than one deal-bootstrap
standard error, fewer than 0.5% of fixed probe states change their final
probability vector by L1 distance greater than 0.01, and the latest candidate's
best calibrated tuple does not improve the fixed real-play development paired
margin over its immediate predecessor.

An accepted policy starts a new iteration. Repeated promotion failure without
the data-stability conditions is not, by itself, evidence of convergence.

## 17. Dataset Partitions and Records

Use disjoint deal namespaces for:

- training;
- fast hybrid development;
- complete-play development and calibration;
- per-candidate promotion;
- final one-time test.

All rooms, states, and branches inherit the parent deal partition. No promotion
or final-test deal may enter training or parameter selection.

Each branch record contains:

- schema, deal, room, and logical-seat IDs;
- independent shuffle and policy seed identifiers;
- acting and per-seat policy IDs;
- NSFP, residual, first-four-rule, solver, play, and configuration hashes;
- chronological observed auction;
- exact 149-dimensional encoder tensor;
- 14 normalized NSFP legal scores and center;
- local neighborhood and masks;
- forced action and continuation auction;
- raw and divided-by-100 targets;
- first-four terminal state summary;
- DDS terminal score margin;
- stratum, sampling probability, and importance weight.

Shards, checkpoints, league manifests, and run states use atomic write,
fsync, validation, and hash admission. Duplicate deal IDs inside a partition
are rejected.

## 18. Failure and Fallback Semantics

Training, development, and promotion fail immediately on:

- missing or changed checkpoints;
- schema, feature, rule, solver, or policy identity mismatch;
- NaN or infinity in features, Q values, probabilities, losses, or targets;
- probability normalization failure;
- an unavailable native solver;
- an incorrect four-trick boundary;
- accidental entry into the belief-sampled play path;
- data-partition leakage.

In an actual interactive game, an invalid residual checkpoint or probability
calculation logs a structured reason and falls back to deterministic NSFP
argmax. The seat's runtime policy provenance records that fallback so the
belief sampler can select the legacy likelihood adapter.

There is no silent substitution of a Python solver, uniform bid distribution,
different encoder, or different play policy during formal generation or
evaluation.

## 19. Required Verification

### Feature and action tests

- the first 149 input values exactly equal the current NSFP tensor;
- 16 raw logits normalize to the approved 14 legal scores;
- Nil, bid-1, ordinary, and bid-13 neighborhoods and masks are exact;
- complete-deal and DDS information never enter residual inference;
- target arithmetic and fixed division by 100 are exact and unclipped.

### Model and statistical tests

- all five members have separate parameters and deterministic bootstrap keys;
- synthetic conditional means are recovered;
- illegal outputs have zero loss gradient;
- ensemble mean and population standard deviation are exact;
- mu minus lambda times sigma is applied before local softmax.

### Probability tests

- every 14-vector is finite, nonnegative, and sums to one;
- geometric tail probability is monotone in distance;
- rho = 1 produces the uniform-tail boundary;
- T = 0 and epsilon = 0 produce deterministic local argmax;
- fixed-seed empirical action frequencies match declared probabilities;
- acting and belief adapters return bit-identical distributions.

### Hybrid evaluator tests

- exactly four complete tricks are played;
- Nil and non-Nil first-four rule routing is correct;
- exactly one terminal solver call occurs per branch;
- no importance-sampling proposal or deployed exact-play decision occurs;
- corresponding later bid random variates are shared across local branches;
- baseline result reuse and the fourteen-game hard bound are enforced.

### Reproducibility and pipeline tests

- one random tape reproduces an auction and branch manifest after restart;
- candidate and incumbent receive common random numbers;
- per-seat likelihood adapters follow policy provenance;
- split and shard resume identity is exact;
- bootstrap and promotion resample whole duplicate deals;
- equal policies do not pass strict promotion;
- a known superior synthetic policy passes with adequate power.

Before long cloud generation, run an end-to-end throughput benchmark on the
target pod and report deals/hour, branch tail latency, native solver build ID,
CPU utilization, GPU fit throughput, and projected storage.

## 20. Operational Observability

Every run reports:

- all artifact and policy hashes;
- data ranges, partitions, and continuation version;
- local and tail action frequencies;
- effective sample weights and reservoir composition;
- Q error, calibration, and ensemble disagreement;
- selected lambda, T, epsilon, and rho;
- belief proposal effective sample size by seat-policy combination;
- hybrid and real-play duplicate margins;
- Nil rate/success, contract success, and bags as diagnostics;
- promotion confidence interval and decision;
- fallback counts and reasons.

The RTX 5090 is used for ensemble fitting and GPU smoke tests. Hybrid branch
generation is CPU/native-solver work and must use outer deal parallelism without
nested Python process pools.

## 21. Implementation Boundary

This specification is one coherent implementation project:

1. preserve the current NSFP encoder and local residual input;
2. implement the fast hybrid counterfactual evaluator;
3. train the five-member local residual-Q ensemble;
4. produce and sample the calibrated 14-action stochastic policy;
5. expose exact per-seat policy likelihoods to deployed belief sampling;
6. run iterative league training, two-stage calibration, and duplicate
   promotion;
7. provide resumable cloud execution and the required verification.

It does not retrain card play, alter Spades rules, train DDS, or use
full-action Q learning. The only learned actions are the local two or three;
nonlocal actions receive probability solely through the geometric tail.
