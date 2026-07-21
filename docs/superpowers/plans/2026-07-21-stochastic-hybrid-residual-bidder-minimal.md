# Minimal Stochastic Residual Bidder Experiment

Date: 2026-07-21

Status: active

## Goal

Obtain evidence quickly: generate local counterfactual Q labels, train the
existing five-member residual model, and compare it with frozen NSFP in paired
two-room fast-hybrid matches.

## Fixed decisions

- Replace only the acting bidder. The existing endgame belief bidder and card
  play pipeline remain unchanged.
- Frozen `bid_nsfp.pt` supplies the center action and the exact 149-value input.
- Learn only the local lower and upper advantages; the center value is zero.
- Targets are acting-partnership score-margin differences divided by 100.
- A training branch plays exactly four tricks with the existing Nil/non-Nil
  rules and calls the native DDS terminal solver exactly once.
- A policy may output a 14-action distribution and use a deterministic policy
  tape. Whether nonzero temperature helps is an evaluation question.
- First evaluation uses the same fast four-trick-plus-DDS approximation as
  training. Real deployed play comes only after a positive fast result.

## Explicitly deferred

- Changes to belief weighting or `RuleExactFirst4NilPlayer`.
- League, reservoir, importance sampling, iterative policy improvement, formal
  promotion gates, bootstrap stopping rules, GUI integration, and deployment.
- Atomic dataset shards, SQLite admission, crash-recovery state machines, and
  adversarial model/checkpoint tamper resistance.
- Further hardening of Tasks 1--4 unless a defect blocks the experiment.

## Milestone 1: local data loop

Implement one reusable hybrid generator and a small command that:

1. creates a deterministic duplicate deal;
2. runs frozen NSFP baseline auctions in both rooms;
3. records the four candidate-partnership bidding decisions across the rooms;
4. evaluates each legal local action while all other bidding decisions use
   frozen NSFP;
5. plays four tricks and performs one DDS solve per branch;
6. emits one row per decision containing the 167-vector, two targets, mask,
   center action, deal ID, room, and physical seat;
7. writes a simple pickle-free NPZ file.

Acceptance smoke: eight fixed deals reproduce byte-identical numeric arrays;
each branch ends at four tricks with an empty table and 36 cards; the solver is
called once; center-relative targets and team-1 signs are correct; no exact
belief-play path is entered.

## Milestone 2: first fitted model

Generate 1,000 remote deals as a speed/storage check, then approximately 10,000
training deals plus a disjoint validation set. Train the existing five-member
ensemble with masked MSE and save one experimental checkpoint. Report holdout
MSE, zero-predictor MSE, lower/upper sign accuracy, and examples per second.

## Milestone 3: fast paired result

On disjoint fixed deals, place the candidate partnership North/South in one room
and East/West in the other against frozen NSFP. Combine the two room scores into
one candidate-relative duplicate margin. Compare deterministic `T=0` first,
then a small predeclared set of nonzero temperatures. Report deal count, paired
mean margin, standard error, and win/tie/loss counts.

## Execution rule

Do not stop for approval between these milestones. Stop only for an unavailable
required artifact, an external permission/cost decision, or evidence that the
four-trick evaluator is semantically wrong.
