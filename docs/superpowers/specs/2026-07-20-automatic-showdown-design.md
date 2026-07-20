# Automatic Showdown Design

**Status:** Discussion approved; awaiting review of this written specification
**Date:** 2026-07-20

## Goal

During GUI play, automatically offer a showdown when the scoring-relevant
outcome of the deal is identical under every legal continuation from the
current complete-trick boundary.

A scoring-relevant outcome is fixed exactly when:

1. the final North/South and East/West team trick totals are fixed; and
2. for every Nil or Blind Nil bidder, whether that player finishes with at
   least one trick is fixed.

Per-seat trick totals do not otherwise need to be fixed. This distinction is
intentional: normal contracts are scored from team totals, while Nil and Blind
Nil depend on the individual bidder taking zero versus at least one trick.

## Scope

The feature is enabled in every GUI game mode:

- single hand;
- 500-point match;
- fixed-seed hand;
- four-AI test;
- remote two-human versus two-AI play.

It does not change the general-purpose game driver, training rollouts,
evaluation programs, acting player, bidding model, or existing minimax
objective. Those paths continue to play every card normally.

## Trigger Policy

The detector runs only after a completed trick has been awarded and cleared
from the table. It never triggers in the middle of a trick.

Detection begins when at most five tricks remain. If a check returns
`variable`, `timeout`, or an operational error, play continues normally. The
next completed trick produces another opportunity to check. A completed
thirteenth trick proceeds directly to normal settlement without a redundant
check.

Each check has a one-second wall-clock budget. The budget includes waiting for
the process-wide native solver lock. A search that cannot prove either
uniqueness or variability within that budget returns `timeout` and stops; it
must not continue in a background thread after the caller resumes play.

## Outcome Signature

The native search represents a terminal scoring-relevant outcome as:

```text
(team_0_final_tricks, nil_broken_mask)
```

`team_0_final_tricks` is North/South's final team trick total. East/West's
total is therefore `13 - team_0_final_tricks`.

`nil_broken_mask` has one bit for each seat that bid Nil or Blind Nil. A set
bit means that bidder has taken at least one trick by the end of the deal.
Already-broken Nil bids remain set in every descendant state. Seats with
normal bids do not contribute bits.

Two terminal states have the same outcome if and only if these signatures are
equal.

## Native Search Architecture

Add a dedicated forced-outcome entry point beside the current fastest C++
double-dummy solver. It reuses the existing:

- `NativeState` representation;
- Spades legal-action generation;
- trick winner calculation;
- `make_move` / `unmake_move` machinery;
- Zobrist-style state hashing; and
- process-wide native-call lock.

The new search has a separate result structure and transposition table. It
does not reuse minimax values, alpha-beta score pruning, quick-trick score
bounds, or the assumption that each partnership chooses optimally. Those
optimizations answer a different question and cannot prove invariance across
all legal continuations.

The Python wrapper exposes a result with one of three statuses:

```text
Fixed(signature) | Variable | Timeout
```

The native implementation also reports diagnostic node and elapsed-time
counts, but diagnostics do not affect game behavior.

## Search Algorithm

For a state `s`, the search behaves as follows:

1. Check the monotonic-clock deadline. If expired, return `Timeout`.
2. At a terminal state, return `Fixed(signature(s))`.
3. Look up the state in the forced-outcome transposition table.
4. Enumerate every legal card for the current player.
5. Recursively search the first child and retain its signature if fixed.
6. If any child returns `Variable`, return `Variable` immediately.
7. If two fixed children produce different signatures, return `Variable`
   immediately.
8. If every child is fixed with the same signature, return that
   `Fixed(signature)` result.
9. If a child times out before variability or uniqueness has been proved,
   return `Timeout`.

Only exact `Fixed` and `Variable` results are cached. `Timeout` is never
cached.

The forced-outcome transposition key contains all information that affects
future legal play or the outcome signature:

- all remaining hand bitsets;
- current player and trick leader;
- cards and seats in the current trick;
- whether Spades have been broken;
- completed-trick count;
- current North/South team trick total; and
- the current Nil-broken mask.

The cache is scoped or generation-tagged so fixed bids and seat-to-team
mapping cannot leak between independent calls.

The initial implementation does not remove actions using minimax-oriented
equivalent-card filtering. Such filtering may be added only after a separate
proof and oracle tests establish that it preserves the complete set of
terminal signatures. Missing the one-second deadline is acceptable; a false
showdown is not.

## Deterministic Completion

After and only after the native search returns `Fixed`, Python completes a
deep copy of the state along one deterministic legal line. At every turn it
sorts the legal cards by stable card ID and plays the first card, awarding
tricks through the shared `SpadesRules` implementation.

The completion produces:

- the remaining ordered card plays and trick winners;
- final per-seat trick counts;
- the final team trick totals; and
- the final Nil/Blind Nil outcomes.

The generated terminal signature is checked against the signature proved by
the native search. A mismatch is treated as an internal error and suppresses
the showdown.

This deterministic line is not part of the uniqueness proof. It exists to
keep normal state shape, per-seat tallies, complete 13-trick replay history,
and existing scoring code intact. Since the signature is already proven
unique, any legal completion has the correct scoring-relevant result.

## Python Integration Boundary

Provide one shared Python service for both local HTTP play and the remote game
server. It accepts a complete authoritative `GameState`, applies the trigger
preconditions, invokes the native checker with the remaining deadline, and,
on `Fixed`, builds the deterministic completion.

The service validates that:

- the phase is playing;
- the table is empty at a complete-trick boundary;
- all four hands have the same remaining size between one and five;
- remaining and already-played cards are unique and form a consistent deck;
- all four bids are complete; and
- teams are `[0, 1, 0, 1]`.

Invalid input is an operational failure, never permission to show down.

### Local HTTP play

Add a separate showdown-check endpoint. Its payload contains all four actual
remaining hands plus public bids, trick history, trick totals, leader, and
Spades-broken state. It uses a new full-information state builder rather than
the placeholder-hand builder used for AI decisions.

The complete state is passed only to the forced-outcome service. It is never
passed to `RuleExactFirst4NilPlayer`, its acting bidder, or its acting card
player. The existing `/api/choose-action` privacy boundary remains unchanged.

The response status is `fixed`, `variable`, or `timeout`. A `fixed` response
also contains the projected result and deterministic continuation. Endpoint
or native-library failures are logged by the client and treated like
`timeout`.

### Remote play

The remote `GameRoom` already owns all four hands, so it calls the same
service directly after each completed trick. It does not serialize hidden
hands to clients unless a fixed outcome has been proved.

## Local Game Flow

The pure card and trick reducers remain responsible for ordinary play. The
asynchronous local coordinators check for showdown immediately after
`finalizeTrick` when the trigger conditions hold.

On a fixed result:

1. The game enters a `showdownPending` UI state and all further human and AI
   actions stop.
2. All four remaining hands are rendered face up.
3. A showdown panel displays projected North/South and East/West trick totals
   and the success or failure of every Nil/Blind Nil bid.
4. The player presses `Confirm settlement`.
5. The stored deterministic continuation is applied without per-card
   animation through the normal card/trick reducers.
6. Existing scoring and result overlays run unchanged.

The score itself is not finalized or displayed until confirmation. The
projected trick and Nil result is displayed before confirmation, as requested.

The four-AI test still requires the human spectator to confirm; it does not
auto-dismiss the showdown.

## Remote Confirmation Flow

When a remote showdown is proved, the server pauses the room and broadcasts a
showdown offer containing all remaining hands and the projected result. Each
of the two connected humans must confirm separately.

- Confirmation messages identify the active showdown and the sender's seat.
- Duplicate confirmations are ignored.
- A confirmation from a seat outside the room or for a stale showdown is
  rejected.
- After confirming, that client sees `Waiting for partner confirmation` and
  cannot act further.
- Only after both human seats have confirmed does the server apply the stored
  deterministic continuation, score the deal, and broadcast the final result.

If a participant disconnects while confirmation is pending, the room follows
the existing remote-disconnection behavior and terminates rather than treating
the disconnect as consent.

## UI Presentation

During `showdownPending`, ordinary turn controls are disabled. Existing
face-up replay hand components should be reused where practical to show each
seat's complete remaining hand; opponent card-back fans are replaced with
face-up spreads only for this state.

The panel shows:

```text
Automatic showdown
Final tricks: NS X, EW Y
North Nil: success
East Blind Nil: failed
[Confirm settlement]
```

Nil rows are omitted when there are no Nil or Blind Nil bids. In remote play,
the confirmation button changes to a disabled waiting message after that
client confirms.

## Failure Handling

The following conditions silently preserve normal play, with a diagnostic log
or console warning where appropriate:

- native solver unavailable;
- solver lock not acquired within the remaining budget;
- native search timeout;
- HTTP/network failure;
- malformed or inconsistent authoritative state;
- deterministic completion/signature mismatch; or
- an unexpected native status.

No failure path may reveal hands or partially mutate the live game state.
Analysis and deterministic completion operate on copies until confirmation.

## Test Strategy

Development follows test-driven development. A small Python exhaustive oracle
supports legal states with one to three remaining tricks and computes the full
set of terminal signatures.

Native-versus-oracle tests compare fixed/variable status and signatures across
random legal small states. Focused fixtures cover:

- fixed team tricks without Nil;
- variable team tricks;
- fixed team tricks but variable Nil outcome;
- fixed Nil success and fixed Nil failure;
- multiple Nil bidders;
- an already-broken Nil;
- following suit, unbroken Spades, and forced Spade leads;
- timeout results not contaminating a later complete search; and
- deterministic completion legality and signature agreement.

Backend tests cover complete-state validation, tri-state responses, and the
fact that native-lock wait time counts toward the budget.

Frontend tests cover:

- checks only after completed tricks;
- no checks with more than five tricks remaining;
- continued play after `variable`, `timeout`, or request failure;
- face-up hands and paused actions after `fixed`;
- confirmation before settlement;
- complete 13-trick replay after deterministic completion; and
- all four local GUI modes.

Remote tests use two fake connections to prove that one confirmation cannot
settle, two distinct confirmations do settle, and duplicate or stale
confirmations do not count.

Regression tests run the existing `solve` and `solve_with_q` corpus to ensure
the new native entry point does not alter current card selection or values.

Deadline tests verify prompt cancellation, but do not require every five-trick
position to finish within one second. Representative fixed and variable
five-trick positions are benchmarked separately to report hit rate and latency
without creating machine-dependent unit-test failures.

## Acceptance Criteria

The feature is complete when:

1. No showdown is offered unless every legal continuation has the same team
   trick totals and Nil/Blind Nil outcomes.
2. Checks occur only at complete-trick boundaries with at most five tricks
   remaining.
3. A check consumes no more than its one-second budget except for small,
   measured call/return overhead and leaves no background search running.
4. Timeout and error paths continue the game without revealing hidden cards.
5. A successful showdown reveals all remaining cards and displays the fixed
   result before settlement.
6. Local settlement requires one explicit confirmation; remote settlement
   requires one confirmation from each connected human.
7. Confirmed settlement uses a legal completion and preserves normal scoring,
   per-seat tallies, and a complete replay.
8. All GUI modes listed in scope use the feature, while training and evaluation
   behavior remains unchanged.
