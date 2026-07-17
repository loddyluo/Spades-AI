# Spades Rules and Determinism Fixes

## Goal

Fix issues 3, 4, and 6 without changing the proposal-likelihood behavior under review in issues 5 and 7.

The observable requirements are:

1. Playing any spade breaks spades, including a legal spade lead made by a player who holds only spades.
2. Fourth hand overtakes an opponent's winning spade with the lowest available higher spade when spades were led.
3. Given the same AI-visible information state and the same strategy configuration, exact play returns the same final action across repeated calls, process restarts, and HTTP/WebSocket state representations.

## Scope

### Issue 3: any spade breaks spades

- In `gui/src/game.js`, `applyCard` sets `spadesBroken` whenever the played card's suit is spades.
- In `Spades_AI_GO-MCTS/spades_ai/game/state.py`, immutable state transitions use the same rule and the surrounding documentation is corrected.
- In `gui/backend.py`, state reconstruction treats spades as broken if either the payload flag is true or any publicly played card in completed/current tricks is a spade. Both `spades_broken` and `trump_broken` receive the normalized value.
- `README.md` states that spades cannot be led before being broken unless the leader holds only spades, and that any played spade breaks them.
- Generated `gui/dist` assets are not edited.

### Issue 4: fourth-hand spade overtake

- In `strategy/rule_based_first4_player.py`, the fourth-hand branch first handles the case where an off-suit trick has already been trumped and following the led non-spade suit cannot win.
- Otherwise, a higher card in the led suit is a winning card even when that suit is spades; the player selects the lowest such card.
- Existing teammate-winning and discard behavior remains unchanged.

### Issue 6: deterministic exact play

- `RuleExactFirst4Player` creates a fresh `random.Random` for each exact-play decision from a stable cryptographic digest of the AI-visible information state.
- The digest includes a version tag, observing seat, own remaining cards, all hand sizes, bids and teams, turn and trick leader, trick counts, broken-spades state, completed public trick history, and the current table.
- Opponents' hidden card identities are deliberately excluded. This keeps equivalent HTTP placeholder states and WebSocket states deterministic without allowing hidden information to influence sampling.
- Serialization uses explicit ordered primitive values and a stable digest, never Python's process-randomized `hash()`.
- The existing RNG object continues to flow through proposal generation, importance sampling, and fallback determinization, so one decision uses one deterministic random stream.
- The guarantee applies to identical visible state plus identical strategy configuration. It does not promise identical actions after configuration, model, solver, or checkpoint changes.

## Non-goals

- Do not change issue 5's max/min-Q proposal weighting in this patch.
- Do not change issue 7's bidding position encoding, `min(p, 2)` callers, opener rotation, derived features, or checkpoint.
- Do not make legacy `RLExactPlayer` or `TruncatedMCTSStrategy` deterministic in this patch; the deployed `RuleExactFirst4Player` path and its nil subclass are the target.
- Do not refactor the generic game-state architecture or exact solver.

## Testing

Tests are written and observed failing before production changes.

- JavaScript regression: a forced spade lead changes `spadesBroken` to true, while a non-spade play does not.
- Python GO-state regression: a legal forced spade lead changes `spades_broken` to true.
- Backend regressions: a spade in either a completed trick or current trick overrides a false payload flag.
- Rule-player regressions: fourth hand overtakes an opponent's led spade with the lowest winning spade; an off-suit trick already trumped still causes the lowest led-suit card to be played.
- Determinism regressions: equivalent visible states with different hidden-opponent card identities produce the same seed/random stream, public or own-hand changes alter it, and repeated exact-play calls return the same final action.

After focused tests pass, run the repository's relevant Python test set, the GUI's Node tests, and the GUI production build.

## Error and compatibility behavior

- Backend normalization is monotonic: public evidence may change false to true, but never true to false.
- The state fingerprint is private implementation detail and is version-tagged so future schema changes can be intentional.
- Existing model/checkpoint inputs and solver scoring semantics are unchanged.
