# Automatic Showdown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an exact, one-second-budget automatic showdown that reveals all remaining cards and waits for confirmation when every legal continuation has identical team trick totals and Nil outcomes.

**Architecture:** Extend the fastest native double-dummy library with an independent all-continuations uniqueness search returning `fixed`, `variable`, or `timeout`. A shared Python service validates authoritative states and generates one deterministic legal completion; the local HTTP GUI and authoritative remote server consume that service without passing full hands to the acting AI.

**Tech Stack:** C++17, Python 3.10+, `ctypes`, `pytest`, JavaScript ES modules, Node test runner, React 18, Vite, asyncio, and WebSockets.

## Global Constraints

- The terminal signature is `(North/South final team tricks, per-seat Nil/Blind Nil broken mask)`; non-Nil per-seat totals need not be invariant.
- Detect only after a completed trick has been awarded and cleared, with one to five tricks remaining.
- The one-second wall-clock budget includes waiting for the process-wide native lock. Timeout stops the search and continues normal play without revealing cards.
- Do not use minimax values, score pruning, sampling, or unproved equivalent-card filtering to prove uniqueness.
- Local settlement requires one confirmation. Remote settlement requires one confirmation from each connected human.
- Generate and verify a deterministic legal completion only after `fixed`; preserve all 13 tricks for replay.
- Do not change training, evaluation, the general driver, bidding, or acting-player behavior.
- Preserve every pre-existing dirty-worktree change. Never stage an unrelated hunk from a file that was already modified before this feature.
- Do not commit regenerated architecture-specific `.so` files.

## File Map

**Create:**

- `trick_taking/forced_outcome.py` — shared validation, completion, and result types.
- `tests/test_forced_outcome_native.py` — exhaustive Python oracle and native parity tests.
- `tests/test_forced_outcome.py` — service and deterministic-completion tests.
- `tests/test_game_server_showdown.py` — remote confirmation tests with fake sockets.
- `gui/src/showdown.js` — pure display selectors and confirmation panel.
- `gui/src/showdown.test.js` — display tests.

**Modify:**

- `trick_taking/solvers/exact_double_dummy_cpp_fastest_core.cpp`
- `trick_taking/solvers/exact_double_dummy_cpp_fastest.py`
- `tests/test_exact_solver_thread_safety.py`
- `gui/backend.py`
- `tests/test_gui_backend.py`
- `gui/src/game.js`
- `gui/src/game.test.js`
- `gui/src/App.jsx`
- `gui/src/styles.css`
- `gui/package.json`
- `gui/game_server.py`

---

### Task 1: Native Forced-Outcome Search

**Files:**
- Create: `tests/test_forced_outcome_native.py`
- Modify: `trick_taking/solvers/exact_double_dummy_cpp_fastest_core.cpp`
- Modify: `trick_taking/solvers/exact_double_dummy_cpp_fastest.py`
- Modify: `tests/test_exact_solver_thread_safety.py`

**Interfaces:**
- Consumes: existing `NativeState`, legal-play rules, `make_move`, `unmake_move`, and `_NATIVE_SOLVER_LOCK`.
- Produces: `ExactDoubleDummyCppFastestSolver.analyze_forced_outcome(state, time_budget_seconds=1.0) -> dict` with `status`, `team0_final_tricks`, `nil_broken_mask`, `nodes_searched`, and `elapsed_ms`.

- [ ] **Step 1: Write the Python oracle and failing API tests**

The oracle must enumerate all legal cards:

```python
def terminal_signatures(state: GameState) -> set[tuple[int, int]]:
    if state.tricks_played >= 13:
        team0 = state.tricks_won[0] + state.tricks_won[2]
        nil_mask = sum(
            1 << seat
            for seat, bid in enumerate(state.max_bid)
            if bid in ("nil", "blind_nil") and state.tricks_won[seat] > 0
        )
        return {(team0, nil_mask)}
    player = state.turn
    result: set[tuple[int, int]] = set()
    for card in SpadesRules().playable(state, state.hands[player], player):
        result.update(terminal_signatures(apply_reference_card(state, player, card)))
    return result
```

Use these exact two-trick fixtures, with seat 0 leading, Spades broken,
`tricks_played=11`:

```python
TEAM_VARIABLE = (
    [["TD", "QH"], ["6H", "KS"], ["8S", "5H"], ["2H", "3S"]],
    [3, 3, 3, 2],
    ["bid_1"] * 4,
    {(6, 0), (7, 0)},
)
NIL_ONLY_VARIABLE = (
    [["4C", "5H"], ["6H", "AS"], ["QS", "8H"], ["7H", "3C"]],
    [0, 4, 4, 3],
    ["nil", "bid_1", "bid_1", "bid_1"],
    {(5, 0), (5, 1)},
)
FIXED_NIL = (
    [["4H", "9S"], ["TH", "KS"], ["7H", "5D"], ["3S", "4S"]],
    [0, 4, 4, 3],
    ["nil", "bid_1", "bid_1", "bid_1"],
    {(4, 0)},
)
```

Assert the oracle output first, then assert the missing native method returns
`variable`, `variable`, and `fixed` with signature `(4, 0)`.

- [ ] **Step 2: Run RED**

Run `pytest -q tests/test_forced_outcome_native.py -x`.

Expected: `AttributeError` for `analyze_forced_outcome`.

- [ ] **Step 3: Split filtered and unfiltered legal generation**

Keep existing minimax behavior and add an all-actions entry point:

```cpp
static void legal_actions_impl(
    const NativeState* s,
    int32_t player_id,
    ActionList& actions,
    bool remove_equivalent
) {
    actions.clear();
    uint64_t allowed = s->hand_bits[player_id];
    if (allowed == 0) return;
    if (s->table_count == 0 && !s->spades_broken) {
        const uint64_t non_spades = allowed & ~0x1FFFULL;
        if (non_spades != 0) allowed = non_spades;
    } else if (s->table_count > 0) {
        const int32_t suit = s->table_suits[0];
        const uint64_t mask = 0x1FFFULL << (suit * 13);
        if ((allowed & mask) != 0) allowed &= mask;
    }
    while (allowed) {
        const int32_t cid = ctz64(allowed);
        actions.push(cid);
        allowed &= allowed - 1;
    }
    if (remove_equivalent) filter_equivalent(s, player_id, actions);
}

static void legal_actions(const NativeState* s, int32_t pid, ActionList& out) {
    legal_actions_impl(s, pid, out, true);
}

static void legal_actions_all(const NativeState* s, int32_t pid, ActionList& out) {
    legal_actions_impl(s, pid, out, false);
}
```

- [ ] **Step 4: Implement the separate deadline-aware DFS and TT**

Add C ABI types:

```cpp
enum ForcedOutcomeStatus : int32_t {
    FORCED_FIXED = 1,
    FORCED_VARIABLE = 2,
    FORCED_TIMEOUT = 3,
};

struct ForcedOutcomeResult {
    int32_t status;
    int32_t team0_final_tricks;
    uint32_t nil_broken_mask;
    uint64_t nodes_searched;
    double elapsed_ms;
};
```

Use a separate generation-tagged table with primary and verification hashes.
The key covers hands, table cards/seats, turn, leader, Spades broken,
completed tricks, team-0 current tricks, and current Nil mask. Cache only
exact fixed/variable results.

Implement the union rule:

```cpp
static ForcedValue forced_search(NativeState* s, ForcedContext& ctx) {
    ctx.nodes++;
    if (std::chrono::steady_clock::now() >= ctx.deadline)
        return {FORCED_TIMEOUT, -1, 0};
    if (s->tricks_played >= 13) return terminal_forced_value(s);

    ForcedValue cached;
    if (forced_tt_lookup(s, ctx, &cached)) return cached;

    ActionList actions;
    legal_actions_all(s, s->turn, actions);
    bool have = false;
    ForcedValue first{FORCED_FIXED, -1, 0};
    for (int i = 0; i < actions.count; i++) {
        const UndoInfo undo = make_move(s, s->turn, actions.cards[i]);
        const ForcedValue child = forced_search(s, ctx);
        unmake_move(s, undo);
        if (child.status == FORCED_TIMEOUT) return child;
        if (child.status == FORCED_VARIABLE) return forced_store_variable(s, ctx);
        if (!have) {
            first = child;
            have = true;
        } else if (first.team0_final_tricks != child.team0_final_tricks ||
                   first.nil_broken_mask != child.nil_broken_mask) {
            return forced_store_variable(s, ctx);
        }
    }
    forced_tt_store(s, ctx, first);
    return first;
}
```

Expose
`analyze_forced_outcome_native(const NativeState*, int64_t budget_us, ForcedOutcomeResult*)`.

- [ ] **Step 5: Add the `ctypes` wrapper with timed lock acquisition**

Define `_ForcedOutcomeResult`, configure the new symbol, and implement:

```python
def analyze_forced_outcome(self, state: GameState, time_budget_seconds: float = 1.0):
    self._validate_state(state)
    started = time.monotonic()
    budget = max(0.0, float(time_budget_seconds))
    if budget == 0.0 or not self.native_available:
        return self._forced_timeout_result(started)
    if not _NATIVE_SOLVER_LOCK.acquire(timeout=budget):
        return self._forced_timeout_result(started)
    try:
        remaining = budget - (time.monotonic() - started)
        if remaining <= 0.0:
            return self._forced_timeout_result(started)
        native = self._to_native_state(state)
        output = _ForcedOutcomeResult()
        self._lib.analyze_forced_outcome_native(
            ctypes.byref(native),
            max(1, int(remaining * 1_000_000)),
            ctypes.byref(output),
        )
    finally:
        _NATIVE_SOLVER_LOCK.release()
    labels = {1: "fixed", 2: "variable", 3: "timeout"}
    return {
        "status": labels.get(int(output.status), "timeout"),
        "team0_final_tricks": int(output.team0_final_tricks),
        "nil_broken_mask": int(output.nil_broken_mask),
        "nodes_searched": int(output.nodes_searched),
        "elapsed_ms": (time.monotonic() - started) * 1000.0,
    }
```

- [ ] **Step 6: Test deadlines, lock wait, and cache safety**

Hold `_NATIVE_SOLVER_LOCK` through the existing fake library, call the new API
with `0.01`, and assert `timeout`, elapsed below 100 ms, and zero forced native
calls. Then run budget zero followed by a normal fixed search to prove timeout
does not pollute the TT.

- [ ] **Step 7: Run GREEN and regressions**

Run:

```bash
pytest -q tests/test_forced_outcome_native.py tests/test_exact_solver_thread_safety.py
pytest -q test_exact_solver_sample2.py tests/test_rule_exact_first4_player.py
```

Expected: all selected tests pass. Leave rebuilt `.so` files unstaged.

- [ ] **Step 8: Checkpoint safely**

Run `git diff --check` on the four task files and inspect `git status`. Because
the Python wrapper was already dirty at planning time, do not commit this task
unless the feature-only hunks can be staged without the earlier changes.

---

### Task 2: Shared Python Service and Deterministic Completion

**Files:**
- Create: `trick_taking/forced_outcome.py`
- Create: `tests/test_forced_outcome.py`

**Interfaces:**
- Consumes: Task 1's native wrapper.
- Produces: `check_for_showdown`, `outcome_signature`, `apply_showdown_continuation`, `ShowdownCheck`, and `ShowdownResolution.to_payload()`.

- [ ] **Step 1: Write failing service tests**

Require these types:

```python
@dataclass(frozen=True)
class ShowdownPlay:
    seat: int
    card: Card

@dataclass(frozen=True)
class ShowdownResolution:
    team_tricks: tuple[int, int]
    nil_outcomes: tuple[bool | None, bool | None, bool | None, bool | None]
    continuation: tuple[ShowdownPlay, ...]
    final_tricks_won: tuple[int, int, int, int]

@dataclass(frozen=True)
class ShowdownCheck:
    status: Literal["fixed", "variable", "timeout"]
    resolution: ShowdownResolution | None = None
```

Assert the fixed Nil fixture yields eight continuation cards, 13 final tricks,
team totals `(4, 9)`, and Nil outcomes `(True, None, None, None)`. Assert
mid-trick, six-card, duplicate-card, incomplete-bid, and signature-mismatch
states are rejected without returning fixed.

- [ ] **Step 2: Run RED**

Run `pytest -q tests/test_forced_outcome.py -x`.

Expected: import failure for `trick_taking.forced_outcome`.

- [ ] **Step 3: Implement signature and deterministic play**

```python
def outcome_signature(state: GameState) -> tuple[int, int]:
    team0 = sum(state.tricks_won[p] for p in range(4) if state.teams[p] == 0)
    nil_mask = sum(
        1 << p
        for p, bid in enumerate(state.max_bid)
        if bid in ("nil", "blind_nil") and state.tricks_won[p] > 0
    )
    return team0, nil_mask

def deterministic_continuation(state: GameState):
    resolved = copy.deepcopy(state)
    plays: list[ShowdownPlay] = []
    rules = SpadesRules()
    while not rules.end_trickgame(resolved):
        seat = resolved.turn
        legal = rules.playable(resolved, resolved.hands[seat], seat)
        card = min(legal, key=lambda candidate: candidate.card_id)
        plays.append(ShowdownPlay(seat, card))
        resolved.play_card_to_table(seat, card)
        if card.suit == Suit.SPADES:
            resolved.spades_broken = resolved.trump_broken = True
        resolved.turn = (seat + 1) % 4
        if resolved.trick_complete:
            winner = rules.winner_trick(resolved)
            resolved.complete_trick(winner)
            resolved.turn = resolved.trick_leader = winner
    return tuple(plays), resolved
```

Implement `apply_showdown_continuation(state, plays)` with the same mutation
sequence on a deep copy, validating the expected seat and held card at every
step and requiring a 13-trick terminal state. The remote server uses this
function after both confirmations.

- [ ] **Step 4: Implement validation, service dispatch, and serialization**

Validate complete-trick boundary, equal 1–5 card hands, deck uniqueness,
history/played consistency, complete bids, teams, and trick totals. Return
variable/timeout unchanged. For fixed, generate the line, verify its signature,
and serialize continuation cards in frontend rank-suit format such as `AS`.

- [ ] **Step 5: Run GREEN**

Run `pytest -q tests/test_forced_outcome.py tests/test_forced_outcome_native.py`.

Expected: all tests pass.

- [ ] **Step 6: Commit isolated new files**

After `git diff --check`, stage only the two new files, inspect the cached diff,
and commit `feat: add showdown resolution service`.

---

### Task 3: Local Full-Information HTTP Endpoint

**Files:**
- Modify: `gui/backend.py`
- Modify: `tests/test_gui_backend.py`

**Interfaces:**
- Consumes: Task 2's service.
- Produces: POST `/api/check-showdown` returning `{ok, status, resolution?}`.

- [ ] **Step 1: Write failing builder and endpoint tests**

Use `remainingHands`, completed tricks, bids, current trick, tricks won,
leader, and Spades-broken fields. Assert the full builder preserves all four
actual hands and the provider serializes a fixed result. Reject duplicates,
history mismatch, and an in-progress trick. Retain the existing test that
acting-state construction uses hidden-hand placeholders.

- [ ] **Step 2: Run RED**

Run `pytest -q tests/test_gui_backend.py -x`.

Expected: missing `build_full_showdown_state` or `check_showdown`.

- [ ] **Step 3: Add a separate authoritative builder**

Parse hands exactly with:

```python
hands = [
    [parse_card_code(str(code)) for code in hand]
    for hand in payload["remainingHands"]
]
```

Populate all solver-relevant `GameState` fields and cross-check payload
`tricksWon` against history-derived winners. Do not call this builder from
`choose_action`.

- [ ] **Step 4: Add provider and route dispatch**

```python
def check_showdown(self, payload: dict[str, Any]) -> dict[str, Any]:
    state = build_full_showdown_state(payload)
    return check_for_showdown(
        state,
        self.exact_solver,
        time_budget_seconds=1.0,
    ).to_payload()
```

Dispatch `/api/check-showdown` separately from `/api/choose-action`. Return 200
for all normal statuses, 400 for validation errors, and 500 for unexpected
failures. Do not touch mutable players or `_decision_lock` on this path.

- [ ] **Step 5: Run GREEN and privacy regressions**

Run `pytest -q tests/test_gui_backend.py tests/test_rule_exact_first4_player.py`.

Expected: all tests pass.

- [ ] **Step 6: Checkpoint safely**

Run `git diff --check -- gui/backend.py tests/test_gui_backend.py`. Leave the
already-dirty backend unstaged unless feature-only hunks can be isolated.

---

### Task 4: Local JavaScript Detection and Settlement

**Files:**
- Modify: `gui/src/game.js`
- Modify: `gui/src/game.test.js`

**Interfaces:**
- Consumes: Task 3's endpoint.
- Produces: `shouldCheckShowdown`, `buildShowdownPayload`, `applyShowdownOffer`, `confirmLocalShowdown`, and paused local coordinators.

- [ ] **Step 1: Write failing trigger and reducer tests**

Assert no detection with six cards or an in-progress trick, detection with five
cards at an empty table, fixed response creates pending state without scoring,
variable/timeout/error preserve state, and confirmation yields phase
`finished`, 13 completed tricks, and backend-projected final trick counts.

- [ ] **Step 2: Run RED**

Run `cd gui && npm test`.

Expected: missing showdown exports.

- [ ] **Step 3: Add state shape, trigger, and payload**

```javascript
export function shouldCheckShowdown(state) {
  if (state.phase !== 'playing' || state.trickComplete || state.currentTrick.length !== 0) return false;
  if (state.showdown) return false;
  const sizes = state.hands.map((hand) => hand.length);
  return sizes.every((size) => size === sizes[0]) && sizes[0] >= 1 && sizes[0] <= 5;
}
```

Initialize and clone `showdown: null`. `buildShowdownPayload` includes complete
remaining hands only for this endpoint.

- [ ] **Step 4: Add bounded request and pending reducer**

POST to `/api/check-showdown` with an `AbortController` guard of 1.1 seconds.
Only `fixed` calls `applyShowdownOffer`; every other result returns the current
state unchanged. Do not apply continuation or calculate score yet.

- [ ] **Step 5: Apply continuation only on confirmation**

```javascript
export function confirmLocalShowdown(state) {
  if (!state.showdown || state.showdown.status !== 'pending') return state;
  let next = { ...cloneState(state), showdown: null };
  for (const play of state.showdown.resolution.continuation) {
    if (next.currentPlayer !== play.seat) throw new Error('Showdown continuation seat mismatch');
    next = applyCard(next, play.seat, play.card, 'showdown');
    if (next.trickComplete) next = finalizeTrick(next);
  }
  if (next.phase !== 'finished' || next.completedTricks.length !== 13)
    throw new Error('Showdown continuation did not produce a complete hand');
  return next;
}
```

- [ ] **Step 6: Detect immediately after trick collection**

After `finalizeTrick`, call one shared asynchronous detection helper. If it
returns pending, emit and stop both `advanceUntilHuman` and
`advanceUntilFinished`. Never detect before the fourth card is collected.

- [ ] **Step 7: Run GREEN**

Run `cd gui && npm test`.

Expected: all game tests pass.

- [ ] **Step 8: Checkpoint safely**

Run `git diff --check -- gui/src/game.js gui/src/game.test.js`; both files were
already changed before this feature, so do not stage earlier hunks.

---

### Task 5: Face-Up UI and Local Confirmation

**Files:**
- Create: `gui/src/showdown.js`
- Create: `gui/src/showdown.test.js`
- Modify: `gui/src/App.jsx`
- Modify: `gui/src/styles.css`
- Modify: `gui/package.json`

**Interfaces:**
- Consumes: Task 4's pending state and confirmation reducer.
- Produces: `ShowdownPanel`, `showdownHandsForDisplay`, face-up seat spreads, and an AI-test confirmation path.

- [ ] **Step 1: Write failing server-render tests**

Use `react-dom/server`:

```javascript
const html = renderToStaticMarkup(ShowdownPanel({
  showdown: FIXED_SHOWDOWN,
  bids: FIXED_BIDS,
  waitingForPartner: false,
  onConfirm: () => {},
}));
assert.match(html, /NS 4/);
assert.match(html, /EW 9/);
assert.match(html, /North Nil.*成功/);
assert.match(html, /确认结算/);
```

Assert `showdownHandsForDisplay` returns all four real hands only while pending.

- [ ] **Step 2: Run RED**

Run `cd gui && node --test src/showdown.test.js`.

Expected: module-not-found for `showdown.js`.

- [ ] **Step 3: Implement the pure display module**

Use `React.createElement` so Node imports the component without JSX tooling.
Accept the current four-seat `bids` array, render team totals and only the
non-null Nil rows with their Nil/Blind Nil labels, and render either `确认结算`
or a disabled `等待搭档确认` button.

- [ ] **Step 4: Integrate into `App.jsx`**

When pending, pass actual cards to each `AiSeat` and render
`ReplayHandSpread` instead of `CardBackFan`; keep the bottom hand face up and
disabled. Local confirmation calls `confirmLocalShowdown`. AI-test mode remains
on the table until confirmation, then builds replay and enters replay view.

- [ ] **Step 5: Add styles that keep all hands readable**

Add `.overlay--showdown` with a light translucent backdrop and no blur, plus
Nil-list, waiting-state, and 1–5-card revealed spread rules.

- [ ] **Step 6: Run GREEN and build**

Set the package script to:

```json
"test": "node --test src/game.test.js src/showdown.test.js"
```

Run `cd gui && npm test && npm run build`.

Expected: tests and Vite build exit 0.

- [ ] **Step 7: Visually inspect all four local modes**

Use a fixed pending state in single-hand, 500-point, fixed-seed, and four-AI
test modes. Verify four readable face-up hands, no score before confirmation,
normal score after confirmation, 500-point accumulation happens exactly once,
and four-AI test does not jump to replay prematurely.

- [ ] **Step 8: Checkpoint safely**

Run `git diff --check` on the five files. Stage new files and clean-file hunks
only; do not absorb the earlier `gui/package.json` change.

---

### Task 6: Remote Server Detection and Confirmation Barrier

**Files:**
- Create: `tests/test_game_server_showdown.py`
- Modify: `gui/game_server.py`

**Interfaces:**
- Consumes: Task 2's service and continuation applier.
- Produces: revealed `game_state.showdown`, sender-aware confirmation, and settlement after two distinct human confirmations.

- [ ] **Step 1: Write failing async tests with `asyncio.run`**

With two fake sockets and a fake fixed checker, assert offer broadcasts all
hands, one confirmation leaves phase playing, the second finishes, duplicate
and stale confirmations do not count, timeout makes no offer, and disconnect
does not count as consent.

- [ ] **Step 2: Run RED**

Run `pytest -q tests/test_game_server_showdown.py -x`.

Expected: missing showdown room methods/fields.

- [ ] **Step 3: Add injected checker and room state**

Add a `showdown_checker=check_for_showdown` constructor dependency plus
`showdown_id`, `showdown_resolution`, `showdown_confirmations`,
`showdown_pending`, and a dedicated `asyncio.Event`. Run the blocking checker
in the executor on a deep-copied state.

- [ ] **Step 4: Broadcast revealed hands only after fixed**

The pending payload is:

```python
{
    "id": self.showdown_id,
    "revealedHands": [
        [_card_to_str(card) for card in _sort_hand_for_display(hand)]
        for hand in self.state.hands
    ],
    "teamTricks": list(self.showdown_resolution.team_tricks),
    "nilOutcomes": list(self.showdown_resolution.nil_outcomes),
    "confirmedSeats": sorted(self.showdown_confirmations),
}
```

Ordinary `game_state` messages continue hiding opponents.

- [ ] **Step 5: Make incoming actions sender-aware**

Change `receive_action(sender_seat, action)`. Accept a showdown confirmation
only for the active ID and a connected human seat; add it to a set and signal
the dedicated event. Ordinary bid/play actions are accepted only from the
currently expected human seat. Pass `my_seat` from the WebSocket handler. The
confirmation-wait loop broadcasts refreshed `confirmedSeats` after every
newly accepted confirmation so both clients see the same barrier state.

- [ ] **Step 6: Integrate after every completed trick**

After awarding and broadcasting a trick, check when 1–5 remain. On fixed,
pause, wait until the original two human seats confirm, apply the stored legal
continuation, broadcast the completed state, set `Phase.SCORING`, and return.
A missing connection raises and terminates the room rather than confirming.

- [ ] **Step 7: Run GREEN**

Run `pytest -q tests/test_game_server_showdown.py tests/test_gui_backend.py`.

Expected: all tests pass.

- [ ] **Step 8: Commit isolated server work**

Run `git diff --check -- gui/game_server.py tests/test_game_server_showdown.py`.
The server was clean at planning time; stage only these two files, inspect the
cached diff, and commit `feat: add remote showdown confirmation barrier`.

---

### Task 7: Remote Client Reveal and Per-Player Confirmation

**Files:**
- Modify: `gui/src/game.js`
- Modify: `gui/src/game.test.js`
- Modify: `gui/src/App.jsx`
- Modify: `gui/src/showdown.js`
- Modify: `gui/src/showdown.test.js`

**Interfaces:**
- Consumes: Task 6's `game_state.showdown` protocol.
- Produces: privacy-preserving parsing, one confirmation message per client, and waiting-for-partner presentation.

- [ ] **Step 1: Write failing remote parser tests**

Assert a message with showdown uses all four `revealedHands`, stores the ID,
and marks `locallyConfirmed` from `confirmedSeats`. Assert an ordinary message
still creates opponent placeholder arrays and has no revealed cards.

- [ ] **Step 2: Run RED**

Run `cd gui && npm test`.

Expected: remote showdown assertions fail.

- [ ] **Step 3: Parse reveal state without weakening ordinary privacy**

In `remoteStateFromServer`, use `revealedHands` only when `msg.showdown` exists;
otherwise retain current own-hand-plus-count placeholders. Export and test
`showdownWaitingForPartner(showdown, mySeat)`.

- [ ] **Step 4: Send and refresh confirmations**

The button sends one `{type: 'showdown_confirm', showdownId}` message and
disables locally. Later `game_state` messages refresh `confirmedSeats`.
`hand_over` clears showdown and applies final score/trick totals.

- [ ] **Step 5: Run GREEN and build**

Run `cd gui && npm test && npm run build`.

Expected: all tests pass and Vite exits 0.

- [ ] **Step 6: Manual two-client check**

Verify the first confirmer sees `等待搭档确认`, the second still has an enabled
button, and both receive the same final result only after the second confirms.

- [ ] **Step 7: Checkpoint safely**

Run `git diff --check` on the five task files and preserve all pre-feature
`game.js`/test hunks.

---

### Task 8: Full Verification and Five-Trick Benchmark

**Files:**
- Verify only; do not modify generated native libraries.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: fresh correctness, build, regression, deadline, and performance evidence.

- [ ] **Step 1: Run all Python tests**

Run `pytest -q`.

Expected: zero failures; record exact passed/skipped counts.

- [ ] **Step 2: Run all frontend checks**

Run `cd gui && npm test && npm run build`.

Expected: both commands exit 0.

- [ ] **Step 3: Run randomized native-oracle parity**

Run `pytest -q tests/test_forced_outcome_native.py -k randomized_oracle_parity`
with at least 200 deterministic legal states containing up to three tricks.

Expected: exact status/signature agreement for every state.

- [ ] **Step 4: Benchmark five-trick positions**

Generate at least 100 deterministic legal five-card-per-hand states and call
the wrapper with a 1.0-second budget. Report fixed/variable/timeout counts,
median, p95, and maximum elapsed milliseconds. Individual timeouts are valid;
under idle local load wrapper elapsed must not exceed 1.10 seconds.

- [ ] **Step 5: Prove no training/evaluation integration**

Run:

```bash
rg -n "check_for_showdown|analyze_forced_outcome" rl evaluate strategy trick_taking/driver.py
```

Expected: no call sites in those paths.

- [ ] **Step 6: Inspect final state**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Confirm no generated `.so`, checkpoint, benchmark output, screenshot, or
unrelated pre-existing hunk is staged. Report pre-existing and feature changes
separately.
