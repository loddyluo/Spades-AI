import assert from 'node:assert/strict';
import test from 'node:test';

import {
  PACE,
  advanceUntilFinished,
  advanceUntilHuman,
  applyCard,
  applyShowdownOffer,
  buildAiPayload,
  buildReplayAnalysisPayload,
  buildReplayRecord,
  buildReplaySnapshot,
  buildShowdownPayload,
  computeScores,
  confirmLocalShowdown,
  createInitialGame,
  dealHands,
  determineTrickWinner,
  finalizeTrick,
  getLegalCards,
  parseReplayImport,
  remoteCompletedTricksFromServer,
  remoteStateFromServer,
  shouldCheckShowdown,
  showdownWaitingForPartner,
} from './game.js';

function playingState(card) {
  return {
    phase: 'playing',
    currentPlayer: 0,
    spadesBroken: false,
    hands: [[card], [], [], []],
    bids: [null, null, null, null],
    tricksWon: [0, 0, 0, 0],
    currentTrick: [],
    completedTricks: [],
    log: [],
  };
}

test('leading a forced spade breaks spades', () => {
  const state = playingState({ code: 'AS', rank: 'A', suit: 'S' });

  const next = applyCard(state, 0, 'AS');

  assert.equal(next.spadesBroken, true);
});

test('leading a non-spade does not break spades', () => {
  const state = playingState({ code: 'AH', rank: 'A', suit: 'H' });

  const next = applyCard(state, 0, 'AH');

  assert.equal(next.spadesBroken, false);
});

test('an AI play keeps the exact decision diagnostics on the play record', () => {
  const state = playingState({ code: 'AH', rank: 'A', suit: 'H' });
  const analysis = {
    schema_version: 1,
    mode: 'exact_is_determinized',
    chosen_card: 'AH',
    action_scores: [{ action: 'AH', value: 0 }],
  };

  const next = applyCard(state, 0, 'AH', 'test-ai', analysis);

  assert.equal(next.currentTrick[0].aiAnalysis, analysis);
});

test('collecting a trick keeps AI diagnostics for the finished replay', () => {
  const analysis = {
    schema_version: 1,
    mode: 'exact_is_determinized',
    chosen_card: 'AH',
    action_scores: [{ action: 'AH', value: 0 }],
  };
  const state = {
    phase: 'playing',
    currentPlayer: -1,
    leader: 0,
    trickNumber: 1,
    spadesBroken: false,
    hands: [[], [], [], []],
    bids: [null, null, null, null],
    tricksWon: [0, 0, 0, 0],
    currentTrick: [
      { seat: 0, card: { code: 'AH', rank: 'A', suit: 'H' }, aiAnalysis: analysis },
      { seat: 1, card: { code: 'KH', rank: 'K', suit: 'H' } },
      { seat: 2, card: { code: 'QH', rank: 'Q', suit: 'H' } },
      { seat: 3, card: { code: 'JH', rank: 'J', suit: 'H' } },
    ],
    completedTricks: [],
    trickComplete: true,
    trickWinner: 0,
    lastPlayedSeat: 3,
    showdown: null,
    log: [],
  };

  const next = finalizeTrick(state);

  assert.deepEqual(next.completedTricks[0].cards[0].aiAnalysis, analysis);
  assert.equal(next.completedTricks[0].cards[1].aiAnalysis, undefined);
});

test('a non-spade cannot reset an already-broken state', () => {
  const state = playingState({ code: 'AH', rank: 'A', suit: 'H' });
  state.spadesBroken = true;

  const next = applyCard(state, 0, 'AH');

  assert.equal(next.spadesBroken, true);
});

test('a fourth-card spade stays broken through trick completion', () => {
  const state = playingState({ code: '2S', rank: '2', suit: 'S' });
  state.hands = [[], [], [], [{ code: '2S', rank: '2', suit: 'S' }]];
  state.currentPlayer = 3;
  state.currentTrick = [
    { seat: 0, card: { code: 'AH', rank: 'A', suit: 'H' } },
    { seat: 1, card: { code: 'KH', rank: 'K', suit: 'H' } },
    { seat: 2, card: { code: 'QH', rank: 'Q', suit: 'H' } },
  ];

  const next = applyCard(state, 3, '2S');

  assert.equal(next.trickComplete, true);
  assert.equal(next.trickWinner, 3);
  assert.equal(next.spadesBroken, true);
});


function showdownBoundary(handSize = 5) {
  const deck = [];
  const suits = ['C', 'D', 'H', 'S'];
  const ranks = ['2', '3', '4', '5', '6', '7'];
  for (let seat = 0; seat < 4; seat += 1) {
    deck.push(ranks.slice(0, handSize).map((rank) => ({
      code: `${rank}${suits[seat]}`,
      rank,
      suit: suits[seat],
    })));
  }
  return {
    seed: 1,
    humanSeat: 2,
    firstSeat: 0,
    phase: 'playing',
    currentPlayer: 0,
    leader: 0,
    trickNumber: 14 - handSize,
    spadesBroken: true,
    hands: deck,
    bids: [
      { value: 2, type: 'normal' },
      { value: 0, type: 'nil' },
      { value: 3, type: 'normal' },
      { value: 1, type: 'normal' },
    ],
    tricksWon: [2, 0, 3, 3],
    currentTrick: [],
    completedTricks: [],
    trickComplete: false,
    trickWinner: -1,
    lastPlayedSeat: -1,
    lastBidSeat: -1,
    score: null,
    showdown: null,
    log: [],
  };
}


test('showdown detection starts only at an empty boundary with at most five tricks', () => {
  assert.equal(shouldCheckShowdown(showdownBoundary(5)), true);
  assert.equal(shouldCheckShowdown(showdownBoundary(6)), false);

  const midTrick = showdownBoundary(5);
  midTrick.currentTrick.push({ seat: 0, card: midTrick.hands[0][0] });
  assert.equal(shouldCheckShowdown(midTrick), false);

  const heldTrick = showdownBoundary(5);
  heldTrick.trickComplete = true;
  assert.equal(shouldCheckShowdown(heldTrick), false);

  const alreadyOffered = showdownBoundary(5);
  alreadyOffered.showdown = { status: 'pending' };
  assert.equal(shouldCheckShowdown(alreadyOffered), false);
});


test('showdown payload is the only local payload containing all four hands', () => {
  const state = showdownBoundary(5);

  const payload = buildShowdownPayload(state);

  assert.deepEqual(payload.remainingHands, state.hands.map((hand) => hand.map((card) => card.code)));
  assert.deepEqual(payload.tricksWon, state.tricksWon);
  assert.deepEqual(payload.currentTrick, []);
});


test('acting bidder payload carries the reproducible deal seed', () => {
  const state = showdownBoundary(5);

  const payload = buildAiPayload(state);

  assert.equal(payload.seed, state.seed);
  assert.equal(payload.firstSeat, state.firstSeat);
  assert.equal(payload.currentPlayer, state.currentPlayer);
  assert.equal('remainingHands' in payload, false);
});


test('a failed AI request pauses a local game without applying a fallback bid', async () => {
  const state = createInitialGame(101, 1);
  const priorFetch = globalThis.fetch;
  const emitted = [];
  globalThis.fetch = async () => ({
    ok: false,
    status: 500,
    async text() {
      return JSON.stringify({ ok: false, error: 'worker OOM' });
    },
  });

  try {
    await assert.rejects(
      () => advanceUntilHuman(state, (step) => emitted.push(step)),
      /AI 后端请求失败.*worker OOM/,
    );
    assert.deepEqual(state.bids, [null, null, null, null]);
    assert.equal(state.log.length, 1);
    assert.deepEqual(emitted, []);
  } finally {
    globalThis.fetch = priorFetch;
  }
});


test('AI test mode rejects a backend-declared bidding fallback', async () => {
  const state = createInitialGame(202, 0);
  const priorFetch = globalThis.fetch;
  const emitted = [];
  globalThis.fetch = async () => ({
    ok: true,
    async json() {
      return {
        ok: true,
        kind: 'bid',
        ai: 'rule_exact',
        bid: { value: 3, type: 'normal' },
        detail: 'residual_fallback_nsfp',
      };
    },
  });

  try {
    await assert.rejects(
      () => advanceUntilFinished(state, (step) => emitted.push(step)),
      /AI 后端触发 fallback.*residual_fallback_nsfp/,
    );
    assert.deepEqual(state.bids, [null, null, null, null]);
    assert.deepEqual(emitted, []);
  } finally {
    globalThis.fetch = priorFetch;
  }
});


test('a backend-declared card fallback pauses before playing the card', async () => {
  const state = createInitialGame(303, 1);
  state.phase = 'playing';
  state.currentPlayer = 0;
  state.leader = 0;
  state.bids = Array.from({ length: 4 }, () => ({ value: 3, type: 'normal' }));
  const priorFetch = globalThis.fetch;
  const emitted = [];
  globalThis.fetch = async () => ({
    ok: true,
    async json() {
      return {
        ok: true,
        kind: 'play',
        ai: 'rule_exact',
        card: state.hands[0][0].code,
        detail: 'exact_no_match_fallback',
      };
    },
  });

  try {
    await assert.rejects(
      () => advanceUntilHuman(state, (step) => emitted.push(step)),
      /AI 后端触发 fallback.*exact_no_match_fallback/,
    );
    assert.equal(state.hands[0].length, 13);
    assert.deepEqual(state.currentTrick, []);
    assert.deepEqual(emitted, []);
  } finally {
    globalThis.fetch = priorFetch;
  }
});


test('a fixed offer pauses without completing or scoring the hand', () => {
  const state = showdownBoundary(5);
  const response = {
    ok: true,
    status: 'fixed',
    resolution: {
      teamTricks: [7, 6],
      nilOutcomes: [null, true, null, null],
      finalTricksWon: [4, 0, 3, 6],
      continuation: [],
    },
  };

  const next = applyShowdownOffer(state, response);

  assert.notEqual(next, state);
  assert.equal(next.phase, 'playing');
  assert.equal(next.score, null);
  assert.equal(next.showdown.status, 'pending');
  assert.deepEqual(next.hands, state.hands);

  for (const status of ['variable', 'timeout']) {
    assert.equal(applyShowdownOffer(state, { ok: true, status }), state);
  }
  assert.equal(applyShowdownOffer(state, { ok: false, status: 'fixed' }), state);
});


function oneTrickShowdownState() {
  const state = showdownBoundary(1);
  state.hands = [
    [{ code: 'AH', rank: 'A', suit: 'H' }],
    [{ code: 'KH', rank: 'K', suit: 'H' }],
    [{ code: 'QH', rank: 'Q', suit: 'H' }],
    [{ code: 'JH', rank: 'J', suit: 'H' }],
  ];
  state.tricksWon = [3, 3, 3, 3];
  state.trickNumber = 13;
  state.completedTricks = Array.from({ length: 12 }, (_, index) => ({
    trickNumber: index + 1,
    winner: index % 4,
    cards: [],
  }));
  state.showdown = {
    status: 'pending',
    resolution: {
      teamTricks: [7, 6],
      nilOutcomes: [null, false, null, null],
      finalTricksWon: [4, 3, 3, 3],
      continuation: [
        { seat: 0, card: 'AH' },
        { seat: 1, card: 'KH' },
        { seat: 2, card: 'QH' },
        { seat: 3, card: 'JH' },
      ],
    },
  };
  return state;
}


test('local confirmation applies the stored line and settles exactly once', () => {
  const state = oneTrickShowdownState();

  const finished = confirmLocalShowdown(state);

  assert.equal(finished.phase, 'finished');
  assert.equal(finished.showdown, null);
  assert.equal(finished.completedTricks.length, 13);
  assert.deepEqual(finished.tricksWon, [4, 3, 3, 3]);
  assert.deepEqual(finished.score, { northSouth: 32, eastWest: -85 });
  assert.equal(confirmLocalShowdown(finished), finished);
});


test('local confirmation rejects a continuation that disagrees with the projection', () => {
  const state = oneTrickShowdownState();
  state.showdown.resolution.finalTricksWon = [3, 4, 3, 3];

  assert.throws(
    () => confirmLocalShowdown(state),
    /projected trick totals/,
  );
});


test('local coordinator checks only after collecting the completed trick', async () => {
  const state = showdownBoundary(5);
  state.trickNumber = 8;
  state.tricksWon = [2, 0, 2, 3];
  state.completedTricks = Array.from({ length: 7 }, (_, index) => ({
    trickNumber: index + 1,
    winner: index % 4,
    cards: [],
  }));
  state.currentTrick = [
    { seat: 0, card: { code: 'AH', rank: 'A', suit: 'H' } },
    { seat: 1, card: { code: 'KH', rank: 'K', suit: 'H' } },
    { seat: 2, card: { code: 'QH', rank: 'Q', suit: 'H' } },
    { seat: 3, card: { code: 'JH', rank: 'J', suit: 'H' } },
  ];
  state.trickComplete = true;
  state.trickWinner = 0;
  state.currentPlayer = -1;
  const priorFetch = globalThis.fetch;
  const priorHold = PACE.trickHold;
  let posted = null;
  PACE.trickHold = 0;
  globalThis.fetch = async (_url, options) => {
    posted = JSON.parse(options.body);
    return {
      ok: true,
      async json() {
        return {
          ok: true,
          status: 'fixed',
          resolution: {
            teamTricks: [7, 6],
            nilOutcomes: [null, true, null, null],
            finalTricksWon: [3, 0, 4, 6],
            continuation: [],
          },
        };
      },
    };
  };

  try {
    const pending = await advanceUntilHuman(state);
    assert.equal(pending.showdown.status, 'pending');
    assert.equal(pending.currentTrick.length, 0);
    assert.equal(pending.completedTricks.length, 8);
    assert.equal(posted.currentTrick.length, 0);
    assert.deepEqual(posted.remainingHands, state.hands.map((hand) => hand.map((card) => card.code)));
  } finally {
    globalThis.fetch = priorFetch;
    PACE.trickHold = priorHold;
  }
});


test('a failed automatic showdown request pauses instead of silently continuing', async () => {
  const state = showdownBoundary(5);
  state.trickNumber = 8;
  state.tricksWon = [2, 0, 2, 3];
  state.completedTricks = Array.from({ length: 7 }, (_, index) => ({
    trickNumber: index + 1,
    winner: index % 4,
    cards: [],
  }));
  state.currentTrick = [
    { seat: 0, card: { code: 'AH', rank: 'A', suit: 'H' } },
    { seat: 1, card: { code: 'KH', rank: 'K', suit: 'H' } },
    { seat: 2, card: { code: 'QH', rank: 'Q', suit: 'H' } },
    { seat: 3, card: { code: 'JH', rank: 'J', suit: 'H' } },
  ];
  state.trickComplete = true;
  state.trickWinner = 0;
  state.currentPlayer = -1;
  const priorFetch = globalThis.fetch;
  const priorHold = PACE.trickHold;
  PACE.trickHold = 0;
  globalThis.fetch = async () => ({
    ok: false,
    status: 503,
    async text() {
      return 'showdown worker unavailable';
    },
  });

  try {
    await assert.rejects(
      () => advanceUntilHuman(state),
      /Showdown check failed.*showdown worker unavailable/,
    );
  } finally {
    globalThis.fetch = priorFetch;
    PACE.trickHold = priorHold;
  }
});


function remoteMessage(showdown = null) {
  return {
    type: 'game_state',
    seat: 0,
    phase: 'playing',
    currentPlayer: 2,
    leader: 2,
    trickNumber: 12,
    spadesBroken: true,
    hand: ['AS', '2H'],
    handSizes: [2, 2, 2, 2],
    bids: [
      { value: 0, type: 'nil' },
      { value: 2, type: 'normal' },
      { value: 3, type: 'normal' },
      { value: 2, type: 'normal' },
    ],
    tricksWon: [0, 4, 4, 3],
    currentTrick: [],
    completedTricks: [],
    showdown,
  };
}


test('remote showdown parsing reveals all hands and tracks this player confirmation', () => {
  const message = remoteMessage({
    id: 7,
    revealedHands: [
      ['AS', '2H'],
      ['KS', '3H'],
      ['QS', '4H'],
      ['JS', '5H'],
    ],
    teamTricks: [4, 9],
    nilOutcomes: [true, null, null, null],
    confirmedSeats: [0],
  });

  const state = remoteStateFromServer(message, 0);

  assert.deepEqual(state.hands.map((hand) => hand.map((card) => card.code)), message.showdown.revealedHands);
  assert.equal(state.showdown.id, 7);
  assert.equal(state.showdown.locallyConfirmed, true);
  assert.deepEqual(state.showdown.resolution.teamTricks, [4, 9]);
  assert.equal(showdownWaitingForPartner(state.showdown, 0), true);
  assert.equal(showdownWaitingForPartner(state.showdown, 2), false);
});


test('ordinary remote state retains opponent card privacy', () => {
  const message = remoteMessage(null);

  const state = remoteStateFromServer(message, 0);

  assert.deepEqual(state.hands[0].map((card) => card.code), ['AS', '2H']);
  assert.equal(state.hands[1].length, 2);
  assert.equal(state.hands[1].every((card) => card === undefined), true);
  assert.equal(state.showdown, null);
});


test('remote state retains the shared seed and public history for replay', () => {
  const seed = 20260724;
  const message = remoteMessage(null);
  message.completedTricks = [{
    trickNumber: 1,
    winner: 0,
    cards: [
      { seat: 0, card: 'AC' },
      { seat: 1, card: 'KC' },
      { seat: 2, card: 'QC' },
      { seat: 3, card: 'JC' },
    ],
  }];

  const state = remoteStateFromServer(message, 0, seed);
  const snapshot = buildReplaySnapshot({
    ...state,
    phase: 'finished',
    score: { northSouth: 42, eastWest: -20 },
  });

  assert.equal(snapshot.seed, seed);
  assert.deepEqual(
    snapshot.hands.map((hand) => hand.map((card) => card.code)),
    dealHands(seed).map((hand) => hand.map((card) => card.code)),
  );
  assert.deepEqual(
    snapshot.plays.map((play) => [play.seat, play.card.code]),
    [[0, 'AC'], [1, 'KC'], [2, 'QC'], [3, 'JC']],
  );
});


test('replay record exports a portable versioned game history', () => {
  const snapshot = {
    seed: 42,
    humanSeat: 2,
    bids: [
      { value: 3, type: 'normal' },
      { value: 0, type: 'nil' },
      { value: 2, type: 'normal' },
      { value: 4, type: 'normal' },
    ],
    hands: [
      [{ code: 'AC', rank: 'A', suit: 'C' }],
      [{ code: 'KC', rank: 'K', suit: 'C' }],
      [{ code: 'QC', rank: 'Q', suit: 'C' }],
      [{ code: 'JC', rank: 'J', suit: 'C' }],
    ],
    completedTricks: [{
      trickNumber: 1,
      winner: 0,
      cards: [
        { seat: 0, card: { code: 'AC', rank: 'A', suit: 'C' } },
        { seat: 1, card: { code: 'KC', rank: 'K', suit: 'C' } },
        { seat: 2, card: { code: 'QC', rank: 'Q', suit: 'C' } },
        { seat: 3, card: { code: 'JC', rank: 'J', suit: 'C' } },
      ],
    }],
    tricksWon: [1, 0, 0, 0],
    score: { northSouth: 31, eastWest: -40 },
  };

  assert.deepEqual(buildReplayRecord(snapshot), {
    format: 'spades-ai-replay',
    version: 1,
    seed: 42,
    viewSeat: 2,
    seats: ['North', 'East', 'South', 'West'],
    bids: snapshot.bids,
    initialHands: [['AC'], ['KC'], ['QC'], ['JC']],
    tricks: [{
      trickNumber: 1,
      leader: 0,
      winner: 0,
      plays: [
        { seat: 0, card: 'AC' },
        { seat: 1, card: 'KC' },
        { seat: 2, card: 'QC' },
        { seat: 3, card: 'JC' },
      ],
    }],
    tricksWon: [1, 0, 0, 0],
    score: { northSouth: 31, eastWest: -40 },
  });
});


function completeReplayRecord(seed = 20260804, viewSeat = 0) {
  const initialHands = dealHands(seed).map((hand) => hand.map((card) => ({ ...card })));
  const remainingHands = initialHands.map((hand) => hand.map((card) => ({ ...card })));
  const bids = [
    { value: 2, type: 'normal' },
    { value: 3, type: 'normal' },
    { value: 4, type: 'normal' },
    { value: 2, type: 'normal' },
  ];
  const tricks = [];
  const tricksWon = [0, 0, 0, 0];
  let leader = 0;
  let spadesBroken = false;

  for (let trickNumber = 1; trickNumber <= 13; trickNumber += 1) {
    const currentTrick = [];
    for (let offset = 0; offset < 4; offset += 1) {
      const seat = (leader + offset) % 4;
      const legalCards = getLegalCards(remainingHands[seat], currentTrick, spadesBroken);
      const card = legalCards[0];
      remainingHands[seat] = remainingHands[seat].filter((candidate) => candidate.code !== card.code);
      currentTrick.push({ seat, card: { ...card } });
      spadesBroken = spadesBroken || card.suit === 'S';
    }
    const winner = determineTrickWinner(currentTrick);
    tricksWon[winner] += 1;
    tricks.push({
      trickNumber,
      leader,
      winner,
      plays: currentTrick.map((entry) => ({ seat: entry.seat, card: entry.card.code })),
    });
    leader = winner;
  }

  return {
    format: 'spades-ai-replay',
    version: 1,
    seed,
    viewSeat,
    seats: ['North', 'East', 'South', 'West'],
    bids,
    initialHands: initialHands.map((hand) => hand.map((card) => card.code)),
    tricks,
    tricksWon,
    score: computeScores(bids, tricksWon),
  };
}


test('detailed AI diagnostics survive replay import and export', () => {
  const record = completeReplayRecord(20260804, 2);
  const analysis = {
    schema_version: 1,
    seat: record.tricks[4].plays[0].seat,
    chosen_card: record.tricks[4].plays[0].card,
    mode: 'exact_is_determinized',
    samples: 1,
    action_scores: [{ action: record.tricks[4].plays[0].card, value: 0 }],
    debug: { samples: [] },
  };
  record.tricks[4].plays[0].aiAnalysis = analysis;

  const [option] = parseReplayImport(record);
  const exported = buildReplayRecord(option.snapshot);

  assert.deepEqual(exported.tricks[4].plays[0].aiAnalysis, analysis);
  assert.deepEqual(option.snapshot.plays[16].aiAnalysis, analysis);
});


test('replay solver payload identifies one action without resending AI diagnostics', () => {
  const record = completeReplayRecord(20260804, 2);
  record.tricks[3].plays[0].aiAnalysis = {
    schema_version: 1,
    mode: 'exact_is_determinized',
  };
  const [option] = parseReplayImport(record);

  const payload = buildReplayAnalysisPayload(option.snapshot, 12);

  assert.equal(payload.seed, record.seed);
  assert.equal(payload.playIndex, 12);
  assert.equal(payload.firstLeader, record.tricks[0].leader);
  assert.equal(payload.initialHands.flat().length, 52);
  assert.equal(payload.plays.length, 52);
  assert.equal('aiAnalysis' in payload.plays[12], false);
});


test('remote hand-over history exposes replay diagnostics only in the final record', () => {
  const raw = [{
    trickNumber: 1,
    winner: 0,
    cards: [{
      seat: 0,
      card: 'AS',
      aiAnalysis: { schema_version: 1, mode: 'single_action_direct' },
    }],
  }];

  const parsed = remoteCompletedTricksFromServer(raw);

  assert.equal(parsed[0].cards[0].card.code, 'AS');
  assert.equal(parsed[0].cards[0].aiAnalysis.mode, 'single_action_direct');
});


test('portable replay records round-trip through strict import validation', () => {
  const record = completeReplayRecord(20260804, 2);

  const options = parseReplayImport(record);

  assert.equal(options.length, 1);
  assert.equal(options[0].snapshot.humanSeat, 2);
  assert.equal(options[0].snapshot.plays.length, 52);
  assert.equal(options[0].snapshot.completedTricks.length, 13);
  assert.deepEqual(buildReplayRecord(options[0].snapshot), record);
});


test('replay import sorts every hand using the standard display order', () => {
  const sortedRecord = completeReplayRecord(20260804);
  const shuffledRecord = {
    ...sortedRecord,
    initialHands: sortedRecord.initialHands.map((hand) => [...hand].reverse()),
  };

  const [option] = parseReplayImport(shuffledRecord);

  assert.deepEqual(
    option.snapshot.hands.map((hand) => hand.map((card) => card.code)),
    sortedRecord.initialHands,
  );
});


test('DeepSeek team-match records import with model seat labels and real team scores', () => {
  const record = completeReplayRecord(20260805);
  const currentPayoff = record.score.northSouth - record.score.eastWest;
  const document = {
    format: 'spades-ai-deepseek-team-match',
    version: 1,
    games: [{
      seed: record.seed,
      winner: currentPayoff > 0 ? 'current_spades_ai' : 'deepseek-v4-flash',
      current_ai_payoff: currentPayoff,
      deepseek_payoff: -currentPayoff,
      bids: record.bids.map((bid) => bid.type === 'nil' ? 'nil' : `bid_${bid.value}`),
      initial_hands: Object.fromEntries(record.initialHands.map((hand, seat) => [String(seat), hand])),
      tricks: record.tricks.map((trick, index) => ({
        index,
        leader: trick.leader,
        winner: trick.winner,
        cards: trick.plays,
      })),
      tricks_won: record.tricksWon,
      seat_assignment: {
        0: 'current_spades_ai',
        1: 'deepseek-v4-flash',
        2: 'current_spades_ai',
        3: 'deepseek-v4-flash',
      },
    }],
  };

  const [option] = parseReplayImport(document);

  assert.match(option.label, /种子 20260805/);
  assert.match(option.snapshot.seatNames[0], /当前 AI/);
  assert.match(option.snapshot.seatNames[1], /DeepSeek/);
  assert.deepEqual(option.snapshot.score, record.score);
  assert.equal(option.snapshot.plays.length, 52);
});


test('a replay summary exposes all embedded hands as selectable options', () => {
  const first = { ...completeReplayRecord(20260804), label: '第一局' };
  const second = { ...completeReplayRecord(20260805), label: '第二局' };
  const summary = {
    format: 'spades-deepseek-8-team-match-summary',
    version: 1,
    replay_records: [first, second],
  };

  const options = parseReplayImport(summary);

  assert.deepEqual(options.map((option) => option.label), ['第一局', '第二局']);
  assert.deepEqual(options.map((option) => option.snapshot.seed), [20260804, 20260805]);
});


test('a duplicate-match summary exposes both tables for replay', () => {
  const tableA = { ...completeReplayRecord(20260804), label: '副牌 20260804 · A 桌' };
  const tableB = { ...completeReplayRecord(20260804), label: '副牌 20260804 · B 桌' };
  const summary = {
    format: 'spades-deepseek-duplicate-match-summary',
    version: 1,
    replay_records: [tableA, tableB],
  };

  const options = parseReplayImport(summary);

  assert.deepEqual(options.map((option) => option.label), [
    '副牌 20260804 · A 桌',
    '副牌 20260804 · B 桌',
  ]);
});


test('replay import rejects illegal cards and index-only summaries with actionable errors', () => {
  const illegal = structuredClone(completeReplayRecord(20260804));
  illegal.tricks[0].plays[0].card = illegal.initialHands[1][0];

  assert.throws(() => parseReplayImport(illegal), /并不持有/);
  assert.throws(
    () => parseReplayImport({
      format: 'spades-deepseek-8-team-match-summary',
      version: 1,
      games: [],
    }),
    /只含统计索引/,
  );
});
