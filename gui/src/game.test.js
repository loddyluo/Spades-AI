import assert from 'node:assert/strict';
import test from 'node:test';

import {
  PACE,
  advanceUntilFinished,
  advanceUntilHuman,
  applyCard,
  applyShowdownOffer,
  buildAiPayload,
  buildRegretRecord,
  buildReplayRecord,
  buildReplaySnapshot,
  buildShowdownPayload,
  computeScores,
  confirmLocalShowdown,
  createInitialGame,
  dealHands,
  determineTrickWinner,
  fetchRegretJob,
  formatRegret,
  getLegalCards,
  parseReplayImport,
  REGRET_EPSILON,
  actionOutcomeDistribution,
  enumerateFinalOutcomes,
  qSamplesForCard,
  regretBadges,
  regretDecisionAt,
  regretSummary,
  regretTeamOf,
  remoteStateFromServer,
  shouldCheckShowdown,
  showdownWaitingForPartner,
  sortHandByRegret,
  startRegretAnalysis,
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

/* ── 完整复盘 (full replay regret) helpers ──────────────────────────── */

function regretDecisionFixture() {
  return {
    playIndex: 20,
    trickNumber: 6,
    seat: 1,
    forced: false,
    actualCard: 'QS',
    legalCards: ['AS', 'KS', 'QS'],
    bestCard: 'AS',
    playedQ: 4,
    playedRegret: 8,
    actions: [
      { card: 'AS', q: 12, regret: 0, source: 'solver' },
      { card: 'QS', q: 4, regret: 8, source: 'solver' },
      { card: 'KS', q: 2, regret: 10, source: 'equivalent:AS' },
    ],
    proposals: [{ weight: 1, q: { AS: 12, KS: 2, QS: 4 } }],
  };
}

test('regret team split follows the fixed partnership seats', () => {
  assert.deepEqual([0, 1, 2, 3].map(regretTeamOf), [0, 1, 0, 1]);
});

test('regretDecisionAt finds the decision that precedes a play index', () => {
  const analysis = { decisions: [{ playIndex: 16 }, { playIndex: 20 }] };
  assert.equal(regretDecisionAt(analysis, 20).playIndex, 20);
  assert.equal(regretDecisionAt(analysis, 19), null);
  assert.equal(regretDecisionAt(null, 20), null);
  assert.equal(regretDecisionAt({}, 20), null);
});

test('formatRegret renders zero, small losses and missing values', () => {
  assert.equal(formatRegret(0), '0');
  assert.equal(formatRegret(-0.0000001), '0');
  assert.equal(formatRegret(12.345), '12.3');
  assert.equal(formatRegret(12.345, 0), '12');
  assert.equal(formatRegret(null), '—');
  assert.equal(formatRegret(undefined), '—');
  assert.equal(formatRegret(Number.NaN), '—');
});

test('regretBadges marks the zero-regret action as best', () => {
  const badges = regretBadges(regretDecisionFixture());
  assert.equal(badges.AS.best, true);
  assert.equal(badges.QS.best, false);
  assert.equal(badges.QS.regret, 8);
  assert.equal(badges.KS.source, 'equivalent:AS');
  assert.deepEqual(regretBadges(null), {});
});

test('regretBadges never marks a card without a Q as best', () => {
  const decision = {
    actions: [{ card: 'AS', q: null, regret: 0, source: 'forced' }],
  };
  assert.equal(regretBadges(decision).AS.best, false);
});

test('sortHandByRegret puts the best card first and keeps unknowns last', () => {
  const cards = ['QS', '2H', 'AS', 'KS'].map((code) => ({
    code,
    rank: code.slice(0, -1),
    suit: code.slice(-1),
  }));
  const sorted = sortHandByRegret(cards, regretDecisionFixture());
  // 2H is not in the analysed action set, so it sinks to the end.
  assert.deepEqual(sorted.map((card) => card.code), ['AS', 'QS', 'KS', '2H']);
  // The input array is left untouched.
  assert.deepEqual(cards.map((card) => card.code), ['QS', '2H', 'AS', 'KS']);
});

test('sortHandByRegret is a no-op without an analysed decision', () => {
  const cards = [{ code: 'QS' }, { code: 'AS' }];
  assert.deepEqual(sortHandByRegret(cards, null).map((card) => card.code), ['QS', 'AS']);
  assert.deepEqual(sortHandByRegret(cards, { actions: [] }).map((card) => card.code), ['QS', 'AS']);
});

test('regretSummary defends against a partial analysis result', () => {
  assert.deepEqual(regretSummary(null), {
    totalRegret: 0,
    analyzedActions: 0,
    totalDecisions: 0,
    meanRegret: 0,
    perSeat: [],
    worst: [],
  });
  const summary = regretSummary({
    summary: { totalRegret: 12.5, analyzedActions: 26, totalDecisions: 36, perSeat: [{ seat: 0 }] },
  });
  assert.equal(summary.totalRegret, 12.5);
  assert.equal(summary.perSeat.length, 1);
  assert.deepEqual(summary.worst, []);
});

test('startRegretAnalysis posts the record and returns the job id', async () => {
  const original = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return { ok: true, json: async () => ({ ok: true, jobId: 'abc123' }) };
  };
  try {
    const jobId = await startRegretAnalysis({ format: 'spades-ai-replay', version: 1 });
    assert.equal(jobId, 'abc123');
    assert.equal(calls[0].url, '/api/analyze-replay');
    assert.equal(calls[0].options.method, 'POST');
    assert.match(calls[0].options.body, /spades-ai-replay/);
  } finally {
    globalThis.fetch = original;
  }
});

test('startRegretAnalysis surfaces backend failures instead of silently continuing', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: false,
    status: 400,
    text: async () => JSON.stringify({ error: '复盘记录无效' }),
  });
  try {
    await assert.rejects(() => startRegretAnalysis({}), /复盘记录无效/);
  } finally {
    globalThis.fetch = original;
  }
});

test('fetchRegretJob reports the running progress and the finished result', async () => {
  const original = globalThis.fetch;
  const responses = [
    { ok: true, json: async () => ({ ok: true, status: 'running', progress: { done: 3, total: 36 } }) },
    { ok: true, json: async () => ({ ok: true, status: 'done', result: { decisions: [] } }) },
  ];
  globalThis.fetch = async () => responses.shift();
  try {
    const running = await fetchRegretJob('job1');
    assert.equal(running.status, 'running');
    assert.equal(running.progress.done, 3);
    const done = await fetchRegretJob('job1');
    assert.equal(done.status, 'done');
    assert.deepEqual(done.result, { decisions: [] });
  } finally {
    globalThis.fetch = original;
  }
});

test('fetchRegretJob rejects when the backend reports an error payload', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ ok: false, error: '未知的 jobId' }) });
  try {
    await assert.rejects(() => fetchRegretJob('nope'), /未知的 jobId/);
  } finally {
    globalThis.fetch = original;
  }
});

/* ── 完整复盘导出的文件必须能被自己读回来 ──────────────────────────── */

function regretAnalysisFixture() {
  return {
    ok: true,
    exactThreshold: 36,
    decisions: [{ playIndex: 16, seat: 2, actualCard: '6D', actions: [] }],
    summary: { analyzedActions: 1, totalRegret: 0 },
  };
}

test('a regret export round-trips back through parseReplayImport', () => {
  const record = completeReplayRecord(20260804);
  const snapshot = parseReplayImport(record)[0].snapshot;
  const analysis = regretAnalysisFixture();

  const exported = buildRegretRecord(snapshot, analysis);
  assert.equal(exported.format, 'spades-ai-regret-replay');
  assert.equal(exported.version, 1);
  assert.equal(exported.replay.format, 'spades-ai-replay');

  // Read the file back the way the menu does: JSON in, options out.
  const options = parseReplayImport(JSON.parse(JSON.stringify(exported)));
  assert.equal(options.length, 1);
  assert.deepEqual(options[0].analysis, analysis);
  assert.equal(options[0].snapshot.plays.length, 52);
  assert.equal(options[0].snapshot.seed, 20260804);
  assert.match(options[0].label, /含遗憾分析/);
});

test('a regret export keeps the exact replay record it was built from', () => {
  const record = completeReplayRecord(20260804);
  const snapshot = parseReplayImport(record)[0].snapshot;
  const exported = buildRegretRecord(snapshot, regretAnalysisFixture());

  // The option carries the inner record so the backend gets the same bytes,
  // not a snapshot re-serialisation.
  assert.deepEqual(exported.replay.initialHands, record.initialHands);
  assert.deepEqual(exported.replay.tricks, record.tricks);
  const options = parseReplayImport(exported);
  assert.equal(options[0].record.format, 'spades-ai-replay');
});

test('the unversioned regret file written by earlier builds still imports', () => {
  const record = completeReplayRecord(20260804);
  const legacy = { replay: record, analysis: regretAnalysisFixture() };

  const options = parseReplayImport(legacy);
  assert.equal(options.length, 1);
  assert.deepEqual(options[0].analysis, regretAnalysisFixture());
  assert.equal(options[0].snapshot.seed, 20260804);
});

test('a legacy regret file without an analysis imports as a plain replay', () => {
  const record = completeReplayRecord(20260804);
  const options = parseReplayImport({ replay: record, analysis: null });
  assert.equal(options[0].analysis, null);
  assert.equal(options[0].snapshot.plays.length, 52);
});

test('a bundled replay is validated as strictly as a bare one', () => {
  const broken = structuredClone(completeReplayRecord(20260804));
  broken.tricks[0].winner = (broken.tricks[0].winner + 1) % 4;

  assert.throws(
    () => parseReplayImport({
      format: 'spades-ai-regret-replay',
      version: 1,
      replay: broken,
      analysis: regretAnalysisFixture(),
    }),
    /赢家/,
  );
});

test('a malformed regret bundle fails loudly instead of importing quietly', () => {
  const record = completeReplayRecord(20260804);
  assert.throws(
    () => parseReplayImport({ format: 'spades-ai-regret-replay', version: 2, replay: record, analysis: regretAnalysisFixture() }),
    /版本/,
  );
  assert.throws(
    () => parseReplayImport({ format: 'spades-ai-regret-replay', version: 1, analysis: regretAnalysisFixture() }),
    /缺少 replay/,
  );
  assert.throws(
    () => parseReplayImport({ format: 'spades-ai-regret-replay', version: 1, replay: record }),
    /analysis\.decisions/,
  );
  assert.throws(
    () => parseReplayImport({ format: 'spades-ai-regret-replay', version: 1, replay: record, analysis: { decisions: 'nope' } }),
    /analysis\.decisions/,
  );
  assert.throws(
    () => parseReplayImport({ replay: record, analysis: { decisions: 3 } }),
    /analysis\.decisions/,
  );
});

test('regretBadges treats float noise as the best action', () => {
  const decision = {
    actions: [
      { card: 'AS', q: 10, regret: 0, source: 'solver' },
      { card: 'KS', q: 10 - 1e-13, regret: 1e-13, source: 'solver' },
      { card: 'QS', q: 4, regret: 6, source: 'solver' },
    ],
  };
  const badges = regretBadges(decision);
  assert.equal(badges.AS.best, true);
  assert.equal(badges.KS.best, true, 'a 1e-13 difference is not a real loss');
  assert.equal(badges.QS.best, false);
  assert.ok(REGRET_EPSILON === 1e-9);
});

/* ── 把期望 Q 反推成「我方得墩数 / Nil 打成」的分布 ─────────────────── */

// 用户给的例子：我方（队 0，座位 0+2）叫 5，对方（队 1，座位 1+3）叫 6。
const EXAMPLE_BIDS = ['bid_3', 'bid_3', 'bid_2', 'bid_3'];

function tricksForQ(q, bids = EXAMPLE_BIDS, team = 0) {
  const outcomes = enumerateFinalOutcomes(bids).get(q) ?? [];
  return [...new Set(outcomes.map((entry) => entry.teamTricks[team]))].sort((a, b) => a - b);
}

test('Q inverts back to the trick count the user worked out by hand', () => {
  // 我方叫 5 / 对方叫 6：−10 → 6 墩，+83 → 8 墩，+74 → 9 墩。
  assert.deepEqual(tricksForQ(-10), [6]);
  assert.deepEqual(tricksForQ(83), [8]);
  assert.deepEqual(tricksForQ(74), [9]);
});

test('the same Q inverts to the other side symmetrically', () => {
  // 队 1 的 6 墩就是队 0 的 7 墩。
  assert.deepEqual(tricksForQ(-10, EXAMPLE_BIDS, 1), [7]);
});

test('enumerateFinalOutcomes covers every split exactly once', () => {
  const outcomes = [...enumerateFinalOutcomes(EXAMPLE_BIDS).values()].flat();
  assert.equal(outcomes.length, 560); // C(16,3) ways to split 13 tricks
  assert.ok(outcomes.every((entry) => entry.tricks.reduce((a, b) => a + b, 0) === 13));
});

test('a one-sided bid total makes the Q map strictly invertible', () => {
  // 两个队合计叫牌 5+6 = 11 < 13，至少一队必然打成，Q 与墩数一一对应。
  const byScore = enumerateFinalOutcomes(EXAMPLE_BIDS);
  const seen = new Map();
  for (const [q, outcomes] of byScore) {
    const tricks = [...new Set(outcomes.map((entry) => entry.teamTricks[0]))];
    assert.equal(tricks.length, 1, `Q=${q} 应当是唯一的`);
    seen.set(q, tricks[0]);
  }
  assert.equal(seen.size, byScore.size);
});

test('the distribution weights proposals by their importance-sampling weight', () => {
  const distribution = actionOutcomeDistribution({
    bids: EXAMPLE_BIDS,
    seat: 0,
    qSamples: [{ weight: 0.75, q: 83 }, { weight: 0.25, q: -10 }],
  });
  assert.equal(distribution.team, 0);
  assert.equal(distribution.sampleCount, 2);
  assert.ok(Math.abs(distribution.meanTricks - (0.75 * 8 + 0.25 * 6)) < 1e-9);
  assert.equal(distribution.mostLikelyTricks, 8);
  const byTricks = new Map(distribution.trickDistribution.map((row) => [row.tricks, row.probability]));
  assert.ok(Math.abs(byTricks.get(8) - 0.75) < 1e-12);
  assert.ok(Math.abs(byTricks.get(6) - 0.25) < 1e-12);
  const mass = distribution.trickDistribution.reduce((sum, row) => sum + row.probability, 0);
  assert.ok(Math.abs(mass - 1) < 1e-12);
});

test('numbers that cannot be a final score are dropped and reported', () => {
  const distribution = actionOutcomeDistribution({
    bids: EXAMPLE_BIDS,
    seat: 0,
    qSamples: [{ weight: 0.5, q: 83 }, { weight: 0.5, q: 40 }], // 40 不可能出现
  });
  assert.equal(distribution.matchedSamples, 1);
  assert.ok(Math.abs(distribution.coveredWeight - 0.5) < 1e-12);
  assert.ok(Math.abs(distribution.totalWeight - 1) < 1e-12);
  // 分布只在能解释的那部分权重上归一化。
  const byTricks = new Map(distribution.trickDistribution.map((row) => [row.tricks, row.probability]));
  assert.ok(Math.abs(byTricks.get(8) - 1) < 1e-12);
});

test('an ambiguous Q splits its weight evenly and says so', () => {
  // 双方都叫到很高时，同一个 Q 可能对应多种终局。
  const bids = ['bid_7', 'bid_7', 'bid_7', 'bid_7'];
  const byScore = enumerateFinalOutcomes(bids);
  const ambiguous = [...byScore.entries()].find(([, outcomes]) => (
    new Set(outcomes.map((entry) => entry.teamTricks[0])).size > 1
  ));
  if (!ambiguous) {
    // 这一副叫牌下 Q 恰好仍可逆，就没有可断言的歧义，跳过。
    return;
  }
  const [q, outcomes] = ambiguous;
  const distribution = actionOutcomeDistribution({
    bids, seat: 0, qSamples: [{ weight: 1, q }],
  });
  const distinct = new Set(outcomes.map((entry) => entry.teamTricks[0])).size;
  assert.ok(distribution.ambiguityShare > 0);
  const byTricks = new Map(distribution.trickDistribution.map((row) => [row.tricks, row.probability]));
  const covered = [...byTricks.values()].filter((value) => value > 0);
  assert.equal(covered.length, distinct);
  for (const probability of covered) assert.ok(Math.abs(probability - 1 / distinct) < 1e-12);
});

test('our Nil bidder gets a made/broken probability from the same samples', () => {
  // 座位 2 叫 Nil，队友（座位 0）叫 3；队 0 总分 3 墩即可打成。
  const bids = ['bid_3', 'bid_4', 'nil', 'bid_2'];
  const byScore = enumerateFinalOutcomes(bids);
  const sample = (q) => actionOutcomeDistribution({ bids, seat: 0, qSamples: [{ weight: 1, q }] });

  const made = [...byScore.entries()].find(([, outcomes]) => (
    outcomes.every((entry) => entry.nilMade[2])
  ));
  const broken = [...byScore.entries()].find(([, outcomes]) => (
    outcomes.every((entry) => !entry.nilMade[2])
  ));

  if (made) {
    const distribution = sample(made[0]);
    assert.deepEqual(distribution.ourNilSeats, [2]);
    assert.ok(Math.abs(distribution.nil[0].madeProbability - 1) < 1e-12);
  }
  if (broken) {
    const distribution = sample(broken[0]);
    assert.ok(Math.abs(distribution.nil[0].madeProbability) < 1e-12);
    assert.ok(Math.abs(distribution.nil[0].failedProbability - 1) < 1e-12);
  }
});

test('no Nil on our side means no Nil distribution', () => {
  const distribution = actionOutcomeDistribution({
    bids: EXAMPLE_BIDS,
    seat: 0,
    qSamples: [{ weight: 1, q: 83 }],
  });
  assert.deepEqual(distribution.ourNilSeats, []);
  assert.deepEqual(distribution.nil, []);
});

test('the opponent Nil is marginalized, not reported as ours', () => {
  const bids = ['bid_3', 'nil', 'bid_2', 'bid_3'];
  const distribution = actionOutcomeDistribution({
    bids, seat: 0, qSamples: [{ weight: 1, q: 83 }],
  });
  assert.deepEqual(distribution.ourNilSeats, []);
  assert.deepEqual(distribution.opponentNilSeats, [1]);
  assert.equal(distribution.nil.length, 0);
});

test('qSamplesForCard falls back to the equivalent representative', () => {
  // 求解器把等大牌张合并掉了：这张牌自己在提案表里没有条目，
  // 但代表牌在同一个世界里与它严格等值，所以直接借用它的逐提案 Q。
  const decision = {
    actions: [
      { card: 'KH', q: 160, source: 'solver' },
      { card: 'QH', q: 160, source: 'equivalent:KH' },
      { card: '2C', q: 40, source: 'solver' },
    ],
    proposals: [
      { weight: 0.6, q: { KH: 160, '2C': 40 } },
      { weight: 0.4, q: { KH: 100, '2C': 20 } },
    ],
  };
  assert.deepEqual(qSamplesForCard(decision, 'QH'), [
    { weight: 0.6, q: 160 },
    { weight: 0.4, q: 100 },
  ]);
  // 有自己条目的牌仍然走自己的数据。
  assert.deepEqual(qSamplesForCard(decision, 'KH'), [
    { weight: 0.6, q: 160 },
    { weight: 0.4, q: 100 },
  ]);
  // 既没有条目、来源也不是等价牌的，才是真的没有。
  assert.deepEqual(
    qSamplesForCard({ actions: [{ card: 'AS', q: null, source: 'forced' }], proposals: [] }, 'AS'),
    [],
  );
});

test('qSamplesForCard reads one card out of the per-proposal table', () => {
  const decision = {
    proposals: [
      { weight: 0.5, q: { AS: 83, KS: 74 } },
      { weight: 0.5, q: { AS: -10 } },
    ],
  };
  assert.deepEqual(qSamplesForCard(decision, 'AS'), [
    { weight: 0.5, q: 83 },
    { weight: 0.5, q: -10 },
  ]);
  assert.deepEqual(qSamplesForCard(decision, 'KS'), [{ weight: 0.5, q: 74 }]);
  assert.deepEqual(qSamplesForCard(decision, 'QS'), []);
  assert.deepEqual(qSamplesForCard(null, 'AS'), []);
});

test('the distribution accepts both bid shapes in the record', () => {
  const objectBids = [
    { value: 3, type: 'normal' },
    { value: 3, type: 'normal' },
    { value: 2, type: 'normal' },
    { value: 3, type: 'normal' },
  ];
  const fromObjects = actionOutcomeDistribution({
    bids: objectBids, seat: 0, qSamples: [{ weight: 1, q: 83 }],
  });
  const fromStrings = actionOutcomeDistribution({
    bids: EXAMPLE_BIDS, seat: 0, qSamples: [{ weight: 1, q: 83 }],
  });
  assert.deepEqual(fromObjects.trickDistribution, fromStrings.trickDistribution);
});
