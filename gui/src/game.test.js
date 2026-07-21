import assert from 'node:assert/strict';
import test from 'node:test';

import {
  PACE,
  advanceUntilHuman,
  applyCard,
  applyShowdownOffer,
  buildShowdownPayload,
  confirmLocalShowdown,
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
