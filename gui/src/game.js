/*
 * Shared local game helpers for the GUI demo.
 * Input: seeded deck / state snapshots / human actions.
 * Output: immutable state updates and legal-action helpers.
 */

const SUITS = ['C', 'D', 'H', 'S'];
const SUIT_SYMBOLS = { C: '♣', D: '♦', H: '♥', S: '♠' };
const RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
const RANK_VALUES = Object.fromEntries(RANKS.map((rank, index) => [rank, index + 2]));
const PLAYER_NAMES = ['North', 'East', 'South', 'West'];

/**
 * Build a seeded pseudo-random generator.
 * Input: integer seed.
 * Output: function returning a float in [0, 1).
 */
export function createRng(seed) {
  let value = (seed >>> 0) || 1;
  return () => {
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    return ((value >>> 0) % 1000000) / 1000000;
  };
}

/**
 * Create a 52-card deck.
 * Input: none.
 * Output: array of card objects with `code`, `rank`, and `suit`.
 */
export function createDeck() {
  return SUITS.flatMap((suit) => RANKS.map((rank) => ({ code: `${rank}${suit}`, rank, suit })));
}

/**
 * Shuffle an array in place with a seeded RNG.
 * Input: array and RNG function.
 * Output: the same array, shuffled.
 */
export function shuffle(items, rng) {
  for (let index = items.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(rng() * (index + 1));
    [items[index], items[swapIndex]] = [items[swapIndex], items[index]];
  }
  return items;
}

/**
 * Deal a single 52-card hand into four 13-card hands.
 * Input: seed used for the shuffle.
 * Output: four sorted hands.
 */
export function dealHands(seed) {
  const rng = createRng(seed);
  const deck = shuffle(createDeck(), rng);
  return Array.from({ length: 4 }, (_, seat) => sortCards(deck.slice(seat * 13, (seat + 1) * 13)));
}

/**
 * Format a card for display.
 * Input: card object.
 * Output: a short string such as `A♠`.
 */
export function cardLabel(card) {
  return `${card.rank}${SUIT_SYMBOLS[card.suit]}`;
}

/**
 * Format a bid for display.
 * Input: bid object or null.
 * Output: string like "3", "Nil", or "—".
 */
export function bidLabel(bid) {
  if (!bid) return '—';
  return bid.type === 'nil' ? 'Nil' : bid.value.toString();
}

/**
 * Sort cards for display: spades → hearts → clubs → diamonds, and within a
 * suit from high rank to low (so left-to-right reads ♠A♠K… ♥… ♣… ♦…).
 * Input: array of cards.
 * Output: new sorted array.
 */
const DISPLAY_SUIT_ORDER = { S: 0, H: 1, C: 2, D: 3 };
function sortCards(cards) {
  return [...cards].sort((a, b) =>
    (DISPLAY_SUIT_ORDER[a.suit] - DISPLAY_SUIT_ORDER[b.suit])
    || (RANK_VALUES[b.rank] - RANK_VALUES[a.rank]),
  );
}

/**
 * Deep clone a game state (immutability helper).
 * Input: state object.
 * Output: a shallow-deep copy suitable for modification.
 */
function cloneState(state) {
  return {
    ...state,
    hands: state.hands.map((hand) => [...hand]),
    bids: [...state.bids],
    tricksWon: [...state.tricksWon],
    currentTrick: state.currentTrick.map((entry) => ({ ...entry, card: { ...entry.card } })),
    completedTricks: state.completedTricks.map((trick) => ({
      ...trick,
      cards: trick.cards.map((entry) => ({ ...entry, card: { ...entry.card } })),
    })),
    showdown: state.showdown ? {
      ...state.showdown,
      resolution: state.showdown.resolution ? {
        ...state.showdown.resolution,
        teamTricks: [...(state.showdown.resolution.teamTricks || [])],
        nilOutcomes: [...(state.showdown.resolution.nilOutcomes || [])],
        finalTricksWon: [...(state.showdown.resolution.finalTricksWon || [])],
        continuation: (state.showdown.resolution.continuation || []).map((play) => ({ ...play })),
      } : null,
    } : null,
    log: state.log.map((entry) => ({ ...entry })),
  };
}

/**
 * Produce a card ordering key inside a trick.
 * Input: a card and the led suit.
 * Output: a comparable integer key.
 */
function trickStrength(card, ledSuit) {
  if (card.suit === 'S') {
    return 100 + RANK_VALUES[card.rank];
  }
  if (ledSuit && card.suit === ledSuit) {
    return 50 + RANK_VALUES[card.rank];
  }
  return RANK_VALUES[card.rank];
}

/**
 * Determine whether a card is legal for the current trick.
 * Input: hand, current trick, and spade-broken flag.
 * Output: the legal cards from the hand.
 */
export function getLegalCards(hand, currentTrick, spadesBroken) {
  if (currentTrick.length === 0) {
    const nonSpades = hand.filter((card) => card.suit !== 'S');
    return spadesBroken || nonSpades.length === 0 ? [...hand] : nonSpades;
  }

  const ledSuit = currentTrick[0].card.suit;
  const matchingSuit = hand.filter((card) => card.suit === ledSuit);
  return matchingSuit.length > 0 ? matchingSuit : [...hand];
}

/**
 * Compare two cards inside the same trick.
 * Input: the new card, the current best card, and the led suit.
 * Output: positive when the new card wins.
 */
function compareTrickCards(nextCard, bestCard, ledSuit) {
  return trickStrength(nextCard, ledSuit) - trickStrength(bestCard, ledSuit);
}

/**
 * Determine the winner of the current trick.
 * Input: a four-card trick snapshot.
 * Output: the seat index of the winner.
 */
export function determineTrickWinner(currentTrick) {
  const ledSuit = currentTrick[0].card.suit;
  let winner = currentTrick[0].seat;
  let bestCard = currentTrick[0].card;

  for (const entry of currentTrick.slice(1)) {
    if (compareTrickCards(entry.card, bestCard, ledSuit) > 0) {
      bestCard = entry.card;
      winner = entry.seat;
    }
  }

  return winner;
}

/**
 * Create a shallow bid object.
 * Input: a numeric value and optional bid type.
 * Output: a normalized bid record.
 */
export function makeBid(value, type = 'normal') {
  return { value, type };
}

function serializeBid(bid) {
  if (!bid) {
    return null;
  }
  return { value: bid.value, type: bid.type };
}

function serializeTrickCards(cards) {
  return cards.map((entry) => ({ seat: entry.seat, card: entry.card.code }));
}

export function buildAiPayload(state) {
  const seat = state.currentPlayer;
  return {
    seed: state.seed,
    firstSeat: state.firstSeat,
    phase: state.phase,
    currentPlayer: seat,
    leader: state.leader,
    trickNumber: state.trickNumber,
    spadesBroken: state.spadesBroken,
    humanSeat: state.humanSeat,
    remainingHand: state.hands[seat].map((card) => card.code),
    bids: state.bids.map((bid) => serializeBid(bid)),
    completedTricks: state.completedTricks.map((trick) => ({ cards: serializeTrickCards(trick.cards) })),
    currentTrick: serializeTrickCards(state.currentTrick),
    tricksWon: [...state.tricksWon],
  };
}

/**
 * Return whether a state is an eligible complete-trick showdown boundary.
 * Detection is intentionally limited to the final one through five tricks.
 */
export function shouldCheckShowdown(state) {
  if (state.phase !== 'playing' || state.trickComplete || state.currentTrick.length !== 0) {
    return false;
  }
  if (state.showdown) return false;
  if (!Array.isArray(state.hands) || state.hands.length !== 4) return false;
  const sizes = state.hands.map((hand) => hand.length);
  return sizes.every((size) => size === sizes[0]) && sizes[0] >= 1 && sizes[0] <= 5;
}

/** Build the full-information payload used only by /api/check-showdown. */
export function buildShowdownPayload(state) {
  return {
    phase: state.phase,
    currentPlayer: state.currentPlayer,
    leader: state.leader,
    trickNumber: state.trickNumber,
    spadesBroken: state.spadesBroken,
    remainingHands: state.hands.map((hand) => hand.map((card) => card.code)),
    bids: state.bids.map((bid) => serializeBid(bid)),
    completedTricks: state.completedTricks.map((trick) => ({
      cards: serializeTrickCards(trick.cards),
    })),
    currentTrick: serializeTrickCards(state.currentTrick),
    tricksWon: [...state.tricksWon],
  };
}

async function requestShowdownCheck(state) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 1100);
  try {
    const response = await fetch('/api/check-showdown', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildShowdownPayload(state)),
      signal: controller.signal,
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Showdown check failed (${response.status}): ${text}`);
    }
    const payload = await response.json();
    if (!payload.ok) throw new Error(payload.error || 'Showdown backend returned an error');
    return payload;
  } finally {
    clearTimeout(timer);
  }
}

/** Store a fixed offer without playing cards or computing the score. */
export function applyShowdownOffer(state, response) {
  if (!response?.ok || response.status !== 'fixed' || !response.resolution) {
    return state;
  }
  const resolution = response.resolution;
  if (
    !Array.isArray(resolution.teamTricks)
    || !Array.isArray(resolution.nilOutcomes)
    || !Array.isArray(resolution.finalTricksWon)
    || !Array.isArray(resolution.continuation)
  ) {
    return state;
  }
  const next = cloneState(state);
  next.showdown = {
    status: 'pending',
    resolution: {
      teamTricks: [...resolution.teamTricks],
      nilOutcomes: [...resolution.nilOutcomes],
      finalTricksWon: [...resolution.finalTricksWon],
      continuation: resolution.continuation.map((play) => ({ ...play })),
    },
  };
  return next;
}

async function checkShowdownAfterTrick(state) {
  if (!shouldCheckShowdown(state)) return state;
  const response = await requestShowdownCheck(state);
  return applyShowdownOffer(state, response);
}

async function requestAiAction(state) {
  let response;
  try {
    response = await fetch('/api/choose-action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildAiPayload(state)),
    });
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    throw new Error(`无法连接 AI 后端：${detail}`);
  }

  if (!response.ok) {
    const text = await response.text();
    let detail = text;
    try {
      const body = JSON.parse(text);
      detail = body.error || text;
    } catch {
      // Keep the raw response when the backend did not return JSON.
    }
    throw new Error(`AI 后端请求失败（HTTP ${response.status}）：${detail || '无错误详情'}`);
  }

  let payload;
  try {
    payload = await response.json();
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    throw new Error(`AI 后端返回了无效响应：${detail}`);
  }
  if (!payload.ok) {
    throw new Error(payload.error || 'AI 后端返回错误');
  }

  const fallbackReason = payload.fallbackReason ?? payload.fallback_reason;
  const detail = typeof payload.detail === 'string' ? payload.detail : '';
  if (
    payload.fallback === true
    || fallbackReason != null
    || detail.toLowerCase().includes('fallback')
  ) {
    const reason = fallbackReason || detail || '未提供原因';
    throw new Error(`AI 后端触发 fallback：${reason}`);
  }

  return payload;
}

/**
 * Compute the team score using the project scoring rules.
 * Input: seat bids and trick counts.
 * Output: a two-element object containing both team scores.
 */
export function computeScores(bids, tricksWon) {
  const scoreForTeam = (teamSeats) => {
    let score = 0;
    let bidTotal = 0;
    let trickTotal = 0;

    for (const seat of teamSeats) {
      const bid = bids[seat];
      if (!bid) {
        continue;
      }
      trickTotal += tricksWon[seat];
      if (bid.type === 'nil') {
        score += tricksWon[seat] === 0 ? 50 : -50;
      } else {
        bidTotal += bid.value;
      }
    }

    if (bidTotal === 0) {
      return score;
    }

    if (trickTotal >= bidTotal) {
      score += bidTotal * 10;
      score -= (trickTotal - bidTotal) * 9;
      return score;
    }

    return score - bidTotal * 10;
  };

  return {
    northSouth: scoreForTeam([0, 2]),
    eastWest: scoreForTeam([1, 3]),
  };
}

/**
 * Apply a bid and advance the game.
 * Input: a state snapshot, a seat, and a bid.
 * Output: the next immutable state.
 */
export function applyBid(state, seat, bid, source = null) {
  const next = cloneState(state);
  next.bids[seat] = bid;
  next.lastBidSeat = seat; // drives the bid-bubble pop animation
  next.log.push({ kind: 'bid', seat, text: `${PLAYER_NAMES[seat]} 叫牌 ${bidLabel(bid)}${source ? ` [${source}]` : ''}` });

  if (next.bids.every(Boolean)) {
    next.phase = 'playing';
    next.currentPlayer = next.firstSeat;
    next.leader = next.firstSeat;
    next.trickNumber = 1;
    next.log.push({ kind: 'system', text: '叫牌完成，进入出牌阶段' });
  } else {
    next.currentPlayer = (seat + 1) % 4;
  }

  return next;
}

/**
 * Apply a play action and advance the trick.
 * Input: a state snapshot, a seat, and a card code.
 * Output: the next immutable state.
 *
 * When the 4th card lands we do NOT clear the table immediately — instead we
 * mark `trickComplete`/`trickWinner` so the UI can hold the full trick on the
 * felt for a beat (and animate it flying to the winner). `finalizeTrick`
 * performs the actual collection afterwards.
 */
export function applyCard(state, seat, cardCode, source = null) {
  const next = cloneState(state);
  const hand = next.hands[seat];
  const cardIndex = hand.findIndex((card) => card.code === cardCode);
  if (cardIndex < 0) {
    throw new Error(`Seat ${seat} does not hold ${cardCode}`);
  }

  const card = hand[cardIndex];
  const legalCards = getLegalCards(hand, next.currentTrick, next.spadesBroken);
  if (!legalCards.some((candidate) => candidate.code === card.code)) {
    throw new Error(`Illegal card ${cardCode} for seat ${seat}`);
  }

  hand.splice(cardIndex, 1);
  next.currentTrick.push({ seat, card });
  next.lastPlayedSeat = seat; // drives the slide-in-from-seat animation
  next.spadesBroken = next.spadesBroken || card.suit === 'S';
  next.log.push({ kind: 'play', seat, text: `${PLAYER_NAMES[seat]} 出牌 ${cardLabel(card)}${source ? ` [${source}]` : ''}` });

  if (next.currentTrick.length < 4) {
    next.currentPlayer = (seat + 1) % 4;
    return next;
  }

  // 4th card: freeze the full trick on the table; winner collects later.
  next.trickComplete = true;
  next.trickWinner = determineTrickWinner(next.currentTrick);
  next.currentPlayer = -1; // nobody acts during the hold
  return next;
}

/**
 * Collect a completed trick (called after the hold-on-table beat).
 * Input: a state with trickComplete=true.
 * Output: the next immutable state with the table cleared and score updated.
 */
export function finalizeTrick(state) {
  if (!state.trickComplete) return state;
  const next = cloneState(state);
  const winner = next.trickWinner;

  next.tricksWon[winner] += 1;
  next.completedTricks.push({
    trickNumber: next.trickNumber,
    winner,
    cards: next.currentTrick.map((entry) => ({ seat: entry.seat, card: entry.card })),
  });
  next.currentTrick = [];
  next.trickComplete = false;
  next.trickWinner = -1;
  next.lastPlayedSeat = -1;
  next.currentPlayer = winner;
  next.leader = winner;
  next.log.push({ kind: 'system', text: `第 ${next.trickNumber} 墩由 ${PLAYER_NAMES[winner]} 赢下` });

  if (next.trickNumber >= 13) {
    next.phase = 'finished';
    next.score = computeScores(next.bids, next.tricksWon);
    next.log.push({
      kind: 'system',
      text: `牌局结束，NS=${next.score.northSouth}，EW=${next.score.eastWest}`,
    });
    return next;
  }

  next.trickNumber += 1;
  return next;
}

/**
 * Create the initial game state.
 * Input: seed and the human-controlled seat index.
 * Output: a ready-to-play bidding state.
 */
export function createInitialGame(seed, humanSeat, firstSeat = 0) {
  const opener = ((firstSeat % 4) + 4) % 4;
  return {
    seed,
    humanSeat,
    firstSeat: opener,
    phase: 'bidding',
    currentPlayer: opener,
    leader: opener,
    trickNumber: 1,
    spadesBroken: false,
    hands: dealHands(seed),
    bids: [null, null, null, null],
    tricksWon: [0, 0, 0, 0],
    currentTrick: [],
    completedTricks: [],
    trickComplete: false,
    trickWinner: -1,
    lastPlayedSeat: -1,
    lastBidSeat: -1,
    score: null,
    showdown: null,
    log: [{ kind: 'system', text: '新牌局已开始' }],
  };
}

/** Apply a proven continuation and settle only after the local confirmation. */
export function confirmLocalShowdown(state) {
  if (!state.showdown || state.showdown.status !== 'pending') return state;
  const resolution = state.showdown.resolution;
  let next = cloneState(state);
  next.showdown = null;

  for (const play of resolution.continuation) {
    if (next.currentPlayer !== play.seat) {
      throw new Error(`Showdown continuation seat mismatch: expected ${next.currentPlayer}, got ${play.seat}`);
    }
    next = applyCard(next, play.seat, play.card, 'showdown');
    if (next.trickComplete) next = finalizeTrick(next);
  }

  if (next.phase !== 'finished' || next.completedTricks.length !== 13) {
    throw new Error('Showdown continuation did not produce a complete hand');
  }
  if (
    next.tricksWon.length !== resolution.finalTricksWon.length
    || next.tricksWon.some((value, seat) => value !== resolution.finalTricksWon[seat])
  ) {
    throw new Error('Showdown continuation disagrees with projected trick totals');
  }
  const actualTeams = [
    next.tricksWon[0] + next.tricksWon[2],
    next.tricksWon[1] + next.tricksWon[3],
  ];
  if (actualTeams.some((value, team) => value !== resolution.teamTricks[team])) {
    throw new Error('Showdown continuation disagrees with projected team totals');
  }
  for (let seat = 0; seat < 4; seat += 1) {
    if (resolution.nilOutcomes[seat] == null) continue;
    const actualSuccess = next.tricksWon[seat] === 0;
    if (actualSuccess !== resolution.nilOutcomes[seat]) {
      throw new Error('Showdown continuation disagrees with projected Nil outcome');
    }
  }
  return next;
}

/** Sleep helper for pacing AI turns into visible animation steps. */
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/* Animation pacing (ms). The exact solver itself takes 1–4 s for the last
 * 36 cards, so those beats are dominated by computation; these delays only
 * shape the fast (policy / bidding) turns and the trick-collection hold. */
export const PACE = {
  aiStep: 800,        // pause after each AI bid / play
  trickHold: 1500,    // hold a completed 4-card trick before collecting
};

/**
 * Resolve AI turns until the human seat needs attention or the hand ends.
 * Input: the current state, and an optional onStep(state) callback that is
 *        invoked after every visible change so the UI can render each beat.
 * Output: a state where it is either the human's turn or the game is finished.
 *
 * Each AI action and each trick collection is surfaced through onStep with a
 * pause in between, turning the previously-instant resolution into animation.
 */
export async function advanceUntilHuman(state, onStep = null) {
  let next = state;
  const emit = (s) => { next = s; if (onStep) onStep(s); };

  // helper: if a trick is sitting complete on the table, hold then collect.
  const collectIfNeeded = async () => {
    if (next.trickComplete) {
      await sleep(PACE.trickHold);
      emit(finalizeTrick(next));
      const checked = await checkShowdownAfterTrick(next);
      if (checked !== next) emit(checked);
    }
    return !!next.showdown;
  };

  if (await collectIfNeeded()) return next;

  while (next.phase !== 'finished' && !next.showdown && next.currentPlayer !== next.humanSeat) {
    if (next.phase === 'bidding') {
      const action = await requestAiAction(next);
      if (action.kind !== 'bid') throw new Error(`AI 叫牌阶段返回了错误动作类型：${action.kind}`);
      const bidState = applyBid(
        next,
        next.currentPlayer,
        makeBid(action.bid.value, action.bid.type),
        action.ai,
      );
      emit(bidState);
      await sleep(PACE.aiStep);
      continue;
    }

    if (next.phase === 'playing') {
      const action = await requestAiAction(next);
      if (action.kind !== 'play') throw new Error(`AI 出牌阶段返回了错误动作类型：${action.kind}`);
      const playState = applyCard(next, next.currentPlayer, action.card, action.ai);
      emit(playState);
      await sleep(PACE.aiStep);
      if (await collectIfNeeded()) break;
      continue;
    }

    break;
  }

  return next;
}

/**
 * Resolve AI turns until the hand is finished (all four seats are AI).
 * Input: the current state, and an optional onStep(state) callback.
 * Output: the finished state.
 */
export async function advanceUntilFinished(state, onStep = null) {
  let next = state;
  const emit = (s) => { next = s; if (onStep) onStep(s); };

  const collectIfNeeded = async () => {
    if (next.trickComplete) {
      await sleep(PACE.trickHold);
      emit(finalizeTrick(next));
      const checked = await checkShowdownAfterTrick(next);
      if (checked !== next) emit(checked);
    }
    return !!next.showdown;
  };

  if (await collectIfNeeded()) return next;

  while (next.phase !== 'finished' && !next.showdown) {
    if (next.phase === 'bidding') {
      const action = await requestAiAction(next);
      if (action.kind !== 'bid') throw new Error(`AI 叫牌阶段返回了错误动作类型：${action.kind}`);
      const bidState = applyBid(
        next,
        next.currentPlayer,
        makeBid(action.bid.value, action.bid.type),
        action.ai,
      );
      emit(bidState);
      await sleep(PACE.aiStep);
      continue;
    }

    if (next.phase === 'playing') {
      const action = await requestAiAction(next);
      if (action.kind !== 'play') throw new Error(`AI 出牌阶段返回了错误动作类型：${action.kind}`);
      const playState = applyCard(next, next.currentPlayer, action.card, action.ai);
      emit(playState);
      await sleep(PACE.aiStep);
      if (await collectIfNeeded()) break;
      continue;
    }

    break;
  }

  return next;
}

/**
 * Apply a human bid and then fast-forward AI turns (with animation steps).
 * Input: a state, the selected bid, and an optional onStep(state) callback.
 * Output: the next playable state.
 */
export async function submitHumanBid(state, bid, onStep = null) {
  const afterBid = applyBid(state, state.humanSeat, bid);
  if (onStep) onStep(afterBid);
  await sleep(PACE.aiStep);
  return await advanceUntilHuman(afterBid, onStep);
}

/**
 * Apply a human card and then fast-forward AI turns (with animation steps).
 * Input: a state, the chosen card code, and an optional onStep(state) callback.
 * Output: the next playable state.
 */
export async function submitHumanCard(state, cardCode, onStep = null) {
  const afterCard = applyCard(state, state.humanSeat, cardCode);
  if (onStep) onStep(afterCard);
  await sleep(PACE.aiStep);
  return await advanceUntilHuman(afterCard, onStep);
}

/**
 * Return the current legal cards for the human player.
 * Input: a state snapshot.
 * Output: the list of legal cards in the human hand.
 */
export function getHumanLegalCards(state) {
  return sortCards(getLegalCards(state.hands[state.humanSeat], state.currentTrick, state.spadesBroken));
}

/**
 * Return the visible summary for the current hand.
 * Input: a state snapshot.
 * Output: a compact descriptor object used by the UI.
 */
export function summarizeGame(state) {
  return {
    phase: state.phase,
    currentPlayer: state.currentPlayer,
    leader: state.leader,
    trickNumber: state.trickNumber,
    spadesBroken: state.spadesBroken,
    bids: state.bids.map((bid) => bidLabel(bid)),
    tricksWon: [...state.tricksWon],
    score: state.score,
  };
}

/**
 * Build a replay snapshot from a finished hand.
 * Input: a finished game state.
 * Output: immutable data for the replay UI (initial hands + play sequence).
 */
export function buildReplaySnapshot(state) {
  const hands = dealHands(state.seed).map((hand) =>
    hand.map((card) => ({ ...card })),
  );
  const plays = state.completedTricks.flatMap((trick) =>
    trick.cards.map((entry) => ({
      seat: entry.seat,
      card: { ...entry.card },
      trickNumber: trick.trickNumber,
      winner: trick.winner,
    })),
  );
  return {
    seed: state.seed,
    humanSeat: state.humanSeat,
    bids: state.bids.map((bid) => (bid ? { ...bid } : null)),
    tricksWon: [...state.tricksWon],
    score: state.score ? { ...state.score } : null,
    hands,
    plays,
    completedTricks: state.completedTricks.map((trick) => ({
      trickNumber: trick.trickNumber,
      winner: trick.winner,
      cards: trick.cards.map((entry) => ({ seat: entry.seat, card: { ...entry.card } })),
    })),
  };
}

/**
 * Build a portable, versioned record from a replay snapshot.
 * Cards are exported as compact rank+suit codes so the file is independent
 * of the current UI representation.
 */
export function buildReplayRecord(snapshot) {
  return {
    format: 'spades-ai-replay',
    version: 1,
    seed: snapshot.seed,
    viewSeat: snapshot.humanSeat,
    seats: [...PLAYER_NAMES],
    bids: snapshot.bids.map((bid) => serializeBid(bid)),
    initialHands: snapshot.hands.map((hand) => hand.map((card) => card.code)),
    tricks: snapshot.completedTricks.map((trick) => ({
      trickNumber: trick.trickNumber,
      leader: trick.cards[0]?.seat ?? null,
      winner: trick.winner,
      plays: serializeTrickCards(trick.cards),
    })),
    tricksWon: [...snapshot.tricksWon],
    score: snapshot.score ? { ...snapshot.score } : null,
  };
}

/** Replay animation pacing (ms). */
export const REPLAY_PACE = {
  cardStep: 650,
  trickHold: 1200,
};

/* ───────────────────────────────────────────────────────────────────
 * Remote (networked) game client — WebSocket to game_server.py
 * ─────────────────────────────────────────────────────────────────── */

/**
 * Parse a card code string (e.g. "SA") into the frontend card object.
 */
function parseCardCode(code) {
  if (!code || code.length < 2) return { code: '??', rank: '?', suit: '?' };
  const rank = code.slice(0, -1);
  const suit = code.slice(-1);
  return { code, rank, suit };
}

/**
 * Convert a server game_state message into the frontend game state format.
 * This lets the existing rendering components work unchanged in remote mode.
 *
 * @param {object} msg — server game_state message
 * @param {number} mySeat — this client's seat
 * @param {number} dealSeed — shared seed used to reconstruct all four hands
 * @returns {object} frontend-compatible game state
 */
export function remoteStateFromServer(msg, mySeat, dealSeed = 0) {
  const myCards = (msg.hand || []).map(parseCardCode);
  const handSizes = msg.handSizes || [13, 13, 13, 13];
  const remoteShowdown = msg.showdown && Array.isArray(msg.showdown.revealedHands)
    ? msg.showdown
    : null;

  // A proved showdown is the sole exception to ordinary hidden-card privacy.
  const hands = remoteShowdown
    ? remoteShowdown.revealedHands.map((hand) => hand.map(parseCardCode))
    : [null, null, null, null].map((_, seat) =>
      seat === mySeat ? myCards : new Array(handSizes[seat])
    );

  // Convert currentTrick cards from string codes to card objects
  const currentTrick = (msg.currentTrick || []).map((entry) => ({
    seat: entry.seat,
    card: parseCardCode(entry.card),
  }));

  // Convert completedTricks cards
  const completedTricks = (msg.completedTricks || []).map((trick) => ({
    trickNumber: trick.trickNumber,
    winner: trick.winner,
    cards: trick.cards.map((entry) => ({
      seat: entry.seat,
      card: parseCardCode(entry.card),
    })),
  }));

  const confirmedSeats = remoteShowdown?.confirmedSeats || [];
  const showdown = remoteShowdown ? {
    status: 'pending',
    id: remoteShowdown.id,
    confirmedSeats: [...confirmedSeats],
    locallyConfirmed: confirmedSeats.includes(mySeat),
    resolution: {
      teamTricks: [...(remoteShowdown.teamTricks || [])],
      nilOutcomes: [...(remoteShowdown.nilOutcomes || [])],
      finalTricksWon: [],
      continuation: [],
    },
  } : null;

  return {
    seed: dealSeed,
    humanSeat: mySeat,
    firstSeat: msg.leader,
    phase: msg.phase,
    currentPlayer: msg.currentPlayer,
    leader: msg.leader,
    trickNumber: msg.trickNumber,
    spadesBroken: msg.spadesBroken,
    hands,
    bids: msg.bids || [null, null, null, null],
    tricksWon: msg.tricksWon || [0, 0, 0, 0],
    currentTrick,
    completedTricks,
    trickComplete: !!msg.trickComplete,
    trickWinner: msg.trickWinner != null ? msg.trickWinner : -1,
    lastPlayedSeat: msg.lastPlayedSeat != null ? msg.lastPlayedSeat : -1,
    lastBidSeat: msg.lastBidSeat != null ? msg.lastBidSeat : -1,
    score: null,
    showdown,
    log: msg.log || [],
  };
}


/** Whether this remote client has confirmed and is waiting for its partner. */
export function showdownWaitingForPartner(showdown, mySeat) {
  if (!showdown || showdown.status !== 'pending') return false;
  return (showdown.confirmedSeats || []).includes(mySeat);
}
