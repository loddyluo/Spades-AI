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
 * Sort cards by suit and rank.
 * Input: array of cards.
 * Output: new sorted array.
 */
function sortCards(cards) {
  const order = (card) => SUITS.indexOf(card.suit) * 100 + RANK_VALUES[card.rank];
  return [...cards].sort((a, b) => order(a) - order(b));
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

function buildAiPayload(state) {
  const seat = state.currentPlayer;
  return {
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

async function requestAiAction(state) {
  const response = await fetch('/api/choose-action', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(buildAiPayload(state)),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`AI backend request failed (${response.status}): ${text}`);
  }

  const payload = await response.json();
  if (!payload.ok) {
    throw new Error(payload.error || 'AI backend returned an error');
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
 * Choose a simple AI bid.
 * Input: current state and the seat index.
 * Output: a normal bid object.
 */
function chooseAIBid(state, seat) {
  const hand = state.hands[seat];
  const spades = hand.filter((card) => card.suit === 'S').length;
  const highCards = hand.filter((card) => RANK_VALUES[card.rank] >= 11).length;
  const estimate = Math.max(0, Math.min(13, Math.round(1 + spades * 0.7 + highCards * 0.25)));
  return makeBid(Math.max(estimate, 1), 'normal');
}

/**
 * Choose a simple AI card.
 * Input: current state and the seat index.
 * Output: a legal card object.
 */
function chooseAICard(state, seat) {
  const hand = state.hands[seat];
  const legalCards = sortCards(getLegalCards(hand, state.currentTrick, state.spadesBroken));

  if (state.currentTrick.length === 0) {
    return legalCards[0];
  }

  const ledSuit = state.currentTrick[0].card.suit;
  const currentWinner = state.currentTrick.reduce((best, entry) => {
    return compareTrickCards(entry.card, best.card, ledSuit) > 0 ? entry : best;
  });

  const winningCards = legalCards.filter((card) => compareTrickCards(card, currentWinner.card, ledSuit) > 0);
  if (winningCards.length > 0) {
    return winningCards[0];
  }

  return legalCards[0];
}

/**
 * Apply a bid and advance the game.
 * Input: a state snapshot, a seat, and a bid.
 * Output: the next immutable state.
 */
export function applyBid(state, seat, bid, source = null) {
  const next = cloneState(state);
  next.bids[seat] = bid;
  next.log.push({ kind: 'bid', seat, text: `${PLAYER_NAMES[seat]} 叫牌 ${bidLabel(bid)}${source ? ` [${source}]` : ''}` });

  if (next.bids.every(Boolean)) {
    next.phase = 'playing';
    next.currentPlayer = 0;
    next.leader = 0;
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
  const ledSuit = next.currentTrick.length > 0 ? next.currentTrick[0].card.suit : card.suit;
  next.currentTrick.push({ seat, card });
  next.spadesBroken = next.spadesBroken || (card.suit === 'S' && ledSuit !== 'S');
  next.log.push({ kind: 'play', seat, text: `${PLAYER_NAMES[seat]} 出牌 ${cardLabel(card)}${source ? ` [${source}]` : ''}` });

  if (next.currentTrick.length < 4) {
    next.currentPlayer = (seat + 1) % 4;
    return next;
  }

  const winner = determineTrickWinner(next.currentTrick);
  next.tricksWon[winner] += 1;
  next.completedTricks.push({
    trickNumber: next.trickNumber,
    winner,
    cards: next.currentTrick.map((entry) => ({ seat: entry.seat, card: entry.card })),
  });
  next.currentTrick = [];
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
export function createInitialGame(seed, humanSeat) {
  return {
    seed,
    humanSeat,
    phase: 'bidding',
    currentPlayer: 0,
    leader: 0,
    trickNumber: 1,
    spadesBroken: false,
    hands: dealHands(seed),
    bids: [null, null, null, null],
    tricksWon: [0, 0, 0, 0],
    currentTrick: [],
    completedTricks: [],
    score: null,
    log: [{ kind: 'system', text: `新牌局已创建，seed=${seed}` }],
  };
}

/**
 * Resolve AI turns until the human seat needs attention or the hand ends.
 * Input: the current state.
 * Output: a state where it is either the human's turn or the game is finished.
 */
export async function advanceUntilHuman(state) {
  let next = cloneState(state);

  while (next.phase !== 'finished' && next.currentPlayer !== next.humanSeat) {
    if (next.phase === 'bidding') {
      // 优先尝试后端 AI，如果后端不可用则回退到简单 AI
      try {
        const action = await requestAiAction(next);
        if (action.kind !== 'bid') {
          throw new Error(`AI returned ${action.kind} during bidding`);
        }
        next = applyBid(next, next.currentPlayer, makeBid(action.bid.value, action.bid.type), action.ai);
      } catch (err) {
        console.warn('Backend AI failed, using fallback AI:', err);
        next = applyBid(next, next.currentPlayer, chooseAIBid(next, next.currentPlayer), 'fallback');
      }
      continue;
    }

    if (next.phase === 'playing') {
      try {
        const action = await requestAiAction(next);
        if (action.kind !== 'play') {
          throw new Error(`AI returned ${action.kind} during playing`);
        }
        next = applyCard(next, next.currentPlayer, action.card, action.ai);
      } catch (err) {
        console.warn('Backend AI failed, using fallback AI:', err);
        next = applyCard(next, next.currentPlayer, chooseAICard(next, next.currentPlayer).code, 'fallback');
      }
      continue;
    }

    break;
  }

  return next;
}

/**
 * Apply a human bid and then fast-forward AI turns.
 * Input: a state and the selected bid.
 * Output: the next playable state.
 */
export async function submitHumanBid(state, bid) {
  const afterBid = applyBid(state, state.humanSeat, bid);
  return await advanceUntilHuman(afterBid);
}

/**
 * Apply a human card and then fast-forward AI turns.
 * Input: a state and the chosen card code.
 * Output: the next playable state.
 */
export async function submitHumanCard(state, cardCode) {
  const afterCard = applyCard(state, state.humanSeat, cardCode);
  return await advanceUntilHuman(afterCard);
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