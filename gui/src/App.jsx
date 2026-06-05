/*
 * Spades table UI — immersive felt table with one human seat at the bottom
 * and three Python-backed AI seats around it.
 *
 * All game logic / API calls live in ./game.js and are reused unchanged.
 * This file owns presentation + match flow (single hand vs. race-to-500):
 * seat→screen mapping (you always sit at the bottom), real playing cards,
 * a fanned hand, a central trick area, bidding chips, a status pill, the
 * mode-select screen, the cumulative scoreboard, and the result overlays.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  advanceUntilHuman,
  bidLabel,
  buildReplaySnapshot,
  createInitialGame,
  getHumanLegalCards,
  makeBid,
  submitHumanBid,
  submitHumanCard,
  summarizeGame,
} from './game';

const SEAT_NAMES = ['North', 'East', 'South', 'West'];
const SUIT_SYMBOL = { S: '♠', H: '♥', D: '♦', C: '♣' };
const SUIT_CLASS = { S: 'suit-spade', H: 'suit-heart', D: 'suit-diamond', C: 'suit-club' };
const TARGET_SCORE = 500;

// team(0) = seats 0 & 2, team(1) = seats 1 & 3
const teamOf = (seat) => seat % 2;

const MODE_LABELS = {
  single: '一局制',
  match500: '500 分赛',
  fixedSeed: '给定种子',
};

const seedFromUrl = () => {
  const raw = new URLSearchParams(window.location.search).get('seed');
  if (!raw) return null;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
};

/** Fresh deal seed for random modes (single / 500-match). */
const randomDealSeed = () => Math.floor(Math.random() * 2_147_483_647);

const normalizeSeed = (value) => {
  const parsed = Number.parseInt(String(value).trim(), 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
};

/* ── A single rendered playing card (face up or face down) ──────────── */
function PlayingCard({ card, faceDown = false, size = 'md', legal = false,
                       disabled = false, onPlay = null, style = null,
                       className = '', static: isStatic = false }) {
  if (faceDown || !card) {
    return <div className={`pcard pcard--back size-${size} ${className}`} style={style} />;
  }
  const sym = SUIT_SYMBOL[card.suit];
  const body = (
    <>
      <span className="pcard__corner pcard__corner--tl">
        <b>{card.rank}</b><i>{sym}</i>
      </span>
      <span className="pcard__pip">{sym}</span>
      <span className="pcard__corner pcard__corner--br">
        <b>{card.rank}</b><i>{sym}</i>
      </span>
    </>
  );
  const cls = `pcard size-${size} ${SUIT_CLASS[card.suit] ?? 'suit-spade'} ${legal ? 'is-legal' : ''} ${className}`;
  if (isStatic || !onPlay) {
    return <div className={cls} style={style} aria-label={`${card.rank}${card.suit}`}>{body}</div>;
  }
  const clickable = legal && !disabled;
  return (
    <button
      type="button"
      className={cls}
      style={style}
      disabled={!clickable}
      onClick={clickable ? () => onPlay(card.code) : undefined}
      aria-label={`${card.rank}${card.suit}`}
    >
      {body}
    </button>
  );
}

/* ── A small fan of card-backs for an AI seat (count only) ──────────── */
function CardBackFan({ count }) {
  const shown = Math.min(count, 5);
  const center = (shown - 1) / 2;
  return (
    <div className="back-fan" aria-label={`${count} cards`}>
      {Array.from({ length: shown }, (_, i) => (
        <div
          key={i}
          className="pcard pcard--back size-xs back-fan__card"
          style={{ '--i': i - center }}
        />
      ))}
      <span className="back-fan__count">{count}</span>
    </div>
  );
}

/* ── Bid / Won badges (shared by AI seats and the human) ────────────── */
function TallyBadges({ bid, won, hideBid, justBid }) {
  const numericBid = bid && bid.type !== 'nil' ? bid.value : null;
  const isNil = bid && bid.type === 'nil';
  const made = numericBid != null && won >= numericBid;       // contract met
  const nilOk = isNil && won === 0;
  return (
    <div className="tally">
      <span className={`tally__bid ${justBid ? 'is-pop' : ''}`}>
        叫 {hideBid ? '·' : (isNil ? 'Nil' : numericBid != null ? numericBid : '—')}
      </span>
      <span className={`tally__won ${made || nilOk ? 'is-made' : ''} ${isNil && won > 0 ? 'is-broken' : ''}`}>
        吃 {won}
      </span>
    </div>
  );
}

/* ── One AI seat badge (top / left / right) ─────────────────────────── */
function AiSeat({ pos, seat, summary, game, active }) {
  const hasBid = !!game.bids[seat];
  const justBid = game.phase === 'bidding' && game.lastBidSeat === seat && hasBid;
  return (
    <div className={`seat seat--${pos} ${active ? 'is-active' : ''} team-${teamOf(seat)}`}>
      <div className="seat__plate">
        <span className="seat__avatar">{SEAT_NAMES[seat][0]}</span>
        <div className="seat__meta">
          <strong>{SEAT_NAMES[seat]}</strong>
          <TallyBadges bid={game.bids[seat]} won={summary.tricksWon[seat]}
                       hideBid={game.phase === 'bidding' && !hasBid} justBid={justBid} />
        </div>
      </div>
      <CardBackFan count={game.hands[seat].length} />
    </div>
  );
}

/* ── The played card for a seat sitting at screen position `pos` ────── */
function TrickSlot({ pos, entry, justPlayed, collecting, winnerPos }) {
  if (!entry) return <div className={`slot slot--${pos}`} />;
  const cls = [
    'slot__card',
    justPlayed ? `slot__card--in-${pos}` : '',
    collecting ? `slot__card--collect-${winnerPos}` : '',
  ].join(' ');
  return (
    <div className={`slot slot--${pos}`}>
      <PlayingCard card={entry.card} size="md" static className={cls} />
    </div>
  );
}

/* ── Face-up hand spread for replay (all seats show their cards) ────── */
function ReplayHandSpread({ cards, pos, size = 'sm', highlightCode = null }) {
  const spread = Math.min(6, pos === 'bottom' ? 56 / Math.max(1, cards.length) : 42 / Math.max(1, cards.length));
  const isVertical = pos === 'left' || pos === 'right';
  return (
    <div className={`replay-hand replay-hand--${pos} ${isVertical ? 'is-vertical' : ''}`} style={{ '--n': cards.length }}>
      {cards.map((card, i) => {
        const center = (cards.length - 1) / 2;
        const offset = (i - center) * (isVertical ? 14 : 1);
        const rot = isVertical ? 0 : (i - center) * spread;
        const highlight = highlightCode === card.code;
        return (
          <PlayingCard
            key={card.code}
            card={card}
            size={size}
            static
            className={`replay-hand__card ${highlight ? 'is-highlight' : ''}`}
            style={{
              '--rot': `${rot}deg`,
              '--idx': i,
              '--off': offset,
            }}
          />
        );
      })}
    </div>
  );
}

/** Rebuild replay table state from cursor position. */
function rebuildReplayState(snapshot, playIndex, trickComplete) {
  const hands = snapshot.hands.map((hand) => hand.map((card) => ({ ...card })));
  const currentTrick = [];
  let complete = false;
  let trickWinner = -1;
  let lastPlayedSeat = -1;
  let activeTrick = 1;

  for (let i = 0; i < playIndex; i += 1) {
    const play = snapshot.plays[i];
    hands[play.seat] = hands[play.seat].filter((card) => card.code !== play.card.code);
    currentTrick.push({ seat: play.seat, card: play.card });
    lastPlayedSeat = play.seat;
    activeTrick = play.trickNumber;

    if (currentTrick.length >= 4) {
      if (trickComplete && i === playIndex - 1) {
        complete = true;
        trickWinner = play.winner;
      } else {
        currentTrick.length = 0;
      }
    }
  }

  return {
    remainingHands: hands,
    currentTrick,
    trickComplete: complete,
    trickWinner,
    lastPlayedSeat,
    activeTrick,
  };
}

function replayTricksWonAt(snapshot, playIndex, trickComplete) {
  const collected = playIndex === 0
    ? 0
    : trickComplete && playIndex % 4 === 0
      ? Math.floor(playIndex / 4) - 1
      : Math.floor(playIndex / 4);
  const won = [0, 0, 0, 0];
  for (let t = 0; t < collected; t += 1) {
    won[snapshot.completedTricks[t].winner] += 1;
  }
  return won;
}

function ReplaySeatPanel({ seat, snapshot, tricksWon, pos, cards, highlightCode, isHuman = false }) {
  return (
    <div className={`replay-seat replay-seat--${pos}`}>
      <div className={`replay-seat__label team-${teamOf(seat)}`}>
        <span className="replay-seat__avatar">{SEAT_NAMES[seat][0]}</span>
        <div className="replay-seat__meta">
          <strong>{SEAT_NAMES[seat]}{isHuman ? <> <em>(You)</em></> : null}</strong>
          <TallyBadges bid={snapshot.bids[seat]} won={tricksWon[seat]} />
        </div>
      </div>
      <ReplayHandSpread
        cards={cards}
        pos={pos}
        size={pos === 'bottom' ? 'lg' : 'sm'}
        highlightCode={highlightCode}
      />
    </div>
  );
}

/* ── Full-hand replay screen (manual step-by-step only) ─────────────── */
function ReplayScreen({ snapshot, onExit }) {
  const [phase, setPhase] = useState('ready'); // ready | done
  const [playIndex, setPlayIndex] = useState(0);
  const [trickComplete, setTrickComplete] = useState(false);
  const [view, setView] = useState(() => rebuildReplayState(snapshot, 0, false));

  const applyCursor = (index, complete, nextPhase = 'ready') => {
    setPlayIndex(index);
    setTrickComplete(complete);
    setView(rebuildReplayState(snapshot, index, complete));
    setPhase(nextPhase);
  };

  const posOf = (seat) => ['bottom', 'left', 'top', 'right'][(seat - snapshot.humanSeat + 4) % 4];
  const seatAt = (pos) => [0, 1, 2, 3].find((s) => posOf(s) === pos);

  const { remainingHands, currentTrick, trickWinner, lastPlayedSeat, activeTrick } = view;
  const tricksWon = replayTricksWonAt(snapshot, playIndex, trickComplete);

  const trickByPos = {};
  for (const entry of currentTrick) trickByPos[posOf(entry.seat)] = entry;

  const justPlayedPos = !trickComplete && lastPlayedSeat >= 0 ? posOf(lastPlayedSeat) : null;
  const winnerPos = trickComplete && trickWinner >= 0 ? posOf(trickWinner) : null;

  const lastPlay = playIndex > 0 ? snapshot.plays[playIndex - 1] : null;
  const nextPlay = playIndex < snapshot.plays.length ? snapshot.plays[playIndex] : null;
  const highlightCode = lastPlay?.card.code ?? null;

  const resetReplay = () => applyCursor(0, false, 'ready');

  const stepForward = () => {
    if (phase === 'done') return;

    if (trickComplete) {
      if (playIndex >= snapshot.plays.length) applyCursor(playIndex, false, 'done');
      else applyCursor(playIndex, false, 'ready');
      return;
    }

    if (playIndex >= snapshot.plays.length) {
      setPhase('done');
      return;
    }

    const nextIndex = playIndex + 1;
    const nextComplete = nextIndex % 4 === 0;
    applyCursor(nextIndex, nextComplete, 'ready');
  };

  const stepBack = () => {
    if (phase === 'done') {
      applyCursor(snapshot.plays.length, true, 'ready');
      return;
    }

    if (trickComplete) {
      applyCursor(playIndex - 1, false, 'ready');
      return;
    }

    if (playIndex === 0) return;

    if (playIndex % 4 === 0) {
      applyCursor(playIndex, true, 'ready');
      return;
    }

    applyCursor(playIndex - 1, false, 'ready');
  };

  const canStepBack = phase === 'done' || playIndex > 0 || trickComplete;
  const canStepForward = phase !== 'done' && (trickComplete || playIndex < snapshot.plays.length);

  let statusText = '四家手牌已摊开，点击「下一步」开始复盘';
  if (phase === 'done') {
    statusText = '复盘结束';
  } else if (trickComplete) {
    statusText = `第 ${activeTrick} 墩由 ${SEAT_NAMES[trickWinner]} 赢下 · 点击下一步收墩`;
  } else if (lastPlay) {
    statusText = `${SEAT_NAMES[lastPlay.seat]} 出 ${lastPlay.card.rank}${SUIT_SYMBOL[lastPlay.card.suit]}`;
  } else if (nextPlay) {
    statusText = `下一步：${SEAT_NAMES[nextPlay.seat]} 出牌`;
  }

  const progress = snapshot.plays.length > 0 ? Math.round((playIndex / snapshot.plays.length) * 100) : 0;

  return (
    <div className="felt felt--replay">
      <header className="topbar">
        <div className="brand">
          <button className="brand__back" onClick={onExit} title="返回结算">←</button>
          <span className="brand__pip">♠</span> 复盘回放
        </div>
        <div className="topbar__right">
          <div className="replay-meta">
            <span>种子 {snapshot.seed}</span>
            <strong>第 {activeTrick} 墩 · {progress}%</strong>
          </div>
        </div>
      </header>

      <main className="stage stage--replay">
        <div className="stage__top">
          <ReplaySeatPanel
            seat={seatAt('top')}
            snapshot={snapshot}
            tricksWon={tricksWon}
            pos="top"
            cards={remainingHands[seatAt('top')]}
            highlightCode={highlightCode}
          />
        </div>

        <div className="stage__left">
          <ReplaySeatPanel
            seat={seatAt('left')}
            snapshot={snapshot}
            tricksWon={tricksWon}
            pos="left"
            cards={remainingHands[seatAt('left')]}
            highlightCode={highlightCode}
          />
        </div>

        <div className="table">
          <div className={`table__felt ${trickComplete ? 'is-collecting' : ''}`}>
            {['top', 'left', 'right', 'bottom'].map((p) => (
              <TrickSlot
                key={p}
                pos={p}
                entry={trickByPos[p]}
                justPlayed={!trickComplete && justPlayedPos === p}
                collecting={trickComplete}
                winnerPos={winnerPos}
              />
            ))}
            <div className="status">
              <span className="status__text">{statusText}</span>
              <span className="status__trick">复盘 · 第 {activeTrick} 墩</span>
            </div>
          </div>
        </div>

        <div className="stage__right">
          <ReplaySeatPanel
            seat={seatAt('right')}
            snapshot={snapshot}
            tricksWon={tricksWon}
            pos="right"
            cards={remainingHands[seatAt('right')]}
            highlightCode={highlightCode}
          />
        </div>

        <div className="stage__hand">
          <ReplaySeatPanel
            seat={snapshot.humanSeat}
            snapshot={snapshot}
            tricksWon={tricksWon}
            pos="bottom"
            cards={remainingHands[snapshot.humanSeat]}
            highlightCode={highlightCode}
            isHuman
          />
        </div>
      </main>

      <footer className="replay-controls">
        <button className="btn-ghost" onClick={resetReplay} disabled={!canStepBack}>重新摊开</button>
        <button className="btn-ghost" onClick={stepBack} disabled={!canStepBack}>上一步</button>
        <button className="btn-new" onClick={stepForward} disabled={!canStepForward}>下一步</button>
        <button className="btn-ghost" onClick={onExit}>返回结算</button>
      </footer>
    </div>
  );
}

/* ── Mode-select screen ─────────────────────────────────────────────── */
function ModeMenu({ onPick, onFixedSeedStart, urlSeed }) {
  const [seedInput, setSeedInput] = useState(urlSeed != null ? String(urlSeed) : '123');
  const [seat, setSeat] = useState(0);
  const [seedError, setSeedError] = useState('');

  const handleFixedStart = () => {
    const seed = normalizeSeed(seedInput);
    if (seed == null) {
      setSeedError('请输入非负整数种子');
      return;
    }
    setSeedError('');
    onFixedSeedStart(seed, seat);
  };

  return (
    <div className="menu">
      <div className="menu__brand"><span className="brand__pip">♠</span> Spades AI</div>
      <p className="menu__sub">选择对战模式</p>
      <div className="menu__cards">
        <button className="mode-card" onClick={() => onPick('single')}>
          <span className="mode-card__icon">🃏</span>
          <strong>一局制</strong>
          <span className="mode-card__desc">每局随机发牌，打完一局即结算。</span>
        </button>
        <button className="mode-card mode-card--gold" onClick={() => onPick('match500')}>
          <span className="mode-card__icon">🏆</span>
          <strong>500 分赛</strong>
          <span className="mode-card__desc">每局随机发牌，逐局累计至 500 分。</span>
        </button>
        <div className="mode-card mode-card--seed">
          <span className="mode-card__icon">🎲</span>
          <strong>给定种子</strong>
          <span className="mode-card__desc">输入种子复现同一副牌，只打一局。</span>
          <div className="seed-form">
            <label className="seed-form__field">
              <span>种子</span>
              <input
                type="number"
                min="0"
                step="1"
                value={seedInput}
                onChange={(e) => { setSeedInput(e.target.value); setSeedError(''); }}
                placeholder="例如 12345"
              />
            </label>
            <label className="seed-form__field">
              <span>座位</span>
              <select value={seat} onChange={(e) => setSeat(Number(e.target.value))}>
                {SEAT_NAMES.map((label, i) => <option key={label} value={i}>{i} · {label}</option>)}
              </select>
            </label>
            <button type="button" className="btn-new seed-form__go" onClick={handleFixedStart}>开始对局</button>
          </div>
          {seedError ? <p className="seed-form__error">{seedError}</p> : null}
        </div>
      </div>
    </div>
  );
}

/* ── main app ──────────────────────────────────────────────────────── */
export default function App() {
  const urlSeed = seedFromUrl();
  const [screen, setScreen] = useState('menu');     // 'menu' | 'game' | 'replay'
  const [mode, setMode] = useState('single');        // 'single' | 'match500' | 'fixedSeed'
  const [humanSeat, setHumanSeat] = useState(0);
  const [busy, setBusy] = useState(false);
  const [game, setGame] = useState(() => createInitialGame(0, 0));
  const [replaySnapshot, setReplaySnapshot] = useState(null);

  // 500-match cumulative state
  const [matchScore, setMatchScore] = useState({ ns: 0, ew: 0 });
  const [handNo, setHandNo] = useState(1);
  const [matchOver, setMatchOver] = useState(false);
  const [matchFirstSeat, setMatchFirstSeat] = useState(0); // rotates each hand in 500-match
  const settledSeedRef = useRef(null);   // guards against double-counting a hand

  // Deal a hand. Random modes should omit `seed` or pass randomDealSeed().
  const dealHand = async (seat, seed = randomDealSeed(), firstSeat = 0) => {
    setBusy(true);
    try {
      const resolvedSeat = Number.isInteger(seat) ? seat : humanSeat;
      setHumanSeat(resolvedSeat);
      const fresh = createInitialGame(seed, resolvedSeat, firstSeat);
      setGame(fresh);
      setGame(await advanceUntilHuman(fresh, setGame));
    } finally {
      setBusy(false);
    }
  };

  // Start a brand-new match in the given mode (resets cumulative score).
  const startMatch = async (chosenMode, seat = humanSeat) => {
    setMode(chosenMode);
    setMatchScore({ ns: 0, ew: 0 });
    setHandNo(1);
    setMatchOver(false);
    setMatchFirstSeat(0);
    settledSeedRef.current = null;
    setScreen('game');
    await dealHand(seat);
  };

  const startFixedSeedMatch = async (seed, seat) => {
    setMode('fixedSeed');
    setMatchScore({ ns: 0, ew: 0 });
    setHandNo(1);
    setMatchOver(false);
    settledSeedRef.current = null;
    setHumanSeat(seat);
    setScreen('game');
    await dealHand(seat, seed);
  };

  // Next hand within a running 500-match (keeps cumulative score).
  const nextHand = async () => {
    const nextFirstSeat = (matchFirstSeat + 1) % 4;
    setMatchFirstSeat(nextFirstSeat);
    setHandNo((n) => n + 1);
    await dealHand(humanSeat, randomDealSeed(), nextFirstSeat);
  };

  const handleBid = async (bid) => {
    if (busy || game.phase !== 'bidding' || game.currentPlayer !== game.humanSeat) return;
    setBusy(true);
    try {
      setGame(await submitHumanBid(game, bid, setGame));
    } finally {
      setBusy(false);
    }
  };

  const handlePlay = async (cardCode) => {
    if (busy || game.phase !== 'playing' || game.currentPlayer !== game.humanSeat) return;
    setBusy(true);
    try {
      setGame(await submitHumanCard(game, cardCode, setGame));
    } finally {
      setBusy(false);
    }
  };

  const summary = summarizeGame(game);
  const finished = game.phase === 'finished';

  // Save a replay snapshot once per finished hand.
  useEffect(() => {
    if (screen !== 'game' || !finished || !summary.score) return;
    setReplaySnapshot(buildReplaySnapshot(game));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finished, game.seed, screen]);

  const enterReplay = () => {
    if (!replaySnapshot) setReplaySnapshot(buildReplaySnapshot(game));
    setScreen('replay');
  };

  // ── 500-match: accumulate this hand's score exactly once ─────────────
  useEffect(() => {
    if (mode !== 'match500' || screen !== 'game') return;
    if (!finished || !summary.score) return;
    if (settledSeedRef.current === game.seed) return;  // already counted
    settledSeedRef.current = game.seed;
    setMatchScore((prev) => {
      const ns = prev.ns + summary.score.northSouth;
      const ew = prev.ew + summary.score.eastWest;
      if ((ns >= TARGET_SCORE || ew >= TARGET_SCORE) && ns !== ew) {
        setMatchOver(true);
      }
      return { ns, ew };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finished, game.seed, mode, screen]);

  const legalCards = game.phase === 'playing' ? getHumanLegalCards(game) : [];
  const legalSet = useMemo(() => new Set(legalCards.map((c) => c.code)), [legalCards]);

  // seat → screen position (you are always at the bottom)
  const posOf = (seat) => ['bottom', 'left', 'top', 'right'][(seat - game.humanSeat + 4) % 4];
  const seatAt = (pos) => [0, 1, 2, 3].find((s) => posOf(s) === pos);

  const humanHand = game.hands[game.humanSeat];
  const myTurn = game.currentPlayer === game.humanSeat;
  const isBidding = game.phase === 'bidding';
  const isPlaying = game.phase === 'playing';

  // current-trick entries keyed by screen position
  const trickByPos = {};
  for (const entry of game.currentTrick) trickByPos[posOf(entry.seat)] = entry;

  // animation hints
  const justPlayedPos = game.lastPlayedSeat >= 0 ? posOf(game.lastPlayedSeat) : null;
  const collecting = !!game.trickComplete;
  const winnerPos = collecting && game.trickWinner >= 0 ? posOf(game.trickWinner) : null;

  // status text in the center of the table
  let statusText = '';
  if (finished) statusText = '本局结束';
  else if (busy && !myTurn) statusText = '对手出牌中…';
  else if (isBidding && myTurn) statusText = '请叫牌';
  else if (isBidding) statusText = `${SEAT_NAMES[game.currentPlayer]} 叫牌中…`;
  else if (isPlaying && myTurn) statusText = '请出牌';
  else if (isPlaying) statusText = `${SEAT_NAMES[game.currentPlayer]} 出牌中…`;

  const spread = Math.min(7, 56 / Math.max(1, humanHand.length)); // deg between cards

  // ── mode-select screen ──
  if (screen === 'menu') {
    return (
      <div className="felt felt--menu">
        <ModeMenu
          onPick={(m) => { void startMatch(m); }}
          onFixedSeedStart={(seed, seat) => { void startFixedSeedMatch(seed, seat); }}
          urlSeed={urlSeed}
        />
      </div>
    );
  }

  if (screen === 'replay' && replaySnapshot) {
    return (
      <ReplayScreen
        snapshot={replaySnapshot}
        onExit={() => setScreen('game')}
      />
    );
  }

  // which scoreboard numbers to show in the top bar
  const boardNS = mode === 'match500' ? matchScore.ns : (summary.score ? summary.score.northSouth : 0);
  const boardEW = mode === 'match500' ? matchScore.ew : (summary.score ? summary.score.eastWest : 0);

  // overlay variant for the finished hand
  const myTeam = teamOf(game.humanSeat);
  const teamWon = (nsScore, ewScore) => (myTeam === 0 ? nsScore >= ewScore : ewScore > nsScore);

  return (
    <div className="felt">
      {/* ── top bar ── */}
      <header className="topbar">
        <div className="brand">
          <button className="brand__back" onClick={() => setScreen('menu')} disabled={busy} title="返回模式选择">←</button>
          <span className="brand__pip">♠</span> Spades
        </div>
        <div className="topbar__right">
          <div className="scoreboard">
            <div className="score score--ns"><span>NS</span><strong>{boardNS}</strong></div>
            <div className="score score--ew"><span>EW</span><strong>{boardEW}</strong></div>
          </div>
          <div className="match-info">
            <span>模式</span>
            <strong>{MODE_LABELS[mode] ?? mode}</strong>
          </div>
          {mode === 'match500' ? (
            <div className="match-info"><span>局数</span><strong>第 {handNo} 局</strong></div>
          ) : null}
          <div className="match-info"><span>种子</span><strong>{game.seed}</strong></div>
          <label className="ctl">
            <span>座位</span>
            <select value={humanSeat} onChange={(e) => setHumanSeat(Number(e.target.value))} disabled={busy || (screen === 'game' && !finished)}>
              {SEAT_NAMES.map((label, seat) => <option key={label} value={seat}>{seat} · {label}</option>)}
            </select>
          </label>
        </div>
      </header>

      {/* ── table ── */}
      <main className="stage">
        <div className="stage__top">
          <AiSeat pos="top" seat={seatAt('top')} summary={summary} game={game}
                  active={game.currentPlayer === seatAt('top')} />
        </div>
        <div className="stage__left">
          <AiSeat pos="left" seat={seatAt('left')} summary={summary} game={game}
                  active={game.currentPlayer === seatAt('left')} />
        </div>

        <div className="table">
          <div className={`table__felt ${collecting ? 'is-collecting' : ''}`}>
            {['top', 'left', 'right', 'bottom'].map((p) => (
              <TrickSlot
                key={p}
                pos={p}
                entry={trickByPos[p]}
                justPlayed={!collecting && justPlayedPos === p}
                collecting={collecting}
                winnerPos={winnerPos}
              />
            ))}
            <div className={`status ${busy && !myTurn ? 'is-busy' : ''} ${myTurn && !finished ? 'is-you' : ''}`}>
              {busy && !myTurn ? <span className="spinner" /> : null}
              <span className="status__text">{statusText}</span>
              {isPlaying ? <span className="status__trick">第 {summary.trickNumber} 墩</span> : null}
            </div>
          </div>
        </div>

        <div className="stage__right">
          <AiSeat pos="right" seat={seatAt('right')} summary={summary} game={game}
                  active={game.currentPlayer === seatAt('right')} />
        </div>

        {/* ── human area ── */}
        <div className="stage__hand">
          <div className={`me ${myTurn && !finished ? 'is-active' : ''} team-${myTeam}`}>
            <span className="me__avatar">{SEAT_NAMES[game.humanSeat][0]}</span>
            <div className="me__meta">
              <strong>{SEAT_NAMES[game.humanSeat]} <em>(You)</em></strong>
              <TallyBadges bid={game.bids[game.humanSeat]} won={summary.tricksWon[game.humanSeat]}
                           hideBid={isBidding && !game.bids[game.humanSeat]} />
            </div>
          </div>

          {isBidding && myTurn ? (
            <div className="bidbar">
              <button className="chip chip--nil" disabled={busy} onClick={() => handleBid(makeBid(0, 'nil'))}>Nil</button>
              {Array.from({ length: 13 }, (_, i) => i + 1).map((b) => (
                <button key={b} className="chip" disabled={busy} onClick={() => handleBid(makeBid(b))}>{b}</button>
              ))}
            </div>
          ) : null}

          <div className="fan" style={{ '--n': humanHand.length }} key={game.seed}>
            {humanHand.map((card, i) => {
              const center = (humanHand.length - 1) / 2;
              const legal = isPlaying && myTurn && legalSet.has(card.code);
              const playable = isPlaying && myTurn;
              return (
                <PlayingCard
                  key={card.code}
                  card={card}
                  size="lg"
                  legal={legal}
                  disabled={!playable || (playable && !legal)}
                  onPlay={handlePlay}
                  className={`fan__card ${playable && !legal ? 'is-muted' : ''}`}
                  style={{ '--rot': `${(i - center) * spread}deg`, '--idx': i }}
                />
              );
            })}
          </div>
        </div>
      </main>

      {/* ── compact log ── */}
      <aside className="mini-log">
        {game.log.slice(-5).map((entry, i) => (
          <div key={`${entry.kind}-${i}`} className={`mini-log__row log-${entry.kind}`}>{entry.text}</div>
        ))}
      </aside>

      {/* ── result overlays ── */}
      {finished && summary.score ? (
        mode === 'fixedSeed' ? (
          <ResultOverlay
            eyebrow="牌局结束"
            subtitle={`种子 ${game.seed}`}
            ns={summary.score.northSouth}
            ew={summary.score.eastWest}
            verdict={teamWon(summary.score.northSouth, summary.score.eastWest) ? '你的队伍获胜 🎉' : '你的队伍落败'}
            buttonLabel="返回菜单"
            onButton={() => setScreen('menu')}
            replayLabel="复盘回放"
            onReplay={enterReplay}
            busy={busy}
          />
        ) : mode === 'single' ? (
          <ResultOverlay
            eyebrow="牌局结束"
            subtitle={`种子 ${game.seed}`}
            ns={summary.score.northSouth}
            ew={summary.score.eastWest}
            verdict={teamWon(summary.score.northSouth, summary.score.eastWest) ? '你的队伍获胜 🎉' : '你的队伍落败'}
            buttonLabel="再来一局"
            onButton={() => { void dealHand(humanSeat); }}
            replayLabel="复盘回放"
            onReplay={enterReplay}
            busy={busy}
          />
        ) : matchOver ? (
          <ResultOverlay
            eyebrow={`500 分赛结束 · 共 ${handNo} 局`}
            ns={matchScore.ns}
            ew={matchScore.ew}
            verdict={teamWon(matchScore.ns, matchScore.ew) ? '你的队伍赢得整场 🏆' : '你的队伍败北'}
            buttonLabel="返回菜单"
            onButton={() => setScreen('menu')}
            replayLabel="复盘上一局"
            onReplay={enterReplay}
            busy={busy}
          />
        ) : (
          <ResultOverlay
            eyebrow={`第 ${handNo} 局结束`}
            subtitle="本局得分"
            ns={summary.score.northSouth}
            ew={summary.score.eastWest}
            cumulative={matchScore}
            verdict={`目标 ${TARGET_SCORE} 分`}
            buttonLabel="下一局"
            onButton={() => { void nextHand(); }}
            replayLabel="复盘回放"
            onReplay={enterReplay}
            busy={busy}
          />
        )
      ) : null}
    </div>
  );
}

/* ── Reusable result overlay ────────────────────────────────────────── */
function ResultOverlay({ eyebrow, subtitle, ns, ew, cumulative, verdict, buttonLabel, onButton,
                         replayLabel, onReplay, busy }) {
  return (
    <div className="overlay">
      <div className="overlay__card">
        <p className="overlay__eyebrow">{eyebrow}</p>
        {subtitle ? <p className="overlay__subtitle">{subtitle}</p> : null}
        <div className="overlay__scores">
          <div className={ns >= ew ? 'win' : ''}><span>North / South</span><strong>{ns}</strong></div>
          <div className={ew > ns ? 'win' : ''}><span>East / West</span><strong>{ew}</strong></div>
        </div>
        {cumulative ? (
          <div className="overlay__cumulative">
            <span>累计</span>
            <strong className="c-ns">NS {cumulative.ns}</strong>
            <strong className="c-ew">EW {cumulative.ew}</strong>
          </div>
        ) : null}
        <p className="overlay__verdict">{verdict}</p>
        <div className="overlay__actions">
          {onReplay ? (
            <button className="btn-ghost" onClick={onReplay} disabled={busy}>{replayLabel}</button>
          ) : null}
          <button className="btn-new" onClick={onButton} disabled={busy}>{buttonLabel}</button>
        </div>
      </div>
    </div>
  );
}
