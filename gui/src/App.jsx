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
  advanceUntilFinished,
  advanceUntilHuman,
  bidLabel,
  buildReplaySnapshot,
  confirmLocalShowdown,
  createInitialGame,
  getHumanLegalCards,
  makeBid,
  remoteStateFromServer,
  showdownWaitingForPartner,
  submitHumanBid,
  submitHumanCard,
  summarizeGame,
} from './game';
import { ShowdownPanel, showdownHandsForDisplay } from './showdown';

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
  aiTest: '测试 AI',
  remote: '远程对战',
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
function AiSeat({ pos, seat, summary, game, active, revealedCards = null }) {
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
      {revealedCards ? (
        <ReplayHandSpread cards={revealedCards} pos={pos} size="sm" />
      ) : (
        <CardBackFan count={game.hands[seat].length} />
      )}
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

function ReplaySeatPanel({ seat, snapshot, tricksWon, pos, cards, highlightCode, isViewSeat = false, viewLabel = '(You)' }) {
  return (
    <div className={`replay-seat replay-seat--${pos}`}>
      <div className={`replay-seat__label team-${teamOf(seat)}`}>
        <span className="replay-seat__avatar">{SEAT_NAMES[seat][0]}</span>
        <div className="replay-seat__meta">
          <strong>{SEAT_NAMES[seat]}{isViewSeat ? <> <em>{viewLabel}</em></> : null}</strong>
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
function ReplayScreen({ snapshot, onExit, viewLabel = '(You)' }) {
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
            isViewSeat
            viewLabel={viewLabel}
          />
        </div>
      </main>

      <footer className="replay-controls">
        <button className="btn-ghost" onClick={resetReplay} disabled={!canStepBack}>重新摊开</button>
        <button className="btn-ghost" onClick={stepBack} disabled={!canStepBack}>上一步</button>
        <button className="btn-new" onClick={stepForward} disabled={!canStepForward}>下一步</button>
        <button className="btn-ghost" onClick={onExit}>{viewLabel === '(视角)' ? '返回菜单' : '返回结算'}</button>
      </footer>
    </div>
  );
}

/* ── Mode-select screen ─────────────────────────────────────────────── */
function ModeMenu({ onPick, onFixedSeedStart, onAiTestStart, onRemoteStart, urlSeed }) {
  const [seedInput, setSeedInput] = useState(urlSeed != null ? String(urlSeed) : '123');
  const [testSeedInput, setTestSeedInput] = useState(urlSeed != null ? String(urlSeed) : '123');
  const [remoteSeedInput, setRemoteSeedInput] = useState(urlSeed != null ? String(urlSeed) : '123');
  const [remoteRoomInput, setRemoteRoomInput] = useState('');
  const [remoteUrlInput, setRemoteUrlInput] = useState('localhost:8765');
  const [seat, setSeat] = useState(0);
  const [viewSeat, setViewSeat] = useState(0);
  const [remoteSeat, setRemoteSeat] = useState(0);
  const [seedError, setSeedError] = useState('');
  const [testSeedError, setTestSeedError] = useState('');
  const [remoteError, setRemoteError] = useState('');

  const handleFixedStart = () => {
    const seed = normalizeSeed(seedInput);
    if (seed == null) {
      setSeedError('请输入非负整数种子');
      return;
    }
    setSeedError('');
    onFixedSeedStart(seed, seat);
  };

  const handleAiTestStart = () => {
    const seed = normalizeSeed(testSeedInput);
    if (seed == null) {
      setTestSeedError('请输入非负整数种子');
      return;
    }
    setTestSeedError('');
    onAiTestStart(seed, viewSeat);
  };

  const handleRemoteStart = () => {
    const seed = normalizeSeed(remoteSeedInput);
    if (seed == null) {
      setRemoteError('请输入非负整数种子');
      return;
    }
    const room = remoteRoomInput.trim();
    if (!room) {
      setRemoteError('请输入房间号');
      return;
    }
    const url = remoteUrlInput.trim();
    if (!url) {
      setRemoteError('请输入服务器地址');
      return;
    }
    setRemoteError('');
    onRemoteStart(url, room.toUpperCase(), seed, remoteSeat);
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
        <div className="mode-card mode-card--ai-test">
          <span className="mode-card__icon">🤖</span>
          <strong>测试 AI</strong>
          <span className="mode-card__desc">指定种子让四家 AI 自动对局，结束后复盘。</span>
          <div className="seed-form">
            <label className="seed-form__field">
              <span>种子</span>
              <input
                type="number"
                min="0"
                step="1"
                value={testSeedInput}
                onChange={(e) => { setTestSeedInput(e.target.value); setTestSeedError(''); }}
                placeholder="例如 12345"
              />
            </label>
            <label className="seed-form__field">
              <span>复盘视角</span>
              <select value={viewSeat} onChange={(e) => setViewSeat(Number(e.target.value))}>
                {SEAT_NAMES.map((label, i) => <option key={label} value={i}>{i} · {label}</option>)}
              </select>
            </label>
            <button type="button" className="btn-new seed-form__go" onClick={handleAiTestStart}>开始对局</button>
          </div>
          {testSeedError ? <p className="seed-form__error">{testSeedError}</p> : null}
        </div>
        <div className="mode-card mode-card--remote">
          <span className="mode-card__icon">🌐</span>
          <strong>远程对战</strong>
          <span className="mode-card__desc">两人各在一台电脑，通过网络对战两个 AI。</span>
          <div className="seed-form">
            <label className="seed-form__field">
              <span>服务器</span>
              <input
                type="text"
                value={remoteUrlInput}
                onChange={(e) => { setRemoteUrlInput(e.target.value); setRemoteError(''); }}
                placeholder="IP:端口 (云服务器请用 wss://域名:8443)"
              />
            </label>
            <label className="seed-form__field">
              <span>房间号</span>
              <input
                type="text"
                value={remoteRoomInput}
                onChange={(e) => { setRemoteRoomInput(e.target.value); setRemoteError(''); }}
                placeholder="例如 ABCD"
                style={{ textTransform: 'uppercase' }}
              />
            </label>
            <label className="seed-form__field">
              <span>种子</span>
              <input
                type="number"
                min="0"
                step="1"
                value={remoteSeedInput}
                onChange={(e) => { setRemoteSeedInput(e.target.value); setRemoteError(''); }}
                placeholder="例如 12345"
              />
            </label>
            <label className="seed-form__field">
              <span>座位</span>
              <select value={remoteSeat} onChange={(e) => setRemoteSeat(Number(e.target.value))}>
                {SEAT_NAMES.map((label, i) => <option key={label} value={i}>{i} · {label}</option>)}
              </select>
            </label>
            <p className="seed-form__hint">搭档座位自动为对家 ({(remoteSeat + 2) % 4})</p>
            <button type="button" className="btn-new seed-form__go" onClick={handleRemoteStart}>连接</button>
          </div>
          {remoteError ? <p className="seed-form__error">{remoteError}</p> : null}
        </div>
      </div>
    </div>
  );
}

/* ── main app ──────────────────────────────────────────────────────── */
export default function App() {
  const urlSeed = seedFromUrl();
  const [screen, setScreen] = useState('menu');     // 'menu' | 'game' | 'replay'
  const [mode, setMode] = useState('single');        // 'single' | 'match500' | 'fixedSeed' | 'aiTest'
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

  // Remote (networked) game state
  const [remote, setRemote] = useState({
    status: 'idle',     // 'connecting' | 'joined' | 'waiting' | 'your_turn' | 'finished'
    error: '',
    mySeat: -1,
    opponentSeat: -1,
    legalCards: null,   // Set of code strings
    legalBids: null,    // [{value, type}, ...]
    serverUrl: 'localhost:8765',
    roomCode: '',
    seed: '',
  });
  const wsRef = useRef(null);
  const sentShowdownRef = useRef(null);

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

  const startAiTest = async (seed, viewSeat = 0) => {
    setMode('aiTest');
    setMatchScore({ ns: 0, ew: 0 });
    setHandNo(1);
    setMatchOver(false);
    settledSeedRef.current = null;
    setHumanSeat(viewSeat);
    setScreen('game');
    setBusy(true);
    try {
      const fresh = createInitialGame(seed, viewSeat, 0);
      setGame(fresh);
      const finalState = await advanceUntilFinished(fresh, setGame);
      setGame(finalState);
      if (finalState.phase === 'finished') {
        setReplaySnapshot(buildReplaySnapshot(finalState));
        setScreen('replay');
      }
    } finally {
      setBusy(false);
    }
  };

  // Next hand within a running 500-match (keeps cumulative score).
  const nextHand = async () => {
    const nextFirstSeat = (matchFirstSeat + 1) % 4;
    setMatchFirstSeat(nextFirstSeat);
    setHandNo((n) => n + 1);
    await dealHand(humanSeat, randomDealSeed(), nextFirstSeat);
  };

  // ── Remote (networked) game handlers ───────────────────────────

  const connectRemote = async (serverUrl, roomCode, seed, seat) => {
    setRemote((r) => ({ ...r, status: 'connecting', error: '', roomCode, seed: String(seed) }));
    try {
      // Normalise user input into a WebSocket URL.
      // Supported inputs: wss://host, ws://host, https://host, http://host, host:port
      let url;
      if (serverUrl.startsWith('wss://') || serverUrl.startsWith('ws://')) {
        url = serverUrl;
      } else if (serverUrl.startsWith('https://')) {
        url = serverUrl.replace(/^https/, 'wss');
      } else if (serverUrl.startsWith('http://')) {
        url = serverUrl.replace(/^http/, 'ws');
      } else {
        // Heuristic: ports 443/8443 → wss, others → ws
        const securePort = /:(443|8443)$/.test(serverUrl);
        url = (securePort ? 'wss://' : 'ws://') + serverUrl;
      }
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ type: 'join', room: roomCode, seed, seat }));
        setRemote((r) => ({ ...r, status: 'joined', mySeat: seat, roomCode, seed: String(seed) }));
      };

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
          case 'joined':
            setRemote((r) => ({ ...r, status: 'waiting', error: '' }));
            break;
          case 'opponent_joined':
            setRemote((r) => ({
              ...r,
              status: 'playing',
              opponentSeat: msg.opponentSeat,
              mySeat: msg.yourSeat,
            }));
            break;
          case 'game_state': {
            const mySeat = msg.seat;
            const remoteGame = remoteStateFromServer(msg, mySeat);
            setGame(remoteGame);
            setHumanSeat(mySeat);
            if (!remoteGame.showdown) sentShowdownRef.current = null;
            if (remoteGame.showdown) {
              setRemote((r) => ({
                ...r,
                status: 'playing',
                legalCards: null,
                legalBids: null,
              }));
            }
            break;
          }
          case 'your_turn':
            setRemote((r) => ({
              ...r,
              status: 'your_turn',
              legalCards: msg.legalCards ? new Set(msg.legalCards) : null,
              legalBids: msg.legalBids || null,
            }));
            break;
          case 'waiting':
            setRemote((r) => ({
              ...r,
              status: 'playing',
              legalCards: null,
              legalBids: null,
            }));
            break;
          case 'hand_over': {
            sentShowdownRef.current = null;
            setRemote((r) => ({ ...r, status: 'finished', legalCards: null, legalBids: null }));
            // Apply score to game state
            setGame((g) => ({ ...g, phase: 'finished', score: msg.score,
              tricksWon: msg.tricksWon || g.tricksWon, showdown: null }));
            break;
          }
          case 'error':
            setRemote((r) => ({ ...r, error: msg.message }));
            break;
        }
      };

      ws.onclose = () => {
        setRemote((r) => ({ ...r, status: r.status === 'finished' ? 'finished' : 'idle',
          error: r.status !== 'finished' ? '连接已断开' : '' }));
        wsRef.current = null;
      };

      ws.onerror = () => {
        setRemote((r) => ({ ...r, error: '无法连接到服务器' }));
      };
    } catch (err) {
      setRemote((r) => ({ ...r, status: 'idle', error: String(err) }));
    }
  };

  const disconnectRemote = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    sentShowdownRef.current = null;
    setRemote({
      status: 'idle', error: '', mySeat: -1, opponentSeat: -1,
      legalCards: null, legalBids: null, serverUrl: 'localhost:8765',
      roomCode: '', seed: '',
    });
    setScreen('menu');
    setMode('single');
  };

  const handleBid = async (bid) => {
    // Remote mode: send via WebSocket
    if (mode === 'remote') {
      if (busy || game.showdown || game.phase !== 'bidding' || remote.status !== 'your_turn') return;
      setBusy(true);
      try {
        wsRef.current?.send(JSON.stringify({ type: 'bid', bid }));
        setRemote((r) => ({ ...r, status: 'playing', legalBids: null }));
      } finally {
        setBusy(false);
      }
      return;
    }
    // Local mode
    if (busy || game.showdown || game.phase !== 'bidding' || game.currentPlayer !== game.humanSeat) return;
    setBusy(true);
    try {
      setGame(await submitHumanBid(game, bid, setGame));
    } finally {
      setBusy(false);
    }
  };

  const handlePlay = async (cardCode) => {
    // Remote mode: send via WebSocket
    if (mode === 'remote') {
      if (busy || game.showdown || game.phase !== 'playing' || remote.status !== 'your_turn') return;
      setBusy(true);
      try {
        wsRef.current?.send(JSON.stringify({ type: 'play', card: cardCode }));
        setRemote((r) => ({ ...r, status: 'playing', legalCards: null }));
      } finally {
        setBusy(false);
      }
      return;
    }
    // Local mode
    if (busy || game.showdown || game.phase !== 'playing' || game.currentPlayer !== game.humanSeat) return;
    setBusy(true);
    try {
      setGame(await submitHumanCard(game, cardCode, setGame));
    } finally {
      setBusy(false);
    }
  };

  const handleShowdownConfirm = () => {
    if (busy || !game.showdown || game.showdown.status !== 'pending') return;
    if (mode === 'remote') {
      const showdownId = game.showdown.id;
      if (
        sentShowdownRef.current === showdownId
        || showdownWaitingForPartner(game.showdown, game.humanSeat)
      ) return;
      sentShowdownRef.current = showdownId;
      wsRef.current?.send(JSON.stringify({ type: 'showdown_confirm', showdownId }));
      setGame((current) => {
        if (current.showdown?.id !== showdownId) return current;
        const confirmedSeats = new Set(current.showdown.confirmedSeats || []);
        confirmedSeats.add(current.humanSeat);
        return {
          ...current,
          showdown: {
            ...current.showdown,
            locallyConfirmed: true,
            confirmedSeats: [...confirmedSeats],
          },
        };
      });
      return;
    }
    setBusy(true);
    try {
      const settled = confirmLocalShowdown(game);
      setGame(settled);
      if (mode === 'aiTest') {
        setReplaySnapshot(buildReplaySnapshot(settled));
        setScreen('replay');
      }
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

  const isSpectator = mode === 'aiTest';
  const isRemote = mode === 'remote';
  const showdownPending = game.showdown?.status === 'pending';
  const revealedHands = showdownHandsForDisplay(game);
  const waitingForPartner = isRemote
    && (
      sentShowdownRef.current === game.showdown?.id
      || showdownWaitingForPartner(game.showdown, game.humanSeat)
    );

  const legalCards = isRemote
    ? []
    : (game.phase === 'playing' ? getHumanLegalCards(game) : []);
  const localLegalSet = useMemo(() => new Set(legalCards.map((c) => c.code)), [legalCards]);
  const legalSet = isRemote ? (remote.legalCards || new Set()) : localLegalSet;

  // seat → screen position (you are always at the bottom)
  const posOf = (seat) => ['bottom', 'left', 'top', 'right'][(seat - game.humanSeat + 4) % 4];
  const seatAt = (pos) => [0, 1, 2, 3].find((s) => posOf(s) === pos);

  const humanHand = (game.hands[game.humanSeat] || []).filter(Boolean);
  const myTurn = isRemote
    ? (!showdownPending && remote.status === 'your_turn')
    : (!showdownPending && !isSpectator && game.currentPlayer === game.humanSeat);
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
  else if (showdownPending) statusText = '结果已固定，等待确认结算';
  else if (isRemote && remote.status === 'connecting') statusText = '连接中…';
  else if (isRemote && remote.status === 'joined') statusText = '已加入，等待对手…';
  else if (isRemote && remote.status === 'waiting') statusText = '等待对手加入…';
  else if (isRemote && remote.error) statusText = remote.error;
  else if (isRemote && myTurn && isBidding) statusText = '请叫牌';
  else if (isRemote && myTurn && isPlaying) statusText = '请出牌';
  else if (isRemote && isBidding) statusText = `${SEAT_NAMES[game.currentPlayer]} 叫牌中…`;
  else if (isRemote && isPlaying) statusText = `${SEAT_NAMES[game.currentPlayer]} 出牌中…`;
  else if (isSpectator && busy) statusText = 'AI 对局中…';
  else if (busy && !myTurn) statusText = '对手出牌中…';
  else if (isBidding && myTurn) statusText = '请叫牌';
  else if (isBidding) statusText = `${SEAT_NAMES[game.currentPlayer]} 叫牌中…`;
  else if (isPlaying && myTurn) statusText = '请出牌';
  else if (isPlaying) statusText = `${SEAT_NAMES[game.currentPlayer]} 出牌中…`;

  const spread = Math.min(7, 56 / Math.max(1, humanHand.length)); // deg between cards

  const startRemoteGame = (serverUrl, roomCode, seed, seat) => {
    setMode('remote');
    setMatchScore({ ns: 0, ew: 0 });
    setHandNo(1);
    setMatchOver(false);
    settledSeedRef.current = null;
    setHumanSeat(seat);
    // Placeholder state — the server will send authoritative hands via
    // game_state shortly. We must NOT use createInitialGame here because
    // its JS PRNG produces different hands from Python's Mersenne Twister.
    setGame({
      seed,
      humanSeat: seat,
      firstSeat: 0,
      phase: 'bidding',
      currentPlayer: -1,
      leader: -1,
      trickNumber: 1,
      spadesBroken: false,
      hands: [[], [], [], []].map(() => new Array(13)),  // 13 back-cards each
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
      log: [{ kind: 'system', text: '连接中…' }],
    });
    setScreen('game');
    connectRemote(serverUrl, roomCode, seed, seat);
  };

  // ── mode-select screen ──
  if (screen === 'menu') {
    return (
      <div className="felt felt--menu">
        <ModeMenu
          onPick={(m) => { void startMatch(m); }}
          onFixedSeedStart={(seed, seat) => { void startFixedSeedMatch(seed, seat); }}
          onAiTestStart={(seed, viewSeat) => { void startAiTest(seed, viewSeat); }}
          onRemoteStart={(serverUrl, roomCode, seed, seat) => {
            startRemoteGame(serverUrl, roomCode, seed, seat);
          }}
          urlSeed={urlSeed}
        />
      </div>
    );
  }

  if (screen === 'replay' && replaySnapshot) {
    return (
      <ReplayScreen
        snapshot={replaySnapshot}
        viewLabel={mode === 'aiTest' ? '(视角)' : '(You)'}
        onExit={() => setScreen(mode === 'aiTest' ? 'menu' : 'game')}
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
          <button className="brand__back" onClick={() => { if (isRemote) disconnectRemote(); else setScreen('menu'); }} disabled={busy && !isRemote} title={isRemote ? '断开连接' : '返回模式选择'}>←</button>
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
          {isRemote ? (
            <>
              <div className="match-info"><span>房间</span><strong>{remote.roomCode || '—'}</strong></div>
              <div className="match-info"><span>种子</span><strong>{remote.seed || '—'}</strong></div>
              <div className="match-info"><span>你的座位</span><strong>{remote.mySeat >= 0 ? `${remote.mySeat} · ${SEAT_NAMES[remote.mySeat]}` : '—'}</strong></div>
              <div className="match-info"><span>搭档</span><strong>{remote.opponentSeat >= 0 ? `${remote.opponentSeat} · ${SEAT_NAMES[remote.opponentSeat]}` : (remote.status === 'waiting' || remote.status === 'joined' ? '等待加入…' : '—')}</strong></div>
              <button className="ctl__disc" onClick={disconnectRemote} title="断开连接">断开</button>
            </>
          ) : (
            <>
              {mode === 'match500' ? (
                <div className="match-info"><span>局数</span><strong>第 {handNo} 局</strong></div>
              ) : null}
              <div className="match-info"><span>种子</span><strong>{game.seed}</strong></div>
              {!isSpectator ? (
                <label className="ctl">
                  <span>座位</span>
                  <select value={humanSeat} onChange={(e) => setHumanSeat(Number(e.target.value))} disabled={busy || (screen === 'game' && !finished)}>
                    {SEAT_NAMES.map((label, seat) => <option key={label} value={seat}>{seat} · {label}</option>)}
                  </select>
                </label>
              ) : null}
            </>
          )}
        </div>
      </header>

      {/* ── table ── */}
      <main className="stage">
        <div className="stage__top">
          <AiSeat pos="top" seat={seatAt('top')} summary={summary} game={game}
                  active={!showdownPending && game.currentPlayer === seatAt('top')}
                  revealedCards={revealedHands?.[seatAt('top')] ?? null} />
        </div>
        <div className="stage__left">
          <AiSeat pos="left" seat={seatAt('left')} summary={summary} game={game}
                  active={!showdownPending && game.currentPlayer === seatAt('left')}
                  revealedCards={revealedHands?.[seatAt('left')] ?? null} />
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
                  active={!showdownPending && game.currentPlayer === seatAt('right')}
                  revealedCards={revealedHands?.[seatAt('right')] ?? null} />
        </div>

        {/* ── human area ── */}
        <div className="stage__hand">
          {isSpectator ? (
            <AiSeat
              pos="bottom"
              seat={game.humanSeat}
              summary={summary}
              game={game}
              active={!showdownPending && game.currentPlayer === game.humanSeat}
              revealedCards={revealedHands?.[game.humanSeat] ?? null}
            />
          ) : (
            <>
              <div className={`me ${myTurn && !finished ? 'is-active' : ''} team-${myTeam}`}>
                <span className="me__avatar">{SEAT_NAMES[game.humanSeat][0]}</span>
                <div className="me__meta">
                  <strong>{SEAT_NAMES[game.humanSeat]} <em>(You)</em></strong>
                  <TallyBadges bid={game.bids[game.humanSeat]} won={summary.tricksWon[game.humanSeat]}
                               hideBid={isBidding && !game.bids[game.humanSeat]} />
                </div>
              </div>

              {isBidding && myTurn ? (
                isRemote && remote.legalBids ? (
                  <div className="bidbar">
                    {remote.legalBids.map((b) => (
                      <button key={`${b.value}-${b.type}`}
                        className={`chip${b.type === 'nil' ? ' chip--nil' : ''}`}
                        disabled={busy}
                        onClick={() => handleBid(makeBid(b.value, b.type))}>
                        {b.type === 'nil' ? 'Nil' : b.value}
                      </button>
                    ))}
                  </div>
                ) : !isRemote ? (
                  <div className="bidbar">
                    <button className="chip chip--nil" disabled={busy} onClick={() => handleBid(makeBid(0, 'nil'))}>Nil</button>
                    {Array.from({ length: 13 }, (_, i) => i + 1).map((b) => (
                      <button key={b} className="chip" disabled={busy} onClick={() => handleBid(makeBid(b))}>{b}</button>
                    ))}
                  </div>
                ) : null
              ) : null}

              {showdownPending ? (
                <ReplayHandSpread cards={humanHand} pos="bottom" size="lg" />
              ) : (
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
              )}
            </>
          )}
        </div>
      </main>

      {/* ── compact log ── */}
      <aside className="mini-log">
        {game.log.slice(-5).map((entry, i) => (
          <div key={`${entry.kind}-${i}`} className={`mini-log__row log-${entry.kind}`}>{entry.text}</div>
        ))}
      </aside>

      {showdownPending ? (
        <ShowdownPanel
          showdown={game.showdown}
          bids={game.bids}
          waitingForPartner={waitingForPartner}
          onConfirm={handleShowdownConfirm}
        />
      ) : null}

      {/* ── result overlays ── */}
      {finished && summary.score ? (
        isRemote ? (
          <ResultOverlay
            eyebrow="牌局结束"
            subtitle={`房间 ${remote.roomCode} · 种子 ${remote.seed}`}
            ns={summary.score.northSouth}
            ew={summary.score.eastWest}
            verdict={teamWon(summary.score.northSouth, summary.score.eastWest) ? '你的队伍获胜 🎉' : '你的队伍落败'}
            buttonLabel="断开并返回"
            onButton={disconnectRemote}
            busy={busy}
          />
        ) : mode === 'fixedSeed' ? (
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
