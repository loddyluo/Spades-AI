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
  createInitialGame,
  getHumanLegalCards,
  makeBid,
  submitHumanBid,
  submitHumanCard,
  summarizeGame,
} from './game';

const SEAT_NAMES = ['North', 'East', 'South', 'West'];
const SUIT_SYMBOL = { S: '♠', H: '♥', D: '♦', C: '♣' };
const SUIT_IS_RED = { S: false, H: true, D: true, C: false };
const TARGET_SCORE = 500;

// team(0) = seats 0 & 2, team(1) = seats 1 & 3
const teamOf = (seat) => seat % 2;

// Deterministic seed for reproducible deals.
const DEFAULT_SEED = 123;
const seedFromUrl = () => {
  const raw = new URLSearchParams(window.location.search).get('seed');
  const parsed = raw ? Number.parseInt(raw, 10) : NaN;
  return Number.isFinite(parsed) ? parsed : DEFAULT_SEED;
};
const fixedSeed = () => seedFromUrl();

/* ── A single rendered playing card (face up or face down) ──────────── */
function PlayingCard({ card, faceDown = false, size = 'md', legal = false,
                       disabled = false, onPlay = null, style = null,
                       className = '' }) {
  if (faceDown || !card) {
    return <div className={`pcard pcard--back size-${size} ${className}`} style={style} />;
  }
  const red = SUIT_IS_RED[card.suit];
  const sym = SUIT_SYMBOL[card.suit];
  const clickable = legal && !disabled && onPlay;
  return (
    <button
      type="button"
      className={`pcard size-${size} ${red ? 'is-red' : 'is-black'} ${legal ? 'is-legal' : ''} ${className}`}
      style={style}
      disabled={!clickable}
      onClick={clickable ? () => onPlay(card.code) : undefined}
      aria-label={`${card.rank}${card.suit}`}
    >
      <span className="pcard__corner pcard__corner--tl">
        <b>{card.rank}</b><i>{sym}</i>
      </span>
      <span className="pcard__pip">{sym}</span>
      <span className="pcard__corner pcard__corner--br">
        <b>{card.rank}</b><i>{sym}</i>
      </span>
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
      <PlayingCard card={entry.card} size="md" className={cls} />
    </div>
  );
}

/* ── Mode-select screen ─────────────────────────────────────────────── */
function ModeMenu({ onPick }) {
  return (
    <div className="menu">
      <div className="menu__brand"><span className="brand__pip">♠</span> Spades AI</div>
      <p className="menu__sub">选择对战模式</p>
      <div className="menu__cards">
        <button className="mode-card" onClick={() => onPick('single')}>
          <span className="mode-card__icon">🃏</span>
          <strong>一局制</strong>
          <span className="mode-card__desc">打完一局即结算，看本局胜负。</span>
        </button>
        <button className="mode-card mode-card--gold" onClick={() => onPick('match500')}>
          <span className="mode-card__icon">🏆</span>
          <strong>500 分赛</strong>
          <span className="mode-card__desc">逐局累计，先到 500 分且领先者获胜。</span>
        </button>
      </div>
    </div>
  );
}

/* ── main app ──────────────────────────────────────────────────────── */
export default function App() {
  const [screen, setScreen] = useState('menu');     // 'menu' | 'game'
  const [mode, setMode] = useState('single');        // 'single' | 'match500'
  const [humanSeat, setHumanSeat] = useState(0);
  const [busy, setBusy] = useState(false);
  const [game, setGame] = useState(() => createInitialGame(fixedSeed(), 0));

  // 500-match cumulative state
  const [matchScore, setMatchScore] = useState({ ns: 0, ew: 0 });
  const [handNo, setHandNo] = useState(1);
  const [matchOver, setMatchOver] = useState(false);
  const settledSeedRef = useRef(null);   // guards against double-counting a hand

  // Deal a fresh random hand; AI turns stream in via onStep=setGame.
  const dealHand = async (seat) => {
    setBusy(true);
    try {
      const fresh = createInitialGame(fixedSeed(), seat);
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
    settledSeedRef.current = null;
    setScreen('game');
    await dealHand(seat);
  };

  // Next hand within a running 500-match (keeps cumulative score).
  const nextHand = async () => {
    setHandNo((n) => n + 1);
    await dealHand(humanSeat);
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
    return <div className="felt felt--menu"><ModeMenu onPick={(m) => { void startMatch(m); }} /></div>;
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
          {mode === 'match500' ? (
            <div className="match-info"><span>500 分赛</span><strong>第 {handNo} 局</strong></div>
          ) : (
            <div className="match-info"><span>模式</span><strong>一局制</strong></div>
          )}
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
        mode === 'single' ? (
          <ResultOverlay
            eyebrow="牌局结束"
            ns={summary.score.northSouth}
            ew={summary.score.eastWest}
            verdict={teamWon(summary.score.northSouth, summary.score.eastWest) ? '你的队伍获胜 🎉' : '你的队伍落败'}
            buttonLabel="再来一局"
            onButton={() => { void dealHand(humanSeat); }}
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
            busy={busy}
          />
        )
      ) : null}
    </div>
  );
}

/* ── Reusable result overlay ────────────────────────────────────────── */
function ResultOverlay({ eyebrow, subtitle, ns, ew, cumulative, verdict, buttonLabel, onButton, busy }) {
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
        <button className="btn-new" onClick={onButton} disabled={busy}>{buttonLabel}</button>
      </div>
    </div>
  );
}
