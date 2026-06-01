/*
 * Spades table UI with one human seat and Python-backed AI seats.
 */
import { useEffect, useMemo, useState } from 'react';
import {
  advanceUntilHuman,
  bidLabel,
  cardLabel,
  createInitialGame,
  getHumanLegalCards,
  makeBid,
  submitHumanCard,
  submitHumanBid,
  summarizeGame,
} from './game';

const SEAT_NAMES = ['North', 'East', 'South', 'West'];

function SeatBadge({ seat, title, subtitle, active, count }) {
  return (
    <div className={`seat-badge seat-${seat} ${active ? 'active' : ''}`}>
      <div className="seat-badge__title">
        <span>{title}</span>
        <strong>P{seat}</strong>
      </div>
      <div className="seat-badge__meta">{subtitle}</div>
      <div className="seat-badge__count">{count} cards</div>
    </div>
  );
}

function TrickCard({ entry }) {
  return (
    <div className={`trick-card suit-${entry.card.suit}`}>
      <span>P{entry.seat}</span>
      <strong>{cardLabel(entry.card)}</strong>
    </div>
  );
}

function HandCard({ card, legal, disabled, onPlay }) {
  return (
    <button
      className={`hand-card suit-${card.suit} ${legal ? 'legal' : 'locked'}`}
      disabled={disabled || !legal}
      onClick={() => legal && !disabled && onPlay(card.code)}
    >
      <span>{card.rank}</span>
      <strong>{cardLabel(card)}</strong>
    </button>
  );
}

function ActionPanel({ game, onBid, onPlay, onNewGame, seedValue, setSeedValue, humanSeat, setHumanSeat, busy }) {
  const summary = summarizeGame(game);
  const legalCards = useMemo(() => getHumanLegalCards(game), [game]);

  return (
    <section className="panel action-panel">
      <div className="panel__header">
        <div>
          <p className="eyebrow">当前决策</p>
          <h2>{summary.phase === 'bidding' ? '请选择叫牌' : summary.phase === 'playing' ? '请选择出牌' : '牌局已结束'}</h2>
        </div>
        <button className="ghost" onClick={onNewGame} disabled={busy}>重新发牌</button>
      </div>

      <div className="control-row">
        <label>
          Seed
          <input value={seedValue} onChange={(event) => setSeedValue(event.target.value)} disabled={busy} />
        </label>
        <label>
          人类座位
          <select value={humanSeat} onChange={(event) => setHumanSeat(Number(event.target.value))} disabled={busy}>
            {SEAT_NAMES.map((label, seat) => <option key={label} value={seat}>{seat} - {label}</option>)}
          </select>
        </label>
      </div>

      <div className="summary-grid">
        <div><span>阶段</span><strong>{summary.phase}</strong></div>
        <div><span>当前玩家</span><strong>P{summary.currentPlayer}</strong></div>
        <div><span>墩数</span><strong>{summary.trickNumber}</strong></div>
        <div><span>黑桃已断</span><strong>{summary.spadesBroken ? '是' : '否'}</strong></div>
      </div>

      <div className="bid-strip">
        <button className="bid-chip special" onClick={() => onBid(makeBid(0, 'nil'))} disabled={busy || summary.phase !== 'bidding'}>Nil</button>
        {Array.from({ length: 13 }, (_, index) => index + 1).map((bid) => (
          <button key={bid} className="bid-chip" onClick={() => onBid(makeBid(bid))} disabled={busy || summary.phase !== 'bidding'}>
            {bid}
          </button>
        ))}
      </div>

      {summary.phase === 'playing' && (
        <div>
          <p className="hint">当前可出牌只显示合法牌。点击即可落子。{busy ? ' 后端 AI 正在计算下一步。' : ''}</p>
          <div className="hand-grid compact">
            {legalCards.map((card) => (
              <HandCard key={card.code} card={card} legal disabled={busy} onPlay={onPlay} />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

export default function App() {
  const [seedValue, setSeedValue] = useState('42');
  const [humanSeat, setHumanSeat] = useState(0);
  const [busy, setBusy] = useState(false);
  const [game, setGame] = useState(() => createInitialGame(42, 0));

  const startNewGame = async (seed, seat) => {
    setBusy(true);
    try {
      setGame(await advanceUntilHuman(createInitialGame(seed, seat)));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    void startNewGame(42, 0);
  }, []);

  const newGame = () => {
    const parsedSeed = Number.parseInt(seedValue, 10);
    const nextSeed = Number.isFinite(parsedSeed) ? parsedSeed : 42;
    void startNewGame(nextSeed, humanSeat);
  };

  const handleBid = async (bid) => {
    if (busy || game.phase !== 'bidding' || game.currentPlayer !== humanSeat) {
      return;
    }
    setBusy(true);
    try {
      setGame(await submitHumanBid(game, bid));
    } finally {
      setBusy(false);
    }
  };

  const handlePlay = async (cardCode) => {
    if (busy || game.phase !== 'playing' || game.currentPlayer !== humanSeat) {
      return;
    }
    setBusy(true);
    try {
      setGame(await submitHumanCard(game, cardCode));
    } finally {
      setBusy(false);
    }
  };

  const currentHumanHand = game.hands[humanSeat];
  const legalCards = game.phase === 'playing' ? getHumanLegalCards(game) : [];
  const legalSet = new Set(legalCards.map((card) => card.code));
  const summary = summarizeGame(game);

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Spades AI Table</p>
          <h1>最小可运行的手动对局界面</h1>
          <p className="hero__copy">前端通过 `/api/choose-action` 调用 Python 后端，由后端 AI 自己编码当前剩余手牌与全部公开历史，再返回叫牌或出牌结果。</p>
        </div>
        <div className="hero__stats">
          <div>
            <span>NS</span>
            <strong>{summary.score ? summary.score.northSouth : '—'}</strong>
          </div>
          <div>
            <span>EW</span>
            <strong>{summary.score ? summary.score.eastWest : '—'}</strong>
          </div>
        </div>
      </header>

      <section className="table-stage panel">
        <SeatBadge seat={0} title="North" subtitle={`Bid: ${bidLabel(game.bids[0])}`} active={game.currentPlayer === 0} count={game.hands[0].length} />
        <SeatBadge seat={3} title="West" subtitle={`Bid: ${bidLabel(game.bids[3])}`} active={game.currentPlayer === 3} count={game.hands[3].length} />

        <div className="table-core">
          <div className="table-core__glow" />
          <h2>当前墩</h2>
          <p>{game.currentTrick.length === 0 ? '尚未出牌' : `已出 ${game.currentTrick.length} 张`}</p>
          <div className="trick-grid">
            {game.currentTrick.map((entry) => <TrickCard key={`${entry.seat}-${entry.card.code}`} entry={entry} />)}
          </div>
          <div className="trick-summary">
            <span>叫牌顺序：{game.bids.map((bid, seat) => `P${seat}:${bidLabel(bid)}`).join('  ')}</span>
            <span>已赢墩：{game.tricksWon.map((count, seat) => `P${seat}:${count}`).join('  ')}</span>
          </div>
        </div>

        <SeatBadge seat={1} title="East" subtitle={`Bid: ${bidLabel(game.bids[1])}`} active={game.currentPlayer === 1} count={game.hands[1].length} />
        <SeatBadge seat={2} title="South" subtitle={`Bid: ${bidLabel(game.bids[2])}`} active={game.currentPlayer === 2} count={game.hands[2].length} />
      </section>

      <section className="bottom-grid">
        <ActionPanel
          game={game}
          onBid={handleBid}
          onPlay={handlePlay}
          onNewGame={newGame}
          seedValue={seedValue}
          setSeedValue={setSeedValue}
          humanSeat={humanSeat}
          setHumanSeat={setHumanSeat}
          busy={busy}
        />

        <section className="panel hand-panel">
          <div className="panel__header">
            <div>
              <p className="eyebrow">我的手牌</p>
              <h2>P{humanSeat} - {SEAT_NAMES[humanSeat]}</h2>
            </div>
            <span className="pill">{currentHumanHand.length} 张</span>
          </div>
          <div className="hand-grid">
            {currentHumanHand.map((card) => (
              <HandCard
                key={card.code}
                card={card}
                legal={game.phase === 'playing' && legalSet.has(card.code) && game.currentPlayer === humanSeat}
                disabled={busy}
                onPlay={handlePlay}
              />
            ))}
          </div>
        </section>

        <section className="panel log-panel">
          <div className="panel__header">
            <div>
              <p className="eyebrow">动作日志</p>
              <h2>最近事件</h2>
            </div>
          </div>
          <div className="log-list">
            {game.log.slice(-12).map((entry, index) => (
              <div key={`${entry.kind}-${index}`} className={`log-item log-${entry.kind}`}>
                <span>{entry.kind}</span>
                <p>{entry.text}</p>
              </div>
            ))}
          </div>
        </section>
      </section>
    </main>
  );
}
