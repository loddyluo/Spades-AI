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
  REGRET_EPSILON,
  advanceUntilFinished,
  advanceUntilHuman,
  bidLabel,
  buildRegretRecord,
  buildReplayRecord,
  buildReplaySnapshot,
  confirmLocalShowdown,
  createInitialGame,
  fetchRegretJob,
  formatRegret,
  getHumanLegalCards,
  makeBid,
  parseReplayImport,
  regretBadges,
  regretDecisionAt,
  regretSummary,
  remoteStateFromServer,
  showdownWaitingForPartner,
  sortHandByRegret,
  startRegretAnalysis,
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
  importReplay: '导入复盘',
  fullReplay: '完整复盘',
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
                       className = '', static: isStatic = false, badge = null }) {
  if (faceDown || !card) {
    return <div className={`pcard pcard--back size-${size} ${className}`} style={style} />;
  }
  const sym = SUIT_SYMBOL[card.suit];
  // 完整复盘会在牌角打一个「期望遗憾」角标；对局/普通回放不传 badge，
  // 渲染结果与之前完全一致。
  const chip = badge ? (
    <span className={`pcard__badge ${badge.tone ?? ''}`} title={badge.title ?? undefined}>
      {badge.text}
    </span>
  ) : null;
  const body = (
    <>
      <span className="pcard__corner pcard__corner--tl">
        <b>{card.rank}</b><i>{sym}</i>
      </span>
      <span className="pcard__pip">{sym}</span>
      <span className="pcard__corner pcard__corner--br">
        <b>{card.rank}</b><i>{sym}</i>
      </span>
      {chip}
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
function TrickSlot({ pos, entry, justPlayed, collecting, winnerPos, badge = null }) {
  if (!entry) return <div className={`slot slot--${pos}`} />;
  const cls = [
    'slot__card',
    justPlayed ? `slot__card--in-${pos}` : '',
    collecting ? `slot__card--collect-${winnerPos}` : '',
  ].join(' ');
  return (
    <div className={`slot slot--${pos}`}>
      <PlayingCard card={entry.card} size="md" static className={cls} badge={badge} />
    </div>
  );
}

/* ── Face-up hand spread for replay (all seats show their cards) ────── */
function ReplayHandSpread({ cards, pos, size = 'sm', highlightCode = null, badges = null,
                            flat = false }) {
  // 完整复盘要读每张牌角上的遗憾数字，重叠扇形会挡掉它们，所以走平铺模式：
  // 牌按普通文档流一张挨一张排，放不下就换行，永不重叠，也永不遮挡角标。
  if (flat) {
    return (
      <div className={`replay-hand replay-hand--${pos} is-flat`}>
        {cards.map((card) => (
          <PlayingCard
            key={card.code}
            card={card}
            size={size}
            static
            className={`replay-hand__card ${highlightCode === card.code ? 'is-highlight' : ''}`}
            badge={badges ? badges[card.code] ?? null : null}
          />
        ))}
      </div>
    );
  }
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
            badge={badges ? badges[card.code] ?? null : null}
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

function ReplaySeatPanel({ seat, snapshot, tricksWon, pos, cards, highlightCode, badges = null, flat = false, isViewSeat = false, viewLabel = '(You)' }) {
  const seatName = snapshot.seatNames?.[seat] ?? SEAT_NAMES[seat];
  return (
    <div className={`replay-seat replay-seat--${pos} ${flat ? 'is-flat' : ''}`}>
      <div className={`replay-seat__label team-${teamOf(seat)}`}>
        <span className="replay-seat__avatar">{seatName[0]}</span>
        <div className="replay-seat__meta">
          <strong>{seatName}{isViewSeat ? <> <em>{viewLabel}</em></> : null}</strong>
          <TallyBadges bid={snapshot.bids[seat]} won={tricksWon[seat]} />
        </div>
      </div>
      <ReplayHandSpread
        cards={cards}
        pos={pos}
        size={pos === 'bottom' ? 'lg' : 'sm'}
        highlightCode={highlightCode}
        badges={badges}
        flat={flat}
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
  const replaySeatNames = snapshot.seatNames ?? SEAT_NAMES;

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

  const exportRecord = () => {
    const json = `${JSON.stringify(buildReplayRecord(snapshot), null, 2)}\n`;
    const blob = new Blob([json], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `spades-replay-seed-${snapshot.seed}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  let statusText = '四家手牌已摊开，点击「下一步」开始复盘';
  if (phase === 'done') {
    statusText = '复盘结束';
  } else if (trickComplete) {
    statusText = `第 ${activeTrick} 墩由 ${replaySeatNames[trickWinner]} 赢下 · 点击下一步收墩`;
  } else if (lastPlay) {
    statusText = `${replaySeatNames[lastPlay.seat]} 出 ${lastPlay.card.rank}${SUIT_SYMBOL[lastPlay.card.suit]}`;
  } else if (nextPlay) {
    statusText = `下一步：${replaySeatNames[nextPlay.seat]} 出牌`;
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
        <button className="btn-ghost" onClick={exportRecord}>导出记录</button>
        <button className="btn-ghost" onClick={onExit}>{viewLabel === '(视角)' ? '返回菜单' : '返回结算'}</button>
      </footer>
    </div>
  );
}

/* ── 完整复盘：把每个动作的期望遗憾标在牌上 + 侧栏明细 ────────────── */
/** Turn one regret number into the corner chip shown on a card. */
function regretChip(regret, { blocked = false, title = '' } = {}) {
  if (blocked) {
    return { text: '×', tone: 'is-blocked', title: title || '本墩不能出这张牌' };
  }
  if (regret == null) {
    return { text: '—', tone: 'is-unknown', title: title || '这一步没有算出 Q 值' };
  }
  if (regret < 1e-9) {
    return { text: '0', tone: 'is-best', title: title || 'AI 眼中的最优动作' };
  }
  const tone = regret >= 30 ? 'is-blunder' : regret >= 8 ? 'is-loss' : 'is-small-loss';
  return { text: formatRegret(regret, regret >= 10 ? 0 : 1), tone, title: title || `期望遗憾 ${formatRegret(regret, 2)} 分` };
}

/** Chips for every card still in the acting seat's hand. */
function handChips(hand, decision) {
  if (!decision) return null;
  const info = regretBadges(decision);
  const chips = {};
  for (const card of hand) {
    const entry = info[card.code];
    chips[card.code] = entry
      ? regretChip(entry.regret, { title: entry.best ? 'AI 眼中的最优动作（遗憾 0）' : undefined })
      : regretChip(null, { blocked: true });
  }
  return chips;
}

/** The chip for one card already on the table, from its own decision. */
function playedChip(decision, cardCode) {
  if (!decision) return null;
  const info = regretBadges(decision)[cardCode];
  if (!info) return null;
  return regretChip(info.regret, {
    title: `实际出牌 · 期望遗憾 ${formatRegret(info.regret, 2)} 分`,
  });
}

function RegretDecisionRow({ decision, seatName, selected, onSelect }) {
  const [open, setOpen] = useState(false);
  const regret = decision.forced ? 0 : decision.playedRegret;
  const tone = decision.forced || regret == null
    ? 'is-flat'
    : regret >= 30 ? 'is-blunder'
      : regret >= 8 ? 'is-loss'
        : regret > REGRET_EPSILON ? 'is-small-loss' : 'is-best';
  return (
    <li className={`regret-row ${selected ? 'is-selected' : ''} ${tone}`}>
      <button type="button" className="regret-row__main" onClick={() => onSelect(decision.playIndex)}>
        <span className="regret-row__where">第 {decision.trickNumber} 墩</span>
        <span className={`regret-row__seat team-${teamOf(decision.seat)}`}>{seatName}</span>
        <span className="regret-row__cards">
          {decision.actualCard}
          {!decision.forced && decision.bestCard !== decision.actualCard
            ? <em>→ {decision.bestCard}</em>
            : null}
        </span>
        <span className={`regret-row__value ${tone}`}>
          {decision.forced ? '唯一选择' : decision.degraded ? '未求解' : formatRegret(regret)}
        </span>
      </button>
      {decision.forced ? null : (
        <>
          <button
            type="button"
            className="regret-row__toggle"
            onClick={() => setOpen((value) => !value)}
            aria-expanded={open}
          >
            {open ? '收起明细' : `明细 (${decision.actions.length} 个动作)`}
          </button>
          {open ? (
            <div className="regret-detail">
              <table>
                <thead>
                  <tr><th>动作</th><th>期望 Q</th><th>期望遗憾</th><th>来源</th></tr>
                </thead>
                <tbody>
                  {decision.actions.map((action) => (
                    <tr
                      key={action.card}
                      className={action.card === decision.actualCard ? 'is-actual' : ''}
                    >
                      <td>{action.card}{action.card === decision.actualCard ? ' ·实际' : ''}</td>
                      <td>{action.q == null ? '—' : action.q.toFixed(2)}</td>
                      <td>{formatRegret(action.regret, 2)}</td>
                      <td>{action.source === 'solver' ? '求解器' : action.source}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {decision.proposals?.length ? (
                <details className="regret-detail__proposals">
                  <summary>
                    IS 提案明细（{decision.proposals.length} 份，权重和
                    {' '}{decision.proposals.reduce((sum, p) => sum + p.weight, 0).toFixed(4)}）
                  </summary>
                  <table>
                    <thead>
                      <tr>
                        <th>权重</th>
                        {decision.legalCards.map((code) => <th key={code}>{code}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {decision.proposals.map((proposal, index) => (
                        <tr key={index}>
                          <td>{proposal.weight.toFixed(5)}</td>
                          {decision.legalCards.map((code) => (
                            <td key={code}>
                              {proposal.q[code] == null ? '—' : proposal.q[code].toFixed(1)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </details>
              ) : null}
            </div>
          ) : null}
        </>
      )}
    </li>
  );
}

export function RegretScreen({ snapshot, analysis, onExit, viewLabel = '(视角)' }) {
  // 打开就直接停在后 9 墩的第一个决策点上：前 4 墩没有遗憾可看，
  // 让用户手动点 16 次「下一步」毫无意义。
  const startIndex = analysis?.decisions?.[0]?.playIndex ?? 0;
  const [phase, setPhase] = useState('ready');
  const [playIndex, setPlayIndex] = useState(startIndex);
  const [trickComplete, setTrickComplete] = useState(false);
  const [view, setView] = useState(() => rebuildReplayState(snapshot, startIndex, false));
  const replaySeatNames = snapshot.seatNames ?? SEAT_NAMES;
  const summary = useMemo(() => regretSummary(analysis), [analysis]);

  // 每个遗憾数字都出自某一份超参数配置；尤其是 multiplier_clip_factor ≠ 1 时
  // AI 优化的目标被重新缩放，必须把出处写在旁边而不是靠记忆。
  const provenance = useMemo(() => {
    const config = analysis?.config;
    if (!config?.path) return null;
    const name = String(config.path).split('/').pop();
    const effective = config.effective ?? {};
    const clip = effective.multiplier_clip;
    const factor = effective.multiplier_clip_factor;
    const clipNote = factor === 1
      ? `裁剪 ${clip}×${factor}（无效果）`
      : `裁剪 ${clip}×${factor}（生效：尾部输分被重标定）`;
    return {
      text: `AI：${analysis?.ai ?? '—'} · 超参：${name} · ${clipNote}`,
      title: `${config.path}\nsha256: ${config.sha256 ?? '—'}`,
    };
  }, [analysis]);

  const applyCursor = (index, complete, nextPhase = 'ready') => {
    setPlayIndex(index);
    setTrickComplete(complete);
    setView(rebuildReplayState(snapshot, index, complete));
    setPhase(nextPhase);
  };

  const posOf = (seat) => ['bottom', 'left', 'top', 'right'][(seat - snapshot.humanSeat + 4) % 4];
  const seatAt = (pos) => [0, 1, 2, 3].find((s) => posOf(s) === pos);

  const { remainingHands, currentTrick, trickWinner, lastPlayedSeat } = view;
  const tricksWon = replayTricksWonAt(snapshot, playIndex, trickComplete);

  const nextPlay = playIndex < snapshot.plays.length ? snapshot.plays[playIndex] : null;
  const lastPlay = playIndex > 0 ? snapshot.plays[playIndex - 1] : null;
  const highlightCode = lastPlay?.card.code ?? null;
  // 光标停在墩边界时 `view.activeTrick` 还停在上一墩；这里显示正在看的那一墩。
  const displayTrick = trickComplete
    ? (lastPlay?.trickNumber ?? 1)
    : (nextPlay?.trickNumber ?? lastPlay?.trickNumber ?? 1);

  // The decision being annotated is the one about to be made at this cursor;
  // the card already on the table carries the regret of the decision that made it.
  const currentDecision = regretDecisionAt(analysis, playIndex);
  const playedDecision = playIndex > 0 ? regretDecisionAt(analysis, playIndex - 1) : null;
  const actingSeat = nextPlay?.seat ?? -1;
  const chips = useMemo(
    () => (currentDecision && actingSeat >= 0
      ? handChips(remainingHands[actingSeat] ?? [], currentDecision)
      : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [currentDecision, actingSeat, playIndex, trickComplete],
  );
  const playedBadge = lastPlay ? playedChip(playedDecision, lastPlay.card.code) : null;

  const trickByPos = {};
  for (const entry of currentTrick) trickByPos[posOf(entry.seat)] = entry;

  const justPlayedPos = !trickComplete && lastPlayedSeat >= 0 ? posOf(lastPlayedSeat) : null;
  const winnerPos = trickComplete && trickWinner >= 0 ? posOf(trickWinner) : null;

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
    applyCursor(nextIndex, nextIndex % 4 === 0, 'ready');
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

  const jumpTo = (index) => applyCursor(index, false, 'ready');

  const jumpToNextMistake = () => {
    const candidates = (analysis?.decisions ?? [])
      .filter((decision) => !decision.forced && (decision.playedRegret ?? 0) > 0)
      .sort((a, b) => b.playedRegret - a.playedRegret);
    const next = candidates.find((decision) => decision.playIndex > playIndex) ?? candidates[0];
    if (next) jumpTo(next.playIndex);
  };

  const canStepBack = phase === 'done' || playIndex > 0 || trickComplete;
  const canStepForward = phase !== 'done' && (trickComplete || playIndex < snapshot.plays.length);

  const exportRecord = () => {
    const json = `${JSON.stringify(buildRegretRecord(snapshot, analysis), null, 2)}\n`;
    const blob = new Blob([json], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `spades-regret-seed-${snapshot.seed}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  let statusText = '四家明牌 + 后 9 墩每个动作的期望遗憾';
  if (phase === 'done') {
    statusText = '复盘结束';
  } else if (trickComplete) {
    statusText = `第 ${displayTrick} 墩由 ${replaySeatNames[trickWinner]} 赢下 · 点击下一步收墩`;
  } else if (currentDecision) {
    if (currentDecision.forced) {
      statusText = `第 ${displayTrick} 墩 · ${replaySeatNames[actingSeat]} 只有一张合法牌，遗憾 0`;
    } else if (currentDecision.degraded) {
      statusText = `第 ${displayTrick} 墩 · ${replaySeatNames[actingSeat]} 有多个合法牌，`
        + `但这一步没有走成精确求解（${currentDecision.mode ?? '未知'}），遗憾未知`;
    } else {
      statusText = `第 ${displayTrick} 墩 · ${replaySeatNames[actingSeat]} 实际出 ${currentDecision.actualCard}`
        + `（期望遗憾 ${formatRegret(currentDecision.playedRegret)}）`
        + `· AI 最优 ${currentDecision.bestCard}`;
    }
  } else if (nextPlay) {
    statusText = `下一步：${replaySeatNames[nextPlay.seat]} 出牌（这一手不在后 9 墩分析范围内）`;
  }

  const progress = snapshot.plays.length > 0 ? Math.round((playIndex / snapshot.plays.length) * 100) : 0;

  return (
    <div className="felt felt--replay felt--regret">
      <header className="topbar">
        <div className="brand">
          <button className="brand__back" onClick={onExit} title="返回菜单">←</button>
          <span className="brand__pip">♠</span> 完整复盘
        </div>
        <div className="topbar__right">
          <div className="replay-meta">
            <span>种子 {snapshot.seed}</span>
            <strong>第 {displayTrick} 墩 · {progress}%</strong>
          </div>
        </div>
      </header>

      <main className="stage stage--regret">
        <div className="stage__top">
          <ReplaySeatPanel
            seat={seatAt('top')}
            snapshot={snapshot}
            tricksWon={tricksWon}
            pos="top"
            cards={seatAt('top') === actingSeat ? sortHandByRegret(remainingHands[seatAt('top')], currentDecision) : remainingHands[seatAt('top')]}
            highlightCode={highlightCode}
            badges={seatAt('top') === actingSeat ? chips : null}
            flat
          />
        </div>

        <div className="stage__left">
          <ReplaySeatPanel
            seat={seatAt('left')}
            snapshot={snapshot}
            tricksWon={tricksWon}
            pos="left"
            cards={seatAt('left') === actingSeat ? sortHandByRegret(remainingHands[seatAt('left')], currentDecision) : remainingHands[seatAt('left')]}
            highlightCode={highlightCode}
            badges={seatAt('left') === actingSeat ? chips : null}
            flat
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
                badge={p === justPlayedPos ? playedBadge : null}
              />
            ))}
            <div className="status">
              <span className="status__text">{statusText}</span>
              <span className="status__trick">完整复盘 · 第 {displayTrick} 墩</span>
            </div>
          </div>
        </div>

        <div className="stage__right">
          <ReplaySeatPanel
            seat={seatAt('right')}
            snapshot={snapshot}
            tricksWon={tricksWon}
            pos="right"
            cards={seatAt('right') === actingSeat ? sortHandByRegret(remainingHands[seatAt('right')], currentDecision) : remainingHands[seatAt('right')]}
            highlightCode={highlightCode}
            badges={seatAt('right') === actingSeat ? chips : null}
            flat
          />
        </div>

        <aside className="regret-panel">
          <div className="regret-panel__summary">
            <div>
              <span>后 9 墩总遗憾</span>
              <strong>{formatRegret(summary.totalRegret, 1)}</strong>
            </div>
            <div>
              <span>平均每手</span>
              <strong>{formatRegret(summary.meanRegret, 2)}</strong>
            </div>
            <div>
              <span>已分析动作</span>
              <strong>{summary.analyzedActions}/{summary.totalDecisions}</strong>
            </div>
          </div>
          <div className="regret-panel__seats">
            {summary.perSeat.map((entry) => (
              <span key={entry.seat} className={`team-${teamOf(entry.seat)}`}>
                {replaySeatNames[entry.seat]} {formatRegret(entry.totalRegret, 1)}
              </span>
            ))}
          </div>
          <ol className="regret-panel__list">
            {(analysis?.decisions ?? []).map((decision) => (
              <RegretDecisionRow
                key={decision.playIndex}
                decision={decision}
                seatName={replaySeatNames[decision.seat]}
                selected={decision.playIndex === playIndex}
                onSelect={jumpTo}
              />
            ))}
          </ol>
          {provenance ? (
            <p className="regret-panel__provenance" title={provenance.title}>
              {provenance.text}
            </p>
          ) : null}
          <p className="regret-panel__hint">
            遗憾 = 同一批 IS 提案下，期望 Q 与最优动作之差（0 队分数 − 1 队分数，单位：分）。
            牌角数字为该动作的期望遗憾，手牌按遗憾从小到大排序。
          </p>
        </aside>

        <div className="stage__hand">
          <ReplaySeatPanel
            seat={snapshot.humanSeat}
            snapshot={snapshot}
            tricksWon={tricksWon}
            pos="bottom"
            cards={snapshot.humanSeat === actingSeat
              ? sortHandByRegret(remainingHands[snapshot.humanSeat], currentDecision)
              : remainingHands[snapshot.humanSeat]}
            highlightCode={highlightCode}
            badges={snapshot.humanSeat === actingSeat ? chips : null}
            flat
            isViewSeat
            viewLabel={viewLabel}
          />
        </div>
      </main>

      <footer className="replay-controls">
        <button className="btn-ghost" onClick={resetReplay} disabled={!canStepBack}>重新摊开</button>
        <button className="btn-ghost" onClick={stepBack} disabled={!canStepBack}>上一步</button>
        <button className="btn-new" onClick={stepForward} disabled={!canStepForward}>下一步</button>
        <button className="btn-ghost" onClick={jumpToNextMistake}>下一个失误</button>
        <button className="btn-ghost" onClick={exportRecord}>导出记录 + 分析</button>
        <button className="btn-ghost" onClick={onExit}>返回菜单</button>
      </footer>
    </div>
  );
}

/* ── Mode-select screen ─────────────────────────────────────────────── */
function ModeMenu({ onPick, onFixedSeedStart, onAiTestStart, onReplayStart, onFullReplayStart, onRemoteStart, urlSeed, regret }) {
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
  const [replayOptions, setReplayOptions] = useState([]);
  const [replayIndex, setReplayIndex] = useState(0);
  const [replayViewSeat, setReplayViewSeat] = useState(0);
  const [replayFileName, setReplayFileName] = useState('');
  const [replayError, setReplayError] = useState('');
  const [fullOptions, setFullOptions] = useState([]);
  const [fullIndex, setFullIndex] = useState(0);
  const [fullFileName, setFullFileName] = useState('');
  const [fullError, setFullError] = useState('');

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

  const handleReplayFile = async (event) => {
    const file = event.target.files?.[0];
    setReplayOptions([]);
    setReplayIndex(0);
    setReplayFileName(file?.name ?? '');
    setReplayError('');
    if (!file) return;
    try {
      const text = await file.text();
      let document;
      try {
        document = JSON.parse(text);
      } catch (err) {
        const detail = err instanceof Error ? err.message : String(err);
        throw new Error(`JSON 解析失败：${detail}`);
      }
      const options = parseReplayImport(document);
      setReplayOptions(options);
      setReplayViewSeat(options[0].snapshot.humanSeat);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      setReplayError(`导入失败：${detail}`);
    }
  };

  const handleReplaySelection = (event) => {
    const nextIndex = Number(event.target.value);
    setReplayIndex(nextIndex);
    setReplayViewSeat(replayOptions[nextIndex]?.snapshot.humanSeat ?? 0);
  };

  const handleReplayStart = () => {
    const selected = replayOptions[replayIndex];
    if (!selected) {
      setReplayError('请先选择有效的复盘记录');
      return;
    }
    setReplayError('');
    onReplayStart({ ...selected.snapshot, humanSeat: replayViewSeat });
  };

  const handleFullFile = async (event) => {
    const file = event.target.files?.[0];
    setFullOptions([]);
    setFullIndex(0);
    setFullFileName(file?.name ?? '');
    setFullError('');
    if (!file) return;
    try {
      const text = await file.text();
      let document;
      try {
        document = JSON.parse(text);
      } catch (err) {
        const detail = err instanceof Error ? err.message : String(err);
        throw new Error(`JSON 解析失败：${detail}`);
      }
      setFullOptions(parseReplayImport(document));
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      setFullError(`导入失败：${detail}`);
    }
  };

  const handleFullStart = () => {
    const selected = fullOptions[fullIndex];
    if (!selected) {
      setFullError('请先选择有效的复盘记录');
      return;
    }
    setFullError('');
    // 文件里已经带着算好的遗憾分析时直接复用，不必再花十几分钟重算一遍。
    onFullReplayStart(
      selected.snapshot,
      selected.record ?? buildReplayRecord(selected.snapshot),
      selected.analysis ?? null,
    );
  };

  const fullSelected = fullOptions[fullIndex] ?? null;

  const regretBusy = regret?.status === 'analyzing';
  const regretProgress = regret?.progress ?? { done: 0, total: 0, current: null };

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
        <div className="mode-card mode-card--replay">
          <span className="mode-card__icon">🎞️</span>
          <strong>导入复盘</strong>
          <span className="mode-card__desc">
            导入 GUI 记录、完整复盘导出的文件、DeepSeek 队式赛单局或含多局的完整汇总。
          </span>
          <div className="seed-form">
            <label className="seed-form__field">
              <span>JSON 记录</span>
              <input
                type="file"
                accept=".json,application/json"
                onChange={(event) => { void handleReplayFile(event); }}
              />
            </label>
            {replayFileName ? <p className="seed-form__hint">已读取：{replayFileName}</p> : null}
            {replayOptions.length > 0 ? (
              <>
                <label className="seed-form__field">
                  <span>牌局</span>
                  <select value={replayIndex} onChange={handleReplaySelection}>
                    {replayOptions.map((option, index) => (
                      <option key={`${option.snapshot.seed}-${index}`} value={index}>{option.label}</option>
                    ))}
                  </select>
                </label>
                <label className="seed-form__field">
                  <span>复盘视角</span>
                  <select value={replayViewSeat} onChange={(event) => setReplayViewSeat(Number(event.target.value))}>
                    {(replayOptions[replayIndex]?.snapshot.seatNames ?? SEAT_NAMES).map((label, seatIndex) => (
                      <option key={`${label}-${seatIndex}`} value={seatIndex}>{seatIndex} · {label}</option>
                    ))}
                  </select>
                </label>
                <button type="button" className="btn-new seed-form__go" onClick={handleReplayStart}>打开复盘</button>
              </>
            ) : null}
          </div>
          {replayError ? <p className="seed-form__error">{replayError}</p> : null}
        </div>
        <div className="mode-card mode-card--full-replay">
          <span className="mode-card__icon">🔍</span>
          <strong>完整复盘</strong>
          <span className="mode-card__desc">
            导入同一份记录，用后 9 墩的 AI pipeline 算出每个动作的期望遗憾并标注在牌上；
            也可以读回自己「导出记录 + 分析」的文件，直接复用里面的结果。
          </span>
          <div className="seed-form">
            <label className="seed-form__field">
              <span>JSON 记录</span>
              <input
                type="file"
                accept=".json,application/json"
                onChange={(event) => { void handleFullFile(event); }}
                disabled={regretBusy}
              />
            </label>
            {fullFileName ? <p className="seed-form__hint">已读取：{fullFileName}</p> : null}
            {fullOptions.length > 0 ? (
              <>
                <label className="seed-form__field">
                  <span>牌局</span>
                  <select value={fullIndex} onChange={(event) => setFullIndex(Number(event.target.value))}>
                    {fullOptions.map((option, index) => (
                      <option key={`${option.snapshot.seed}-${index}`} value={index}>{option.label}</option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  className="btn-new seed-form__go"
                  onClick={handleFullStart}
                  disabled={regretBusy}
                >
                  {regretBusy ? '分析中…' : fullSelected?.analysis ? '打开（复用文件内分析）' : '开始完整复盘'}
                </button>
                {fullSelected?.analysis ? (
                  <p className="seed-form__hint">
                    这份文件里已经带着算好的遗憾分析，会直接打开，不再重新求解。
                  </p>
                ) : null}
              </>
            ) : null}
            {regretBusy ? (
              <p className="seed-form__hint">
                正在逐手重算后 9 墩的期望遗憾：{regretProgress.done}/{regretProgress.total || '?'}
                {regretProgress.current
                  ? `（第 ${regretProgress.current.trickNumber} 墩 · 座位 ${regretProgress.current.seat}）`
                  : ''}
                。四家全部动作要跑 30 多轮完整精确求解，视机器性能可能需要十几分钟到一小时，请勿关闭页面。
              </p>
            ) : null}
          </div>
          {fullError ? <p className="seed-form__error">{fullError}</p> : null}
          {regret?.status === 'error' && regret.error ? (
            <p className="seed-form__error">分析失败：{regret.error}</p>
          ) : null}
        </div>
        <div className="mode-card mode-card--remote">
          <span className="mode-card__icon">🌐</span>
          <strong>远程对战</strong>
          <span className="mode-card__desc">两人各在一台电脑，通过网络对战两个 AI，结束后可复盘。</span>
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
  const [mode, setMode] = useState('single');        // 'single' | 'match500' | 'fixedSeed' | 'aiTest' | 'importReplay'
  const [humanSeat, setHumanSeat] = useState(0);
  const [busy, setBusy] = useState(false);
  const [game, setGame] = useState(() => createInitialGame(0, 0));
  const [replaySnapshot, setReplaySnapshot] = useState(null);
  const [aiError, setAiError] = useState('');

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
  const remoteCloseExpectedRef = useRef(false);
  const remoteFinishedRef = useRef(false);

  // ── full-replay (完整复盘) job state ─────────────────────────────────
  // status: 'idle' | 'analyzing' | 'ready' | 'error'
  const [regret, setRegret] = useState({
    status: 'idle',
    jobId: '',
    snapshot: null,
    progress: { done: 0, total: 0, current: null },
    analysis: null,
    error: '',
  });

  const startFullReplay = async (snapshot, record, analysis = null) => {
    if (analysis) {
      // 导出的完整复盘文件自带分析结果，直接进复盘页，不打扰后端。
      setRegret({
        status: 'ready',
        jobId: '',
        snapshot: { ...snapshot },
        progress: { done: 0, total: 0, current: null },
        analysis,
        error: '',
      });
      setScreen('regret');
      return;
    }
    setRegret({
      status: 'analyzing',
      jobId: '',
      snapshot: { ...snapshot },
      progress: { done: 0, total: 0, current: null },
      analysis: null,
      error: '',
    });
    try {
      const jobId = await startRegretAnalysis(record);
      setRegret((prev) => ({ ...prev, jobId }));
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      setRegret((prev) => ({ ...prev, status: 'error', error: detail }));
    }
  };

  // Poll the analysis job until it settles. The backend runs the whole thing
  // in its own process, so this loop only reads a small JSON snapshot.
  useEffect(() => {
    if (regret.status !== 'analyzing' || !regret.jobId) return undefined;
    let cancelled = false;
    let timer = null;

    const tick = async () => {
      try {
        const payload = await fetchRegretJob(regret.jobId);
        if (cancelled) return;
        if (payload.status === 'done') {
          setRegret((prev) => ({
            ...prev,
            status: 'ready',
            analysis: payload.result ?? null,
            progress: payload.progress ?? prev.progress,
          }));
          setScreen('regret');
          return;
        }
        if (payload.status === 'error') {
          setRegret((prev) => ({
            ...prev,
            status: 'error',
            error: payload.error || '分析失败',
          }));
          return;
        }
        setRegret((prev) => ({ ...prev, progress: payload.progress ?? prev.progress }));
        timer = setTimeout(() => { void tick(); }, 2000);
      } catch (err) {
        if (cancelled) return;
        const detail = err instanceof Error ? err.message : String(err);
        setRegret((prev) => ({ ...prev, status: 'error', error: detail }));
      }
    };

    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [regret.status, regret.jobId]);

  const pauseForAiError = (err) => {
    const message = err instanceof Error ? err.message : String(err);
    console.error('AI backend failed; game paused without fallback:', err);
    setAiError(message || '未知错误');
  };

  // Deal a hand. Random modes should omit `seed` or pass randomDealSeed().
  const dealHand = async (seat, seed = randomDealSeed(), firstSeat = 0) => {
    setAiError('');
    setBusy(true);
    try {
      const resolvedSeat = Number.isInteger(seat) ? seat : humanSeat;
      setHumanSeat(resolvedSeat);
      const fresh = createInitialGame(seed, resolvedSeat, firstSeat);
      setGame(fresh);
      setGame(await advanceUntilHuman(fresh, setGame));
    } catch (err) {
      pauseForAiError(err);
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
    setAiError('');
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
    } catch (err) {
      pauseForAiError(err);
    } finally {
      setBusy(false);
    }
  };

  const startImportedReplay = (snapshot) => {
    setMode('importReplay');
    setAiError('');
    setReplaySnapshot(snapshot);
    setScreen('replay');
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
    remoteCloseExpectedRef.current = false;
    remoteFinishedRef.current = false;
    setAiError('');
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
            const remoteGame = remoteStateFromServer(msg, mySeat, seed);
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
            remoteFinishedRef.current = true;
            sentShowdownRef.current = null;
            setRemote((r) => ({ ...r, status: 'finished', legalCards: null, legalBids: null }));
            // Apply score + seed to game state.
            // completedTricks is already correct from the last game_state
            // (remoteStateFromServer parsed card strings into objects), so
            // leave it untouched.
            setGame((g) => ({
              ...g,
              seed: msg.seed ?? g.seed,
              phase: 'finished',
              score: msg.score,
              tricksWon: msg.tricksWon || g.tricksWon,
              showdown: null,
            }));
            break;
          }
          case 'error':
            setRemote((r) => ({ ...r, error: msg.message }));
            if (msg.fatal === true || msg.code === 'ai_fallback') {
              setAiError(msg.message || '远程 AI 触发 fallback');
            }
            break;
        }
      };

      ws.onclose = () => {
        const interrupted = !remoteCloseExpectedRef.current && !remoteFinishedRef.current;
        setRemote((r) => ({ ...r, status: r.status === 'finished' ? 'finished' : 'idle',
          error: r.status !== 'finished' ? '连接已断开' : '' }));
        wsRef.current = null;
        remoteCloseExpectedRef.current = false;
        if (interrupted) {
          setAiError('远程 AI 后端连接已断开，牌局已暂停。');
        }
      };

      ws.onerror = () => {
        setRemote((r) => ({ ...r, error: '无法连接到服务器' }));
        setAiError('无法连接远程 AI 后端，牌局已暂停。');
      };
    } catch (err) {
      setRemote((r) => ({ ...r, status: 'idle', error: String(err) }));
      pauseForAiError(err);
    }
  };

  const disconnectRemote = () => {
    remoteCloseExpectedRef.current = true;
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
    setAiError('');
    setScreen('menu');
    setMode('single');
  };

  const handleBid = async (bid) => {
    // Remote mode: send via WebSocket
    if (mode === 'remote') {
      if (busy || aiError || game.showdown || game.phase !== 'bidding' || remote.status !== 'your_turn') return;
      setBusy(true);
      try {
        if (!wsRef.current) throw new Error('远程连接不可用');
        wsRef.current.send(JSON.stringify({ type: 'bid', bid }));
        setRemote((r) => ({ ...r, status: 'playing', legalBids: null }));
      } catch (err) {
        pauseForAiError(err);
      } finally {
        setBusy(false);
      }
      return;
    }
    // Local mode
    if (busy || aiError || game.showdown || game.phase !== 'bidding' || game.currentPlayer !== game.humanSeat) return;
    setBusy(true);
    try {
      setGame(await submitHumanBid(game, bid, setGame));
    } catch (err) {
      pauseForAiError(err);
    } finally {
      setBusy(false);
    }
  };

  const handlePlay = async (cardCode) => {
    // Remote mode: send via WebSocket
    if (mode === 'remote') {
      if (busy || aiError || game.showdown || game.phase !== 'playing' || remote.status !== 'your_turn') return;
      setBusy(true);
      try {
        if (!wsRef.current) throw new Error('远程连接不可用');
        wsRef.current.send(JSON.stringify({ type: 'play', card: cardCode }));
        setRemote((r) => ({ ...r, status: 'playing', legalCards: null }));
      } catch (err) {
        pauseForAiError(err);
      } finally {
        setBusy(false);
      }
      return;
    }
    // Local mode
    if (busy || aiError || game.showdown || game.phase !== 'playing' || game.currentPlayer !== game.humanSeat) return;
    setBusy(true);
    try {
      setGame(await submitHumanCard(game, cardCode, setGame));
    } catch (err) {
      pauseForAiError(err);
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
      try {
        if (!wsRef.current) throw new Error('远程连接不可用');
        wsRef.current.send(JSON.stringify({ type: 'showdown_confirm', showdownId }));
      } catch (err) {
        sentShowdownRef.current = null;
        pauseForAiError(err);
        return;
      }
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
    setReplaySnapshot(buildReplaySnapshot(game));
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
  const myTurn = aiError
    ? false
    : isRemote
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
  if (aiError) statusText = 'AI 后端错误，本局已暂停';
  else if (finished) statusText = '本局结束';
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
    setReplaySnapshot(null);
    // Placeholder state — the server will send authoritative hands and
    // history via game_state shortly. The shared deal seed is retained so a
    // finished remote hand can be reconstructed by the replay screen.
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
          onReplayStart={startImportedReplay}
          onFullReplayStart={(snapshot, record, analysis) => {
            void startFullReplay(snapshot, record, analysis);
          }}
          onRemoteStart={(serverUrl, roomCode, seed, seat) => {
            startRemoteGame(serverUrl, roomCode, seed, seat);
          }}
          urlSeed={urlSeed}
          regret={regret}
        />
      </div>
    );
  }

  if (screen === 'regret' && regret.snapshot && regret.analysis) {
    return (
      <RegretScreen
        snapshot={regret.snapshot}
        analysis={regret.analysis}
        onExit={() => setScreen('menu')}
      />
    );
  }

  if (screen === 'replay' && replaySnapshot) {
    const replayFromMenu = mode === 'aiTest' || mode === 'importReplay';
    return (
      <ReplayScreen
        snapshot={replaySnapshot}
        viewLabel={replayFromMenu ? '(视角)' : '(You)'}
        onExit={() => setScreen(replayFromMenu ? 'menu' : 'game')}
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
            replayLabel="复盘回放"
            onReplay={enterReplay}
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

      {aiError ? (
        <AiErrorOverlay
          message={aiError}
          seed={game.seed}
          onExit={() => {
            setAiError('');
            if (isRemote) disconnectRemote();
            else setScreen('menu');
          }}
        />
      ) : null}
    </div>
  );
}

/* ── Fatal AI error overlay (local and remote) ───────────────────────── */
function AiErrorOverlay({ message, seed, onExit }) {
  return (
    <div className="overlay overlay--error" role="alertdialog" aria-modal="true" aria-labelledby="ai-error-title">
      <div className="overlay__card ai-error">
        <p className="overlay__eyebrow">AI 后端错误</p>
        <h2 id="ai-error-title" className="ai-error__title">本局已暂停</h2>
        <p className="ai-error__summary">没有使用 fallback 继续叫牌或出牌。</p>
        <div className="ai-error__detail">
          <span>错误详情</span>
          <code>{message}</code>
        </div>
        <p className="overlay__subtitle">种子 {seed}</p>
        <div className="overlay__actions">
          <button className="btn-new" onClick={onExit}>返回菜单</button>
        </div>
      </div>
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
