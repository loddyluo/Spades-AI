/*
 * Render smoke test for 完整复盘 (RegretScreen).
 *
 * The replay/regret logic is covered by node --test unit tests, but nothing
 * there exercises the JSX wiring that actually paints the annotations on the
 * cards.  This script builds a legal 13-trick record in JS, synthesises an
 * analysis for the last nine tricks, server-renders RegretScreen through
 * Vite's SSR pipeline, and asserts the badges / panel really appear.
 *
 * Usage: npm run smoke
 */
import { createServer } from 'vite';
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  actionOutcomeDistribution,
  enumerateFinalOutcomes,
  qSamplesForCard,
} from '../src/game.js';

import {
  applyBid,
  applyCard,
  buildReplayRecord,
  buildReplaySnapshot,
  createDeck,
  createInitialGame,
  createRng,
  finalizeTrick,
  getLegalCards,
  makeBid,
  parseReplayImport,
} from '../src/game.js';

const here = dirname(fileURLToPath(import.meta.url));
const guiRoot = join(here, '..');
const scratch = join(guiRoot, '.regret-smoke');

/** Play one complete legal hand with a deterministic card chooser. */
function buildLegalRecord(seed = 987654321) {
  let state = createInitialGame(seed, 0);
  const rng = createRng(seed);
  for (let seat = 0; seat < 4; seat += 1) {
    state = applyBid(state, (state.firstSeat + seat) % 4, makeBid(4));
  }
  for (let trick = 0; trick < 13; trick += 1) {
    for (let play = 0; play < 4; play += 1) {
      const seat = state.currentPlayer;
      const legal = getLegalCards(state.hands[seat], state.currentTrick, state.spadesBroken);
      if (legal.length === 0) throw new Error(`seat ${seat} has no legal card`);
      const card = legal[Math.floor(rng() * legal.length)];
      state = applyCard(state, seat, card.code);
    }
    state = finalizeTrick(state);
  }
  return buildReplayRecord(buildReplaySnapshot(state));
}

/** Synthetic analysis: every legal card from trick 5 on gets a fake regret. */
function buildAnalysis(record) {
  // getLegalCards speaks card objects, so map the record's codes back to cards.
  const byCode = new Map(createDeck().map((card) => [card.code, card]));
  const hands = record.initialHands.map((hand) => hand.map((code) => byCode.get(code)));
  const plays = record.tricks.flatMap((trick) =>
    trick.plays.map((play) => ({ seat: play.seat, card: play.card, trickNumber: trick.trickNumber })),
  );
  const decisions = [];
  let spadesBroken = false;
  let currentTrick = [];
  plays.forEach((play, index) => {
    const remaining = 52 - index;
    if (remaining <= 36) {
      // Legal cards must be computed against the real in-progress trick,
      // otherwise followers would be credited with cards they cannot play.
      const legal = getLegalCards(hands[play.seat], currentTrick, spadesBroken);
      const forced = legal.length <= 1;
      const actions = legal.map((card, position) => ({
        card: card.code,
        q: forced ? null : 20 - position * 3 - (play.seat % 2),
        regret: forced ? 0 : position * 3,
        source: forced ? 'forced' : 'solver',
      }));
      // Two IS proposals whose Q values are scores a real final standing can
      // actually produce, so the click-through distribution panel has something
      // to invert (an unreachable score would simply be dropped).
      const reachable = [...enumerateFinalOutcomes(bidsFixture).keys()].sort((x, y) => y - x);
      const proposals = forced ? [] : [
        { weight: 0.6, q: Object.fromEntries(legal.map((c, i) => [c.code, reachable[i % reachable.length]])) },
        { weight: 0.4, q: Object.fromEntries(legal.map((c, i) => [c.code, reachable[(i + 3) % reachable.length]])) },
      ];
      decisions.push({
        playIndex: index,
        trickNumber: play.trickNumber,
        seat: play.seat,
        actualCard: play.card,
        forced,
        legalCards: legal.map((card) => card.code),
        chosenCard: legal[0]?.code ?? play.card,
        bestCard: legal[0]?.code ?? play.card,
        playedQ: forced ? null : 20,
        playedRegret: 0,
        actions,
        proposals,
      });
    }
    hands[play.seat] = hands[play.seat].filter((card) => card.code !== play.card);
    currentTrick.push({ seat: play.seat, card: byCode.get(play.card) });
    if (currentTrick.length === 4) currentTrick = [];
    if (play.card.endsWith('S')) spadesBroken = true;
  });
  return {
    ok: true,
    ai: 'solver_leaf_mlp_exact_residual_q_100k',
    config: {
      path: '/repo/configs/8.yaml',
      sha256: '0'.repeat(64),
      effective: { multiplier_clip: 40, multiplier_clip_factor: 1 },
    },
    decisions,
    summary: {
      analyzedActions: decisions.filter((d) => !d.forced).length,
      totalDecisions: decisions.length,
      totalRegret: 12,
      meanRegret: 2,
      perSeat: [0, 1, 2, 3].map((seat) => ({ seat, totalRegret: seat, worstRegret: seat })),
      worst: [],
    },
  };
}

/**
 * The click-through panel is a pure function of the decision, so exercise the
 * real inversion helpers on a real analysed decision and check the numbers.
 */
function panelInvertsACard(analysis) {
  const decision = analysis.decisions.find(
    (entry) => !entry.forced && entry.proposals.length > 0 && entry.actions.length > 0,
  );
  if (!decision) return false;
  const card = decision.actions[0].card;
  const samples = qSamplesForCard(decision, card);
  if (samples.length === 0) return false;
  const distribution = actionOutcomeDistribution({
    bids: bidsFixture,
    seat: decision.seat,
    qSamples: samples,
  });
  const mass = distribution.trickDistribution.reduce((sum, row) => sum + row.probability, 0);
  return distribution.coveredWeight > 0
    && distribution.team === decision.seat % 2
    && Math.abs(mass - 1) < 1e-9
    && distribution.trickDistribution.length === 14
    && Number.isFinite(distribution.meanTricks);
}

/** `.stage--regret` must define a rectangular `panel` area for the side panel. */
function regretGridIsWellFormed() {
  const css = readFileSync(join(guiRoot, 'src', 'styles.css'), 'utf8');
  const block = css.match(/\.stage--regret \{([\s\S]*?)\n\}/);
  if (!block) return false;
  const columns = block[1].match(/grid-template-columns:([\s\S]*?);/);
  const areas = block[1].match(/grid-template-areas:([\s\S]*?);/);
  if (!columns || !areas) return false;

  const columnCount = (columns[1].match(/minmax\(/g) ?? []).length;
  const rows = areas[1].split('"').filter((part) => part.trim());
  if (columnCount === 0 || rows.length === 0) return false;
  if (rows.some((row) => row.trim().split(/\s+/).length !== columnCount)) return false;

  const names = [...new Set(rows.flatMap((row) => row.trim().split(/\s+/)))];
  if (!names.includes('panel')) return false;
  return names.every((name) => {
    const cells = [];
    rows.forEach((row, rowIndex) => {
      row.trim().split(/\s+/).forEach((cell, columnIndex) => {
        if (cell === name) cells.push([rowIndex, columnIndex]);
      });
    });
    const rowSpan = Math.max(...cells.map((c) => c[0])) - Math.min(...cells.map((c) => c[0])) + 1;
    const colSpan = Math.max(...cells.map((c) => c[1])) - Math.min(...cells.map((c) => c[1])) + 1;
    return cells.length === rowSpan * colSpan;
  });
}

/** The flat-hand override must not strip the card's own positioning context. */
function flatCardKeepsItsPositioningContext() {
  const css = readFileSync(join(guiRoot, 'src', 'styles.css'), 'utf8');
  const rule = css.match(/\.replay-hand\.is-flat \.replay-hand__card \{([^}]*)\}/);
  if (!rule) return false;
  const body = rule[1];
  return /position:\s*relative/.test(body) && !/position:\s*static/.test(body);
}

function fail(message) {
  console.error(`[regret-smoke] FAIL: ${message}`);
  process.exitCode = 1;
}

rmSync(scratch, { recursive: true, force: true });
mkdirSync(scratch, { recursive: true });

const bidsFixture = ['bid_3', 'bid_3', 'bid_2', 'bid_3'];
const record = buildLegalRecord();
// The record must survive the very same importer the GUI uses.
const snapshot = parseReplayImport(record)[0].snapshot;
const analysis = buildAnalysis(record);
analysis.bids = bidsFixture;

writeFileSync(join(scratch, 'fixture.json'), JSON.stringify({ record, analysis }));
writeFileSync(
  join(scratch, 'entry.jsx'),
  `import { renderToStaticMarkup } from 'react-dom/server';
import { RegretScreen } from '../src/App';
import { parseReplayImport } from '../src/game';
import fixture from './fixture.json';
const snapshot = parseReplayImport(fixture.record)[0].snapshot;
export const html = renderToStaticMarkup(
  <RegretScreen snapshot={snapshot} analysis={fixture.analysis} onExit={() => {}} />,
);
`,
);

const server = await createServer({
  root: guiRoot,
  logLevel: 'error',
  server: { middlewareMode: true },
  appType: 'custom',
});

try {
  const module = await server.ssrLoadModule('/.regret-smoke/entry.jsx');
  const html = module.html;
  const badges = html.match(/pcard__badge [a-z-]+/g) ?? [];
  const rows = html.match(/class="regret-row /g) ?? [];
  // 完整复盘的硬性要求：四家手牌平铺不重叠，否则角标会被上一张牌盖住。
  const flatHands = (html.match(/replay-hand replay-hand--[a-z]+ is-flat/g) ?? []).length;
  const fannedCards = (html.match(/--rot:/g) ?? []).length;
  const handCards = (html.match(/replay-hand__card/g) ?? []).length;
  const openIndex = analysis.decisions[0].playIndex;
  const selectableCards = (html.match(/is-selectable/g) ?? []).length;
  const expectedCards = 4 * (13 - Math.floor(openIndex / 4));

  if (analysis.decisions.length !== 36) {
    fail(`expected 36 analysed decisions, built ${analysis.decisions.length}`);
  } else if (badges.length === 0) {
    fail('no regret badges were rendered on any card');
  } else if (rows.length !== analysis.decisions.length) {
    fail(`side panel rendered ${rows.length} rows, expected ${analysis.decisions.length}`);
  } else if (flatHands !== 4) {
    fail(`expected 4 flat hands, found ${flatHands}`);
  } else if (fannedCards !== 0) {
    fail(`${fannedCards} cards still carry a fan rotation; they would overlap`);
  } else if (handCards !== expectedCards) {
    fail(`rendered ${handCards} hand cards, expected ${expectedCards}`);
  } else if (!html.includes('regret-panel__summary') || !html.includes('status__text')) {
    fail('side panel summary or status line is missing');
  } else if (selectableCards < expectedCards - 9 && selectableCards === 0) {
    fail(
      'no hand card rendered as a clickable button: the regret screen marks cards '
      + '`static`, so the `onSelect` branch must come before the static early-return',
    );
  } else if (!panelInvertsACard(analysis)) {
    fail('clicking a card must yield a trick-count / Nil probability distribution');
  } else if (!html.includes('regret-panel__provenance')
    || !html.includes('/repo/configs/8.yaml')
    || !html.includes('无效果')) {
    fail('the side panel must state which hyperparameter config produced the numbers');
  } else if (!regretGridIsWellFormed()) {
    fail(
      'the .stage--regret grid is malformed: every grid-template-areas row must '
      + 'have one cell per column and each named area must be rectangular, '
      + 'otherwise `.regret-panel` loses its `panel` area and gets auto-placed',
    );
  } else if (!flatCardKeepsItsPositioningContext()) {
    fail(
      'the flat-hand card rule must keep `position: relative`; with `static` the '
      + 'rank/suit corners, the centre pip and the regret badge (all absolute) '
      + 'escape to the felt and the cards render blank',
    );
  } else {
    console.log(`[regret-smoke] clickable cards rendered: ${selectableCards}`);
    console.log(
      `[regret-smoke] OK — ${analysis.decisions.length} decisions, `
      + `${rows.length} panel rows, ${badges.length} card badges`,
    );
    console.log(
      `[regret-smoke] ${flatHands} flat hands, ${handCards} cards, no fan rotation`,
    );
    console.log(`[regret-smoke] card badge tones: ${[...new Set(badges)].sort().join(', ')}`);
  }
} catch (error) {
  fail(`render threw: ${error?.stack ?? error}`);
} finally {
  await server.close();
  rmSync(scratch, { recursive: true, force: true });
}
