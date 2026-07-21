import assert from 'node:assert/strict';
import test from 'node:test';

import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

import { ShowdownPanel, showdownHandsForDisplay } from './showdown.js';


const FIXED_SHOWDOWN = {
  status: 'pending',
  resolution: {
    teamTricks: [4, 9],
    nilOutcomes: [true, null, false, null],
    finalTricksWon: [0, 5, 4, 4],
    continuation: [],
  },
};

const FIXED_BIDS = [
  { value: 0, type: 'nil' },
  { value: 1, type: 'normal' },
  { value: 0, type: 'blind_nil' },
  { value: 1, type: 'normal' },
];


test('showdown panel displays projected team and Nil outcomes before settlement', () => {
  const html = renderToStaticMarkup(React.createElement(ShowdownPanel, {
    showdown: FIXED_SHOWDOWN,
    bids: FIXED_BIDS,
    waitingForPartner: false,
    onConfirm: () => {},
  }));

  assert.match(html, /自动摊牌/);
  assert.match(html, /NS[^0-9]*4/);
  assert.match(html, /EW[^0-9]*9/);
  assert.match(html, /North Nil[^<]*成功/);
  assert.match(html, /South Blind Nil[^<]*失败/);
  assert.match(html, /确认结算/);
});


test('a separately confirmed remote player sees the partner barrier', () => {
  const html = renderToStaticMarkup(React.createElement(ShowdownPanel, {
    showdown: FIXED_SHOWDOWN,
    bids: FIXED_BIDS,
    waitingForPartner: true,
    onConfirm: () => {},
  }));

  assert.match(html, /等待搭档确认/);
  assert.match(html, /disabled/);
});


test('all real hands are exposed only while a showdown is pending', () => {
  const hands = [[{ code: 'AS' }], [{ code: 'KH' }], [{ code: 'QD' }], [{ code: 'JC' }]];

  const shown = showdownHandsForDisplay({ hands, showdown: FIXED_SHOWDOWN });

  assert.deepEqual(shown, hands);
  assert.notEqual(shown, hands);
  assert.equal(showdownHandsForDisplay({ hands, showdown: null }), null);
  assert.equal(showdownHandsForDisplay({ hands, showdown: { status: 'done' } }), null);
});
