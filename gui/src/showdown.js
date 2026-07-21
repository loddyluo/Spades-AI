import React from 'react';


const SEAT_NAMES = ['North', 'East', 'South', 'West'];


/** Return copied face-up hands only while an offer is awaiting confirmation. */
export function showdownHandsForDisplay(game) {
  if (!game?.showdown || game.showdown.status !== 'pending') return null;
  if (!Array.isArray(game.hands) || game.hands.length !== 4) return null;
  return game.hands.map((hand) => hand.filter(Boolean).map((card) => ({ ...card })));
}


function nilLabel(bid) {
  return bid?.type === 'blind_nil' ? 'Blind Nil' : 'Nil';
}


/** Pure confirmation panel shared by local and remote showdown flows. */
export function ShowdownPanel({ showdown, bids, waitingForPartner = false, onConfirm }) {
  if (!showdown || showdown.status !== 'pending' || !showdown.resolution) return null;
  const { teamTricks, nilOutcomes } = showdown.resolution;
  const nilRows = (nilOutcomes || []).flatMap((success, seat) => {
    if (success == null) return [];
    return [React.createElement(
      'li',
      { key: seat, className: success ? 'showdown__nil-success' : 'showdown__nil-failure' },
      `${SEAT_NAMES[seat]} ${nilLabel(bids?.[seat])}：${success ? '成功' : '失败'}`,
    )];
  });

  return React.createElement(
    'div',
    { className: 'overlay overlay--showdown', role: 'dialog', 'aria-modal': 'true' },
    React.createElement(
      'section',
      { className: 'overlay__card showdown__card' },
      React.createElement('p', { className: 'overlay__eyebrow' }, '自动摊牌'),
      React.createElement('p', { className: 'showdown__explanation' }, '所有合法打法的计分结果均相同。确认前不会结算。'),
      React.createElement(
        'div',
        { className: 'overlay__scores showdown__scores' },
        React.createElement('div', null,
          React.createElement('span', null, 'North / South'),
          React.createElement('strong', null, `NS ${teamTricks?.[0] ?? '—'}`)),
        React.createElement('div', null,
          React.createElement('span', null, 'East / West'),
          React.createElement('strong', null, `EW ${teamTricks?.[1] ?? '—'}`)),
      ),
      nilRows.length > 0
        ? React.createElement('ul', { className: 'showdown__nil-list' }, nilRows)
        : null,
      React.createElement(
        'button',
        {
          type: 'button',
          className: 'btn-new showdown__confirm',
          disabled: waitingForPartner,
          onClick: waitingForPartner ? undefined : onConfirm,
        },
        waitingForPartner ? '等待搭档确认' : '确认结算',
      ),
    ),
  );
}
