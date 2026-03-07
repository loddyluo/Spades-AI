#!/usr/bin/env python3
"""Automated Spades evaluation: watch DRQN AI play full games with detailed output.

Runs multiple rounds of DRQN (Team 0) vs DRQN (Team 1), printing every bid
and every card played so you can judge the model's strength.

Usage:
    python watch_spades_ai.py [--checkpoint PATH] [--num-rounds 5] [--seed 42]
"""

import argparse
import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import rlcard
import torch
from rlcard.utils.agent_utils import load_agent_from_checkpoint
from rlcard.agents.random_agent import RandomAgent

# ──────────────────────── Display helpers ────────────────────────

SUIT_SYMBOLS = {'S': '\u2660', 'H': '\u2665', 'D': '\u2666', 'C': '\u2663'}
RANK_NAMES = {
    '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
    '8': '8', '9': '9', 'T': '10', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A',
}
PLAYER_NAMES = {0: 'South(T0)', 1: 'West (T1)', 2: 'North(T0)', 3: 'East (T1)'}


def card_str(c):
    return f"{SUIT_SYMBOLS.get(c[0], c[0])}{RANK_NAMES.get(c[1], c[1])}"


def sort_hand(hand):
    suit_order = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
    rank_order = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    return sorted(hand, key=lambda c: (suit_order.get(c[0], 9), rank_order.index(c[1]) if c[1] in rank_order else 99))


def rank_value(rank):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    return ranks.index(rank) if rank in ranks else -1


def trick_winner(trick_order):
    if len(trick_order) < 4:
        return None
    lead_suit = trick_order[0][1][0]
    best_trump = None
    best_lead = None
    for pid, c in trick_order:
        suit, rank = c[0], c[1]
        val = rank_value(rank)
        if suit == 'S':
            if best_trump is None or val > best_trump[1]:
                best_trump = (pid, val)
        elif suit == lead_suit:
            if best_lead is None or val > best_lead[1]:
                best_lead = (pid, val)
    return best_trump[0] if best_trump else best_lead[0]


# ──────────────────────── Main ────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Watch DRQN AI play Spades')
    p.add_argument('--checkpoint', type=str,
                   default='../experiments/spades_selfplay_drqn/checkpoint_drqn.pt')
    p.add_argument('--num-rounds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--vs-random', action='store_true',
                   help='Team 1 uses RandomAgent instead of DRQN')
    p.add_argument('--verbose', action='store_true', default=True,
                   help='Print every card played (default: on)')
    p.add_argument('--quiet', action='store_true',
                   help='Only show round summaries, not individual tricks')
    return p.parse_args()


def run_one_round(env, agents, round_num, verbose=True):
    """Play one full round and return (team0_score, team1_score, detail_dict)."""
    state, player_id = env.reset()

    # Reset LSTM hidden states
    for ag in agents:
        if hasattr(ag, 'reset_hidden_states'):
            ag.reset_hidden_states()

    print(f'\n{"="*64}')
    print(f'  ROUND {round_num}')
    print(f'{"="*64}')

    # Show initial hands
    if verbose:
        print('\n  Dealt hands:')
        for p in range(4):
            hand = sort_hand([c.get_index() for c in env.game.players[p].hand])
            hand_str = '  '.join(card_str(c) for c in hand)
            print(f'    {PLAYER_NAMES[p]}: {hand_str}')

    # ── Play through all actions ──
    trick_order = []
    trick_cards = [None] * 4
    trick_id = 0
    bidding_done = False

    while not env.is_over():
        raw_obs = state.get('raw_obs', {})
        phase = raw_obs.get('phase', 0)

        action, info = agents[player_id].eval_step(state)
        action_name = env._decode_action(action)
        name = PLAYER_NAMES[player_id]

        if phase == 0:
            # Bidding
            if action_name == 'pass':
                if verbose:
                    print(f'  {name}: pass')
            elif action_name == 'blind_nil':
                print(f'  {name}: ** BLIND NIL **')
            elif action_name == 'nil':
                print(f'  {name}: Nil')
            elif action_name.startswith('bid_'):
                n = action_name.split('_')[1]
                if verbose:
                    print(f'  {name}: bids {n}')
        else:
            if not bidding_done:
                bidding_done = True
                # Print bid summary
                bids = raw_obs.get('bids', [])
                is_nil = raw_obs.get('is_nil', [False]*4)
                is_blind = raw_obs.get('is_blind_nil', [False]*4)
                print('\n  ── Bids ──')
                for p in range(4):
                    if is_blind[p]:
                        b = 'Blind Nil'
                    elif is_nil[p]:
                        b = 'Nil'
                    else:
                        b = str(bids[p])
                    print(f'    {PLAYER_NAMES[p]}: {b}')

                t0_bid = sum(bids[p] for p in [0, 2] if not is_nil[p] and not is_blind[p])
                t1_bid = sum(bids[p] for p in [1, 3] if not is_nil[p] and not is_blind[p])
                print(f'    Team 0 total bid: {t0_bid}  |  Team 1 total bid: {t1_bid}')
                print()

            # Card play
            if action_name in env.card2id:
                if len(trick_order) == 0:
                    trick_cards = [None] * 4
                    if verbose:
                        print(f'  ── Trick #{trick_id+1} ──')
                trick_cards[player_id] = action_name
                trick_order.append((player_id, action_name))

                if verbose:
                    lead_mark = ' (lead)' if len(trick_order) == 1 else ''
                    print(f'    {name}: {card_str(action_name)}{lead_mark}')

        state, player_id = env.step(action)

        # Trick complete?
        if len(trick_order) == 4:
            winner = trick_winner(trick_order)
            winner_name = PLAYER_NAMES[winner]
            if verbose:
                print(f'    --> Winner: {winner_name}')
            trick_id += 1
            trick_order = []
            trick_cards = [None] * 4

    # ── Round result ──
    scores = env.game.judger.judge_game(env.game.players)
    team0_score = scores[0]
    team1_score = scores[1]

    bids_final = [p.bid for p in env.game.players]
    tricks_final = [p.tricks for p in env.game.players]
    is_nil_final = [p.is_nil for p in env.game.players]
    is_blind_final = [p.is_blind_nil for p in env.game.players]

    print(f'\n  ── Round {round_num} Result ──')
    print(f'  {"Player":<14s} {"Bid":>5s}  {"Tricks":>6s}')
    print(f'  {"─"*30}')
    for p in range(4):
        if is_blind_final[p]:
            b = 'BNil'
            result = 'OK' if tricks_final[p] == 0 else 'FAIL'
        elif is_nil_final[p]:
            b = 'Nil'
            result = 'OK' if tricks_final[p] == 0 else 'FAIL'
        else:
            b = str(bids_final[p])
            result = ''
        print(f'  {PLAYER_NAMES[p]:<14s} {b:>5s}  {tricks_final[p]:>4d}  {result}')

    t0_tricks = tricks_final[0] + tricks_final[2]
    t1_tricks = tricks_final[1] + tricks_final[3]
    t0_bid = sum(bids_final[p] for p in [0, 2] if not is_nil_final[p] and not is_blind_final[p])
    t1_bid = sum(bids_final[p] for p in [1, 3] if not is_nil_final[p] and not is_blind_final[p])
    t0_over = max(0, t0_tricks - t0_bid)
    t1_over = max(0, t1_tricks - t1_bid)

    print(f'\n  Team 0: bid {t0_bid}, got {t0_tricks} tricks, overtricks {t0_over}  -->  {team0_score:+d} pts')
    print(f'  Team 1: bid {t1_bid}, got {t1_tricks} tricks, overtricks {t1_over}  -->  {team1_score:+d} pts')

    detail = {
        'bids': bids_final,
        'tricks': tricks_final,
        'team0_bid': t0_bid, 'team1_bid': t1_bid,
        'team0_tricks': t0_tricks, 'team1_tricks': t1_tricks,
        'team0_score': team0_score, 'team1_score': team1_score,
    }
    return team0_score, team1_score, detail


def main():
    args = parse_args()
    verbose = not args.quiet

    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.abspath(os.path.join(ROOT_DIR, ckpt_path))
    if not os.path.exists(ckpt_path):
        print(f'Error: checkpoint not found: {ckpt_path}')
        sys.exit(1)

    env_config = {
        'allow_raw_data': True,
        'game_enable_blind_nil': True,
        'seed': args.seed,
    }
    env = rlcard.make('spades', config=env_config)

    print(f'Loading DRQN model from: {ckpt_path}')
    device = torch.device('cpu')
    agent, agent_type = load_agent_from_checkpoint(ckpt_path, device=device, env=env)
    print(f'Agent type: {agent_type}')

    if args.vs_random:
        print('Mode: DRQN (Team 0) vs Random (Team 1)')
        random_agent = RandomAgent(num_actions=env.num_actions)
        if isinstance(agent, list):
            agents = [agent[0], random_agent, agent[2], random_agent]
        else:
            agents = [agent, random_agent, agent, random_agent]
    else:
        print('Mode: DRQN (Team 0) vs DRQN (Team 1)  (mirror match)')
        if isinstance(agent, list):
            agents = agent
        else:
            agents = [agent, agent, agent, agent]

    env.set_agents(agents)

    # ── Run rounds ──
    total_t0 = 0
    total_t1 = 0
    t0_wins = 0
    t1_wins = 0
    ties = 0
    all_details = []

    for r in range(1, args.num_rounds + 1):
        s0, s1, detail = run_one_round(env, agents, r, verbose=verbose)
        total_t0 += s0
        total_t1 += s1
        if s0 > s1:
            t0_wins += 1
        elif s1 > s0:
            t1_wins += 1
        else:
            ties += 1
        all_details.append(detail)

    # ── Final summary ──
    print(f'\n{"="*64}')
    print(f'  FINAL SUMMARY  ({args.num_rounds} rounds)')
    print(f'{"="*64}')

    print(f'\n  Team 0 total: {total_t0:+d}  ({t0_wins}W / {args.num_rounds - t0_wins - ties}L / {ties}T)')
    print(f'  Team 1 total: {total_t1:+d}  ({t1_wins}W / {args.num_rounds - t1_wins - ties}L / {ties}T)')
    print(f'  Avg score per round:  Team 0 = {total_t0/args.num_rounds:+.1f}  |  Team 1 = {total_t1/args.num_rounds:+.1f}')

    # Bid accuracy analysis
    total_accurate = 0
    total_underbid = 0
    total_overbid = 0
    total_rounds = 0
    for d in all_details:
        for team_bid, team_tricks in [(d['team0_bid'], d['team0_tricks']),
                                       (d['team1_bid'], d['team1_tricks'])]:
            if team_bid > 0:
                total_rounds += 1
                if team_tricks == team_bid:
                    total_accurate += 1
                elif team_tricks > team_bid:
                    total_underbid += 1
                else:
                    total_overbid += 1

    if total_rounds > 0:
        print(f'\n  Bid accuracy analysis ({total_rounds} team-rounds):')
        print(f'    Exact:    {total_accurate:>3d} ({100*total_accurate/total_rounds:.0f}%)')
        print(f'    Underbid: {total_underbid:>3d} ({100*total_underbid/total_rounds:.0f}%) -- overtricks')
        print(f'    Overbid:  {total_overbid:>3d} ({100*total_overbid/total_rounds:.0f}%) -- set (failed)')

    avg_overtricks = []
    for d in all_details:
        for tb, tt in [(d['team0_bid'], d['team0_tricks']), (d['team1_bid'], d['team1_tricks'])]:
            if tb > 0 and tt >= tb:
                avg_overtricks.append(tt - tb)
    if avg_overtricks:
        print(f'    Avg overtricks (when making bid): {np.mean(avg_overtricks):.2f}')

    print()


if __name__ == '__main__':
    main()
