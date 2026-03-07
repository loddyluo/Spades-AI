#!/usr/bin/env python3
"""Command-line interactive Spades game: Human + AI partner vs AI team.

Usage:
    python play_spades_cli.py [--checkpoint PATH] [--human-player 0] [--seed 42]

You (human) play as one player. Your partner and both opponents are controlled
by the DRQN model loaded from the checkpoint.

Team 0: Player 0 (South) & Player 2 (North)
Team 1: Player 1 (West)  & Player 3 (East)
"""

import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import rlcard
import torch
from rlcard.utils.agent_utils import load_agent_from_checkpoint

# ──────────────────────── Display helpers ────────────────────────

SUIT_SYMBOLS = {'S': '\u2660', 'H': '\u2665', 'D': '\u2666', 'C': '\u2663'}
SUIT_NAMES   = {'S': 'Spades', 'H': 'Hearts', 'D': 'Diamonds', 'C': 'Clubs'}
RANK_NAMES   = {
    '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
    '8': '8', '9': '9', 'T': '10', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A',
}
POSITION_NAMES = {0: 'South (You)', 1: 'West', 2: 'North (Partner)', 3: 'East'}


def card_display(card_str):
    """Convert 'SA' -> '♠A', 'HT' -> '♥10'"""
    suit = card_str[0]
    rank = card_str[1]
    return f"{SUIT_SYMBOLS.get(suit, suit)}{RANK_NAMES.get(rank, rank)}"


def sort_hand(hand):
    """Sort hand by suit (S, H, D, C) then rank (2..A)."""
    suit_order = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
    rank_order = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    return sorted(hand, key=lambda c: (suit_order.get(c[0], 9), rank_order.index(c[1]) if c[1] in rank_order else 99))


def print_separator():
    print('─' * 60)


def print_header(text):
    print()
    print_separator()
    print(f'  {text}')
    print_separator()


# ──────────────────────── Game state display ────────────────────────

def display_game_state(raw_obs, human_player, current_trick_cards, trick_id):
    """Print the current game state in a readable format."""
    phase = raw_obs.get('phase', 0)
    bids = raw_obs.get('bids', [None]*4)
    tricks_won = raw_obs.get('tricks_won', [0]*4)
    spades_broken = raw_obs.get('spades_broken', False)
    hand = raw_obs.get('hand', [])

    # Bids & tricks summary
    print()
    print('  Player        Bid    Tricks')
    print('  ─────────────────────────────')
    for p in range(4):
        name = POSITION_NAMES.get(p, f'P{p}')
        bid_str = str(bids[p]) if bids[p] is not None else '?'
        # Check nil/blind nil
        is_nil = raw_obs.get('is_nil', [False]*4)
        is_blind_nil = raw_obs.get('is_blind_nil', [False]*4)
        if is_blind_nil[p]:
            bid_str = 'Blind Nil'
        elif is_nil[p]:
            bid_str = 'Nil'
        marker = ' <--' if p == raw_obs.get('current_player') else ''
        print(f'  {name:<16s} {bid_str:<10s} {tricks_won[p]}{marker}')

    team0_bid = sum(b for i, b in enumerate(bids) if i in (0, 2) and b is not None and not raw_obs.get('is_nil', [False]*4)[i] and not raw_obs.get('is_blind_nil', [False]*4)[i])
    team1_bid = sum(b for i, b in enumerate(bids) if i in (1, 3) and b is not None and not raw_obs.get('is_nil', [False]*4)[i] and not raw_obs.get('is_blind_nil', [False]*4)[i])
    team0_tricks = tricks_won[0] + tricks_won[2]
    team1_tricks = tricks_won[1] + tricks_won[3]

    if phase == 1:
        print()
        print(f'  Team 0 (You): bid {team0_bid}, tricks {team0_tricks}')
        print(f'  Team 1 (Opp): bid {team1_bid}, tricks {team1_tricks}')
        print(f'  Spades broken: {"Yes" if spades_broken else "No"}')
        print(f'  Trick #{trick_id + 1}/13')

    # Current trick
    if current_trick_cards and any(c is not None for c in current_trick_cards):
        print()
        print('  Current trick:')
        for p in range(4):
            if current_trick_cards[p]:
                name = POSITION_NAMES.get(p, f'P{p}')
                print(f'    {name}: {card_display(current_trick_cards[p])}')

    # Hand
    if hand:
        sorted_hand = sort_hand(hand)
        print()
        # Group by suit
        suits_in_hand = {}
        for c in sorted_hand:
            s = c[0]
            suits_in_hand.setdefault(s, []).append(c)

        print('  Your hand:')
        for s in ['S', 'H', 'D', 'C']:
            if s in suits_in_hand:
                cards_str = '  '.join(card_display(c) for c in suits_in_hand[s])
                print(f'    {SUIT_SYMBOLS[s]} {cards_str}')


def display_trick_result(trick_cards, winner, trick_id):
    """Display the completed trick result."""
    print()
    print(f'  --- Trick #{trick_id + 1} result ---')
    for p in range(4):
        if trick_cards[p]:
            name = POSITION_NAMES.get(p, f'P{p}')
            marker = ' ** WINS **' if p == winner else ''
            print(f'    {name}: {card_display(trick_cards[p])}{marker}')


# ──────────────────────── Trick tracking ────────────────────────

def spades_rank_value(rank):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    return ranks.index(rank) if rank in ranks else -1


def compute_trick_winner(trick_order):
    """trick_order: list of (player_id, card_str) in play order."""
    if not trick_order or len(trick_order) < 4:
        return None
    lead_suit = trick_order[0][1][0]
    highest_trump = None
    highest_lead = None
    for idx, (pid, card_str) in enumerate(trick_order):
        suit = card_str[0]
        rank = card_str[1]
        val = spades_rank_value(rank)
        if suit == 'S':
            if highest_trump is None or val > highest_trump[1]:
                highest_trump = (pid, val)
        elif suit == lead_suit:
            if highest_lead is None or val > highest_lead[1]:
                highest_lead = (pid, val)
    if highest_trump:
        return highest_trump[0]
    if highest_lead:
        return highest_lead[0]
    return trick_order[0][0]


# ──────────────────────── Main game loop ────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Play Spades against DRQN AI in the terminal')
    parser.add_argument('--checkpoint', type=str,
                        default='../experiments/spades_selfplay_drqn/checkpoint_drqn.pt',
                        help='Path to AI checkpoint file')
    parser.add_argument('--human-player', type=int, default=0,
                        help='Player seat for human (0=South, 1=West, 2=North, 3=East)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    return parser.parse_args()


def get_human_bid_action(legal_actions, env, raw_obs, human_player):
    """Prompt human for a bidding action."""
    # Decode the legal actions to readable form
    action_map = {}
    display_list = []
    for action_id in legal_actions:
        action_str = env._decode_action(action_id)
        action_map[action_str] = action_id
        if action_str == 'pass':
            display_list.append(('pass', action_id, 'Pass (decline blind nil)'))
        elif action_str == 'blind_nil':
            display_list.append(('blind_nil', action_id, 'Blind Nil'))
        elif action_str == 'nil':
            display_list.append(('nil', action_id, 'Nil (bid 0)'))
        elif action_str.startswith('bid_'):
            n = action_str.split('_')[1]
            display_list.append((action_str, action_id, f'Bid {n}'))

    print()
    print('  Your bidding options:')
    for i, (_, _, desc) in enumerate(display_list):
        print(f'    [{i}] {desc}')

    while True:
        try:
            choice = input('\n  Enter choice number: ').strip()
            idx = int(choice)
            if 0 <= idx < len(display_list):
                return display_list[idx][1]
            print('  Invalid choice, try again.')
        except (ValueError, EOFError):
            print('  Invalid input, enter a number.')


def get_human_play_action(legal_actions, env, raw_obs, human_player):
    """Prompt human for a card play action."""
    # Decode the legal card actions
    card_actions = []
    for action_id in legal_actions:
        action_str = env._decode_action(action_id)
        card_actions.append((action_str, action_id))

    # Sort by suit then rank
    card_actions = sorted(card_actions, key=lambda x: (
        {'S': 0, 'H': 1, 'D': 2, 'C': 3}.get(x[0][0], 9),
        spades_rank_value(x[0][1]) if len(x[0]) > 1 else -1
    ))

    print()
    print('  Playable cards:')
    for i, (card_str, _) in enumerate(card_actions):
        print(f'    [{i}] {card_display(card_str)}')

    while True:
        try:
            choice = input('\n  Enter choice number: ').strip()
            idx = int(choice)
            if 0 <= idx < len(card_actions):
                return card_actions[idx][1]
            print('  Invalid choice, try again.')
        except (ValueError, EOFError):
            print('  Invalid input, enter a number.')


def main():
    args = parse_args()

    # Resolve checkpoint path
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.abspath(os.path.join(ROOT_DIR, ckpt_path))
    if not os.path.exists(ckpt_path):
        print(f'Error: checkpoint not found: {ckpt_path}')
        sys.exit(1)

    human_player = args.human_player
    POSITION_NAMES[human_player] = POSITION_NAMES.get(human_player, f'P{human_player}').replace('You', 'You')
    # Update position names based on human player
    for p in range(4):
        base = {0: 'South', 1: 'West', 2: 'North', 3: 'East'}[p]
        if p == human_player:
            POSITION_NAMES[p] = f'{base} (You)'
        elif (p % 2) == (human_player % 2):
            POSITION_NAMES[p] = f'{base} (Partner)'
        else:
            POSITION_NAMES[p] = f'{base} (Opp)'

    # Create environment
    env_config = {
        'allow_raw_data': True,
        'game_enable_blind_nil': True,
    }
    if args.seed is not None:
        env_config['seed'] = args.seed
    env = rlcard.make('spades', config=env_config)

    # Load AI agent
    print(f'Loading AI model from: {ckpt_path}')
    device = torch.device('cpu')
    agent, agent_type = load_agent_from_checkpoint(ckpt_path, device=device, env=env)
    print(f'Agent type: {agent_type}')

    # Set agents — all seats use the same model, human seat is just
    # intercepted in the loop below.
    if isinstance(agent, list):
        agents = agent
    else:
        agents = [agent for _ in range(4)]
    env.set_agents(agents)

    # ──── Game loop ────
    print_header('SPADES  -  Human vs DRQN AI')
    print(f'  You are {POSITION_NAMES[human_player]}')
    human_team = 0 if human_player in (0, 2) else 1
    print(f'  Your team: Team {human_team}')
    print()

    while True:
        # Reset for new round
        state, player_id = env.reset()

        # Reset LSTM hidden states for all agents
        for ag in agents:
            if hasattr(ag, 'reset_hidden_states'):
                ag.reset_hidden_states()

        trick_cards = [None] * 4
        trick_order = []
        trick_id = 0

        print_header('New Round')

        game_over = False
        while not env.is_over():
            raw_obs = state.get('raw_obs', {})
            phase = raw_obs.get('phase', 0)
            legal_actions = list(state.get('legal_actions', {}).keys())

            if player_id == human_player:
                # Human's turn
                display_game_state(raw_obs, human_player, trick_cards, trick_id)

                if phase == 0:
                    action = get_human_bid_action(legal_actions, env, raw_obs, human_player)
                else:
                    action = get_human_play_action(legal_actions, env, raw_obs, human_player)

                action_str = env._decode_action(action)
            else:
                # AI's turn
                action, info = agents[player_id].eval_step(state)
                action_str = env._decode_action(action)
                name = POSITION_NAMES.get(player_id, f'P{player_id}')

                if phase == 0:
                    # Show AI bid
                    if action_str == 'pass':
                        print(f'  {name} passes (blind nil decision)')
                    elif action_str == 'blind_nil':
                        print(f'  {name} bids Blind Nil!')
                    elif action_str == 'nil':
                        print(f'  {name} bids Nil')
                    elif action_str.startswith('bid_'):
                        n = action_str.split('_')[1]
                        print(f'  {name} bids {n}')
                else:
                    # Show AI card play
                    print(f'  {name} plays {card_display(action_str)}')

            # Track trick cards
            if phase == 1 and action_str in env.card2id:
                if len(trick_order) == 0:
                    trick_cards = [None] * 4
                trick_cards[player_id] = action_str
                trick_order.append((player_id, action_str))

            # Step the environment
            state, player_id = env.step(action)

            # Check if trick is complete
            if len(trick_order) == 4:
                winner = compute_trick_winner(trick_order)
                display_trick_result(trick_cards, winner, trick_id)
                trick_id += 1
                trick_order = []
                trick_cards = [None] * 4
                if not env.is_over():
                    print_separator()

        # Round over — show results
        payoffs = env.get_payoffs()
        scores = env.game.judger.judge_game(env.game.players)
        team0_score = scores[0]
        team1_score = scores[1]

        bids = [p.bid for p in env.game.players]
        tricks = [p.tricks for p in env.game.players]
        is_nil = [p.is_nil for p in env.game.players]
        is_blind_nil = [p.is_blind_nil for p in env.game.players]

        print_header('Round Result')
        print('  Player           Bid     Tricks')
        print('  ──────────────────────────────────')
        for p in range(4):
            name = POSITION_NAMES.get(p, f'P{p}')
            if is_blind_nil[p]:
                bid_str = 'BNil'
            elif is_nil[p]:
                bid_str = 'Nil'
            else:
                bid_str = str(bids[p])
            print(f'  {name:<18s} {bid_str:<8s} {tricks[p]}')

        print()
        print(f'  Team 0 ({"You" if human_team == 0 else "Opp"}): {team0_score:+d} points')
        print(f'  Team 1 ({"You" if human_team == 1 else "Opp"}): {team1_score:+d} points')

        your_score = team0_score if human_team == 0 else team1_score
        opp_score = team1_score if human_team == 0 else team0_score
        if your_score > opp_score:
            print('\n  ** You win this round! **')
        elif your_score < opp_score:
            print('\n  ** You lose this round. **')
        else:
            print('\n  ** Tie! **')

        # Play again?
        print()
        again = input('  Play another round? [Y/n] ').strip().lower()
        if again in ('n', 'no', 'q', 'quit', 'exit'):
            print('\n  Thanks for playing!')
            break


if __name__ == '__main__':
    main()
