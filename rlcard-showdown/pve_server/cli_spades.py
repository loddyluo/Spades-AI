#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.request
import urllib.error
import random
import time

SPADES_SUITS = ['S', 'H', 'D', 'C']
SPADES_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SPADES_DECK = [f"{s}{r}" for s in SPADES_SUITS for r in SPADES_RANKS]
SPADES_DECK_INDEX = {card: idx for idx, card in enumerate(SPADES_DECK)}
RANK_ORDER_DESC = {rank: idx for idx, rank in enumerate(['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'])}
SUIT_ORDER = {suit: idx for idx, suit in enumerate(SPADES_SUITS)}


def action_id_to_label(action_id):
    if 0 <= action_id <= 51:
        return SPADES_DECK[action_id]
    if action_id == 52:
        return 'pass'
    if action_id == 53:
        return 'blind_nil'
    if action_id == 54:
        return 'nil'
    if 55 <= action_id <= 67:
        return f"bid_{action_id - 54}"
    return None


def label_to_action_id(label):
    if label in SPADES_DECK_INDEX:
        return SPADES_DECK_INDEX[label]
    if label == 'pass':
        return 52
    if label == 'blind_nil':
        return 53
    if label == 'nil':
        return 54
    if label.startswith('bid_'):
        try:
            val = int(label.split('_', 1)[1])
            if 1 <= val <= 13:
                return 54 + val
        except ValueError:
            return None
    return None


def http_post_json(url, payload):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode('utf-8'))


def http_get_json(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode('utf-8'))


def print_state(state, human_player):
    obs = state.get('obs', {})
    phase = state.get('phase')
    current_player = state.get('current_player')
    bids = obs.get('bids', [])
    tricks = obs.get('tricks_won', [])
    spades_broken = obs.get('spades_broken')
    current_trick = obs.get('current_trick', [])
    hand = obs.get('hand', [])
    hand_sizes = state.get('hand_sizes', [])

    if hand:
        hand = sorted(
            hand,
            key=lambda c: (
                SUIT_ORDER.get(c[0], 99),
                RANK_ORDER_DESC.get(c[1], 99),
            ),
        )

    print('\n=== Spades PvE ===')
    print(f"Phase: {phase}")
    print(f"Current Player: {current_player}{' (You)' if current_player == human_player else ''}")
    print(f"Bids: {bids}")
    print(f"Tricks: {tricks}")
    print(f"Spades Broken: {'Yes' if spades_broken else 'No'}")

    if current_trick:
        trick_display = [c if c is not None else '_' for c in current_trick]
        print(f"Current Trick: {trick_display}")

    if hand:
        print(f"Your Hand: {' '.join(hand)}")
    else:
        if phase == 'bidding':
            print('Your Hand: (hidden in blind phase)')
        else:
            print('Your Hand: (empty or hidden)')

    if hand_sizes:
        sizes = ', '.join([str(s) for s in hand_sizes])
        print(f"Hand Sizes: [{sizes}]")


def format_legal_actions(legal_actions):
    labels = []
    for action_id in legal_actions:
        label = action_id_to_label(action_id)
        if label:
            labels.append((action_id, label))
    return labels


def choose_action_interactive(legal_actions, phase, hand):
    labels = format_legal_actions(legal_actions)

    if phase == 'bidding':
        print('Legal Actions:')
        for idx, (_, label) in enumerate(labels, start=1):
            print(f"  [{idx}] {label}")
        raw = input('> 输入编号或动作名: ').strip()
        if raw.isdigit():
            pick = int(raw)
            if 1 <= pick <= len(labels):
                return labels[pick - 1][0]
        return label_to_action_id(raw)

    if phase == 'play':
        legal_cards = [label for _, label in labels if len(label) == 2]
        print(f"Legal Cards: {' '.join(legal_cards)}")
        raw = input('> 输入牌面(如 SA) 或编号: ').strip().upper()
        if raw.isdigit():
            pick = int(raw)
            if 1 <= pick <= len(legal_cards):
                return label_to_action_id(legal_cards[pick - 1])
        if raw in hand and raw in legal_cards:
            return label_to_action_id(raw)
        return label_to_action_id(raw)

    return None


def choose_action_auto(legal_actions):
    if not legal_actions:
        return None
    return random.choice(legal_actions)


def main():
    parser = argparse.ArgumentParser(description='Spades PvE Terminal Client')
    parser.add_argument('--server', default='http://127.0.0.1:5001', help='PvE server base url')
    parser.add_argument('--human', type=int, default=0, help='Human player index (0-3)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--auto', type=int, default=0, help='Auto-play N steps (for smoke test)')
    parser.add_argument('--auto-exit', action='store_true', help='Exit after auto steps without prompting')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay between auto steps')
    args = parser.parse_args()

    base = args.server.rstrip('/')
    try:
        state = http_post_json(f"{base}/reset", {
            'game': 'spades',
            'seed': args.seed,
            'human_player': args.human,
        })
    except urllib.error.URLError as e:
        print(f"Failed to connect to server: {e}")
        sys.exit(1)

    game_id = state.get('game_id')
    if not game_id:
        print('Failed to start game: missing game_id')
        sys.exit(1)

    auto_left = args.auto
    last_trick_id = None

    while True:
        print_state(state, args.human)

        last_trick = state.get('trick')
        if last_trick and last_trick.get('trick_id') != last_trick_id:
            trick_id = last_trick.get('trick_id')
            lead = last_trick.get('lead')
            cards = last_trick.get('cards', [])
            winner = last_trick.get('winner')
            played = {}
            if lead is not None:
                for i, card in enumerate(cards):
                    played[(lead + i) % 4] = card
            display = ' '.join([f"P{pid}:{played.get(pid, '_')}" for pid in range(4)])
            print(f"Trick {trick_id}: {display} | Winner: P{winner}")
            last_trick_id = trick_id

        if state.get('terminal'):
            result = state.get('result')
            if result:
                bids = result.get('bids', [])
                tricks = result.get('tricks_won', [])
                print("\n=== Game Summary ===")
                for pid in range(4):
                    bid = bids[pid] if pid < len(bids) else None
                    got = tricks[pid] if pid < len(tricks) else None
                    print(f"Player {pid}: Bid {bid} | Tricks {got}")
                print(f"Team0: {result['team_scores'][0]} | Team1: {result['team_scores'][1]}")
            else:
                print('Game Over.')
            if args.auto_exit:
                break
            restart = input('Restart? (y/N): ').strip().lower()
            if restart == 'y':
                state = http_post_json(f"{base}/reset", {
                    'game': 'spades',
                    'seed': args.seed,
                    'human_player': args.human,
                })
                game_id = state.get('game_id')
                continue
            break

        legal_actions = state.get('legal_actions', [])
        phase = state.get('phase')
        hand = state.get('obs', {}).get('hand', [])

        if state.get('current_player') != args.human:
            print('Waiting for AI...')
            time.sleep(args.delay)
            try:
                state = http_get_json(f"{base}/state?game_id={game_id}")
            except urllib.error.URLError as e:
                print(f"Failed to fetch state: {e}")
                break
            continue

        if auto_left > 0:
            action_id = choose_action_auto(legal_actions)
            auto_left -= 1
            time.sleep(args.delay)
            if auto_left == 0 and args.auto_exit:
                print('Auto steps completed. Exit.')
                break
        else:
            action_id = choose_action_interactive(legal_actions, phase, hand)

        if action_id is None or action_id not in legal_actions:
            print('Invalid action. Please try again.')
            continue

        try:
            state = http_post_json(f"{base}/step", {
                'game_id': game_id,
                'action': action_id,
            })
        except urllib.error.URLError as e:
            print(f"Failed to send action: {e}")
            break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nExit.')
