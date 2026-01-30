import numpy as np
import os
import json
import io

from rlcard.games.base import Card

def set_seed(seed):
    if seed is not None:
        import subprocess
        import sys

        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
        if 'torch' in installed_packages:
            import torch
            torch.backends.cudnn.deterministic = True
            torch.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)

def get_device():
    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps:0")
        print("--> Running on the GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("--> Running on the GPU")
    else:
        device = torch.device("cpu")
        print("--> Running on the CPU")

    return device    

def _shard_index_path(base_path):
    return base_path + '.index.json'

def save_torch_sharded(obj, base_path, max_shard_size_mb=49):
    '''Save a torch object to disk with optional sharding.

    If the serialized object exceeds max_shard_size_mb, it will be split into
    multiple parts and an index file will be written at base_path + '.index.json'.
    Otherwise, it saves a single file at base_path.

    Args:
        obj: Any torch-serializable object
        base_path (str): Target file path (e.g., model.pth)
        max_shard_size_mb (int): Max size per shard in MB
    '''
    import torch

    max_bytes = int(max_shard_size_mb * 1024 * 1024)
    if max_bytes <= 0:
        raise ValueError('max_shard_size_mb must be positive')

    buffer = io.BytesIO()
    torch.save(obj, buffer)
    data = buffer.getvalue()

    base_dir = os.path.dirname(base_path)
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)

    # If small enough, save as a single file and remove any old shards/index
    if len(data) <= max_bytes:
        torch.save(obj, base_path)
        index_path = _shard_index_path(base_path)
        if os.path.isfile(index_path):
            try:
                with open(index_path, 'r') as f:
                    index = json.load(f)
                for part in index.get('parts', []):
                    if os.path.isfile(part):
                        os.remove(part)
            except Exception:
                pass
            try:
                os.remove(index_path)
            except Exception:
                pass
        return

    # Remove any existing shards/index
    index_path = _shard_index_path(base_path)
    if os.path.isfile(index_path):
        try:
            with open(index_path, 'r') as f:
                index = json.load(f)
            for part in index.get('parts', []):
                if os.path.isfile(part):
                    os.remove(part)
        except Exception:
            pass
        try:
            os.remove(index_path)
        except Exception:
            pass

    # Write shards
    parts = []
    for i in range(0, len(data), max_bytes):
        part_path = f"{base_path}.part-{i // max_bytes:05d}"
        with open(part_path, 'wb') as f:
            f.write(data[i:i + max_bytes])
        parts.append(part_path)

    index = {
        'base_path': base_path,
        'total_size': len(data),
        'max_shard_size_mb': max_shard_size_mb,
        'parts': parts,
    }
    with open(index_path, 'w') as f:
        json.dump(index, f)

def load_torch_sharded(base_path, map_location=None):
    '''Load a torch object from a single file or from sharded parts.

    Args:
        base_path (str): Target file path (e.g., model.pth)
        map_location: torch.load map_location
    '''
    import torch

    if os.path.isfile(base_path):
        return torch.load(base_path, map_location=map_location)

    index_path = _shard_index_path(base_path)
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f'No model file or shard index found for {base_path}')

    with open(index_path, 'r') as f:
        index = json.load(f)

    parts = index.get('parts', [])
    if not parts:
        raise FileNotFoundError(f'No shard parts listed in {index_path}')

    data = bytearray()
    for part in parts:
        if not os.path.isfile(part):
            raise FileNotFoundError(f'Missing shard part: {part}')
        with open(part, 'rb') as f:
            data.extend(f.read())

    buffer = io.BytesIO(data)
    return torch.load(buffer, map_location=map_location)

def init_standard_deck():
    ''' Initialize a standard deck of 52 cards

    Returns:
        (list): A list of Card object
    '''
    suit_list = ['S', 'H', 'D', 'C']
    rank_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
    return res

def init_54_deck():
    ''' Initialize a standard deck of 52 cards, BJ and RJ

    Returns:
        (list): Alist of Card object
    '''
    suit_list = ['S', 'H', 'D', 'C']
    rank_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
    res.append(Card('BJ', ''))
    res.append(Card('RJ', ''))
    return res

def rank2int(rank):
    ''' Get the coresponding number of a rank.

    Args:
        rank(str): rank stored in Card object

    Returns:
        (int): the number corresponding to the rank

    Note:
        1. If the input rank is an empty string, the function will return -1.
        2. If the input rank is not valid, the function will return None.
    '''
    if rank == '':
        return -1
    elif rank.isdigit():
        if int(rank) >= 2 and int(rank) <= 10:
            return int(rank)
        else:
            return None
    elif rank == 'A':
        return 14
    elif rank == 'T':
        return 10
    elif rank == 'J':
        return 11
    elif rank == 'Q':
        return 12
    elif rank == 'K':
        return 13
    return None

def elegent_form(card):
    ''' Get a elegent form of a card string

    Args:
        card (string): A card string

    Returns:
        elegent_card (string): A nice form of card
    '''
    suits = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣','s': '♠', 'h': '♥', 'd': '♦', 'c': '♣' }
    rank = '10' if card[1] == 'T' else card[1]

    return suits[card[0]] + rank

def print_card(cards):
    ''' Nicely print a card or list of cards

    Args:
        card (string or list): The card(s) to be printed
    '''
    if cards is None:
        cards = [None]
    if isinstance(cards, str):
        cards = [cards]

    lines = [[] for _ in range(9)]

    for card in cards:
        if card is None:
            lines[0].append('┌─────────┐')
            lines[1].append('│░░░░░░░░░│')
            lines[2].append('│░░░░░░░░░│')
            lines[3].append('│░░░░░░░░░│')
            lines[4].append('│░░░░░░░░░│')
            lines[5].append('│░░░░░░░░░│')
            lines[6].append('│░░░░░░░░░│')
            lines[7].append('│░░░░░░░░░│')
            lines[8].append('└─────────┘')
        else:
            if isinstance(card, Card):
                elegent_card = elegent_form(card.suit + card.rank)
            else:
                elegent_card = elegent_form(card)
            suit = elegent_card[0]
            rank = elegent_card[1]
            if len(elegent_card) == 3:
                space = elegent_card[2]
            else:
                space = ' '

            lines[0].append('┌─────────┐')
            lines[1].append('│{}{}       │'.format(rank, space))
            lines[2].append('│         │')
            lines[3].append('│         │')
            lines[4].append('│    {}    │'.format(suit))
            lines[5].append('│         │')
            lines[6].append('│         │')
            lines[7].append('│       {}{}│'.format(space, rank))
            lines[8].append('└─────────┘')

    for line in lines:
        print ('   '.join(line))

def reorganize(trajectories, payoffs):
    ''' Reorganize the trajectory to make it RL friendly

    Args:
        trajectory (list): A list of trajectories
        payoffs (list): A list of payoffs for the players. Each entry corresponds to one player

    Returns:
        (list): A new trajectories that can be fed into RL algorithms.

    '''
    num_players = len(trajectories)
    new_trajectories = [[] for _ in range(num_players)]

    for player in range(num_players):
        for i in range(0, len(trajectories[player])-2, 2):
            if i ==len(trajectories[player])-3:
                reward = payoffs[player]
                done =True
            else:
                reward, done = 0, False
            transition = trajectories[player][i:i+3].copy()
            transition.insert(2, reward)
            transition.append(done)

            new_trajectories[player].append(transition)
    return new_trajectories

def remove_illegal(action_probs, legal_actions):
    ''' Remove illegal actions and normalize the
        probability vector

    Args:
        action_probs (numpy.array): A 1 dimention numpy array.
        legal_actions (list): A list of indices of legal actions.

    Returns:
        probd (numpy.array): A normalized vector without legal actions.
    '''
    probs = np.zeros(action_probs.shape[0])
    probs[legal_actions] = action_probs[legal_actions]
    if np.sum(probs) == 0:
        probs[legal_actions] = 1 / len(legal_actions)
    else:
        probs /= sum(probs)
    return probs

def tournament(env, num):
    ''' Evaluate he performance of the agents in the environment

    Args:
        env (Env class): The environment to be evaluated.
        num (int): The number of games to play.

    Returns:
        A list of avrage payoffs for each player
    '''
    payoffs = [0 for _ in range(env.num_players)]
    counter = 0
    while counter < num:
        _, _payoffs = env.run(is_training=False)
        if isinstance(_payoffs, list):
            for _p in _payoffs:
                for i, _ in enumerate(payoffs):
                    payoffs[i] += _p[i]
                counter += 1
        else:
            for i, _ in enumerate(payoffs):
                payoffs[i] += _payoffs[i]
            counter += 1
    for i, _ in enumerate(payoffs):
        payoffs[i] /= counter
    return payoffs

def plot_curve(csv_path, save_path, algorithm):
    ''' Read data from csv file and plot the results
    '''
    import os
    import csv
    import matplotlib.pyplot as plt
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        xs = []
        ys = []
        for row in reader:
            xs.append(int(row['episode']))
            ys.append(float(row['reward']))
        fig, ax = plt.subplots()
        ax.plot(xs, ys, label=algorithm)
        ax.set(xlabel='episode', ylabel='reward')
        ax.legend()
        ax.grid()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)

