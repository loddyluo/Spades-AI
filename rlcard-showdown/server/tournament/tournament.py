import os
import json
from tqdm import tqdm
import numpy as np

from .rlcard_wrap import rlcard

class Tournament(object):
    
    def __init__(self, game, model_ids, num_eval_games=100):
        """ Default for two player games
            For Dou Dizhu, the two peasants use the same model
        """
        self.game = game
        self.model_ids = model_ids
        self.num_eval_games = num_eval_games
        # Load the models
        self.models = [rlcard.models.load(model_id) for model_id in model_ids]

    def launch(self):
        """ Currently for two-player game only
        """
        num_models = len(self.model_ids)
        games_data = []
        payoffs_data = []
        for i in range(num_models):
            for j in range(num_models):
                if j == i:
                    continue
                print(self.game, '-', self.model_ids[i], 'VS', self.model_ids[j])
                if self.game == 'doudizhu':
                    agents = [self.models[i].agents[0], self.models[j].agents[1], self.models[j].agents[2]]
                    names = [self.model_ids[i], self.model_ids[j], self.model_ids[j]]
                    data, payoffs, wins = doudizhu_tournament(self.game, agents, names, self.num_eval_games)
                elif self.game == 'leduc-holdem':
                    agents = [self.models[i].agents[0], self.models[j].agents[1]]
                    names = [self.model_ids[i], self.model_ids[j]]
                    data, payoffs, wins = leduc_holdem_tournament(self.game, agents, names, self.num_eval_games)
                elif self.game == 'spades':
                    agents = [
                        self.models[i].agents[0],
                        self.models[j].agents[1],
                        self.models[i].agents[2],
                        self.models[j].agents[3],
                    ]
                    names = [self.model_ids[i], self.model_ids[j], self.model_ids[i], self.model_ids[j]]
                    data, payoffs, wins = spades_tournament(self.game, agents, names, self.num_eval_games)
                mean_payoff = np.mean(payoffs)
                print('Average payoff:', mean_payoff)
                print()

                for k in range(len(data)):
                    game_data = {}
                    game_data['name'] = self.game
                    game_data['index'] = k
                    game_data['agent0'] = self.model_ids[i]
                    game_data['agent1'] = self.model_ids[j]
                    game_data['win'] = wins[k]
                    game_data['replay'] = data[k]
                    game_data['payoff'] = payoffs[k]

                    games_data.append(game_data)

                payoff_data = {}
                payoff_data['name'] = self.game
                payoff_data['agent0'] = self.model_ids[i]
                payoff_data['agent1'] = self.model_ids[j]
                payoff_data['payoff'] = mean_payoff
                payoffs_data.append(payoff_data)
        return games_data, payoffs_data

def _spades_rank_value(rank):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    return ranks.index(rank) if rank in ranks else -1

def _spades_trick_winner(trick_order):
    # trick_order: list of (player_id, card_str) in play order
    if not trick_order:
        return None
    lead_suit = trick_order[0][1][0]
    highest_trump = None
    highest_lead = None
    for idx, (pid, card_str) in enumerate(trick_order):
        suit = card_str[0]
        rank = card_str[1]
        val = _spades_rank_value(rank)
        if suit == 'S':
            if highest_trump is None or val > highest_trump[1]:
                highest_trump = (idx, val)
        elif suit == lead_suit:
            if highest_trump is None and (highest_lead is None or val > highest_lead[1]):
                highest_lead = (idx, val)
    winner_idx = highest_trump[0] if highest_trump else highest_lead[0]
    return trick_order[winner_idx][0]

def _serialize_spades_obs(raw_obs, current_trick_cards):
    player_id = raw_obs.get('current_player')
    phase = raw_obs.get('phase', 0)
    blind_passed = raw_obs.get('blind_passed', [])
    hand = raw_obs.get('hand', [])
    if phase == 0 and player_id is not None and player_id < len(blind_passed) and not blind_passed[player_id]:
        hand = []
    current_trick = current_trick_cards if current_trick_cards is not None else [None, None, None, None]
    return {
        'hand': hand,
        'bids': raw_obs.get('bids', []),
        'tricks_won': raw_obs.get('tricks_won', []),
        'spades_broken': 1 if raw_obs.get('spades_broken') else 0,
        'current_trick': current_trick,
    }

def doudizhu_tournament(game, agents, names, num_eval_games):
    env = rlcard.make(game, config={'allow_raw_data': True})
    env.set_agents(agents)
    payoffs = []
    json_data = []
    wins = []
    for _ in tqdm(range(num_eval_games)):
        data = {}
        roles = ['landlord', 'peasant', 'peasant']
        data['playerInfo'] = [{'id': i, 'index': i, 'role': roles[i], 'agentInfo': {'name': names[i]}} for i in range(env.num_players)]
        state, player_id = env.reset()
        perfect = env.get_perfect_information()
        data['initHands'] = perfect['hand_cards_with_suit']
        current_hand_cards = perfect['hand_cards_with_suit'].copy()
        for i in range(len(current_hand_cards)):
            current_hand_cards[i] = current_hand_cards[i].split()
        data['moveHistory'] = []
        while not env.is_over():
            action, info = env.agents[player_id].eval_step(state)
            history = {}
            history['playerIdx'] = player_id
            if env.agents[player_id].use_raw:
                _action = action
            else:
                _action = env._decode_action(action)
            history['move'] = _calculate_doudizhu_move(_action, player_id, current_hand_cards)
            history['info'] = info

            data['moveHistory'].append(history)
            state, player_id = env.step(action, env.agents[player_id].use_raw)
        data = json.dumps(str(data))
        #data = json.dumps(data, indent=2, sort_keys=True)
        json_data.append(data)
        if env.get_payoffs()[0] > 0:
            wins.append(True)
        else:
            wins.append(False)
        payoffs.append(env.get_payoffs()[0])
    return json_data, payoffs, wins

def _calculate_doudizhu_move(action, player_id, current_hand_cards):
    if action == 'pass':
        return action
    trans = {'B': 'BJ', 'R': 'RJ'}
    cards_with_suit = []
    for card in action:
        if card in trans:
            cards_with_suit.append(trans[card])
            current_hand_cards[player_id].remove(trans[card])
        else:
            for hand_card in current_hand_cards[player_id]:
                if hand_card[1] == card:
                    cards_with_suit.append(hand_card)
                    current_hand_cards[player_id].remove(hand_card)
                    break
    return ' '.join(cards_with_suit)

def leduc_holdem_tournament(game, agents, names, num_eval_games):
    env = rlcard.make(game, config={'allow_raw_data': True})
    env.set_agents(agents)
    payoffs = []
    json_data = []
    wins = []
    for _ in tqdm(range(num_eval_games)):
        data = {}
        data['playerInfo'] = [{'id': i, 'index': i, 'agentInfo': {'name': names[i]}} for i in range(env.num_players)]
        state, player_id = env.reset()
        perfect = env.get_perfect_information()
        data['initHands'] = perfect['hand_cards']
        data['moveHistory'] = []
        round_history = []
        round_id = 0
        while not env.is_over():
            action, info = env.agents[player_id].eval_step(state)
            history = {}
            history['playerIdx'] = player_id
            if env.agents[player_id].use_raw:
                history['move'] = action
            else:
                history['move'] = env._decode_action(action)

            history['info'] = info
            round_history.append(history)
            state, player_id = env.step(action, env.agents[player_id].use_raw)
            perfect = env.get_perfect_information()
            if round_id < perfect['current_round'] or env.is_over():
                round_id = perfect['current_round']
                data['moveHistory'].append(round_history)
                round_history = []
        perfect = env.get_perfect_information()
        data['publicCard'] = perfect['public_card']
        data = json.dumps(str(data))
        #data = json.dumps(data, indent=2, sort_keys=True)
        json_data.append(data)
        if env.get_payoffs()[0] > 0:
            wins.append(True)
        else:
            wins.append(False)
        payoffs.append(env.get_payoffs()[0])
    return json_data, payoffs, wins

def spades_tournament(game, agents, names, num_eval_games):
    env = rlcard.make(game, config={'allow_raw_data': True})
    env.set_agents(agents)
    payoffs = []
    json_data = []
    wins = []
    for _ in tqdm(range(num_eval_games)):
        data = {}
        data['game'] = 'spades'
        data['version'] = 1
        data['metadata'] = {
            'num_players': env.num_players,
            'team_map': [0, 1, 0, 1],
            'source': 'rlcard'
        }
        data['playerInfo'] = [{'id': i, 'index': i, 'agentInfo': {'name': names[i]}} for i in range(env.num_players)]

        state, player_id = env.reset()
        # Initial hands (perfect information)
        initial_hands = []
        for p in env.game.players:
            initial_hands.append([c.get_index() for c in p.hand])
        data['initialHands'] = initial_hands

        current_trick_cards = [None for _ in range(env.num_players)]
        current_trick_order = []
        trick_id = 0
        data['states'] = []
        data['tricks'] = []

        while not env.is_over():
            raw_obs = state.get('raw_obs', {})
            phase = 'bidding' if raw_obs.get('phase', 0) == 0 else 'play'
            legal_actions = list(state.get('legal_actions', {}).keys())

            action, info = env.agents[player_id].eval_step(state)
            if env.agents[player_id].use_raw:
                action_str = action
                action_id = env.actions.index(action)
            else:
                action_id = action
                action_str = env._decode_action(action)

            # Track trick cards for UI
            if phase == 'play' and isinstance(action_str, str) and action_str in env.card2id:
                if len(current_trick_order) == 0:
                    current_trick_cards = [None for _ in range(env.num_players)]
                current_trick_cards[player_id] = action_str
                current_trick_order.append((player_id, action_str))

            # Step env
            state, player_id = env.step(action, env.agents[player_id].use_raw)
            raw_after = state.get('raw_obs', raw_obs)

            entry = {
                't': len(data['states']),
                'phase': phase,
                'current_player': raw_obs.get('current_player', None),
                'action': action_id,
                'action_str': action_str,
                'legal_actions': legal_actions,
                'obs': _serialize_spades_obs(raw_after, current_trick_cards),
            }
            data['states'].append(entry)

            if len(current_trick_order) == env.num_players:
                winner = _spades_trick_winner(current_trick_order)
                data['tricks'].append({
                    'trick_id': trick_id,
                    'lead': current_trick_order[0][0],
                    'cards': [c for _, c in current_trick_order],
                    'winner': winner
                })
                trick_id += 1
                current_trick_order = []
                current_trick_cards = [None for _ in range(env.num_players)]

        payoffs_game = env.get_payoffs()
        bids = [p.bid for p in env.game.players]
        tricks_won = [p.tricks for p in env.game.players]
        data['result'] = {
            'team_scores': [payoffs_game[0], payoffs_game[1]],
            'player_scores': payoffs_game,
            'bids': bids,
            'tricks_won': tricks_won
        }
        json_data.append(json.dumps(data))
        wins.append(payoffs_game[0] > payoffs_game[1])
        payoffs.append(payoffs_game[0])
    return json_data, payoffs, wins


if __name__=='__main__':
    game = 'leduc-holdem'
    model_ids = ['leduc-holdem-random', 'leduc-holdem-rule-v1', 'leduc-holdem-cfr']
    t = Tournament(game, model_ids)
    games_data = t.launch()
    print(len(games_data))
    print(games_data[0])
    #root_path = './models'
    #agent1 = LeducHoldemDQNModel1(root_path)
    #agent2 = LeducHoldemRandomModel(root_path)
    #agent3 = LeducHoldemRuleModel()
    #agent4 = LeducHoldemCFRModel(root_path)
    #agent5 = LeducHoldemDQNModel2(root_path)
    #t = Tournament(agent1, agent2, agent3, agent4, agent5, 'leduc-holdem')
    ##t.competition()
    #t.evaluate()
