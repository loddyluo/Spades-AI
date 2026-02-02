import os
import sys
import uuid

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'rlcard'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import rlcard
import torch
from rlcard.agents import DQNAgent


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


def _serialize_obs(raw_obs, current_trick):
    player_id = raw_obs.get('current_player')
    phase = raw_obs.get('phase', 0)
    blind_passed = raw_obs.get('blind_passed', [])
    hand = raw_obs.get('hand', [])
    if phase == 0 and player_id is not None and player_id < len(blind_passed) and not blind_passed[player_id]:
        hand = []
    return {
        'hand': hand,
        'bids': raw_obs.get('bids', []),
        'bid_types': raw_obs.get('bid_types', []),
        'tricks_won': raw_obs.get('tricks_won', []),
        'spades_broken': 1 if raw_obs.get('spades_broken') else 0,
        'current_trick': current_trick,
    }


class SpadesSession:
    def __init__(self, seed=None, human_player=0, ai_checkpoint=None):
        self.env = rlcard.make('spades', config={'allow_raw_data': True, 'seed': seed})
        self.human_player = human_player
        self.ai_checkpoint = ai_checkpoint
        self.agents = self._build_agents()
        self.env.set_agents(self.agents)
        self.current_state = None
        self.current_player = None
        self.pending_trick = []
        self.current_trick = [None for _ in range(self.env.num_players)]
        self.last_trick = None
        self.trick_id = 0

    def _load_dqn_agent(self, checkpoint_path):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return None
        device = torch.device('cpu')
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        agent = DQNAgent.from_checkpoint(ckpt)
        if hasattr(agent, 'set_device'):
            agent.set_device(device)
        return agent

    def _build_agents(self):
        ai_agent = self._load_dqn_agent(self.ai_checkpoint)
        if ai_agent is None:
            raise ValueError('ai_checkpoint is required for PvE; no random agent fallback is allowed.')

        # Use the same trained checkpoint for all AI seats
        return [ai_agent for _ in range(self.env.num_players)]

    def reset(self):
        self.current_state, self.current_player = self.env.reset()
        self.pending_trick = []
        self.current_trick = [None for _ in range(self.env.num_players)]
        self.last_trick = None
        self.trick_id = 0
        self._advance_until_human()
        return self._build_response()

    def _apply_action(self, action, use_raw=False):
        # Decode action string for trick tracking
        if use_raw:
            action_str = action
        else:
            action_str = self.env._decode_action(action)

        if isinstance(action_str, str) and action_str in self.env.card2id:
            if len(self.pending_trick) == 0:
                self.current_trick = [None for _ in range(self.env.num_players)]
            self.current_trick[self.current_player] = action_str
            self.pending_trick.append((self.current_player, action_str))

        self.current_state, self.current_player = self.env.step(action, use_raw)

        if len(self.pending_trick) == self.env.num_players:
            winner = _spades_trick_winner(self.pending_trick)
            self.last_trick = {
                'trick_id': self.trick_id,
                'lead': self.pending_trick[0][0],
                'cards': [c for _, c in self.pending_trick],
                'winner': winner,
            }
            self.trick_id += 1
            self.pending_trick = []
            self.current_trick = [None for _ in range(self.env.num_players)]

    def _advance_until_human(self):
        while not self.env.is_over() and self.current_player != self.human_player:
            action, _ = self.env.agents[self.current_player].eval_step(self.current_state)
            self._apply_action(action, self.env.agents[self.current_player].use_raw)

    def step(self, action):
        acting_player = self.current_player
        self._apply_action(action, False)
        if not self.env.is_over():
            self._advance_until_human()
        return self._build_response(last_action={'player': acting_player, 'action': action})

    def _build_response(self, last_action=None):
        raw_obs = self.current_state.get('raw_obs', {})
        phase = 'bidding' if raw_obs.get('phase', 0) == 0 else 'play'
        legal_actions = list(self.current_state.get('legal_actions', {}).keys())
        bid_types = []
        for p in self.env.game.players:
            if p.bid is None:
                bid_types.append(None)
            elif p.is_blind_nil:
                bid_types.append('blind_nil')
            elif p.is_nil:
                bid_types.append('nil')
            else:
                bid_types.append('bid')
        raw_obs['bid_types'] = bid_types
        obs = _serialize_obs(raw_obs, self.current_trick)
        hand_sizes = [len(p.hand) for p in self.env.game.players]

        result = None
        reward = 0
        if self.env.is_over():
            payoffs = self.env.get_payoffs()
            payoffs_list = [int(x) for x in payoffs.tolist()] if hasattr(payoffs, 'tolist') else [int(x) for x in payoffs]
            bids = [p.bid for p in self.env.game.players]
            tricks_won = [p.tricks for p in self.env.game.players]
            result = {
                'team_scores': [payoffs_list[0], payoffs_list[1]],
                'player_scores': payoffs_list,
                'bids': bids,
                'tricks_won': tricks_won,
            }
            reward = payoffs_list[self.human_player]

        return {
            'current_player': self.current_player,
            'phase': phase,
            'obs': obs,
            'legal_actions': legal_actions,
            'hand_sizes': hand_sizes,
            'last_action': last_action,
            'trick': self.last_trick,
            'terminal': self.env.is_over(),
            'reward': reward,
            'result': result,
        }


class SpadesSessionManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self, seed=None, human_player=0, ai_checkpoint=None):
        game_id = f"spades-{uuid.uuid4().hex[:8]}"
        session = SpadesSession(
            seed=seed,
            human_player=human_player,
            ai_checkpoint=ai_checkpoint,
        )
        self.sessions[game_id] = session
        return game_id, session

    def get(self, game_id):
        return self.sessions.get(game_id)
