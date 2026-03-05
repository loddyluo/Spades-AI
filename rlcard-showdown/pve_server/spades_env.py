import os
import sys
import uuid

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'rlcard'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import rlcard
import torch
from rlcard.utils import get_device
from rlcard.utils.agent_utils import load_agent_from_checkpoint


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
    for idx, (_, card_str) in enumerate(trick_order):
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


def _serialize_obs(raw_obs, current_trick, enable_blind_nil=True):
    player_id = raw_obs.get('current_player')
    phase = raw_obs.get('phase', 0)
    blind_passed = raw_obs.get('blind_passed', [])
    hand = raw_obs.get('hand', [])
    if enable_blind_nil and phase == 0 and player_id is not None and player_id < len(blind_passed) and not blind_passed[player_id]:
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
    def __init__(self, seed=None, human_player=0, ai_checkpoint=None, ai_checkpoint_team0=None, ai_checkpoint_team1=None, enable_blind_nil=True):
        self.enable_blind_nil = enable_blind_nil
        self.env = rlcard.make(
            'spades',
            config={'allow_raw_data': True, 'seed': seed, 'game_enable_blind_nil': enable_blind_nil},
        )
        self.human_player = human_player
        if self.human_player < 0 or self.human_player >= self.env.num_players:
            raise ValueError(f'human_player must be in [0, {self.env.num_players - 1}]')

        self.ai_checkpoint = ai_checkpoint
        self.ai_checkpoint_team0 = ai_checkpoint_team0
        self.ai_checkpoint_team1 = ai_checkpoint_team1
        self.agents = self._build_agents()
        self.env.set_agents(self.agents)
        self.current_state = None
        self.current_player = None
        self.pending_trick = []
        self.current_trick = [None for _ in range(self.env.num_players)]
        self.last_trick = None
        self.trick_id = 0

    def _load_agent(self, checkpoint_path):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return None
        device = get_device()
        try:
            agent, _ = load_agent_from_checkpoint(checkpoint_path, device=device, env=self.env)
            return agent
        except RuntimeError as exc:
            msg = str(exc).lower()
            if 'cuda' not in msg:
                raise
            cpu_device = torch.device('cpu')
            agent, _ = load_agent_from_checkpoint(checkpoint_path, device=cpu_device, env=self.env)
            return agent

    def _build_agents(self):
        team0_path = self.ai_checkpoint_team0 or self.ai_checkpoint
        team1_path = self.ai_checkpoint_team1 or self.ai_checkpoint
        if not team0_path or not team1_path:
            raise ValueError('Provide ai_checkpoint, or provide both ai_checkpoint_team0 and ai_checkpoint_team1.')

        if team0_path == team1_path:
            ai_agent = self._load_agent(team0_path)
            if ai_agent is None:
                raise ValueError('ai_checkpoint does not exist or cannot be loaded.')
            if isinstance(ai_agent, list):
                if len(ai_agent) != self.env.num_players:
                    raise ValueError('DMC agent list length must match number of players.')
                return ai_agent
            return [ai_agent for _ in range(self.env.num_players)]

        team0_agent = self._load_agent(team0_path)
        team1_agent = self._load_agent(team1_path)
        if team0_agent is None or team1_agent is None:
            raise ValueError('ai_checkpoint does not exist or cannot be loaded for one of the teams.')
        if isinstance(team0_agent, list) or isinstance(team1_agent, list):
            if not isinstance(team0_agent, list) or not isinstance(team1_agent, list):
                raise ValueError('DMC checkpoints must be provided for both teams when using separate checkpoints.')
            if len(team0_agent) != self.env.num_players or len(team1_agent) != self.env.num_players:
                raise ValueError('DMC agent list length must match number of players.')
            return [team0_agent[0], team1_agent[1], team0_agent[2], team1_agent[3]]
        return [team0_agent, team1_agent, team0_agent, team1_agent]

    def reset(self):
        self.current_state, self.current_player = self.env.reset()
        self.pending_trick = []
        self.current_trick = [None for _ in range(self.env.num_players)]
        self.last_trick = None
        self.trick_id = 0
        self._advance_until_human()
        return self._build_response()

    def _current_legal_actions(self):
        return list(self.current_state.get('legal_actions', {}).keys())

    def _apply_action(self, action, use_raw=False):
        # Decode action string for trick tracking
        if use_raw:
            action_str = action
        else:
            try:
                action_str = self.env._decode_action(action)
            except (TypeError, IndexError):
                raise ValueError(f'invalid action id: {action}')

        if isinstance(action_str, str) and action_str in self.env.card2id:
            if len(self.pending_trick) == 0:
                self.current_trick = [None for _ in range(self.env.num_players)]
            self.current_trick[self.current_player] = action_str
            self.pending_trick.append((self.current_player, action_str))

        try:
            self.current_state, self.current_player = self.env.step(action, use_raw)
        except Exception as exc:
            raise RuntimeError(f'failed to apply action {action}: {exc}') from exc

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
            agent = self.env.agents[self.current_player]
            action, _ = agent.eval_step(self.current_state)
            if not agent.use_raw and action not in self._current_legal_actions():
                raise RuntimeError(f'AI produced illegal action {action} for player {self.current_player}.')
            self._apply_action(action, agent.use_raw)

    def step(self, action):
        if self.current_state is None:
            raise RuntimeError('game is not initialized')
        if self.env.is_over():
            raise RuntimeError('game is already over; call /reset to start a new game')
        if self.current_player != self.human_player:
            raise RuntimeError('not human turn')

        legal_actions = self._current_legal_actions()
        if action not in legal_actions:
            raise ValueError(f'illegal action: {action}; legal actions: {legal_actions}')

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
        obs = _serialize_obs(raw_obs, self.current_trick, self.enable_blind_nil)
        hand_sizes = [len(p.hand) for p in self.env.game.players]

        result = None
        reward = 0
        if self.env.is_over():
            # Raw game scores for UI display.
            judged_scores = self.env.game.judger.judge_game(self.env.game.players)
            team_scores = [int(judged_scores[0]), int(judged_scores[1])]

            # RL payoffs are still exposed for diagnostics.
            payoffs = self.env.get_payoffs()
            payoffs_list = [int(x) for x in payoffs.tolist()] if hasattr(payoffs, 'tolist') else [int(x) for x in payoffs]
            bids = [p.bid for p in self.env.game.players]
            tricks_won = [p.tricks for p in self.env.game.players]
            result = {
                'team_scores': team_scores,
                'player_scores': payoffs_list,
                'bids': bids,
                'tricks_won': tricks_won,
            }
            human_team = 0 if self.human_player in (0, 2) else 1
            reward = team_scores[human_team]

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

    def create_session(self, seed=None, human_player=0, ai_checkpoint=None, ai_checkpoint_team0=None, ai_checkpoint_team1=None, enable_blind_nil=True):
        game_id = f"spades-{uuid.uuid4().hex[:8]}"
        session = SpadesSession(
            seed=seed,
            human_player=human_player,
            ai_checkpoint=ai_checkpoint,
            ai_checkpoint_team0=ai_checkpoint_team0,
            ai_checkpoint_team1=ai_checkpoint_team1,
            enable_blind_nil=enable_blind_nil,
        )
        self.sessions[game_id] = session
        return game_id, session

    def get(self, game_id):
        return self.sessions.get(game_id)
