"""DRQN (Deep Recurrent Q-Network) agent for RLCard.

Extends the standard DQN approach with an LSTM layer so the agent can
leverage temporal patterns within each game episode (e.g. tracking which
cards have been played across tricks).

The three main components are:
  - DRQNNetwork  : nn.Module with embedding → LSTM → MLP head
  - EpisodeMemory: replay buffer that stores full episodes and samples
                   fixed-length sequence slices
  - DRQNAgent    : training / inference logic with hidden-state management
"""

import os
import random
from collections import deque, namedtuple
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from rlcard.utils.utils import remove_illegal

Transition = namedtuple(
    'Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions']
)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class DRQNNetwork(nn.Module):
    """Q-network with LSTM backbone.

    Architecture::

        (batch, seq_len, state_dim)
          -> LayerNorm(state_dim)                per-timestep normalisation
          -> Linear(state_dim, embed_dim)     feature embedding
          -> ReLU
          -> LSTM(embed_dim, hidden_size)     temporal modelling
          -> (take last timestep output)
          -> MLP head  [hidden_size, *mlp_layers, num_actions]
    """

    def __init__(
        self,
        num_actions: int,
        state_shape,
        embed_dim: int = 256,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 1,
        mlp_layers=None,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.embed_dim = embed_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.mlp_layers = mlp_layers or [256]

        state_dim = int(np.prod(state_shape))

        # --- embedding ---
        self.ln = nn.LayerNorm(state_dim)
        self.embed = nn.Linear(state_dim, embed_dim)

        # --- LSTM ---
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        # --- MLP head ---
        head_dims = [lstm_hidden_size] + list(self.mlp_layers)
        layers = []
        for i in range(len(head_dims) - 1):
            layers.append(nn.Linear(head_dims[i], head_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(head_dims[-1], num_actions))
        self.head = nn.Sequential(*layers)

    def forward(self, obs_seq, hidden=None):
        """
        Args:
            obs_seq: (batch, seq_len, state_dim)  or  (batch, state_dim)
            hidden:  optional (h_0, c_0) tuple for LSTM

        Returns:
            q_values: (batch, num_actions) — Q values for last timestep
            hidden:   new (h_n, c_n) tuple
        """
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(1)  # (batch, 1, state_dim)

        batch, seq_len, feat = obs_seq.shape

        # LayerNorm operates on (batch, seq_len, feat) directly — no reshape needed
        x = self.ln(obs_seq)
        x = x.reshape(batch * seq_len, feat)
        x = torch.relu(self.embed(x))
        x = x.reshape(batch, seq_len, self.embed_dim)

        # LSTM
        lstm_out, hidden = self.lstm(x, hidden)  # (batch, seq_len, hidden)

        # Take last timestep
        last = lstm_out[:, -1, :]  # (batch, hidden)

        q_values = self.head(last)  # (batch, num_actions)
        return q_values, hidden

    def share_memory(self):
        """Make parameters accessible across processes (for multi-actor training)."""
        for param in self.parameters():
            param.data.share_memory_()
        return self


# ---------------------------------------------------------------------------
# Estimator (wraps network for train / predict)
# ---------------------------------------------------------------------------

class DRQNEstimator:
    """Wraps DRQNNetwork with numpy ↔ tensor conversion, optimizer, loss."""

    def __init__(
        self,
        num_actions: int = 68,
        learning_rate: float = 0.00005,
        state_shape=None,
        embed_dim: int = 256,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 1,
        mlp_layers=None,
        device=None,
    ):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.state_shape = state_shape
        self.embed_dim = embed_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.mlp_layers = mlp_layers or [256]
        self.device = device or torch.device('cpu')

        self.qnet = DRQNNetwork(
            num_actions=num_actions,
            state_shape=state_shape,
            embed_dim=embed_dim,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            mlp_layers=self.mlp_layers,
        ).to(self.device)
        self.qnet.eval()

        for p in self.qnet.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        self.huber_loss = nn.SmoothL1Loss(reduction='mean', beta=10.0)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=learning_rate)

    # ---- inference --------------------------------------------------------

    def predict_nograd(self, obs_seq, hidden=None):
        """
        Args:
            obs_seq: np.ndarray (batch, seq_len, state_dim)  or  (batch, state_dim)
            hidden:  optional tuple of tensors

        Returns:
            q_values: np.ndarray (batch, num_actions)
            hidden:   tuple of tensors (on device)
        """
        with torch.no_grad():
            t = torch.from_numpy(obs_seq).float().to(self.device)
            q, h = self.qnet(t, hidden)
        return q.cpu().numpy(), h

    # ---- training ---------------------------------------------------------

    def update(self, state_seqs, actions, targets):
        """
        Args:
            state_seqs: np.ndarray (batch, seq_len, state_dim)
            actions:    np.ndarray (batch,) int
            targets:    np.ndarray (batch,) float

        Returns:
            loss: float
        """
        self.optimizer.zero_grad()
        self.qnet.train()

        s = torch.from_numpy(state_seqs).float().to(self.device)
        a = torch.from_numpy(actions).long().to(self.device)
        y = torch.from_numpy(targets).float().to(self.device)

        q_all, _ = self.qnet(s)  # (batch, num_actions)
        q = torch.gather(q_all, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)

        loss = self.huber_loss(q, y)
        loss.backward()
        self.optimizer.step()
        loss_val = loss.item()

        self.qnet.eval()
        return loss_val

    # ---- checkpoint -------------------------------------------------------

    def checkpoint_attributes(self):
        return {
            'qnet': self.qnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'num_actions': self.num_actions,
            'learning_rate': self.learning_rate,
            'state_shape': self.state_shape,
            'embed_dim': self.embed_dim,
            'lstm_hidden_size': self.lstm_hidden_size,
            'lstm_num_layers': self.lstm_num_layers,
            'mlp_layers': self.mlp_layers,
            'device': self.device,
        }

    @classmethod
    def from_checkpoint(cls, ckpt):
        est = cls(
            num_actions=ckpt['num_actions'],
            learning_rate=ckpt['learning_rate'],
            state_shape=ckpt['state_shape'],
            embed_dim=ckpt['embed_dim'],
            lstm_hidden_size=ckpt['lstm_hidden_size'],
            lstm_num_layers=ckpt['lstm_num_layers'],
            mlp_layers=ckpt['mlp_layers'],
            device=ckpt['device'],
        )
        est.qnet.load_state_dict(ckpt['qnet'])
        est.optimizer.load_state_dict(ckpt['optimizer'])
        return est


# ---------------------------------------------------------------------------
# Episode Replay Memory
# ---------------------------------------------------------------------------

class EpisodeMemory:
    """Replay buffer that stores complete episodes and samples fixed-length
    sequential slices for DRQN training.

    Storage: list of episodes, each episode is a list of Transition.
    Sampling: pick *batch_size* random episodes, pick a random slice of
    length *seq_len* from each.  If an episode is shorter than *seq_len*,
    left-pad with zeros.
    """

    def __init__(self, max_episodes: int = 5000, batch_size: int = 32, seq_len: int = 16):
        self.max_episodes = max_episodes
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.episodes = deque(maxlen=max_episodes)
        self._current_episode = []
        self._total_transitions = 0

    @property
    def total_transitions(self):
        return self._total_transitions

    def start_episode(self):
        """Call before feeding transitions of a new episode."""
        if self._current_episode:
            # Auto-commit any pending episode
            self._commit()
        self._current_episode = []

    def save(self, state, action, reward, next_state, legal_actions, done):
        self._current_episode.append(
            Transition(state, action, reward, next_state, done, legal_actions)
        )
        if done:
            self._commit()

    def _commit(self):
        if not self._current_episode:
            return
        # If deque is full, the oldest episode will be evicted automatically
        if len(self.episodes) == self.max_episodes:
            self._total_transitions -= len(self.episodes[0])
        self.episodes.append(self._current_episode)
        self._total_transitions += len(self._current_episode)
        self._current_episode = []

    def can_sample(self):
        return len(self.episodes) >= 1 and self.total_transitions >= self.batch_size

    def sample(self):
        """Sample *batch_size* sequence slices.

        Returns:
            state_seqs       (batch, seq_len, state_dim) np.float32
            action_batch     (batch,)                    np.int64
            reward_batch     (batch,)                    np.float32
            next_state_seqs  (batch, seq_len, state_dim) np.float32
            done_batch       (batch,)                    np.bool_
            legal_actions_batch  list of lists
        """
        state_seqs = []
        next_state_seqs = []
        actions = []
        rewards = []
        dones = []
        legal_actions_list = []

        for _ in range(self.batch_size):
            ep = random.choice(self.episodes)
            ep_len = len(ep)

            # Pick end index: the transition whose Q we want to learn
            end_idx = random.randint(0, ep_len - 1)

            # Build state sequence ending at end_idx (inclusive)
            start_idx = max(0, end_idx - self.seq_len + 1)
            slice_transitions = ep[start_idx : end_idx + 1]

            # Extract state arrays
            s_seq = [t.state for t in slice_transitions]
            ns_seq = [t.next_state for t in slice_transitions]

            # Left-pad if shorter than seq_len
            pad_len = self.seq_len - len(s_seq)
            if pad_len > 0:
                zero = np.zeros_like(s_seq[0])
                s_seq = [zero] * pad_len + s_seq
                ns_seq = [zero] * pad_len + ns_seq

            state_seqs.append(np.stack(s_seq))
            next_state_seqs.append(np.stack(ns_seq))

            # Target transition is the last one in the slice
            target = slice_transitions[-1]
            actions.append(target.action)
            rewards.append(target.reward)
            dones.append(target.done)
            legal_actions_list.append(target.legal_actions)

        return (
            np.array(state_seqs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_state_seqs, dtype=np.float32),
            np.array(dones),
            legal_actions_list,
        )

    def checkpoint_attributes(self):
        return {
            'max_episodes': self.max_episodes,
            'batch_size': self.batch_size,
            'seq_len': self.seq_len,
            'episodes': self.episodes,
        }

    @classmethod
    def from_checkpoint(cls, ckpt):
        inst = cls(ckpt['max_episodes'], ckpt['batch_size'], ckpt['seq_len'])
        inst.episodes = deque(ckpt.get('episodes', []), maxlen=inst.max_episodes)
        inst._total_transitions = sum(len(ep) for ep in inst.episodes)
        return inst


# ---------------------------------------------------------------------------
# DRQN Agent
# ---------------------------------------------------------------------------

class DRQNAgent:
    """Deep Recurrent Q-Network agent.

    Compatible with the RLCard agent interface (step / eval_step / feed).
    Maintains per-player LSTM hidden states for online inference.
    """

    def __init__(
        self,
        num_actions: int = 68,
        state_shape=None,
        # network
        embed_dim: int = 256,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 1,
        mlp_layers=None,
        learning_rate: float = 0.00005,
        # replay
        max_episodes: int = 5000,
        replay_memory_init_size: int = 500,
        batch_size: int = 32,
        seq_len: int = 16,
        # DQN hypers
        update_target_estimator_every: int = 1000,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_steps: int = 20000,
        train_every: int = 1,
        # misc
        device=None,
        save_path=None,
        save_every: int = 5000,
    ):
        self.use_raw = False
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.embed_dim = embed_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.mlp_layers = mlp_layers or [256]
        self.seq_len = seq_len
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.train_every = train_every
        self.save_path = save_path
        self.save_every = save_every

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.total_t = 0
        self.train_t = 0

        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        est_kwargs = dict(
            num_actions=num_actions,
            learning_rate=learning_rate,
            state_shape=state_shape,
            embed_dim=embed_dim,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            mlp_layers=self.mlp_layers,
            device=self.device,
        )
        self.q_estimator = DRQNEstimator(**est_kwargs)
        self.target_estimator = DRQNEstimator(**est_kwargs)

        self.memory = EpisodeMemory(
            max_episodes=max_episodes,
            batch_size=batch_size,
            seq_len=seq_len,
        )

        # Per-player hidden states for online inference
        self.hidden_states = {}

    # ---- hidden state management -----------------------------------------

    def reset_hidden_states(self):
        self.hidden_states = {}

    def _get_hidden(self, player_id):
        return self.hidden_states.get(player_id, None)

    def _set_hidden(self, player_id, hidden):
        self.hidden_states[player_id] = hidden

    # ---- episode boundary ------------------------------------------------

    def feed_episode_start(self):
        self.memory.start_episode()

    # ---- RLCard interface ------------------------------------------------

    def step(self, state):
        """Epsilon-greedy action selection (training)."""
        q_values, player_id = self._predict_with_hidden(state)
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
        legal_actions = list(state['legal_actions'].keys())
        probs = np.ones(len(legal_actions), dtype=float) * epsilon / len(legal_actions)
        best_action_idx = legal_actions.index(np.argmax(q_values))
        probs[best_action_idx] += 1.0 - epsilon
        action_idx = np.random.choice(len(probs), p=probs)
        return legal_actions[action_idx]

    def eval_step(self, state):
        """Greedy action selection (evaluation)."""
        q_values, _ = self._predict_with_hidden(state)
        best_action = np.argmax(q_values)
        info = {}
        info['values'] = {
            state['raw_legal_actions'][i]: float(
                q_values[list(state['legal_actions'].keys())[i]]
            )
            for i in range(len(state['legal_actions']))
        }
        return best_action, info

    def _predict_with_hidden(self, state):
        """Run a single-step forward pass, updating the stored hidden state."""
        player_id = state['raw_obs']['current_player']
        obs = np.expand_dims(state['obs'], 0)  # (1, state_dim)
        obs = np.expand_dims(obs, 1)            # (1, 1, state_dim)

        hidden = self._get_hidden(player_id)
        q_values, new_hidden = self.q_estimator.predict_nograd(obs, hidden)
        self._set_hidden(player_id, new_hidden)

        q_values = q_values[0]  # (num_actions,)

        # Mask illegal actions
        masked = -np.inf * np.ones(self.num_actions, dtype=float)
        legal = list(state['legal_actions'].keys())
        masked[legal] = q_values[legal]
        return masked, player_id

    # ---- feed & train ----------------------------------------------------

    def feed(self, ts):
        """Store a transition and optionally train."""
        state, action, reward, next_state, done = tuple(ts)
        self.memory.save(
            state['obs'], action, reward, next_state['obs'],
            list(next_state['legal_actions'].keys()), done,
        )
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp >= 0 and tmp % self.train_every == 0 and self.memory.can_sample():
            self.train()

    def train(self):
        (state_seqs, action_batch, reward_batch,
         next_state_seqs, done_batch, legal_actions_batch) = self.memory.sample()

        # --- Double DQN target ---
        # Q-network selects best next action
        q_next, _ = self.q_estimator.predict_nograd(next_state_seqs)
        # Mask illegal
        for b in range(self.batch_size):
            mask = np.ones(self.num_actions) * (-np.inf)
            mask[legal_actions_batch[b]] = 0.0
            q_next[b] += mask
        best_next_actions = np.argmax(q_next, axis=1)

        # Target-network evaluates
        q_target, _ = self.target_estimator.predict_nograd(next_state_seqs)
        target_values = reward_batch + (
            ~done_batch
        ).astype(np.float32) * self.discount_factor * q_target[
            np.arange(self.batch_size), best_next_actions
        ]

        loss = self.q_estimator.update(state_seqs, action_batch, target_values)
        print(f'\rINFO - Step {self.total_t}, rl-loss: {loss:.6f}', end='')

        if self.train_t % self.update_target_estimator_every == 0:
            self.target_estimator = deepcopy(self.q_estimator)
            print('\nINFO - Copied model parameters to target network.')

        self.train_t += 1

        if self.save_path and self.train_t % self.save_every == 0:
            self.save_checkpoint(self.save_path)
            print('\nINFO - Saved model checkpoint.')

    # ---- checkpoint ------------------------------------------------------

    def checkpoint_attributes(self):
        return {
            'agent_type': 'DRQNAgent',
            'q_estimator': self.q_estimator.checkpoint_attributes(),
            'memory': self.memory.checkpoint_attributes(),
            'total_t': self.total_t,
            'train_t': self.train_t,
            'num_actions': self.num_actions,
            'state_shape': self.state_shape,
            'embed_dim': self.embed_dim,
            'lstm_hidden_size': self.lstm_hidden_size,
            'lstm_num_layers': self.lstm_num_layers,
            'mlp_layers': self.mlp_layers,
            'seq_len': self.seq_len,
            'replay_memory_init_size': self.replay_memory_init_size,
            'update_target_estimator_every': self.update_target_estimator_every,
            'discount_factor': self.discount_factor,
            'epsilon_start': float(self.epsilons[0]),
            'epsilon_end': float(self.epsilons[-1]),
            'epsilon_decay_steps': self.epsilon_decay_steps,
            'batch_size': self.batch_size,
            'train_every': self.train_every,
            'device': self.device,
            'save_path': self.save_path,
            'save_every': self.save_every,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        print('\nINFO - Restoring DRQN model from checkpoint...')
        agent = cls(
            num_actions=checkpoint['num_actions'],
            state_shape=checkpoint.get('state_shape') or checkpoint['q_estimator']['state_shape'],
            embed_dim=checkpoint.get('embed_dim', 256),
            lstm_hidden_size=checkpoint.get('lstm_hidden_size', 256),
            lstm_num_layers=checkpoint.get('lstm_num_layers', 1),
            mlp_layers=checkpoint.get('mlp_layers') or checkpoint['q_estimator'].get('mlp_layers', [256]),
            learning_rate=checkpoint['q_estimator']['learning_rate'],
            max_episodes=checkpoint['memory']['max_episodes'],
            replay_memory_init_size=checkpoint.get('replay_memory_init_size', 500),
            batch_size=checkpoint['batch_size'],
            seq_len=checkpoint.get('seq_len', 16),
            update_target_estimator_every=checkpoint['update_target_estimator_every'],
            discount_factor=checkpoint['discount_factor'],
            epsilon_start=checkpoint.get('epsilon_start', 0.1),
            epsilon_end=checkpoint.get('epsilon_end', 0.1),
            epsilon_decay_steps=checkpoint['epsilon_decay_steps'],
            train_every=checkpoint.get('train_every', 1),
            device=checkpoint.get('device'),
            save_path=checkpoint.get('save_path'),
            save_every=checkpoint.get('save_every', 5000),
        )
        agent.total_t = checkpoint['total_t']
        agent.train_t = checkpoint['train_t']
        agent.q_estimator = DRQNEstimator.from_checkpoint(checkpoint['q_estimator'])
        agent.target_estimator = deepcopy(agent.q_estimator)
        agent.memory = EpisodeMemory.from_checkpoint(checkpoint['memory'])
        return agent

    def save_checkpoint(self, path, filename='checkpoint_drqn.pt'):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.checkpoint_attributes(), os.path.join(path, filename))


# ---------------------------------------------------------------------------
# Lightweight Actor Agent (inference only, for multi-actor training)
# ---------------------------------------------------------------------------

class DRQNActorAgent:
    """Lightweight DRQN agent for actor processes — inference only.

    Uses a shared DRQNNetwork (whose parameters live in shared memory) to
    select actions via epsilon-greedy.  Does not own a replay buffer or
    optimizer.
    """

    def __init__(self, qnet, num_actions, device='cpu', exp_epsilon=0.05):
        self.use_raw = False
        self.qnet = qnet
        self.num_actions = num_actions
        self.device = torch.device(device) if isinstance(device, str) else device
        self.exp_epsilon = exp_epsilon
        self.hidden_states = {}

    def reset_hidden_states(self):
        self.hidden_states = {}

    def _predict(self, state):
        player_id = state['raw_obs']['current_player']
        obs = torch.from_numpy(state['obs']).float().unsqueeze(0).unsqueeze(0).to(self.device)
        hidden = self.hidden_states.get(player_id)
        with torch.no_grad():
            q_values, new_hidden = self.qnet(obs, hidden)
        self.hidden_states[player_id] = new_hidden
        q_np = q_values[0].cpu().numpy()
        masked = -np.inf * np.ones(self.num_actions, dtype=float)
        legal = list(state['legal_actions'].keys())
        masked[legal] = q_np[legal]
        return masked

    def step(self, state):
        """Epsilon-greedy action selection."""
        q_values = self._predict(state)
        legal_actions = list(state['legal_actions'].keys())
        if random.random() < self.exp_epsilon:
            return random.choice(legal_actions)
        best_action = np.argmax(q_values)
        if best_action in legal_actions:
            return best_action
        return random.choice(legal_actions)

    def eval_step(self, state):
        """Greedy action selection."""
        q_values = self._predict(state)
        best_action = np.argmax(q_values)
        legal_actions = list(state['legal_actions'].keys())
        if best_action not in legal_actions:
            best_action = legal_actions[0]
        info = {}
        info['values'] = {
            state['raw_legal_actions'][i]: float(
                q_values[legal_actions[i]]
            )
            for i in range(len(legal_actions))
        }
        return best_action, info
