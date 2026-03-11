"""Multi-actor DRQN trainer for RLCard.

Architecture::

    ┌──────────────────────────────────────────────┐
    │          Shared DRQNNetwork (CPU)             │
    │          (parameters in shared memory)        │
    └──────┬──────────┬──────────┬─────────────────┘
           │          │          │
      Actor 0    Actor 1   ...  Actor N   (separate processes)
      env.run    env.run        env.run   CPU game simulation
           │          │          │
           ▼          ▼          ▼
    ┌──────────────────────────────────────────────┐
    │         mp.Queue  (episode data)             │
    └─────────────────────┬────────────────────────┘
                          │
              ┌───────────▼────────────┐
              │    Learner (GPU)       │
              │  EpisodeMemory.sample  │
              │  Double DQN update     │
              │  Sync weights → shared │
              └────────────────────────┘

Modelled after the DMC trainer in ``rlcard/agents/dmc_agent/trainer.py``.
"""

import logging
import os
import random
import threading
import time
import timeit
import traceback
from collections import deque
from copy import deepcopy

import numpy as np
import torch
from torch import multiprocessing as mp
from torch import nn

import rlcard
from rlcard.utils import reorganize

from .drqn_agent import (
    DRQNActorAgent,
    DRQNEstimator,
    DRQNNetwork,
    EpisodeMemory,
    Transition,
)

log = logging.getLogger('drqn_trainer')
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] %(message)s'))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Actor process
# ---------------------------------------------------------------------------

def drqn_act(
    actor_id,
    shared_qnet,
    opponent_qnet,
    episode_queue,
    env_id,
    env_config,
    num_actions,
    device,
    exp_epsilon,
    stop_event,
    epsilon_start=0.5,
    epsilon_decay_games=5000,
    bid_shaping_decay_games=20000,
):
    """Actor process: run games endlessly, push episode data into *episode_queue*."""
    try:
        log.info('Actor %d started (device=%s)', actor_id, device)
        env = rlcard.make(env_id, config=env_config)
        env.seed(actor_id * 1000)

        # Learning agent (Team 0: P0, P2)
        agent = DRQNActorAgent(shared_qnet, num_actions, device=device,
                               exp_epsilon=epsilon_start)
        # Opponent agent (Team 1: P1, P3) — uses frozen weights from pool
        opp_agent = DRQNActorAgent(opponent_qnet, num_actions, device=device,
                                   exp_epsilon=epsilon_start)
        env.set_agents([agent, opp_agent, agent, opp_agent])

        games_played = 0
        while not stop_event.is_set():
            # Epsilon decay
            frac = min(1.0, games_played / max(epsilon_decay_games, 1))
            current_epsilon = epsilon_start + (exp_epsilon - epsilon_start) * frac
            agent.exp_epsilon = current_epsilon
            opp_agent.exp_epsilon = current_epsilon

            agent.reset_hidden_states()
            opp_agent.reset_hidden_states()
            trajectories, payoffs = env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)

            # ---- Reward shaping: inject per-trick intermediate rewards ----
            _TRICK_REWARD = 1.0
            _NIL_BREAK_BONUS = 2.0
            _NIL_TAKE_TRICK_PENALTY = -3.0

            for pid in [0, 2]:  # Only learning agents (Team 0)
                teammate_id = (pid + 2) % env.num_players
                opp_players = [1 - pid % 2, 1 - pid % 2 + 2]
                for i, ts in enumerate(trajectories[pid]):
                    if i < len(trajectories[pid]) - 1:  # non-terminal step
                        curr_tw = ts[0]['raw_obs']['tricks_won']
                        next_tw = ts[3]['raw_obs']['tricks_won']

                        pid_is_nil = (
                            ts[0]['raw_obs'].get('is_nil', [False]*4)[pid]
                            or ts[0]['raw_obs'].get('is_blind_nil', [False]*4)[pid])
                        tm_is_nil = (
                            ts[0]['raw_obs'].get('is_nil', [False]*4)[teammate_id]
                            or ts[0]['raw_obs'].get('is_blind_nil', [False]*4)[teammate_id])

                        shaped = 0.0
                        if pid_is_nil:
                            # Nil bidder: penalise own tricks
                            my_gain = next_tw[pid] - curr_tw[pid]
                            if my_gain > 0:
                                shaped += my_gain * _NIL_TAKE_TRICK_PENALTY
                            # Reward non-nil teammate's tricks
                            if not tm_is_nil:
                                shaped += (next_tw[teammate_id] - curr_tw[teammate_id]) * _TRICK_REWARD
                        else:
                            # Normal bidder: overtrick-aware trick reward
                            my_gain = next_tw[pid] - curr_tw[pid]
                            tm_gain = (next_tw[teammate_id] - curr_tw[teammate_id]
                                       if not tm_is_nil else 0)

                            # Compute team bid (non-nil members only)
                            bids = ts[0]['raw_obs']['bids']
                            _is_nil = ts[0]['raw_obs'].get('is_nil', [False]*4)
                            _is_bnil = ts[0]['raw_obs'].get('is_blind_nil', [False]*4)
                            team_bid = 0
                            for p_idx in [pid, teammate_id]:
                                if not (_is_nil[p_idx] or _is_bnil[p_idx]):
                                    b = bids[p_idx]
                                    if b is not None:
                                        team_bid += b

                            team_tricks = curr_tw[pid] + curr_tw[teammate_id]
                            if team_bid > 0 and team_tricks >= team_bid:
                                # Already met bid — overtricks are harmful
                                shaped += my_gain * (-0.5)
                                shaped += tm_gain * (-0.5)
                            else:
                                # Still need tricks to make bid
                                shaped += my_gain * _TRICK_REWARD
                                shaped += tm_gain * _TRICK_REWARD

                        # Bonus for breaking opponent nil
                        for opp in opp_players:
                            opp_nil = (
                                ts[0]['raw_obs'].get('is_nil', [False]*4)[opp]
                                or ts[0]['raw_obs'].get('is_blind_nil', [False]*4)[opp])
                            if opp_nil and curr_tw[opp] == 0 and next_tw[opp] > 0:
                                shaped += _NIL_BREAK_BONUS

                        ts[2] = shaped

            # ---- Bid-quality shaped reward (cosine annealing to 0) ----
            import math
            if bid_shaping_decay_games > 0 and games_played < bid_shaping_decay_games:
                _BID_SHAPING_SCALE = 0.5 * (1 + math.cos(math.pi * games_played / bid_shaping_decay_games)) / 2
            else:
                _BID_SHAPING_SCALE = 0.0

            def _hand_strength(hand_cards):
                """Hand-strength estimator v4.2: 48-dim features with Nil pre-check + NIL_RAW_THRESHOLD.

                Returns (bid: int, raw_strength: float).
                bid is the final recommended bid (0 for Nil hands).
                raw_strength is the linear model output before Nil override.
                """
                _RANK_VAL = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                             '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
                _ALL_SPADE_VALS = set(range(2, 15))
                _MU_SP = 0.6151
                _NIL_RAW_THRESHOLD = 2  # If raw linear bid >= this, don't risk Nil

                # v4.2 optimized coefficients (48-dim)
                _W = [
                    0.701044,   # [0]  sp_tricks_c
                    0.007511,   # [1]  sp_tricks_hi
                    0.881980,   # [2]  sp_tricks_ex
                    -1.999145,  # [3]  sp_0
                    -1.273403,  # [4]  sp_1
                    -0.883269,  # [5]  sp_2
                    0.558984,   # [6]  sp_5
                    0.429923,   # [7]  sp_6
                    0.266541,   # [8]  sp_7p
                    0.370726,   # [9]  short_void
                    0.488241,   # [10] short_1
                    0.220489,   # [11] short_2
                    1.476624,   # [12] bias
                    2.203615, 1.432062, 0.920113, 0.806798, 0.785587, 0.753077, -0.033648,  # d1
                    1.882863, 1.416556, 1.157554, 0.515139, 0.888351, 0.441180, 0.298371,   # d2
                    1.877064, 1.811177, 1.169807, 0.934469, 0.978589, 0.675725, 0.260065,   # d34
                    1.793521, 1.739455, 1.214013, 0.806248, 0.831830, 0.556648, 0.019048,   # d57
                    2.081140, 0.738415, 0.667451, 1.104983, 0.265381, -0.123201, -0.185711, # d8p
                ]

                suits = {'S': set(), 'H': set(), 'D': set(), 'C': set()}
                for c in hand_cards:
                    suits[c[0]].add(c[1])

                # --- Nil pre-check ---
                _nil_ok = False
                sp_count = len(suits['S'])
                if sp_count <= 3:
                    _nil_ok = True
                    # Rule 1a: no big spades (A/K/Q)
                    if any(r in suits['S'] for r in ['A', 'K', 'Q']):
                        _nil_ok = False
                    # Rule 1c: spade high cards (9/T/J) at most 1
                    if _nil_ok:
                        sp_vals = [_RANK_VAL[r] for r in suits['S']]
                        if sum(1 for v in sp_vals if v >= 9) >= 2:
                            _nil_ok = False
                    if _nil_ok:
                        for suit in ['H', 'D', 'C']:
                            vals = sorted([_RANK_VAL[r] for r in suits[suit]])
                            d = len(vals)
                            if d == 0:
                                continue
                            if d <= 3:
                                if d <= 2:
                                    if any(v >= 12 for v in vals):
                                        _nil_ok = False; break
                                else:
                                    if any(v >= 13 for v in vals):
                                        _nil_ok = False; break
                                if any(v >= 12 for v in vals):
                                    _nil_ok = False; break
                                if sum(1 for v in vals if v >= 9) >= 2:
                                    _nil_ok = False; break
                            else:
                                if vals[0] > 5 or vals[1] > 9:
                                    _nil_ok = False; break

                # --- Spade trick estimate ---
                my_vals = sorted([_RANK_VAL[r] for r in suits['S']], reverse=True)
                opp_vals = sorted(_ALL_SPADE_VALS - set(my_vals), reverse=True)
                sp_tricks = 0
                oi = 0
                for mv in my_vals:
                    if oi < len(opp_vals):
                        if mv > opp_vals[oi]:
                            sp_tricks += 1
                        oi += 1
                    else:
                        sp_tricks += 1

                # --- Build 48-dim feature vector and compute strength ---
                strength = _W[12]  # bias

                # [0] mean-centered spade tricks
                strength += _W[0] * (sp_tricks - _MU_SP)
                # [1] piecewise-linear above 5
                if sp_tricks > 5:
                    strength += _W[1] * (sp_tricks - 5)
                # [2] piecewise-linear above 8
                if sp_tricks > 8:
                    strength += _W[2] * (sp_tricks - 8)

                # [3-8] spade length one-hot (3-4 is baseline)
                sp_len = len(suits['S'])
                if sp_len == 0:
                    strength += _W[3]
                elif sp_len == 1:
                    strength += _W[4]
                elif sp_len == 2:
                    strength += _W[5]
                elif sp_len == 5:
                    strength += _W[6]
                elif sp_len == 6:
                    strength += _W[7]
                elif sp_len >= 7:
                    strength += _W[8]

                # [9-11] shortest side suit
                side_lengths = [len(suits[s]) for s in ['H', 'D', 'C']]
                shortest = min(side_lengths)
                if shortest == 0:
                    strength += _W[9]
                elif shortest == 1:
                    strength += _W[10]
                elif shortest == 2:
                    strength += _W[11]

                # [13-47] Side suits: honour x length-bucket
                _SIDE_BASE = 13
                for suit in ['H', 'D', 'C']:
                    cards = suits[suit]
                    d = len(cards)
                    if d == 0:
                        continue
                    has_A, has_K, has_Q = 'A' in cards, 'K' in cards, 'Q' in cards
                    if has_A and has_K and has_Q:   hon = 0
                    elif has_A and has_K:           hon = 1
                    elif has_A and has_Q:           hon = 2
                    elif has_K and has_Q:           hon = 3
                    elif has_A:                     hon = 4
                    elif has_K:                     hon = 5
                    elif has_Q:                     hon = 6
                    else:
                        continue
                    if d == 1:     bucket = 0
                    elif d == 2:   bucket = 1
                    elif d <= 4:   bucket = 2
                    elif d <= 7:   bucket = 3
                    else:          bucket = 4
                    strength += _W[_SIDE_BASE + bucket * 7 + hon]

                raw = strength
                # Nil override: rule-based Nil AND raw < threshold
                if _nil_ok and raw < _NIL_RAW_THRESHOLD:
                    return 0, raw
                return int(max(0, min(13, round(raw)))), raw

            for pid in [0, 2]:  # learning players only
                for i, ts in enumerate(trajectories[pid]):
                    raw_obs = ts[0]['raw_obs']
                    if raw_obs['phase'] == 0 and raw_obs['bids'][pid] is None:
                        action_str = env.actions[ts[1]] if ts[1] < len(env.actions) else None
                        bid_reward = 0.0
                        if action_str and action_str.startswith('bid_'):
                            bid_val = int(action_str.split('_')[1])
                            recommended, _ = _hand_strength(raw_obs.get('hand', []))
                            diff = abs(bid_val - recommended)
                            # Adaptive threshold: low estimate (<=4) → tol=1, high (>=5) → tol=2
                            tol = 1 if recommended <= 4 else 2
                            excess = max(0, diff - tol)
                            if diff == 0:
                                # Exact match: small bonus
                                bid_reward = _BID_SHAPING_SCALE * 0.3
                            elif excess == 0:
                                # Within tolerance: no penalty
                                bid_reward = 0.0
                            else:
                                # Beyond tolerance: exponential penalty
                                bid_reward = _BID_SHAPING_SCALE * -(2 ** excess)
                        elif action_str == 'nil':
                            recommended, _ = _hand_strength(raw_obs.get('hand', []))
                            if recommended == 0:
                                # Estimator also says Nil: good
                                bid_reward = _BID_SHAPING_SCALE * 0.3
                            elif recommended <= 2:
                                # Borderline: no penalty
                                bid_reward = 0.0
                            else:
                                # Strong hand bidding nil: exponential penalty
                                excess = recommended - 2
                                bid_reward = _BID_SHAPING_SCALE * -(2 ** excess)
                        # blind_nil / pass: bid_reward stays 0
                        ts[2] = ts[2] + bid_reward

            # ---- End-of-round bid accuracy bonus (terminal transition) ----
            _BID_ACCURACY_SCALE = 2.0
            for pid in [0, 2]:
                traj = trajectories[pid]
                if not traj:
                    continue
                terminal_ts = traj[-1]
                final_raw = terminal_ts[3]['raw_obs']
                teammate_id = (pid + 2) % 4

                _is_nil = final_raw.get('is_nil', [False] * 4)
                _is_bnil = final_raw.get('is_blind_nil', [False] * 4)
                if _is_nil[pid] or _is_bnil[pid]:
                    continue  # Nil players: skip (already handled by nil rewards)

                player_bid = final_raw['bids'][pid]
                if player_bid is None:
                    continue

                # Compute team totals
                team_bid = 0
                for p_idx in [pid, teammate_id]:
                    if not (_is_nil[p_idx] or _is_bnil[p_idx]):
                        b = final_raw['bids'][p_idx]
                        if b is not None:
                            team_bid += b
                team_tricks = (final_raw['tricks_won'][pid]
                               + final_raw['tricks_won'][teammate_id])

                if team_bid > 0:
                    diff = team_tricks - team_bid
                    if diff == 0:       # Perfect bid
                        terminal_ts[2] += _BID_ACCURACY_SCALE * 2.0
                    elif diff == 1:     # 1 overtrick
                        terminal_ts[2] += _BID_ACCURACY_SCALE * 0.5
                    elif diff < 0:      # Set (failed bid)
                        terminal_ts[2] += _BID_ACCURACY_SCALE * -1.0
                    else:               # Multiple overtricks
                        terminal_ts[2] += _BID_ACCURACY_SCALE * (-0.3 * diff)

            # Only push learning agent's trajectories (Team 0: P0, P2)
            for player_id in [0, 2]:
                episode_data = []
                for ts in trajectories[player_id]:
                    state, action, reward, next_state, done = tuple(ts)
                    episode_data.append((
                        state['obs'].copy(),
                        action,
                        reward,
                        next_state['obs'].copy(),
                        list(next_state['legal_actions'].keys()),
                        done,
                    ))
                if episode_data:
                    episode_queue.put(episode_data)

            games_played += 1
        pass
    except Exception:
        log.error('Exception in actor %d', actor_id)
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DRQNTrainer:
    """Multi-actor distributed DRQN trainer.

    Args:
        env: RLCard environment (used for metadata; actors create their own).
        num_actors: Number of actor processes.
        total_frames: Stop after this many training frames.
        save_interval: Minutes between checkpoint saves.
        save_dir: Directory for checkpoints.
        exp_epsilon: Actor exploration rate.
        batch_size: Learner mini-batch size.
        sync_every: Sync learner weights to shared model every N train steps.
        train_steps_per_sync: Continuous train steps between syncs (GPU saturation).
        max_grad_norm: Gradient clipping.
        Network / memory parameters forwarded to DRQNEstimator / EpisodeMemory.
    """

    def __init__(
        self,
        env,
        env_id='spades',
        env_config=None,
        # parallelism  (int or "auto")
        num_actors=8,
        total_frames=5_000_000,
        save_interval=30,
        save_dir='experiments/spades_selfplay_drqn',
        # actor
        exp_epsilon=0.05,
        # learner  (int or "auto")
        batch_size=256,
        sync_every=100,
        train_steps_per_sync=16,
        max_grad_norm=40,
        learning_rate=0.00003,
        # DQN
        update_target_every=1000,
        discount_factor=0.99,
        # network
        embed_dim=256,
        lstm_hidden_size=256,
        lstm_num_layers=1,
        mlp_layers=None,
        # memory
        max_episodes=8000,
        seq_len=16,
        # misc
        cuda='auto',
        gpu_fraction=0.8,
        # opponent pool
        opponent_pool_size=10,
        opponent_update_every=500,
        # evaluation
        eval_every_frames=100_000,
        eval_num_games=50,
        # epsilon schedule
        epsilon_start=0.5,
        epsilon_decay_games=5000,
        # bid shaping schedule
        bid_shaping_decay_games=20000,
    ):
        self.env = env
        self.env_id = env_id
        self.env_config = env_config or {}
        self.total_frames = total_frames
        self.save_interval = save_interval
        self.save_dir = save_dir
        self.exp_epsilon = exp_epsilon
        self.sync_every = sync_every
        self.train_steps_per_sync = train_steps_per_sync
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.update_target_every = update_target_every
        self.discount_factor = discount_factor
        self.embed_dim = embed_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.mlp_layers = mlp_layers or [256]
        self.max_episodes = max_episodes
        self.seq_len = seq_len
        self.gpu_fraction = gpu_fraction
        self.opponent_pool_size = opponent_pool_size
        self.opponent_update_every = opponent_update_every
        self.eval_every_frames = eval_every_frames
        self.eval_num_games = eval_num_games
        self.epsilon_start = epsilon_start
        self.epsilon_decay_games = epsilon_decay_games
        self.bid_shaping_decay_games = bid_shaping_decay_games

        # Handle "auto" for batch_size and num_actors
        self._auto_batch = (batch_size == 'auto')
        self._auto_actors = (num_actors == 'auto')
        self.batch_size = 256 if self._auto_batch else int(batch_size)
        self.num_actors = 4 if self._auto_actors else int(num_actors)

        self.num_actions = env.num_actions
        self.state_shape = env.state_shape[0][0]

        # Device
        if cuda == 'auto':
            self.training_device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        elif cuda:
            self.training_device = torch.device('cuda:' + str(cuda))
        else:
            self.training_device = torch.device('cpu')

        # Actors always run on CPU
        self.actor_device = 'cpu'

    def _build_network(self):
        return DRQNNetwork(
            num_actions=self.num_actions,
            state_shape=self.state_shape,
            embed_dim=self.embed_dim,
            lstm_hidden_size=self.lstm_hidden_size,
            lstm_num_layers=self.lstm_num_layers,
            mlp_layers=self.mlp_layers,
        )

    def auto_configure(self):
        """Auto-detect GPU memory and CPU cores, compute optimal batch_size and num_actors."""

        # --- CPU: auto num_actors ---
        if self._auto_actors:
            cpu_count = os.cpu_count() or 4
            self.num_actors = max(1, cpu_count - 2)
            log.info('Auto-configured: num_actors=%d (CPU cores=%d)',
                     self.num_actors, cpu_count)

        # --- GPU: auto batch_size ---
        if not self._auto_batch:
            return

        if not torch.cuda.is_available():
            self.batch_size = 64
            log.info('No CUDA available, using batch_size=%d', self.batch_size)
            return

        device = self.training_device
        device_idx = device.index if device.index is not None else 0
        # Force CUDA context initialisation before querying memory stats
        torch.cuda.set_device(device_idx)
        torch.zeros(1, device=device)
        torch.cuda.reset_peak_memory_stats(device_idx)
        torch.cuda.empty_cache()

        # 1. Allocate models + optimizer (fixed overhead)
        learner_qnet = self._build_network().to(device)
        target_qnet = self._build_network().to(device)
        optimizer = torch.optim.Adam(learner_qnet.parameters(), lr=self.learning_rate)
        torch.cuda.synchronize(device_idx)
        fixed_mem = torch.cuda.memory_allocated(device_idx)

        # 2. Profile: forward + backward with a small probe batch
        probe_batch = 32
        dummy_s = torch.randn(probe_batch, self.seq_len, self.state_shape,
                              device=device)
        dummy_a = torch.randint(0, self.num_actions, (probe_batch,), device=device)
        dummy_y = torch.randn(probe_batch, device=device)

        torch.cuda.reset_peak_memory_stats(device_idx)
        optimizer.zero_grad()
        learner_qnet.train()
        q_all, _ = learner_qnet(dummy_s)
        q = torch.gather(q_all, -1, dummy_a.unsqueeze(-1)).squeeze(-1)
        loss = ((q - dummy_y) ** 2).mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device_idx)
        learner_qnet.eval()

        peak_mem = torch.cuda.max_memory_allocated(device_idx)
        batch_mem = peak_mem - fixed_mem
        per_sample_mem = max(batch_mem / probe_batch, 1)

        # 3. Cleanup profiling allocations
        del learner_qnet, target_qnet, optimizer
        del dummy_s, dummy_a, dummy_y, q_all, q, loss
        torch.cuda.empty_cache()

        # 4. Compute optimal batch_size
        _, total_mem = torch.cuda.mem_get_info(device_idx)
        target_mem = total_mem * self.gpu_fraction
        available_for_batch = target_mem - fixed_mem
        optimal_batch = int(available_for_batch / per_sample_mem)

        # Align to multiples of 32, clamp to [32, 8192]
        optimal_batch = max(32, (optimal_batch // 32) * 32)
        optimal_batch = min(optimal_batch, 8192)
        self.batch_size = optimal_batch

        log.info(
            'Auto-configured: batch_size=%d '
            '(per_sample=%.3f MB, fixed=%.1f MB, target %.0f%% of %.0f MB GPU)',
            self.batch_size, per_sample_mem / 1024**2, fixed_mem / 1024**2,
            self.gpu_fraction * 100, total_mem / 1024**2,
        )

    def _evaluate(self, learner_qnet):
        """Run evaluation games: current model vs RandomAgent.

        Returns:
            avg_diff: average (team0_raw_score - team1_raw_score)
            win_rate: fraction of games where team0 wins
        """
        from rlcard.agents.random_agent import RandomAgent

        eval_env = rlcard.make(self.env_id, config=self.env_config)

        eval_qnet = self._build_network()
        eval_qnet.load_state_dict(learner_qnet.state_dict())
        eval_qnet.eval()

        eval_agent = DRQNActorAgent(eval_qnet, self.num_actions,
                                    device='cpu', exp_epsilon=0.0)
        random_agent = RandomAgent(num_actions=self.num_actions)

        eval_env.set_agents([eval_agent, random_agent,
                             eval_agent, random_agent])

        total_diff = 0.0
        wins = 0
        for _ in range(self.eval_num_games):
            eval_agent.reset_hidden_states()
            eval_env.run(is_training=False)
            raw_scores = eval_env.game.judger.judge_game(eval_env.game.players)
            diff = raw_scores[0] - raw_scores[1]
            total_diff += diff
            if diff > 0:
                wins += 1

        avg_diff = total_diff / max(self.eval_num_games, 1)
        win_rate = wins / max(self.eval_num_games, 1)
        return avg_diff, win_rate

    def start(self):
        if self._auto_batch or self._auto_actors:
            self.auto_configure()

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # ----- shared model (CPU, shared memory) for actors -----
        shared_qnet = self._build_network()
        shared_qnet.eval()
        shared_qnet.share_memory()

        # ----- opponent model (CPU, shared memory) for frozen opponents -----
        opponent_qnet = self._build_network()
        opponent_qnet.load_state_dict(shared_qnet.state_dict())
        opponent_qnet.eval()
        opponent_qnet.share_memory()

        # Opponent pool: list of CPU state_dict snapshots
        opponent_pool = [
            {k: v.cpu().clone() for k, v in shared_qnet.state_dict().items()}
        ]
        # Seed pool with randomly-initialised networks for early diversity
        for _ in range(2):
            rand_net = self._build_network()
            opponent_pool.append(
                {k: v.cpu().clone() for k, v in rand_net.state_dict().items()})
            del rand_net

        # ----- learner model (GPU) -----
        learner_qnet = self._build_network().to(self.training_device)
        learner_qnet.load_state_dict(shared_qnet.state_dict())
        learner_qnet.eval()

        target_qnet = deepcopy(learner_qnet)
        target_qnet.eval()

        optimizer = torch.optim.Adam(learner_qnet.parameters(), lr=self.learning_rate)
        mse_loss = nn.MSELoss(reduction='mean')

        memory = EpisodeMemory(
            max_episodes=self.max_episodes,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
        )

        # ----- queues and processes -----
        ctx = mp.get_context('spawn')
        episode_queue = ctx.Queue(maxsize=self.num_actors * 100)
        stop_event = ctx.Event()

        actor_processes = []
        for i in range(self.num_actors):
            p = ctx.Process(
                target=drqn_act,
                args=(
                    i,
                    shared_qnet,
                    opponent_qnet,
                    episode_queue,
                    self.env_id,
                    self.env_config,
                    self.num_actions,
                    self.actor_device,
                    self.exp_epsilon,
                    stop_event,
                    self.epsilon_start,
                    self.epsilon_decay_games,
                    self.bid_shaping_decay_games,
                ),
            )
            p.start()
            actor_processes.append(p)

        log.info('Started %d actor processes', self.num_actors)
        log.info('Training device: %s', self.training_device)
        log.info('Target total frames: %d', self.total_frames)

        # ----- learner loop -----
        frames = 0
        train_t = 0
        loss_buf = deque(maxlen=100)
        ep_count = 0

        timer = timeit.default_timer
        start_time = timer()
        last_save_time = start_time
        last_log_time = start_time

        # ----- evaluation CSV -----
        import csv
        eval_csv_path = os.path.join(self.save_dir, 'eval_performance.csv')
        eval_csv_file = open(eval_csv_path, 'w', newline='')
        eval_writer = csv.DictWriter(
            eval_csv_file,
            fieldnames=['frames', 'avg_score_diff', 'win_rate_vs_random'],
        )
        eval_writer.writeheader()
        last_eval_frames = 0

        def _drain_queue():
            """Pull all available episodes from the queue into memory."""
            nonlocal ep_count
            drained = 0
            while not episode_queue.empty():
                try:
                    ep_data = episode_queue.get_nowait()
                except Exception:
                    break
                memory.start_episode()
                for t in ep_data:
                    memory.save(*t)
                drained += 1
                ep_count += 1
            return drained

        def _train_step():
            """One Double-DQN gradient step on the learner."""
            nonlocal train_t
            (state_seqs, action_batch, reward_batch,
             next_state_seqs, done_batch, legal_actions_batch) = memory.sample()

            device = self.training_device

            # Build legal-action mask on GPU
            legal_mask = torch.full(
                (self.batch_size, self.num_actions), float('-inf'), device=device)
            for b in range(self.batch_size):
                legal_mask[b, legal_actions_batch[b]] = 0.0

            # --- Double DQN (all on GPU) ---
            with torch.no_grad():
                s_next = torch.from_numpy(next_state_seqs).float().to(device)

                # Q-network selects best next action
                q_next, _ = learner_qnet(s_next)
                q_next_masked = q_next + legal_mask
                best_next = q_next_masked.argmax(dim=1)

                # Target-network evaluates
                q_target_vals, _ = target_qnet(s_next)

                r = torch.from_numpy(reward_batch).float().to(device)
                d = torch.from_numpy(done_batch.astype(np.float32)).to(device)

                targets = r + (1.0 - d) * self.discount_factor * \
                    q_target_vals.gather(1, best_next.unsqueeze(1)).squeeze(1)

            # --- gradient step ---
            optimizer.zero_grad()
            learner_qnet.train()

            s = torch.from_numpy(state_seqs).float().to(device)
            a = torch.from_numpy(action_batch).long().to(device)
            q_all, _ = learner_qnet(s)
            q = torch.gather(q_all, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)
            loss = mse_loss(q, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(learner_qnet.parameters(), self.max_grad_norm)
            optimizer.step()

            learner_qnet.eval()
            loss_val = loss.item()
            loss_buf.append(loss_val)
            train_t += 1

            # Update target network
            if train_t % self.update_target_every == 0:
                target_qnet.load_state_dict(learner_qnet.state_dict())
                log.info('Updated target network at train step %d', train_t)

            return loss_val

        try:
            while frames < self.total_frames:
                # Drain episodes from actors
                _drain_queue()

                if not memory.can_sample():
                    time.sleep(0.1)
                    continue

                # Train multiple steps per sync (keep GPU busy)
                for _ in range(self.train_steps_per_sync):
                    if not memory.can_sample():
                        break
                    _train_step()
                    frames += self.batch_size

                # Sync weights to shared model
                if train_t % self.sync_every == 0:
                    shared_qnet.load_state_dict(learner_qnet.state_dict())

                # Update opponent pool periodically
                if (ep_count > 0
                        and ep_count % self.opponent_update_every == 0
                        and ep_count != getattr(self, '_last_opp_update', -1)):
                    self._last_opp_update = ep_count
                    snapshot = {k: v.cpu().clone()
                                for k, v in learner_qnet.state_dict().items()}
                    opponent_pool.append(snapshot)
                    if len(opponent_pool) > self.opponent_pool_size:
                        opponent_pool.pop(0)
                    chosen = random.choice(opponent_pool)
                    opponent_qnet.load_state_dict(chosen)
                    log.info('Updated opponent pool (%d models), '
                             'synced random opponent at ep %d',
                             len(opponent_pool), ep_count)

                # Logging
                now = timer()
                if now - last_log_time > 10:
                    elapsed = now - start_time
                    fps = frames / max(elapsed, 1)
                    avg_loss = np.mean(loss_buf) if loss_buf else 0
                    log.info(
                        'Frames %d | FPS %.1f | Train steps %d | Episodes %d | '
                        'Avg loss %.6f | Memory %d eps',
                        frames, fps, train_t, ep_count, avg_loss,
                        len(memory.episodes),
                    )
                    last_log_time = now

                # Checkpoint
                if now - last_save_time > self.save_interval * 60:
                    self._save_checkpoint(learner_qnet, optimizer, frames, train_t)
                    last_save_time = now

                # Periodic evaluation (async — does not block learner)
                if frames - last_eval_frames >= self.eval_every_frames:
                    eval_frames_snapshot = frames
                    def _eval_bg(qnet, f, writer, csvfile):
                        try:
                            avg_diff, win_rate = self._evaluate(qnet)
                            log.info(
                                'EVAL @ %d frames: avg_score_diff=%.1f, '
                                'win_rate_vs_random=%.2f',
                                f, avg_diff, win_rate,
                            )
                            writer.writerow({
                                'frames': f,
                                'avg_score_diff': f'{avg_diff:.2f}',
                                'win_rate_vs_random': f'{win_rate:.4f}',
                            })
                            csvfile.flush()
                        except Exception:
                            log.error('Eval thread failed')
                            traceback.print_exc()
                    threading.Thread(
                        target=_eval_bg,
                        args=(learner_qnet, eval_frames_snapshot,
                              eval_writer, eval_csv_file),
                        daemon=True,
                    ).start()
                    last_eval_frames = frames

        except KeyboardInterrupt:
            log.info('Training interrupted by user.')
        finally:
            # Final save
            self._save_checkpoint(learner_qnet, optimizer, frames, train_t)
            eval_csv_file.close()

            # Stop actors
            stop_event.set()
            for p in actor_processes:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()

            log.info('Training finished after %d frames, %d train steps.', frames, train_t)

    def _save_checkpoint(self, learner_qnet, optimizer, frames, train_t):
        ckpt_path = os.path.join(self.save_dir, 'checkpoint_drqn.pt')
        torch.save({
            'agent_type': 'DRQNAgent',
            'qnet': learner_qnet.state_dict(),
            'optimizer': optimizer.state_dict(),
            'frames': frames,
            'train_t': train_t,
            # Network config (for reconstruction)
            'num_actions': self.num_actions,
            'state_shape': self.state_shape,
            'embed_dim': self.embed_dim,
            'lstm_hidden_size': self.lstm_hidden_size,
            'lstm_num_layers': self.lstm_num_layers,
            'mlp_layers': self.mlp_layers,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'seq_len': self.seq_len,
            'discount_factor': self.discount_factor,
            'update_target_every': self.update_target_every,
        }, ckpt_path)
        log.info('Saved checkpoint to %s (frames=%d)', ckpt_path, frames)
