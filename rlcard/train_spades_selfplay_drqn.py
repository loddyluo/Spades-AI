"""DRQN self-play training for Spades.

Usage:
    python3 train_spades_selfplay_drqn.py

Configuration is read from ``model_config.yaml`` (keys: ``training``, ``drqn``).
"""

import os
import json
import time
import torch

import rlcard
from rlcard.agents.drqn_agent import DRQNAgent
from rlcard.utils import get_device, set_seed, reorganize, Logger, plot_curve


def _load_config():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(repo_root, 'model_config.yaml')
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    try:
        import yaml
        return yaml.safe_load(content) or {}
    except Exception:
        return json.loads(content)


def train():
    config = _load_config()

    # --- global training params ---
    tcfg = config.get('training', {})
    ENV_ID = 'spades'
    SEED = tcfg.get('seed', 42)
    NUM_EPISODES = tcfg.get('num_episodes', 200000)
    MAX_HOURS = tcfg.get('max_hours')
    max_seconds = MAX_HOURS * 3600 if MAX_HOURS is not None else None
    EVALUATE_EVERY = tcfg.get('evaluate_every', 100)
    REWARD_BETA = tcfg.get('reward_beta', 1.0)
    ENABLE_BLIND_NIL = tcfg.get('game_enable_blind_nil', True)
    OPPONENT_UPDATE_EVERY = tcfg.get('opponent_update_every', 500)

    # --- DRQN-specific params ---
    dcfg = config.get('drqn', {})
    SAVE_PATH = dcfg.get('save_path', 'experiments/spades_selfplay_drqn')
    lstm_hidden_size = dcfg.get('lstm_hidden_size', 256)
    lstm_num_layers = dcfg.get('lstm_num_layers', 1)
    seq_len = dcfg.get('seq_len', 16)
    embed_dim = dcfg.get('embed_dim', 256)
    mlp_layers = dcfg.get('mlp_layers', [256])
    max_episodes = dcfg.get('max_episodes', 5000)
    batch_size = dcfg.get('batch_size', 32)
    learning_rate = dcfg.get('learning_rate', 0.00005)
    replay_memory_init_size = dcfg.get('replay_memory_init_size', 500)
    save_every = dcfg.get('save_every', 5000)

    device = get_device()
    set_seed(SEED)

    # --- environments ---
    env_cfg = {
        'seed': SEED,
        'reward_beta': REWARD_BETA,
        'game_enable_blind_nil': ENABLE_BLIND_NIL,
    }
    train_env = rlcard.make(ENV_ID, config=env_cfg)
    eval_env = rlcard.make(ENV_ID, config=env_cfg)

    # --- agent ---
    agent = DRQNAgent(
        num_actions=train_env.num_actions,
        state_shape=train_env.state_shape[0][0],
        embed_dim=embed_dim,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        mlp_layers=mlp_layers,
        learning_rate=learning_rate,
        max_episodes=max_episodes,
        replay_memory_init_size=replay_memory_init_size,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        save_path=SAVE_PATH,
        save_every=save_every,
    )

    train_env.set_agents([agent, agent, agent, agent])
    eval_env.set_agents([agent, agent, agent, agent])

    # --- opponent for evaluation ---
    opponent_agent = DRQNAgent(
        num_actions=train_env.num_actions,
        state_shape=train_env.state_shape[0][0],
        embed_dim=embed_dim,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        mlp_layers=mlp_layers,
        learning_rate=learning_rate,
        max_episodes=100,
        replay_memory_init_size=0,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )

    def _evaluate_raw_diff(env, num_games):
        total_diff = 0.0
        for _ in range(num_games):
            # Reset hidden states for all agents each game
            for ag in env.agents:
                if hasattr(ag, 'reset_hidden_states'):
                    ag.reset_hidden_states()
            env.run(is_training=False)
            raw_scores = env.game.judger.judge_game(env.game.players)
            total_diff += raw_scores[0] - raw_scores[1]
        return total_diff / float(num_games)

    print('Start DRQN Self-Play training...')
    print(f'  LSTM hidden={lstm_hidden_size}, layers={lstm_num_layers}, seq_len={seq_len}')
    print(f'  embed_dim={embed_dim}, mlp_layers={mlp_layers}')
    print(f'  device={device}')

    start_time = time.time()
    with Logger(SAVE_PATH) as logger:
        for episode in range(NUM_EPISODES):
            if max_seconds is not None and (time.time() - start_time) >= max_seconds:
                print(f'\nReached max_hours={MAX_HOURS}. Stopping training.')
                break

            # --- opponent checkpoint update ---
            if episode % OPPONENT_UPDATE_EVERY == 0:
                if not os.path.exists(SAVE_PATH):
                    os.makedirs(SAVE_PATH)
                agent.save_checkpoint(SAVE_PATH, filename='checkpoint_opponent.pt')
                ckpt_path = os.path.join(SAVE_PATH, 'checkpoint_opponent.pt')
                opponent_agent = DRQNAgent.from_checkpoint(
                    torch.load(ckpt_path, map_location=device, weights_only=False)
                )
                eval_env.set_agents([agent, opponent_agent, agent, opponent_agent])
                print(f'\nUpdated opponent with model from episode {episode}')

            # --- reset hidden states for this game ---
            agent.reset_hidden_states()

            # --- run one game ---
            trajectories, payoffs = train_env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)

            # --- feed per-player episodes sequentially ---
            for player_id in range(train_env.num_players):
                agent.feed_episode_start()
                for ts in trajectories[player_id]:
                    agent.feed(ts)

            # --- evaluation ---
            if episode % EVALUATE_EVERY == 0:
                eval_reward = _evaluate_raw_diff(eval_env, 20)
                logger.log_performance(episode, eval_reward)
                print(f'\nEpisode: {episode}, Agent Team Reward vs Previous Model: {eval_reward}')

        # --- save final model & plot ---
        agent.save_checkpoint(SAVE_PATH, filename='checkpoint_drqn.pt')
        csv_path, fig_path = logger.csv_path, logger.fig_path
        plot_curve(csv_path, fig_path, 'DRQN_SelfPlay')
        print(f'\nTraining finished. Logs saved to {SAVE_PATH}')


if __name__ == '__main__':
    train()
