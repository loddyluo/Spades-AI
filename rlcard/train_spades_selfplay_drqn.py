"""DRQN self-play training for Spades (multi-actor parallel).

Usage:
    python3 train_spades_selfplay_drqn.py

Configuration is read from ``model_config.yaml`` (keys: ``training``, ``drqn``).
Uses the multi-actor DRQNTrainer for high GPU utilisation.
"""

import os
import json

import rlcard
from rlcard.agents.drqn_trainer import DRQNTrainer
from rlcard.utils import set_seed


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
    REWARD_BETA = tcfg.get('reward_beta', 1.0)
    ENABLE_BLIND_NIL = tcfg.get('game_enable_blind_nil', True)

    set_seed(SEED)

    # --- DRQN-specific params ---
    dcfg = config.get('drqn', {})

    env_cfg = {
        'seed': SEED,
        'reward_beta': REWARD_BETA,
        'game_enable_blind_nil': ENABLE_BLIND_NIL,
    }

    # Create a reference env for metadata (num_actions, state_shape)
    env = rlcard.make(ENV_ID, config=env_cfg)

    trainer = DRQNTrainer(
        env=env,
        env_id=ENV_ID,
        env_config=env_cfg,
        # parallelism
        num_actors=dcfg.get('num_actors', 'auto'),
        total_frames=dcfg.get('total_frames', 5_000_000),
        save_interval=dcfg.get('save_interval', 30),
        save_dir=dcfg.get('save_path', 'experiments/spades_selfplay_drqn'),
        # actor
        exp_epsilon=dcfg.get('exp_epsilon', 0.05),
        # learner
        batch_size=dcfg.get('batch_size', 'auto'),
        sync_every=dcfg.get('sync_every', 100),
        train_steps_per_sync=dcfg.get('train_steps_per_sync', 16),
        max_grad_norm=dcfg.get('max_grad_norm', 40),
        learning_rate=dcfg.get('learning_rate', 0.00003),
        # DQN
        update_target_every=dcfg.get('update_target_every', 1000),
        discount_factor=dcfg.get('discount_factor', 0.99),
        # network
        embed_dim=dcfg.get('embed_dim', 256),
        lstm_hidden_size=dcfg.get('lstm_hidden_size', 256),
        lstm_num_layers=dcfg.get('lstm_num_layers', 1),
        mlp_layers=dcfg.get('mlp_layers', [256]),
        # memory
        max_episodes=dcfg.get('max_episodes', 8000),
        seq_len=dcfg.get('seq_len', 16),
        # misc
        cuda=dcfg.get('cuda', 'auto'),
        gpu_fraction=dcfg.get('gpu_fraction', 0.8),
        # opponent pool
        opponent_pool_size=dcfg.get('opponent_pool_size', 10),
        opponent_update_every=dcfg.get('opponent_update_every', 500),
        # evaluation
        eval_every_frames=dcfg.get('eval_every_frames', 100_000),
        eval_num_games=dcfg.get('eval_num_games', 50),
        # epsilon schedule
        epsilon_start=dcfg.get('epsilon_start', 0.5),
        epsilon_decay_games=dcfg.get('epsilon_decay_games', 5000),
        # bid shaping schedule
        bid_shaping_decay_games=dcfg.get('bid_shaping_decay_games', 20000),
    )

    trainer.start()


if __name__ == '__main__':
    train()
