import os
import json
import rlcard
from rlcard.agents import CFRAgent, RandomAgent
from rlcard.utils import set_seed, tournament, Logger, plot_curve


def _load_config():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(repo_root, 'model_config.yaml')
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    try:
        import yaml  # optional dependency
        return yaml.safe_load(content) or {}
    except Exception:
        return json.loads(content)


def _ensure_step_back_supported(env):
    if not env.allow_step_back:
        raise RuntimeError('CFR requires allow_step_back=True when creating the environment.')
    if not hasattr(env.game, 'step_back'):
        raise NotImplementedError(
            'Spades game does not implement step_back. '
            'CFR cannot run until step_back is supported in rlcard/games/spades/game.py.'
        )


def train():
    config = _load_config()
    cfr_cfg = config.get('cfr', {})
    seed = cfr_cfg.get('seed', config.get('training', {}).get('seed', 42))
    reward_beta = config.get('training', {}).get('reward_beta', 1.0)
    enable_blind_nil = config.get('training', {}).get('game_enable_blind_nil', True)

    log_dir = cfr_cfg.get('log_dir', 'experiments/spades_cfr_result')
    num_episodes = cfr_cfg.get('num_episodes', 5000)
    num_eval_games = cfr_cfg.get('num_eval_games', 200)
    evaluate_every = cfr_cfg.get('evaluate_every', 100)
    max_nodes_per_iter = cfr_cfg.get('max_nodes_per_iter')
    max_seconds_per_iter = cfr_cfg.get('max_seconds_per_iter')

    set_seed(seed)

    env = rlcard.make(
        'spades',
        config={
            'seed': seed,
            'allow_step_back': True,
            'reward_beta': reward_beta,
            'game_enable_blind_nil': enable_blind_nil,
        },
    )
    _ensure_step_back_supported(env)

    eval_env = rlcard.make(
        'spades',
        config={'seed': seed, 'reward_beta': reward_beta, 'game_enable_blind_nil': enable_blind_nil},
    )

    agent = CFRAgent(
        env,
        os.path.join(log_dir, 'cfr_model'),
        max_nodes_per_iter=max_nodes_per_iter,
        max_seconds_per_iter=max_seconds_per_iter,
    )
    agent.load()

    eval_env.set_agents([
        agent,
        RandomAgent(num_actions=env.num_actions),
        agent,
        RandomAgent(num_actions=env.num_actions),
    ])

    with Logger(log_dir) as logger:
        for episode in range(num_episodes):
            agent.train()
            if episode % evaluate_every == 0:
                agent.save()
                rewards = tournament(eval_env, num_eval_games)
                eval_reward = rewards[0]
                logger.log_performance(episode, eval_reward)
                print(f'Episode: {episode}, Team Reward vs Random: {eval_reward}')

        csv_path, fig_path = logger.csv_path, logger.fig_path
        plot_curve(csv_path, fig_path, 'CFR_Spades')


if __name__ == '__main__':
    train()
