import os
import json
import time
import torch
import rlcard
from rlcard.agents import NFSPAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)


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
        # Accept JSON content in .yaml (valid YAML)
        return json.loads(content)


def train():
    # Configuration
    config = _load_config()
    ENV_ID = 'spades'
    SEED = config.get('training', {}).get('seed', 42)
    NUM_EPISODES = config.get('training', {}).get('num_episodes', 2000)
    MAX_HOURS = config.get('training', {}).get('max_hours')
    max_seconds = MAX_HOURS * 3600 if MAX_HOURS is not None else None
    EVALUATE_EVERY = config.get('training', {}).get('evaluate_every', 100)
    SAVE_PATH = config.get('training', {}).get('save_path', 'experiments/spades_selfplay_nfsp')
    REWARD_BETA = config.get('training', {}).get('reward_beta', 1.0)
    OPPONENT_UPDATE_EVERY = config.get('training', {}).get('opponent_update_every', 500)

    nfsp_cfg = config.get('nfsp', {})

    # Check device
    device = get_device()

    # Seed
    set_seed(SEED)

    # 1. Make Training Environment
    train_env = rlcard.make(ENV_ID, config={'seed': SEED, 'reward_beta': REWARD_BETA})

    # 2. Make Evaluation Environment
    eval_env = rlcard.make(ENV_ID, config={'seed': SEED, 'reward_beta': REWARD_BETA})

    # 3. Initialize NFSP Agent
    hidden_layers_sizes = nfsp_cfg.get('hidden_layers_sizes', [256, 256])
    q_mlp_layers = nfsp_cfg.get('q_mlp_layers', [256, 256])

    agent = NFSPAgent(
        num_actions=train_env.num_actions,
        state_shape=train_env.state_shape[0][0],
        hidden_layers_sizes=hidden_layers_sizes,
        q_mlp_layers=q_mlp_layers,
        reservoir_buffer_capacity=nfsp_cfg.get('reservoir_buffer_capacity', 20000),
        anticipatory_param=nfsp_cfg.get('anticipatory_param', 0.1),
        batch_size=nfsp_cfg.get('batch_size', 256),
        train_every=nfsp_cfg.get('train_every', 1),
        rl_learning_rate=nfsp_cfg.get('rl_learning_rate', 0.00005),
        sl_learning_rate=nfsp_cfg.get('sl_learning_rate', 0.005),
        min_buffer_size_to_learn=nfsp_cfg.get('min_buffer_size_to_learn', 100),
        q_replay_memory_size=nfsp_cfg.get('q_replay_memory_size', 200000),
        q_replay_memory_init_size=nfsp_cfg.get('q_replay_memory_init_size', 2000),
        q_update_target_estimator_every=nfsp_cfg.get('q_update_target_estimator_every', 1000),
        q_discount_factor=nfsp_cfg.get('q_discount_factor', 0.99),
        q_epsilon_start=nfsp_cfg.get('q_epsilon_start', 1.0),
        q_epsilon_end=nfsp_cfg.get('q_epsilon_end', 0.1),
        q_epsilon_decay_steps=nfsp_cfg.get('q_epsilon_decay_steps', 20000),
        q_batch_size=nfsp_cfg.get('q_batch_size', 256),
        q_train_every=nfsp_cfg.get('q_train_every', 1),
        evaluate_with=nfsp_cfg.get('evaluate_with', 'average_policy'),
        device=device,
        save_path=SAVE_PATH,
        save_every=nfsp_cfg.get('save_every', 5000),
    )

    # 4. Set Agents for Training (All 4 seats share the SAME agent instance)
    train_env.set_agents([agent, agent, agent, agent])

    # 5. Set Agents for Evaluation
    eval_env.set_agents([agent, agent, agent, agent])

    # Initialize Opponent Agent (same structure as main agent)
    opponent_agent = NFSPAgent(
        num_actions=train_env.num_actions,
        state_shape=train_env.state_shape[0][0],
        hidden_layers_sizes=hidden_layers_sizes,
        q_mlp_layers=q_mlp_layers,
        device=device,
        reservoir_buffer_capacity=100,
        batch_size=nfsp_cfg.get('batch_size', 256),
        save_path=None,
    )

    def _evaluate_raw_diff(env, num_games):
        total_diff = 0.0
        for _ in range(num_games):
            env.run(is_training=False)
            raw_scores = env.game.judger.judge_game(env.game.players)
            total_diff += raw_scores[0] - raw_scores[1]
        return total_diff / float(num_games)

    print("Start NFSP Self-Play training...")
    start_time = time.time()
    with Logger(SAVE_PATH) as logger:
        for episode in range(NUM_EPISODES):
            if max_seconds is not None and (time.time() - start_time) >= max_seconds:
                print(f"Reached max_hours={MAX_HOURS}. Stopping training.")
                break

            # Sample policy (best_response or average) for this episode
            agent.sample_episode_policy()

            # Update Opponent Checkpoint Logic
            if episode % OPPONENT_UPDATE_EVERY == 0:
                ckpt_path = os.path.join(SAVE_PATH, 'checkpoint_opponent_nfsp.pt')
                if not os.path.exists(SAVE_PATH):
                    os.makedirs(SAVE_PATH)

                agent.save_checkpoint(SAVE_PATH, filename='checkpoint_opponent_nfsp.pt')
                opponent_agent = NFSPAgent.from_checkpoint(
                    torch.load(ckpt_path, map_location=device, weights_only=False)
                )

                # Team 0: current agent, Team 1: frozen checkpoint
                eval_env.set_agents([agent, opponent_agent, agent, opponent_agent])
                print(f"Updated opponent with model from episode {episode}")

            # --- Training Step ---
            trajectories, payoffs = train_env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions from ALL 4 players to the shared agent
            for i in range(train_env.num_players):
                for ts in trajectories[i]:
                    agent.feed(ts)

            # --- Evaluation Step ---
            if episode % EVALUATE_EVERY == 0:
                eval_reward = _evaluate_raw_diff(eval_env, 20)
                logger.log_performance(episode, eval_reward)
                print(f"Episode: {episode}, Agent Team Reward vs Previous Model: {eval_reward}")

        csv_path, fig_path = logger.csv_path, logger.fig_path
        plot_curve(csv_path, fig_path, 'NFSP_SelfPlay')
        print(f"Training finished. Logs saved to {SAVE_PATH}")


if __name__ == '__main__':
    train()
