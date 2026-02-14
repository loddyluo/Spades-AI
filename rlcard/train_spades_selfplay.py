import os
import json
import time
import torch
import rlcard
from rlcard.agents import RandomAgent, DQNAgent
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
    REWARD_BETA = config.get('training', {}).get('reward_beta', 1.0)
    SAVE_PATH = config.get('dqn', {}).get(
        'save_path',
        config.get('training', {}).get('save_path', 'experiments/spades_selfplay_dqn'),
    )
    ENABLE_BLIND_NIL = config.get('training', {}).get('game_enable_blind_nil', True)
    
    # Check device
    device = get_device()

    # Seed
    set_seed(SEED)

    # 1. Make Training Environment
    train_env = rlcard.make(
        ENV_ID,
        config={'seed': SEED, 'reward_beta': REWARD_BETA, 'game_enable_blind_nil': ENABLE_BLIND_NIL},
    )
    
    # 2. Make Evaluation Environment
    # We use a separate env for eval to avoid messing up agent assignment
    eval_env = rlcard.make(
        ENV_ID,
        config={'seed': SEED, 'reward_beta': REWARD_BETA, 'game_enable_blind_nil': ENABLE_BLIND_NIL},
    )

    # 3. Initialize the Shared Agent (DQN)
    # In Self-Play, we use ONE agent to play all 4 positions, 
    # sharing the experience to learn a general strategy.
    mlp_layers = config.get('model', {}).get('mlp_layers', [256, 256])
    replay_memory_size = config.get('agent', {}).get('replay_memory_size', 100000)
    replay_memory_init_size = config.get('agent', {}).get('replay_memory_init_size', 1000)
    batch_size = config.get('agent', {}).get('batch_size', 128)
    save_every = config.get('agent', {}).get('save_every', 5000)

    agent = DQNAgent(
        num_actions=train_env.num_actions,
        state_shape=train_env.state_shape[0][0],
        mlp_layers=mlp_layers,
        device=device,          # 脚本会自动检测 GPU
        save_path=SAVE_PATH,
        replay_memory_size=replay_memory_size,
        replay_memory_init_size=replay_memory_init_size,
        batch_size=batch_size,
        save_every=save_every
    )
    
    # 4. Set Agents for Training (All 4 seats are the SAME agent instance)
    train_env.set_agents([agent, agent, agent, agent])

    # 5. Set Agents for Evaluation
    # Note: We will dynamically update the opponents during evaluation
    # Initial opponents can be random, or a copy of current agent
    eval_env.set_agents([agent, agent, agent, agent])

    # Variable to keep track of the best model (or just previous checkpoint)
    # Since RLCard agent saving is done inside agent.save(), we need to manually handle loading for opponents.
    # We will save a "checkpoint_opponent.pth" periodically.
    OPPONENT_UPDATE_EVERY = config.get('training', {}).get('opponent_update_every', 500)
    
    # Initialize Opponent Agent (same structure as main agent)
    opponent_agent = DQNAgent(
        num_actions=train_env.num_actions,
        state_shape=train_env.state_shape[0][0],
        mlp_layers=mlp_layers,
        device=device,
        save_path=None,
        replay_memory_size=100, # Minimal buffer for opponent as it doesn't train
        batch_size=batch_size
    )

    def _evaluate_raw_diff(env, num_games):
        total_diff = 0.0
        for _ in range(num_games):
            env.run(is_training=False)
            raw_scores = env.game.judger.judge_game(env.game.players)
            total_diff += raw_scores[0] - raw_scores[1]
        return total_diff / float(num_games)

    print("Start Self-Play training...")
    start_time = time.time()
    with Logger(SAVE_PATH) as logger:
        for episode in range(NUM_EPISODES):
            if max_seconds is not None and (time.time() - start_time) >= max_seconds:
                print(f"Reached max_hours={MAX_HOURS}. Stopping training.")
                break
            
            # Update Opponent Checkpoint Logic
            if episode % OPPONENT_UPDATE_EVERY == 0:
                # Save current agent as a temporary checkpoint to be loaded as opponent
                ckpt_path = os.path.join(SAVE_PATH, 'checkpoint_opponent.pt')
                if not os.path.exists(SAVE_PATH):
                    os.makedirs(SAVE_PATH)
                
                # Use RLCard DQN checkpoint save/load
                agent.save_checkpoint(SAVE_PATH, filename='checkpoint_opponent.pt')
                opponent_agent = DQNAgent.from_checkpoint(torch.load(ckpt_path, map_location=device, weights_only=False))
                
                # Update Evaluation Environment: 
                # Team 0 (Agent): Current Learning Agent
                # Team 1 (Opponent): Frozen Checkpoint
                eval_env.set_agents([agent, opponent_agent, agent, opponent_agent])
                print(f"Updated opponent with model from episode {episode}")

            # --- Training Step ---
            # Generate one episode of data (Agent plays against itself)
            trajectories, payoffs = train_env.run(is_training=True)
            # Reorganize data
            trajectories = reorganize(trajectories, payoffs)

            # Feed ALL transitions from ALL 4 players to the shared agent
            # This allows the agent to learn from every position's perspective
            for i in range(train_env.num_players):
                for ts in trajectories[i]:
                    agent.feed(ts)

            # --- Evaluation Step ---
            if episode % EVALUATE_EVERY == 0:
                # Run tournament: Agent Team vs Opponent Team (Checkpoint)
                # Returns average payoffs for each player
                eval_reward = _evaluate_raw_diff(eval_env, 20)

                # We log the reward of Player 0 (Our Agent)
                # Since P0 and P2 are a team, their rewards should be identical
                logger.log_performance(episode, eval_reward)
                print(f"Episode: {episode}, Agent Team Reward vs Previous Model: {eval_reward}")

        # Save Plot & Model
        csv_path, fig_path = logger.csv_path, logger.fig_path
        plot_curve(csv_path, fig_path, 'DQN_SelfPlay')
        print(f"Training finished. Logs saved to {SAVE_PATH}")

if __name__ == '__main__':
    train()
