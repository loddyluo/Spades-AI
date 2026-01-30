import os
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

def train():
    # Configuration
    ENV_ID = 'spades'
    SEED = 42
    NUM_EPISODES = 2000
    EVALUATE_EVERY = 100
    SAVE_PATH = 'experiments/spades_selfplay_dqn'
    
    # Check device
    device = get_device()

    # Seed
    set_seed(SEED)

    # 1. Make Training Environment
    train_env = rlcard.make(ENV_ID, config={'seed': SEED})
    
    # 2. Make Evaluation Environment
    # We use a separate env for eval to avoid messing up agent assignment
    eval_env = rlcard.make(ENV_ID, config={'seed': SEED})

    # 3. Initialize the Shared Agent (DQN)
    # In Self-Play, we use ONE agent to play all 4 positions, 
    # sharing the experience to learn a general strategy.
    agent = DQNAgent(
        num_actions=train_env.num_actions,
        state_shape=train_env.state_shape[0][0],
        mlp_layers=[256, 256],  # 将 [128, 128] 改为 [256, 256] 以提升表达能力
        device=device,          # 脚本会自动检测 GPU
        save_path=SAVE_PATH,
        replay_memory_size=100000, # 增大缓存
        batch_size=128,            # 增大 Batch Size 以利用 GPU 并行能力
        save_every=5000
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
    OPPONENT_UPDATE_EVERY = 500 # Update opponent every 500 episodes
    
    # Initialize Opponent Agent (same structure as main agent)
    opponent_agent = DQNAgent(
        num_actions=train_env.num_actions,
        state_shape=train_env.state_shape[0][0],
        mlp_layers=[256, 256],
        device=device,
        save_path=None,
        replay_memory_size=100, # Minimal buffer for opponent as it doesn't train
        batch_size=128
    )

    print("Start Self-Play training...")
    with Logger(SAVE_PATH) as logger:
        for episode in range(NUM_EPISODES):
            
            # Update Opponent Checkpoint Logic
            if episode % OPPONENT_UPDATE_EVERY == 0:
                # Save current agent as a temporary checkpoint to be loaded as opponent
                ckpt_path = os.path.join(SAVE_PATH, 'checkpoint_opponent.pth')
                if not os.path.exists(SAVE_PATH):
                    os.makedirs(SAVE_PATH)
                
                # Use standard RLCard save/load
                agent.save(ckpt_path)
                opponent_agent.load(ckpt_path)
                
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
                rewards = tournament(eval_env, num_eval_games=20)
                
                # We log the reward of Player 0 (Our Agent)
                # Since P0 and P2 are a team, their rewards should be identical
                logger.log_performance(episode, rewards[0])
                print(f"Episode: {episode}, Agent Team Reward vs Previous Model: {rewards[0]}")

        # Save Plot & Model
        csv_path, fig_path = logger.csv_path, logger.fig_path
        plot_curve(csv_path, fig_path, 'DQN_SelfPlay')
        print(f"Training finished. Logs saved to {SAVE_PATH}")

if __name__ == '__main__':
    train()
