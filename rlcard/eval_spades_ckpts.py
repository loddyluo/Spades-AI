import argparse
import rlcard
from rlcard.utils import get_device, set_seed, tournament
from rlcard.utils.agent_utils import load_agent_from_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate two checkpoints head-to-head in Spades."
    )
    parser.add_argument("--ckpt-team0", required=True, help="Checkpoint for players 0 and 2")
    parser.add_argument("--ckpt-team1", required=True, help="Checkpoint for players 1 and 3")
    parser.add_argument(
        "--agent-type",
        choices=["dqn", "nfsp"],
        default=None,
        help="Optional agent type override for both checkpoints",
    )
    parser.add_argument(
        "--disable-blind-nil",
        action="store_true",
        help="Disable blind-nil phase and always show hand",
    )
    parser.add_argument("--num-games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    set_seed(args.seed)

    env = rlcard.make(
        "spades",
        config={"seed": args.seed, "game_enable_blind_nil": not args.disable_blind_nil},
    )

    team0_agent, _ = load_agent_from_checkpoint(
        args.ckpt_team0,
        device=device,
        agent_type_override=args.agent_type,
        env=env,
    )
    team1_agent, _ = load_agent_from_checkpoint(
        args.ckpt_team1,
        device=device,
        agent_type_override=args.agent_type,
        env=env,
    )

    if isinstance(team0_agent, list) or isinstance(team1_agent, list):
        if not isinstance(team0_agent, list) or not isinstance(team1_agent, list):
            raise ValueError('DMC checkpoints must be provided for both teams when using separate checkpoints.')
        if len(team0_agent) != env.num_players or len(team1_agent) != env.num_players:
            raise ValueError('DMC agent list length must match number of players.')
        env.set_agents([team0_agent[0], team1_agent[1], team0_agent[2], team1_agent[3]])
    else:
        env.set_agents([team0_agent, team1_agent, team0_agent, team1_agent])

    total_diff = 0.0
    for _ in range(args.num_games):
        env.run(is_training=False)
        raw_scores = env.game.judger.judge_game(env.game.players)
        total_diff += raw_scores[0] - raw_scores[1]
    avg_value = total_diff / float(args.num_games)

    print("Evaluation complete")
    print(f"Avg score diff (Team0 - Team1): {avg_value:.6f}")


if __name__ == "__main__":
    main()
