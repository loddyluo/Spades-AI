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
    parser.add_argument("--num-games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    set_seed(args.seed)

    env = rlcard.make("spades", config={"seed": args.seed})

    team0_agent, _ = load_agent_from_checkpoint(
        args.ckpt_team0,
        device=device,
        agent_type_override=args.agent_type,
    )
    team1_agent, _ = load_agent_from_checkpoint(
        args.ckpt_team1,
        device=device,
        agent_type_override=args.agent_type,
    )

    env.set_agents([team0_agent, team1_agent, team0_agent, team1_agent])

    rewards = tournament(env, args.num_games)
    team0_score = 0.0
    team1_score = 0.0
    if len(rewards) == 4:
        team0_score = rewards[0]
        team1_score = rewards[1]
        avg_diff = team0_score - team1_score
    else:
        avg_diff = rewards[0]

    print("Evaluation complete")
    print(f"Team0 avg score: {team0_score:.6f}")
    print(f"Team1 avg score: {team1_score:.6f}")
    print(f"Avg score diff (Team0 - Team1): {avg_diff:.6f}")


if __name__ == "__main__":
    main()
