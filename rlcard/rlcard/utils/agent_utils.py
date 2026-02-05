import os
import torch

from rlcard.agents import DQNAgent, NFSPAgent


def detect_agent_type_from_checkpoint(checkpoint):
    agent_type = checkpoint.get('agent_type')
    if isinstance(agent_type, str):
        agent_type = agent_type.lower()
        if agent_type in {'dqnagent', 'dqn'}:
            return 'dqn'
        if agent_type in {'nfspagent', 'nfsp'}:
            return 'nfsp'
    if 'policy_network' in checkpoint and 'rl_agent' in checkpoint:
        return 'nfsp'
    if 'q_estimator' in checkpoint and 'memory' in checkpoint:
        return 'dqn'
    raise ValueError('Unable to detect agent type from checkpoint.')


def load_agent_from_checkpoint(ckpt_path, device=None, agent_type_override=None):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    agent_type = agent_type_override or detect_agent_type_from_checkpoint(checkpoint)
    if agent_type == 'dqn':
        agent = DQNAgent.from_checkpoint(checkpoint)
    elif agent_type == 'nfsp':
        agent = NFSPAgent.from_checkpoint(checkpoint)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    if hasattr(agent, 'set_device') and device is not None:
        agent.set_device(device)
    return agent, agent_type
