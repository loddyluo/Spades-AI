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


def _override_checkpoint_device(checkpoint, device):
    if device is None:
        return checkpoint
    checkpoint = dict(checkpoint)
    checkpoint['device'] = device
    if 'q_estimator' in checkpoint:
        q_estimator = dict(checkpoint['q_estimator'])
        q_estimator['device'] = device
        checkpoint['q_estimator'] = q_estimator
    if 'rl_agent' in checkpoint:
        rl_agent = dict(checkpoint['rl_agent'])
        rl_agent['device'] = device
        if 'q_estimator' in rl_agent:
            q_estimator = dict(rl_agent['q_estimator'])
            q_estimator['device'] = device
            rl_agent['q_estimator'] = q_estimator
        checkpoint['rl_agent'] = rl_agent
    return checkpoint


def load_agent_from_checkpoint(ckpt_path, device=None, agent_type_override=None):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint = _override_checkpoint_device(checkpoint, device)
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
