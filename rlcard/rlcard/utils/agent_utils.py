import os
import torch

from rlcard.agents import DQNAgent, NFSPAgent, DRQNAgent
from rlcard.utils import load_torch_sharded


def detect_agent_type_from_checkpoint(checkpoint):
    agent_type = checkpoint.get('agent_type')
    if isinstance(agent_type, str):
        agent_type = agent_type.lower()
        if agent_type in {'dqnagent', 'dqn'}:
            return 'dqn'
        if agent_type in {'nfspagent', 'nfsp'}:
            return 'nfsp'
        if agent_type in {'drqnagent', 'drqn'}:
            return 'drqn'
    if 'policy_network' in checkpoint and 'rl_agent' in checkpoint:
        return 'nfsp'
    if 'lstm_hidden_size' in checkpoint and 'q_estimator' in checkpoint:
        return 'drqn'
    if 'q_estimator' in checkpoint and 'memory' in checkpoint:
        return 'dqn'
    if 'model_state_dict' in checkpoint and isinstance(checkpoint.get('model_state_dict'), list):
        return 'dmc'
    raise ValueError('Unable to detect agent type from checkpoint.')


def _device_to_dmc_device(device):
    if device is None:
        return 'cpu'
    if isinstance(device, torch.device):
        if device.type == 'cuda':
            return '0'
        return 'cpu'
    if isinstance(device, str):
        if device.startswith('cuda'):
            return '0'
        return 'cpu'
    return 'cpu'


def _resolve_dmc_action_shape(env):
    action_shape = env.action_shape
    if action_shape and action_shape[0] is None:
        action_shape = [[env.num_actions] for _ in range(env.num_players)]
    return action_shape


def _load_dmc_agents(checkpoint, env, device):
    from rlcard.agents.dmc_agent.model import DMCModel
    if env is None:
        raise ValueError('DMC requires env to load checkpoint.')
    state_shape = env.state_shape
    action_shape = _resolve_dmc_action_shape(env)
    dmc_device = _device_to_dmc_device(device)
    model = DMCModel(state_shape, action_shape, device=dmc_device)
    model_state_dict = checkpoint.get('model_state_dict', [])
    for idx, agent in enumerate(model.get_agents()):
        if idx < len(model_state_dict):
            agent.load_state_dict(model_state_dict[idx])
    return model.get_agents()


def _load_drqn_agent(checkpoint, device):
    """Load a DRQNAgent from either the full agent checkpoint format or the
    lightweight trainer checkpoint format (which only stores qnet weights)."""
    if 'q_estimator' in checkpoint and 'memory' in checkpoint:
        # Full DRQNAgent checkpoint
        return DRQNAgent.from_checkpoint(checkpoint)

    # Trainer-produced checkpoint: reconstruct a minimal DRQNAgent for inference
    from rlcard.agents.drqn_agent import DRQNEstimator, EpisodeMemory
    agent = DRQNAgent(
        num_actions=checkpoint['num_actions'],
        state_shape=checkpoint['state_shape'],
        embed_dim=checkpoint.get('embed_dim', 256),
        lstm_hidden_size=checkpoint.get('lstm_hidden_size', 256),
        lstm_num_layers=checkpoint.get('lstm_num_layers', 1),
        mlp_layers=checkpoint.get('mlp_layers', [256]),
        learning_rate=checkpoint.get('learning_rate', 0.00003),
        batch_size=checkpoint.get('batch_size', 256),
        seq_len=checkpoint.get('seq_len', 16),
        device=device,
    )
    agent.q_estimator.qnet.load_state_dict(checkpoint['qnet'])
    agent.target_estimator.qnet.load_state_dict(checkpoint['qnet'])
    return agent


def _load_cfr_agent(ckpt_dir, env):
    from rlcard.agents import CFRAgent
    if env is None:
        raise ValueError('CFR requires env to load checkpoint.')
    agent = CFRAgent(env, ckpt_dir)
    agent.load()
    return agent


def _is_cfr_dir(path):
    return os.path.isdir(path) and os.path.exists(os.path.join(path, 'policy.pkl'))


def _is_dmc_dir(path):
    if not os.path.isdir(path):
        return False
    if os.path.exists(os.path.join(path, 'model.tar')):
        return True
    if os.path.exists(os.path.join(path, 'model.tar.index.json')):
        return True
    return False


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


def load_agent_from_checkpoint(ckpt_path, device=None, agent_type_override=None, env=None):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if _is_cfr_dir(ckpt_path):
        agent = _load_cfr_agent(ckpt_path, env)
        return agent, 'cfr'
    if _is_dmc_dir(ckpt_path):
        checkpoint = load_torch_sharded(
            os.path.join(ckpt_path, 'model.tar'),
            map_location=device,
        )
        agent = _load_dmc_agents(checkpoint, env, device)
        return agent, 'dmc'

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint = _override_checkpoint_device(checkpoint, device)
    agent_type = agent_type_override or detect_agent_type_from_checkpoint(checkpoint)
    if agent_type == 'dqn':
        agent = DQNAgent.from_checkpoint(checkpoint)
    elif agent_type == 'nfsp':
        agent = NFSPAgent.from_checkpoint(checkpoint)
    elif agent_type == 'drqn':
        agent = _load_drqn_agent(checkpoint, device)
    elif agent_type == 'dmc':
        agent = _load_dmc_agents(checkpoint, env, device)
    elif agent_type == 'cfr':
        agent = _load_cfr_agent(ckpt_path, env)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    if hasattr(agent, 'set_device') and device is not None:
        agent.set_device(device)
    return agent, agent_type
