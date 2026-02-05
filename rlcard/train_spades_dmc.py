import os
import json
import rlcard
import torch
from rlcard.agents.dmc_agent import DMCTrainer
from rlcard.utils import set_seed


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


def train():
    config = _load_config()
    env_id = 'spades'

    seed = config.get('training', {}).get('seed', 42)
    reward_beta = config.get('training', {}).get('reward_beta', 1.0)
    dmc_cfg = config.get('dmc', {})

    xpid = dmc_cfg.get('xpid', 'spades_dmc')
    save_dir = dmc_cfg.get('save_dir', 'experiments/spades_dmc')
    load_model = dmc_cfg.get('load_model', False)
    cuda = dmc_cfg.get('cuda', '')
    if cuda == '':
        cuda = '0' if torch.cuda.is_available() else ''

    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)

    training_device = dmc_cfg.get('training_device', '0')
    if cuda == '':
        training_device = 'cpu'

    set_seed(seed)
    env = rlcard.make(env_id, config={'seed': seed, 'reward_beta': reward_beta})

    trainer = DMCTrainer(
        env,
        cuda=cuda,
        load_model=load_model,
        xpid=xpid,
        savedir=save_dir,
        save_interval=dmc_cfg.get('save_interval', 30),
        num_actor_devices=dmc_cfg.get('num_actor_devices', 1),
        num_actors=dmc_cfg.get('num_actors', 5),
        training_device=training_device,
        total_frames=dmc_cfg.get('total_frames', 5000000),
        exp_epsilon=dmc_cfg.get('exp_epsilon', 0.01),
        batch_size=dmc_cfg.get('batch_size', 32),
        unroll_length=dmc_cfg.get('unroll_length', 100),
        num_buffers=dmc_cfg.get('num_buffers', 50),
        num_threads=dmc_cfg.get('num_threads', 4),
        max_grad_norm=dmc_cfg.get('max_grad_norm', 40),
        learning_rate=dmc_cfg.get('learning_rate', 0.0001),
        alpha=dmc_cfg.get('alpha', 0.99),
        momentum=dmc_cfg.get('momentum', 0),
        epsilon=dmc_cfg.get('epsilon', 0.00001),
    )

    trainer.start()


if __name__ == '__main__':
    train()
