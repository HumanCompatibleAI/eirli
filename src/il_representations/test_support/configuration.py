"""Benchmark ingredient configurations for unit testing."""
from os import path

from ray import tune

from il_representations import algos

CURRENT_DIR = path.dirname(path.abspath(__file__))
TEST_DATA_DIR = path.abspath(
    path.join(CURRENT_DIR, '..', '..', '..', 'tests', 'data'))
VENV_OPTS_TEST_CONFIG = {
    'venv_parallel': False,
    'n_envs': 2,
}
ENV_DATA_TEST_CONFIG = {
    'data_root': path.join(TEST_DATA_DIR, '..'),
    'atari_demo_paths': {
        'PongNoFrameskip-v4': path.join(TEST_DATA_DIR, 'atari', 'pong.npz'),
        # TODO add test data dir here
    },
    'magical_demo_dirs': {
        'MoveToRegion': path.join(TEST_DATA_DIR, 'magical', 'move-to-region'),
    },
    'dm_control_demo_patterns': {
        'reacher-easy':
        path.join(TEST_DATA_DIR, 'dm_control', 'reacher-easy-*.pkl.gz')
    },

}
ENV_DATA_VENV_OPTS_TEST_CONFIG = {
    'env_data': ENV_DATA_TEST_CONFIG,
    'venv_opts': VENV_OPTS_TEST_CONFIG,
}
ENV_CFG_TEST_CONFIGS = [
    {
        'benchmark_name': 'atari',
        'task_name': 'PongNoFrameskip-v4',
    },
    {
        'benchmark_name': 'magical',
        'task_name': 'MoveToRegion',
    },
    {
        'benchmark_name': 'dm_control',
        'task_name': 'reacher-easy',
    },
    {
        'benchmark_name': 'minecraft',
        'task_name': 'NavigateVectorObf',
        'minecraft_max_env_steps': 100
    }

]
FAST_IL_TRAIN_CONFIG = {
    'bc': {
        'n_epochs': None,
        'n_batches': 1,
    },
    'gail': {
        # ppo_n_steps, ppo_batch_size and disc_batch_size are the smallest
        # "non-trivial" values (disc_batch_size in particular needs to be at
        # least 2 so that the discriminator sees both a positive and a
        # negative)
        'ppo_n_steps': 2,
        'ppo_batch_size': 2,
        'disc_batch_size': 2,
        # imitation requires total_timesteps needs to be at least 4 to let PPO
        # train with a n_envs * n_steps = 2 * 2 = 4 samples per PPO epoch
        'total_timesteps': 4,
        # ppo_n_epochs and disc_n_updates_per_round are at minimum values
        'ppo_n_epochs': 1,
        'disc_n_updates_per_round': 1,
    },
}
REPL_SMOKE_TEST_CONFIG = {
    'batches_per_epoch': 2,
    'n_epochs': 1,
    'algo_params': {
        'representation_dim': 3,
        'batch_size': 7,
    },
}
CHAIN_CONFIG = {
    'spec': {
        'repl': {
            'algo': tune.grid_search([algos.SimCLR]),
        },
        'il_train': {
            # in practice we probably want to try GAIL too
            # (I'd put this in the unit test if it wasn't so slow)
            'algo': tune.grid_search(['bc']),
            'freeze_encoder': tune.grid_search([False])
        },
        'env_cfg': tune.grid_search([ENV_CFG_TEST_CONFIGS[0]]),
    },
    'tune_run_kwargs': {
        'resources_per_trial': {
            'cpu': 2,
            'gpu': 0,
        },
        'num_samples': 1,
    },
    'ray_init_kwargs': {
        # Ray has been mysteriously complaining about the amount of memory
        # available on CircleCI, even though the machines have heaps of RAM.
        # Setting sane defaults so this doesn't happen.
        'object_store_memory': int(0.2 * 1e9),
        'num_cpus': 2,
    },
    'il_train': {
        'device_name': 'cpu',
        **FAST_IL_TRAIN_CONFIG,
    },
    'il_test': {
        'device_name': 'cpu',
        'n_rollouts': 2,
    },
    'repl': {
        'device': 'cpu',
        **REPL_SMOKE_TEST_CONFIG,
    },
    **ENV_DATA_VENV_OPTS_TEST_CONFIG,
}
