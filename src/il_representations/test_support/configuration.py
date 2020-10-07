"""Benchmark ingredient configurations for unit testing."""
from os import path

CURRENT_DIR = path.dirname(path.abspath(__file__))
TEST_DATA_DIR = path.abspath(path.join(CURRENT_DIR, '..', '..', '..', 'tests', 'data'))
COMMON_TEST_CONFIG = {
    'venv_parallel': False,
    'n_envs': 2,
    'n_traj': 1,
}
BENCHMARK_TEST_CONFIGS = [
    {
        'benchmark_name': 'atari',
        'atari_env_id': 'PongNoFrameskip-v4',
        'atari_demo_paths': {
            'PongNoFrameskip-v4': path.join(TEST_DATA_DIR, 'atari', 'pong.npz'),
        },
        **COMMON_TEST_CONFIG,
    },
    {
        'benchmark_name': 'magical',
        'magical_env_prefix': 'MoveToRegion',
        'magical_demo_dirs': {
            'MoveToRegion': path.join(TEST_DATA_DIR, 'magical', 'move-to-region'),
        },
        **COMMON_TEST_CONFIG,
    },
    {
        'benchmark_name': 'dm_control',
        'dm_control_env': 'reacher-easy',
        'dm_control_demo_patterns': {
            'reacher-easy': path.join(TEST_DATA_DIR, 'dm_control', 'reacher-easy-*.pkl.gz'),
        },
        **COMMON_TEST_CONFIG,
    },
]
FAST_IL_TRAIN_CONFIG = {
    'bc': {
        'n_epochs': 1,
    },
    'gail': {
        'total_timesteps': 2,
        'ppo_n_steps': 1,
        'ppo_batch_size': 2,
        'ppo_n_epochs': 1,
        'disc_minibatch_size': 2,
        'disc_batch_size': 2,
    },
}
