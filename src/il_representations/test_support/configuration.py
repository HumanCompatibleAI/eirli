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
}

ENV_DATA_VENV_OPTS_TEST_CONFIG = {
    'env_data': ENV_DATA_TEST_CONFIG,
    'venv_opts': VENV_OPTS_TEST_CONFIG,
}

ENV_CFG_TEST_CONFIGS = [
    {
        'benchmark_name': 'magical',
        'task_name': 'MoveToRegion-Demo-v0',
    },
    {
        'benchmark_name': 'dm_control',
        'task_name': 'reacher-easy',
    },
    {
        'benchmark_name': 'procgen',
        'task_name': 'coinrun',
    },
    # TODO(sam): re-enable this once we're using Atari tasks (2021-04-07)
    # {
    #     'benchmark_name': 'atari',
    #     'task_name': 'PongNoFrameskip-v4',
    # },
    # TODO(sam): re-enable this once we're using Minecraft tasks (2021-04-07)
    # {
    #     'benchmark_name': 'minecraft',
    #     'task_name': 'NavigateVectorObf',
    #     'minecraft_max_env_steps': 100
    # },
]

FAST_IL_TRAIN_CONFIG = {
    'bc': {
        'n_batches': 2,
        'nominal_num_epochs': 3,
        'batch_size': 5,
        'augs': 'translate,rotate,noise',
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
        'disc_augs': 'translate,rotate,noise',
    },
    # we don't need a large shuffle buffer for tests
    'shuffle_buffer_size': 3,
}

FAST_DQN_TRAIN_CONFIG = {
    'n_batches': 2,
    'nominal_num_epochs': 1,
    'optimize_memory': False
}

REPL_SMOKE_TEST_CONFIG = {
    'batches_per_epoch': 2,
    'n_epochs': 1,
    'algo_params': {
        'representation_dim': 3,
        'batch_size': 7,
        'augmenter_kwargs': {
            # note that this will be run against (grayscale) Atari frames, so
            # we can't add color jitter or other augmentations that require RGB
            # frames
            'augmenter_spec': 'translate,rotate,noise',
        }
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
        'env_cfg': tune.grid_search([ENV_CFG_TEST_CONFIGS[2]]),
    },
    'tune_run_kwargs': {
        'resources_per_trial': {
            'cpu': 2,
            'gpu': 0,
        },
        'num_samples': 1,
        'max_failures': 0
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
    'dqn_train': {
        'device_name': 'cpu',
        **FAST_DQN_TRAIN_CONFIG
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

CHAIN_CONFIG_SKOPT = {
    **CHAIN_CONFIG,
    'use_skopt': True,
    'skopt_search_mode': 'max',
    'metric': 'return_mean',
    'stages_to_run': 'REPL_AND_IL',
    'spec': {},
    'env_cfg': ENV_CFG_TEST_CONFIGS[0],
    'skopt_space': {
        'repl:algo_params:augmenter_kwargs:augmenter_spec': [
            "translate", "rotate",
        ],
        'il_train:bc:optimizer_kwargs:lr': (1e-7, 1.0, 'log-uniform'),
    }
}

FAST_JOINT_TRAIN_CONFIG = {
    'bc': {
        'batch_size': 5,
        'augs': 'translate,rotate,noise',
        'short_eval_n_traj': 2,
    },
    'repl': {
        'algo_params': {
            'batch_size': 3,
            'augmenter_kwargs': {
                'augmenter_spec': 'translate,rotate,noise',
            }
        },
    },
    'representation_dim': 3,
    'shuffle_buffer_size': 3,
    'n_batches': 2,
    'final_eval_n_traj': 2,
    **ENV_DATA_VENV_OPTS_TEST_CONFIG,
}
