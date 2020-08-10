import glob
import os

import pytest
from sacred.observers import FileStorageObserver

from il_representations.scripts.il_test import il_test_ex
from il_representations.scripts.il_train import il_train_ex

# HACK(sam): this should be a 'run once' fixture
observer = FileStorageObserver('test_observer')
il_train_ex.observers.append(observer)
il_test_ex.observers.append(observer)


@pytest.mark.parametrize("benchmark_name", ["magical", "dm_control", "atari"])
def test_il_train_test(benchmark_name):
    """Simple smoke test for training/testing IL code."""
    # experiment config

    # benchmark config
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if benchmark_name == 'magical':
        common_cfg = {
            'device_name': 'cpu',
            'benchmark': {
                'benchmark_name': 'magical',
                'magical_env_prefix': 'MoveToRegion',
                'magical_demo_dirs': {
                    'MoveToRegion':
                    os.path.join(this_dir, 'data', 'magical',
                                 'move-to-region'),
                }
            }
        }
    elif benchmark_name == 'dm_control':
        common_cfg = {
            'device_name': 'cpu',
            'benchmark': {
                'benchmark_name': 'dm_control',
                'dm_control_env': 'reacher-easy',
                'dm_control_demo_patterns': {
                    'reacher-easy':
                    os.path.join(this_dir, 'data', 'dm_control',
                                 'reacher-easy-*.pkl.gz'),
                }
            }
        }
    elif benchmark_name == 'atari':
        common_cfg = {
            'device_name': 'cpu',
            'benchmark': {
                'benchmark_name':
                'atari',
                'atari_env_id':
                'PongNoFrameskip-v4',
                'atari_demo_paths':
                [os.path.join(this_dir, 'data', 'atari', 'pong.npz')],
            }
        }
    else:
        raise NotImplementedError(f"How do I handle '{benchmark_name}'?")

    # train
    run_result = il_train_ex.run(config_updates={
        'bc_n_epochs': 1,
        **common_cfg,
    })
    # FIXME(sam): same comment as elsewhere: should have a better way of
    # getting at saved policies.
    log_dir = run_result.observers[0].dir

    # test
    policy_path = glob.glob(os.path.join(log_dir, '*.pt'))[0]
    il_test_ex.run(
        config_updates={
            'n_rollouts': 2,
            'eval_batch_size': 2,
            'policy_path': policy_path,
            **common_cfg,
        })
