import glob
import os
from sacred.observers import FileStorageObserver

from il_representations.scripts.il_test import il_test_ex
from il_representations.scripts.il_train import il_train_ex


def test_il_train_test():
    """Simple smoke test for training/testing IL code."""
    # experiment config
    observer = FileStorageObserver('test_observer')
    il_train_ex.observers.append(observer)
    il_test_ex.observers.append(observer)

    # benchmark config
    this_dir = os.path.dirname(os.path.abspath(__file__))
    common_cfg = {
        'dev_name': 'cpu',
        'benchmark': {
            'benchmark_name': 'magical',
            'magical_env_prefix': 'MoveToRegion',
            'magical_demo_dirs': {
                'MoveToRegion':
                os.path.join(this_dir, 'data', 'magical', 'move-to-region'),
            }
        }
    }

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
    il_test_ex.run(config_updates={
        'n_rollouts': 2,
        'eval_batch_size': 2,
        'policy_path': policy_path,
        **common_cfg,
    })
