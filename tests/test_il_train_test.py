import glob
import os

import pytest
from sacred.observers import FileStorageObserver

from il_representations.scripts.il_test import il_test_ex
from il_representations.scripts.il_train import il_train_ex
from il_representations.test_support.configuration import BENCHMARK_CONFIGS

# HACK(sam): this should be a 'run once' fixture
observer = FileStorageObserver('test_observer')
il_train_ex.observers.append(observer)
il_test_ex.observers.append(observer)


@pytest.mark.parametrize("benchmark_cfg", BENCHMARK_CONFIGS)
def test_il_train_test(benchmark_cfg):
    """Simple smoke test for training/testing IL code."""
    common_cfg = {
        'device_name': 'cpu',
        'benchmark': benchmark_cfg,
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
    il_test_ex.run(
        config_updates={
            'n_rollouts': 2,
            'eval_batch_size': 2,
            'policy_path': policy_path,
            **common_cfg,
        })
