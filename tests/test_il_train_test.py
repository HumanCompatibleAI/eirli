import glob
import os

import pytest

from il_representations.test_support.configuration import BENCHMARK_CONFIGS


@pytest.mark.parametrize("benchmark_cfg", BENCHMARK_CONFIGS)
def test_il_train_test(benchmark_cfg, il_train_ex, il_test_ex, file_observer):
    """Simple smoke test for training/testing IL code."""
    common_cfg = {
        'device_name': 'cpu',
        'benchmark': benchmark_cfg,
    }

    # train
    il_train_ex.run(config_updates={
        'bc_n_epochs': 1,
        **common_cfg,
    })
    # FIXME(sam): same comment as elsewhere: should have a better way of
    # getting at saved policies.
    log_dir = file_observer.dir

    # test
    policy_path = glob.glob(os.path.join(log_dir, '*.pt'))[0]
    il_test_ex.run(
        config_updates={
            'n_rollouts': 2,
            'eval_batch_size': 2,
            'policy_path': policy_path,
            **common_cfg,
        })
