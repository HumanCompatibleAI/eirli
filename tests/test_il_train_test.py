import os

import pytest

from il_representations.envs import auto
from il_representations.test_support.configuration import (
    ENV_CFG_TEST_CONFIGS, ENV_DATA_VENV_OPTS_TEST_CONFIG, FAST_IL_TRAIN_CONFIG,
    VENV_OPTS_TEST_CONFIG)


@pytest.mark.parametrize("env_cfg", ENV_CFG_TEST_CONFIGS)
@pytest.mark.parametrize("algo", ["bc", "gail"])
def test_il_train_test(env_cfg, algo, il_train_ex, il_test_ex,
                       file_observer):
    """Simple smoke test for training/testing IL code."""
    bench_available, why = auto.benchmark_is_available(
        env_cfg['benchmark_name'])
    if not bench_available:
        pytest.skip(why)

    common_cfg = {
        'env_cfg': env_cfg,
        'device_name': 'cpu',
    }

    # train
    il_train_run = il_train_ex.run(config_updates={
        'algo': algo,
        'final_pol_name': final_pol_name,
        # these defaults make training cheap
        **FAST_IL_TRAIN_CONFIG,
        **ENV_DATA_VENV_OPTS_TEST_CONFIG,
        **common_cfg,
    })

    # test
    policy_path = il_train_run.result['model_path']
    il_test_ex.run(
        config_updates={
            'n_rollouts': 2,
            'policy_path': policy_path,
            'venv_opts': VENV_OPTS_TEST_CONFIG,
            **common_cfg,
        })
