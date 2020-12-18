import os

import pytest

from il_representations.test_support.configuration import (
    ENV_CFG_TEST_CONFIGS, ENV_DATA_VENV_OPTS_TEST_CONFIG, FAST_IL_TRAIN_CONFIG,
    VENV_OPTS_TEST_CONFIG)


@pytest.mark.parametrize("env_cfg", ENV_CFG_TEST_CONFIGS) #TODO fix these back, doing this to test one test more quickly
@pytest.mark.parametrize("algo", ["bc", "gail"])
def test_il_train_test(env_cfg, algo, il_train_ex, il_test_ex,
                       file_observer):
    """Simple smoke test for training/testing IL code."""
    common_cfg = {
        'env_cfg': env_cfg,
        'device_name': 'cpu',
    }

    final_pol_name = 'last_test_policy.pt'
    # train
    il_train_ex.run(config_updates={
        'algo': algo,
        'final_pol_name': final_pol_name,
        # these defaults make training cheap
        **FAST_IL_TRAIN_CONFIG,
        **ENV_DATA_VENV_OPTS_TEST_CONFIG,
        **common_cfg,
    })
    # FIXME(sam): same comment as elsewhere: should have a better way of
    # getting at saved policies.
    log_dir = file_observer.dir

    # test
    policy_path = os.path.join(log_dir, final_pol_name)
    il_test_ex.run(
        config_updates={
            'n_rollouts': 2,
            'policy_path': policy_path,
            'venv_opts': VENV_OPTS_TEST_CONFIG,
            **common_cfg,
        })
