import os

import pytest

from il_representations.test_support.configuration import BENCHMARK_TEST_CONFIGS


@pytest.mark.parametrize("benchmark_cfg", BENCHMARK_TEST_CONFIGS)
@pytest.mark.parametrize("algo", ["bc", "gail"])
def test_il_train_test(benchmark_cfg, algo, il_train_ex, il_test_ex,
                       file_observer):
    """Simple smoke test for training/testing IL code."""
    common_cfg = {
        'benchmark': benchmark_cfg,
        'device_name': 'cpu',
    }

    final_pol_name = 'last_test_policy.pt'
    # train
    il_train_ex.run(config_updates={
        'algo': algo,
        'final_pol_name': final_pol_name,
        # these defaults make training cheap
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
            **common_cfg,
        })
