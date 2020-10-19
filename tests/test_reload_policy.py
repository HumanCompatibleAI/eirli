import glob
import os

import pytest

from il_representations.algos import MoCo
from il_representations.test_support.configuration import (BENCHMARK_TEST_CONFIGS,
                                                           FAST_IL_TRAIN_CONFIG)


@pytest.mark.parametrize("algo", ["bc", "gail"])
@pytest.mark.parametrize("freeze_encoder", [False, True])
def test_reload_policy(algo, freeze_encoder, represent_ex, il_train_ex, file_observer):
    """Test saving a policy with one specific representation learner, then loading
    it with the IL code.

    (I only test one IL algorithm, one dataset, and one representation learner
    because the process is roughly the same in all cases)"""
    represent_ex.run(
        config_updates={
            'pretrain_epochs': 1,
            'unit_test_max_train_steps': 2,
            'algo_params': {'representation_dim': 3, 'batch_size': 7},
            'algo': MoCo,
            'n_traj': 1,
            'use_random_rollouts': False,
            'benchmark': BENCHMARK_TEST_CONFIGS[0],
            'ppo_finetune': False,
        })

    # train BC using learnt representation
    encoder_list = glob.glob(
        os.path.join(
            file_observer.dir,
            'checkpoints/representation_encoder/*.ckpt'))
    policy_path = encoder_list[0]
    il_train_ex.run(
        config_updates={
            'algo': algo,
            'device_name': 'cpu',
            # we only test against once benchmark, since this process should be
            # similar for all
            'benchmark': BENCHMARK_TEST_CONFIGS[0],
            'encoder_path': policy_path,
            'freeze_encoder': freeze_encoder,
            **FAST_IL_TRAIN_CONFIG,
        })
