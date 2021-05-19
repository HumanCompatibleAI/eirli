"""Do smoke test of joint_training.py with each repL algorithm."""

import pytest

from il_representations.script_utils import update
from il_representations.test_support.configuration import (
    ENV_CFG_TEST_CONFIGS, FAST_JOINT_TRAIN_CONFIG)

REPL_NAMED_CONFIGS = ['repl_noid', 'repl_vae', 'repl_fd', 'repl_id']


@pytest.mark.parametrize("env_cfg", ENV_CFG_TEST_CONFIGS)
def test_all_benchmarks(joint_train_ex, file_observer, env_cfg):
    jt_config = update(FAST_JOINT_TRAIN_CONFIG, dict(env_cfg=env_cfg))
    joint_train_ex.run(config_updates=jt_config)


@pytest.mark.parametrize("repl_named_config", REPL_NAMED_CONFIGS)
def test_all_repl_algos(joint_train_ex, file_observer, repl_named_config):
    jt_config = update(FAST_JOINT_TRAIN_CONFIG,
                       dict(env_cfg=ENV_CFG_TEST_CONFIGS[0]))
    joint_train_ex.run(config_updates=jt_config,
                       named_configs=[repl_named_config])
