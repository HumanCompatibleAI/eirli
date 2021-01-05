import copy

import pytest
import ray
from ray import tune

from il_representations import algos
from il_representations.envs import auto
from il_representations.scripts.pretrain_n_adapt import StagesToRun
from il_representations.test_support.configuration import (
    CHAIN_CONFIG, ENV_CFG_TEST_CONFIGS)


def test_chain(chain_ex, file_observer):
    try:
        chain_ex.run(config_updates=CHAIN_CONFIG)
    finally:
        # always shut down Ray, in case we get a test failure
        if ray.is_initialized():
            ray.shutdown()


@pytest.mark.parametrize("env_cfg", ENV_CFG_TEST_CONFIGS)
def test_all_benchmarks(chain_ex, file_observer, env_cfg):
    bench_available, why = auto.benchmark_is_available(
        env_cfg['benchmark_name'])
    if not bench_available:
        pytest.skip(why)

    chain_config = copy.deepcopy(CHAIN_CONFIG)
    # don't search over representation learner
    chain_config['spec']['repl']['algo'] \
        = tune.grid_search([algos.SimCLR])
    # try just this benchmark
    chain_config['spec']['env_cfg'] = tune.grid_search([env_cfg])
    try:
        chain_ex.run(config_updates=chain_config)
    finally:
        if ray.is_initialized():
            ray.shutdown()


@pytest.mark.parametrize("stages", list(StagesToRun))
def test_individual_stages(chain_ex, file_observer, stages):
    # test just doing IL, just doing REPL, etc.
    chain_config = copy.deepcopy(CHAIN_CONFIG)
    # again, don't search over representation learner
    chain_config['spec']['repl']['algo'] \
        = tune.grid_search([algos.SimCLR])
    chain_config['stages_to_run'] = stages
    try:
        chain_ex.run(config_updates=chain_config)
    finally:
        if ray.is_initialized():
            ray.shutdown()
