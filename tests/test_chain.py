import copy

import pytest
import ray
from ray import tune

from il_representations import algos
from il_representations.scripts.pretrain_n_adapt import StagesToRun
from il_representations.test_support.configuration import (
    CHAIN_CONFIG, ENV_CFG_TEST_CONFIGS)
from il_representations.utils import hash_configs
from time import time

def test_chain(chain_ex, file_observer):
    try:
        chain_ex.run(config_updates=CHAIN_CONFIG)
    finally:
        # always shut down Ray, in case we get a test failure
        if ray.is_initialized():
            ray.shutdown()


@pytest.mark.parametrize("env_cfg", ENV_CFG_TEST_CONFIGS)
def test_all_benchmarks(chain_ex, file_observer, env_cfg):
    chain_config = copy.deepcopy(CHAIN_CONFIG)
    # don't search over representation learner
    chain_config['spec']['repl']['algo'] \
        = tune.grid_search([algos.SimCLR])
    # try just this benchmark
    chain_config['spec']['env_cfg'] = tune.grid_search([env_cfg])
    chain_config['force_repl_run'] = True
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
    chain_config['force_repl_run'] = True
    try:
        chain_ex.run(config_updates=chain_config)
    finally:
        if ray.is_initialized():
            ray.shutdown()


def test_repl_reuse(chain_ex):
    # test just doing IL, just doing REPL, etc.
    chain_config = copy.deepcopy(CHAIN_CONFIG)
    # again, don't search over representation learner
    chain_config['spec']['repl']['algo'] \
        = tune.grid_search([algos.SimCLR])
    random_id = round(time())
    chain_config['spec']['repl']['exp_ident'] = random_id
    chain_config['stages_to_run'] = StagesToRun.REPL_AND_IL
    chain_config['force_repl_run'] = False

    try:
        first_start_time = time()
        chain_ex.run(config_updates=chain_config)
        first_runtime = time() - first_start_time
    finally:
        if ray.is_initialized():
            ray.shutdown()

    try:
        second_start_time = time()
        chain_ex.run(config_updates=chain_config)
        second_runtime = time() - second_start_time
    finally:
        if ray.is_initialized():
            ray.shutdown()

    try:
        third_start_time = time()
        chain_ex.run(config_updates=chain_config)
        third_runtime = time() - third_start_time
    finally:
        if ray.is_initialized():
            ray.shutdown()

    assert second_runtime < first_runtime
    assert third_runtime > second_runtime


def test_hash_config():
    chain_config = copy.deepcopy(CHAIN_CONFIG)
    repl_config = chain_config['repl']
    repl_config['algo'] = algos.TemporalCPC
    config_hash_1 = hash_configs(repl_config)
    config_hash_2 = hash_configs(repl_config)
    repl_config['algo'] = algos.SimCLR
    diff_config_hash = hash_configs(repl_config)
    assert config_hash_1 == config_hash_2, "Sequential hashes from hash_config for " \
                                           "identical config dict do not match"
    assert diff_config_hash != config_hash_1, "Hashes for different config dicts from " \
                                              "hash_config result in identical hashes"
