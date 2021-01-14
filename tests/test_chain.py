import copy

import pytest
import ray
from ray import tune

from il_representations import algos
from il_representations.scripts.utils import StagesToRun, ReuseRepl
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


def _time_ray_run(ex, config_to_run):
    try:
        start_time = time()
        ex.run(config_updates=config_to_run)
        runtime = time() - start_time
    finally:
        if ray.is_initialized():
            ray.shutdown()
    return runtime


def test_repl_reuse(chain_ex):
    """
    An test of the functionality of reusing repl encoders. This test works
    by running the same chain run experiment (with a unique timestamp ID)
    twice in a row, and confirming that the second time RepL is reused. We
    currently check this somewhat heuristically, by demanding that the
    second run take meaningfully less time. We then modify one element of the
    config, and confirm that repl is now run again, because there are no
    existing matching runs (again, checking via the heuristic of longer
    running time compared to the cache run)
    """

    chain_config = copy.deepcopy(CHAIN_CONFIG)
    chain_config['reuse_repl'] = ReuseRepl.IF_AVAILABLE
    chain_config['spec']['repl']['algo'] \
        = tune.grid_search([algos.SimCLR])
    random_id = round(time())

    chain_config['repl']['exp_ident'] = random_id
    chain_config['repl']['batches_per_epoch'] = 15
    chain_config['stages_to_run'] = StagesToRun.REPL_AND_IL

    first_runtime = _time_ray_run(chain_ex, chain_config)
    second_runtime = _time_ray_run(chain_ex, chain_config)


    modified_config = copy.deepcopy(chain_config)
    modified_config['spec']['repl']['seed'] = tune.grid_search([42])
    third_runtime = _time_ray_run(chain_ex, modified_config)

    assert second_runtime < first_runtime
    assert third_runtime > second_runtime


def test_hash_config():
    """
    A test of the config-hashing used in the repl reuse logic. This test
    is designed to specifically confirm that hashing twice without changes
    to the config dict yields the same hash, and that modifying an element
    of the config dict changes the hash

    """
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
