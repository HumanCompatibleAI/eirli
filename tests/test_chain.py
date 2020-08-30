import copy

import pytest
import ray

from il_representations.test_support.configuration import BENCHMARK_TEST_CONFIGS, CHAIN_CONFIG


@pytest.mark.parametrize("benchmark_cfg", BENCHMARK_TEST_CONFIGS)
def test_chain(benchmark_cfg, chain_ex, file_observer):
    common_config = {
        'benchmark': benchmark_cfg,
        'device_name': 'cpu',
    }
    chain_config = copy.deepcopy(CHAIN_CONFIG)
    chain_config['spec']['rep']['benchmark'] = benchmark_cfg
    chain_config['spec']['il_train'].update(common_config)
    chain_config['spec']['il_test'].update(common_config)

    try:
        chain_ex.run(config_updates=chain_config)
    finally:
        if ray.is_initialized():
            ray.shutdown()
