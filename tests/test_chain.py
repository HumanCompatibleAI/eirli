import copy
import pytest
from il_representations.test_support.configuration import CHAIN_CONFIG
from il_representations.test_support.configuration import BENCHMARK_TEST_CONFIGS

import warnings
for category in [FutureWarning, DeprecationWarning, PendingDeprecationWarning]:
    warnings.filterwarnings("ignore", category=category)


@pytest.mark.parametrize("benchmark_cfg", BENCHMARK_TEST_CONFIGS)
def test_chain(benchmark_cfg, chain_ex, file_observer):
    common_config = {
        'benchmark': benchmark_cfg,
        'device_name': 'cpu'
    }
    chain_config = copy.deepcopy(CHAIN_CONFIG)
    chain_config['spec']['rep']['benchmark'] = benchmark_cfg
    chain_config['spec']['il_train'].update(common_config)
    chain_config['spec']['il_test'].update(common_config)

    chain_ex.run(config_updates=chain_config)


