import os
import copy
import pytest
from sacred.observers import FileStorageObserver
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

    final_pol_name = 'last_test_policy.pt'
    log_dir = file_observer.dir
    policy_path = os.path.join(log_dir, final_pol_name)
    chain_config['spec']['policy_path'].update(policy_path)

    observer = FileStorageObserver('runs/chain_runs')
    chain_ex.observers.append(observer)
    chain_ex.run(config_updates=chain_config)


if __name__ == '__main__':
    from il_representations.scripts.pretrain_n_adapt import chain_ex
    for bm in BENCHMARK_TEST_CONFIGS:
        test_chain(bm, chain_ex)

