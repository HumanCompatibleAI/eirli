import pytest
import ray

from il_representations.test_support.configuration import (
    BENCHMARK_TEST_CONFIGS, CHAIN_CONFIG)


@pytest.mark.parametrize('benchmark_cfg', BENCHMARK_TEST_CONFIGS)
def test_chain(benchmark_cfg, chain_ex, file_observer):
    try:
        chain_ex.run(config_updates={
            **CHAIN_CONFIG,
            'benchmark': benchmark_cfg,
        })
    finally:
        # always shut down Ray, in case we get a test failure
        if ray.is_initialized():
            ray.shutdown()
