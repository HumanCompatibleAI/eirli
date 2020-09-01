import ray

from il_representations.test_support.configuration import (
    BENCHMARK_TEST_CONFIGS, CHAIN_CONFIG)


def test_chain(chain_ex, file_observer):
    try:
        chain_ex.run(config_updates={
            **CHAIN_CONFIG,
            'spec': {
                'benchmark': ray.tune.grid_search(BENCHMARK_TEST_CONFIGS)
            }
        })
    finally:
        # always shut down Ray, in case we get a test failure
        if ray.is_initialized():
            ray.shutdown()
