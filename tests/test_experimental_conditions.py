import inspect

import pytest

from il_representations.test_support.configuration import BENCHMARK_TEST_CONFIGS, REPL_SMOKE_TEST_CONFIG
from il_representations.scripts import experimental_conditions


@pytest.mark.parametrize("experimental_condition",
                         [el[0] for el in inspect.getmembers(experimental_conditions)
                          if el[0].startswith('condition')])
@pytest.mark.parametrize("benchmark_cfg", BENCHMARK_TEST_CONFIGS)
def test_algo(experimental_condition, benchmark_cfg, represent_ex):
    represent_ex.run(named_configs=[experimental_condition], config_updates={
        **REPL_SMOKE_TEST_CONFIG, 'benchmark': benchmark_cfg,
    })
