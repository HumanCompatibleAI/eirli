import inspect

import pytest

from il_representations import algos
from il_representations.test_support.configuration import BENCHMARK_TEST_CONFIGS, REPL_SMOKE_TEST_CONFIG
from il_representations.test_support.utils import is_representation_learner

named_configs_available = ['magical']

@pytest.mark.parametrize("algo", [
    el[1] for el in inspect.getmembers(algos)
    if is_representation_learner(el[1])
])
@pytest.mark.parametrize("benchmark_cfg", BENCHMARK_TEST_CONFIGS)
def test_algo(algo, benchmark_cfg, represent_ex):
    if benchmark_cfg['benchmark_name'] in named_configs_available:
        named_configs = [benchmark_cfg['benchmark_name']]
    else:
        named_configs = []
    represent_ex.run(config_updates={
        **REPL_SMOKE_TEST_CONFIG, 'algo': algo, 'benchmark': benchmark_cfg,
    }, named_configs=named_configs)
