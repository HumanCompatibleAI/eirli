import inspect

import pytest

from il_representations import algos
from il_representations.envs import auto
from il_representations.test_support.configuration import (
    ENV_CFG_TEST_CONFIGS, ENV_DATA_TEST_CONFIG, REPL_SMOKE_TEST_CONFIG)
from il_representations.test_support.utils import is_representation_learner


@pytest.mark.parametrize("algo", [
    el[1] for el in inspect.getmembers(algos)
    if is_representation_learner(el[1])
])
@pytest.mark.parametrize("env_cfg", ENV_CFG_TEST_CONFIGS)
def test_algo(algo, env_cfg, represent_ex):
    bench_available, why = auto.benchmark_is_available(
        env_cfg['benchmark_name'])
    if not bench_available:
        pytest.skip(why)
    represent_ex.run(config_updates={
        **REPL_SMOKE_TEST_CONFIG, 'algo': algo, 'env_cfg': env_cfg,
        'env_data': ENV_DATA_TEST_CONFIG,
    })
