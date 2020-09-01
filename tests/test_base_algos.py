import inspect
import warnings
for category in [FutureWarning, DeprecationWarning, PendingDeprecationWarning]:
    warnings.filterwarnings("ignore", category=category)

import pytest
from sacred.observers import FileStorageObserver

from il_representations import algos
from il_representations.test_support.configuration import BENCHMARK_TEST_CONFIGS, REPL_SMOKE_TEST_CONFIG
from il_representations.test_support.utils import is_representation_learner


@pytest.mark.parametrize("algo", [
    el[1] for el in inspect.getmembers(algos)
    if is_representation_learner(el[1])
])
@pytest.mark.parametrize("benchmark_cfg", BENCHMARK_TEST_CONFIGS)
def test_algo(algo, benchmark_cfg, represent_ex):

    represent_ex.run(config_updates={
        **REPL_SMOKE_TEST_CONFIG, 'algo': algo, 'benchmark': benchmark_cfg,
    })
