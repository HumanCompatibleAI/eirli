import inspect
import warnings
for category in [FutureWarning, DeprecationWarning, PendingDeprecationWarning]:
    warnings.filterwarnings("ignore", category=category)

import pytest
from sacred.observers import FileStorageObserver

from il_representations import algos
from il_representations.scripts.run_rep_learner import represent_ex
from il_representations.test_support.configuration import BENCHMARK_CONFIGS

represent_ex.observers.append(FileStorageObserver('test_observer'))


def is_representation_learner(el):
    try:
        return issubclass(el, algos.RepresentationLearner) and el != algos.RepresentationLearner and el not in algos.WIP_ALGOS
    except TypeError:
        return False


@pytest.mark.parametrize("algo", [el[1] for el in inspect.getmembers(algos) if is_representation_learner(el[1])])
@pytest.mark.parametrize("benchmark_cfg", BENCHMARK_CONFIGS)
def test_algo(algo, benchmark_cfg):
    represent_ex.run(config_updates={'pretrain_epochs': 1,
                                     'algo': algo,
                                     'use_random_rollouts': False,
                                     'benchmark': benchmark_cfg,
                                     'ppo_finetune': False})
