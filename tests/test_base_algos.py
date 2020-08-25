import inspect
import warnings
for category in [FutureWarning, DeprecationWarning, PendingDeprecationWarning]:
    warnings.filterwarnings("ignore", category=category)

import pytest
from sacred.observers import FileStorageObserver

from il_representations import algos
from il_representations.test_support.configuration import BENCHMARK_TEST_CONFIGS


def is_representation_learner(el):
    try:
        return issubclass(el, algos.RepresentationLearner) and el != algos.RepresentationLearner and el not in algos.WIP_ALGOS
    except TypeError:
        return False


@pytest.mark.parametrize("algo", [el[1] for el in inspect.getmembers(algos) if is_representation_learner(el[1])])
@pytest.mark.parametrize("benchmark_cfg", BENCHMARK_TEST_CONFIGS)
def test_algo(algo, benchmark_cfg, represent_ex):
    represent_ex.run(config_updates={'pretrain_epochs': 1,
                                     'timesteps': 32,
                                     'batch_size': 7,
                                     'unit_test_max_train_steps': 2,
                                     'representation_dim': 3,
                                     'algo': algo,
                                     'use_random_rollouts': False,
                                     'benchmark': benchmark_cfg,
                                     'ppo_finetune': False})

