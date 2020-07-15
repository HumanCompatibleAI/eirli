import warnings

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from algos import *
from run_rep_learner import represent_ex
from sacred.observers import FileStorageObserver
import pytest

represent_ex.observers.append(FileStorageObserver('test_observer'))


def is_representation_learner(el):
    try:
        return issubclass(el, RepresentationLearner) and el != RepresentationLearner and el not in WIP_ALGOS
    except TypeError:
        return False


@pytest.mark.parametrize("algo", [el for el in globals().values() if is_representation_learner(el)])
def test_algo(algo):
    print()
    print()
    print(f"Testing: {algo}")
    represent_ex.run(config_updates={'pretrain_epochs': 1, 'timesteps': 500,
                                     'train_from_expert': False, 'algo': algo})