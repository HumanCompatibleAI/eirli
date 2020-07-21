import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
for category in [FutureWarning, DeprecationWarning, PendingDeprecationWarning]:
    warnings.filterwarnings("ignore", category=category)
import inspect
import algos
from run_rep_learner import represent_ex
from sacred.observers import FileStorageObserver
import pytest

represent_ex.observers.append(FileStorageObserver('test_observer'))


def is_representation_learner(el):
    try:
        return issubclass(el, algos.RepresentationLearner) and el != algos.RepresentationLearner and el not in algos.WIP_ALGOS
    except TypeError:
        return False


@pytest.mark.parametrize("algo", [el[1] for el in inspect.getmembers(algos) if is_representation_learner(el[1])])
def test_algo(algo):
    print()
    print()
    print(f"Testing: {algo}")
    represent_ex.run(config_updates={'pretrain_epochs': 1, 'timesteps': 500,
                                     'train_from_expert': False, 'algo': algo})