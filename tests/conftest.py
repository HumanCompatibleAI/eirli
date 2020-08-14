import pytest
from sacred.observers import FileStorageObserver

from il_representations.scripts.il_test import il_test_ex as _il_test_ex
from il_representations.scripts.il_train import il_train_ex as _il_train_ex
from il_representations.scripts.run_rep_learner import \
    represent_ex as _represent_ex


@pytest.fixture
def file_observer():
    return FileStorageObserver('test_observer')


def _observer_fixture(ex, file_observer):
    ex.observers.append(file_observer)
    yield ex
    ex.observers.remove(file_observer)


@pytest.fixture
def represent_ex(file_observer):
    yield from _observer_fixture(_represent_ex, file_observer)


@pytest.fixture
def il_train_ex(file_observer):
    yield from _observer_fixture(_il_train_ex, file_observer)


@pytest.fixture
def il_test_ex(file_observer):
    yield from _observer_fixture(_il_test_ex, file_observer)
