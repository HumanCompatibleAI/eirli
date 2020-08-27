"""Fixtures for pytest."""
import pytest
from sacred.observers import FileStorageObserver

from il_representations.scripts.il_test import il_test_ex as _il_test_ex
from il_representations.scripts.il_train import il_train_ex as _il_train_ex
from il_representations.scripts.pretrain_n_adapt import chain_ex as _chain_ex
from il_representations.scripts.run_rep_learner import \
    represent_ex as _represent_ex


@pytest.fixture
def file_observer():
    # this will get created anew for each test, and added to experiments as
    # necessary
    return FileStorageObserver('runs/test_observer')


def _observer_fixture(ex, file_observer):
    # adds a file storage observer to the given experiment, but remembers to
    # delete it after the test is done, too
    ex.observers.append(file_observer)
    yield ex
    ex.observers.remove(file_observer)


@pytest.fixture
def represent_ex(file_observer):
    # We use this fixtures instead of importing represent_ex directly so that
    # we always have a single FileStorageObserver attached to it. Same deal for
    # the other Sacred experiments, too.
    yield from _observer_fixture(_represent_ex, file_observer)


@pytest.fixture
def il_train_ex(file_observer):
    yield from _observer_fixture(_il_train_ex, file_observer)


@pytest.fixture
def il_test_ex(file_observer):
    yield from _observer_fixture(_il_test_ex, file_observer)


@pytest.fixture
def chain_ex(file_observer):
    yield from _observer_fixture(_chain_ex, file_observer)
