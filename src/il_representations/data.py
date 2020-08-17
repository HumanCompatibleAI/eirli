"""Some utilities for converting between data formats for `imitation` and for
Torch."""

import imitation.data.datasets as il_datasets
import imitation.data.types as il_types
import numpy as np


class TransitionsMinimalDataset(il_datasets.Dataset):
    """Exposes a dict {'obs': <observations ndarray>, 'acts'} as a dataset that
    enumerates `TransitionsMinimal` instances. Useful for interfacing with
    BC."""
    def __init__(self, data_map):
        req_keys = {'obs', 'acts'}
        assert req_keys <= data_map.keys()
        self.dict_dataset = il_datasets.RandomDictDataset(
            {k: data_map[k] for k in req_keys})

    def sample(self, n_samples):
        dict_samples = self.dict_dataset.sample(n_samples)
        # we don't have infos dicts, so we insert some fake ones to make
        # TransitionsMinimal happy
        dummy_infos = np.asarray([{} for _ in range(n_samples)], dtype='object')
        result = il_types.TransitionsMinimal(infos=dummy_infos, **dict_samples)
        assert len(result) == n_samples
        return result

    def size(self):
        return self.dict_dataset.size()
