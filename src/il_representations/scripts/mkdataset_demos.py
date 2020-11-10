"""Make a repL WebDataset from demonstration data."""
import logging
import numbers
import os
import sys

import numpy as np
import sacred
from sacred import Experiment
from tqdm import trange
import webdataset

from il_representations.algos.utils import set_global_seeds
import il_representations.envs.auto as auto_env
from il_representations.envs.config import benchmark_ingredient

sacred.SETTINGS['CAPTURE_MODE'] = 'sys'  # workaround for sacred issue#740
mkdataset_demos_ex = Experiment('mkdataset_demos',
                                ingredients=[benchmark_ingredient])


@mkdataset_demos_ex.config
def default_config():
    # TODO(Sam): support sharding
    out_file = 'dataset.tgz'
    # shuffle order of different trajectories, while retaining order within
    # each single trajectory
    shuffle_traj_order = True

    _ = locals()
    del _


@mkdataset_demos_ex.main
def run(out_file, seed, benchmark):
    set_global_seeds(seed)
    # python built-in logging
    logging.basicConfig(level=logging.INFO)

    # TODO(sam): actually write this
    dataset_dict = auto_env.load_dataset()
    n_items = len(dataset_dict['obs'])
    # keys in dataset_dict: 'obs', 'next_obs', 'acts', 'infos', 'rews', 'dones'
    # numeric_types = (np.ndarray, numbers.Number, np.bool_)

    out_dir = os.path.dirname(out_file)
    if out_dir:
        logging.info(f"Creating output directory '{out_dir}'")
        os.makedirs(out_dir, exist_ok=True)

    # TODO(sam): should figure out precisely what needs to be saved, and save
    # no more than that. Really it should be limited to the data required to
    # instantiate a new environment of the correct type.
    meta_dict = benchmark

    logging.info(f"Will write {n_items} outputs to '{out_file}'")
    with webdataset.TarWriter(out_file, keep_meta=True, compress=True) \
         as writer:
        # first write _metadata.json containing the benchmark config
        writer.dwrite(key='_metadata', json=meta_dict)

        if os.isatty(sys.stdout.fileno()):
            range_iter = trange(n_items, desc='items')
        else:
            range_iter = range(n_items)

        for i in range_iter:
            write_dict = {'__key__': 'frame_%03d' % i}
            for key, array in dataset_dict.items():
                if isinstance(array[0], (dict, numbers.Number, np.bool_)):
                    # dicts (e.g. for 'infos') get stored as pickles
                    # (same goes for basic numeric types)
                    assert isinstance(array[i], (dict, numbers.Number,
                                                 np.bool_))
                    write_dict[key + '.pyd'] = array[i]
                else:
                    # everything else gets stored with minimal array format
                    # (the "tenbin" format from WebDataset)
                    assert isinstance(array[i], np.ndarray), type(array[i])
                    write_dict[key + '.ten'] = array[i]
            writer.write(write_dict)


if __name__ == '__main__':
    # there are no FileObservers for this Sacred experiment; we are just using
    # Sacred to parse arguments
    mkdataset_demos_ex.run_commandline()
