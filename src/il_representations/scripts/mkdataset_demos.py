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
from il_representations.scripts.utils import sacred_copy

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

    dataset_dict = auto_env.load_dataset()
    n_items = len(dataset_dict['obs'])
    # keys in dataset_dict: 'obs', 'next_obs', 'acts', 'infos', 'rews', 'dones'
    # numeric_types = (np.ndarray, numbers.Number, np.bool_)

    out_dir = os.path.dirname(out_file)
    if out_dir:
        logging.info(f"Creating output directory '{out_dir}'")
        os.makedirs(out_dir, exist_ok=True)

    # TODO(sam): refactor the 'benchmark' Sacred experiment so that it splits
    # out the keys necessary to instantiate new environments from the keys
    # necessary to load data (possibly even make a separate 'dataset' config
    # ingredient for the latter). That way we can save just the env stuff in
    # this function.
    color_space = auto_env.load_color_space()
    venv = auto_env.load_vec_env()
    meta_dict = {
        'benchmark_config': sacred_copy(benchmark),
        'action_space': venv.action_space,
        'observation_space': venv.observation_space,
        'color_space': color_space,
    }
    venv.close()

    logging.info(f"Will write {n_items} outputs to '{out_file}'")
    with webdataset.TarWriter(out_file, keep_meta=True, compress=True) \
         as writer:
        # first write _metadata.meta.pickle containing the benchmark config
        writer.dwrite(key='_metadata', meta_pickle=meta_dict)

        if os.isatty(sys.stdout.fileno()):
            range_iter = trange(n_items, desc='items')
        else:
            range_iter = range(n_items)

        for i in range_iter:
            write_dict = {'__key__': 'frame_%03d' % i, 'frame.pyd': i}
            for key, array in dataset_dict.items():
                if isinstance(array[0], (dict, numbers.Number, np.bool_)):
                    # dicts (e.g. for 'infos') get stored as pickles
                    # (same goes for basic numeric types)
                    assert isinstance(array[i], (dict, numbers.Number,
                                                 np.bool_))
                    write_dict[key + '.pickle'] = array[i]
                else:
                    # everything else gets stored with minimal array format
                    assert isinstance(array[i], np.ndarray), type(array[i])
                    # write_dict[key + '.ten'] = array[i]
                    # FIXME(sam): tenbin currently (as of 2020-11-16) seems to
                    # decode every array as a singleton list of arrays (i.e. it
                    # converts arr -> [arr]). I can't figure out why it's doing
                    # this, so I'm going to save things as pickles for now.
                    write_dict[key + '.pickle'] = array[i]
            writer.write(write_dict)


if __name__ == '__main__':
    # there are no FileObservers for this Sacred experiment; we are just using
    # Sacred to parse arguments
    mkdataset_demos_ex.run_commandline()
