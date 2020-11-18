"""Make a repL WebDataset from demonstration data."""
import logging
import numbers
import os
import sys

import numpy as np
import sacred
from sacred import Experiment
from tqdm import tqdm
import webdataset

from il_representations.algos.utils import set_global_seeds
import il_representations.envs.auto as auto_env
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)
from il_representations.envs.utils import serialize_gym_space
from il_representations.scripts.utils import sacred_copy

sacred.SETTINGS['CAPTURE_MODE'] = 'sys'  # workaround for sacred issue#740
mkdataset_demos_ex = Experiment('mkdataset_demos',
                                ingredients=[env_cfg_ingredient,
                                             env_data_ingredient,
                                             venv_opts_ingredient])


@mkdataset_demos_ex.config
def default_config():
    # shuffle order of different trajectories, while retaining order within
    # each single trajectory
    shuffle_traj_order = True
    # overwrite the default destination
    custom_out_file_path = None
    # put an upper limit on number of trajectories to load
    n_traj_total = None
    # TODO(sam): support sharding

    _ = locals()
    del _


@mkdataset_demos_ex.main
def run(seed, env_data, env_cfg, shuffle_traj_order, custom_out_file_path,
        n_traj_total):
    set_global_seeds(seed)
    # python built-in logging
    logging.basicConfig(level=logging.INFO)

    dataset_dict = auto_env.load_dataset(n_traj=n_traj_total)
    n_items = len(dataset_dict['obs'])
    # keys in dataset_dict: 'obs', 'next_obs', 'acts', 'infos', 'rews', 'dones'
    # numeric_types = (np.ndarray, numbers.Number, np.bool_)

    if custom_out_file_path is None:
        # figure out which directory to write to
        pd = '_processed_data_dirs'  # shorthand for key suffix
        if env_cfg['benchmark_name'] == 'magical':
            pfx = env_cfg['magical_env_prefix']
            data_root = env_data[f'magical{pd}'][pfx]['demos']
        elif env_cfg['benchmark_name'] == 'dm_control':
            ename = env_cfg['dm_control_env_name']
            data_root = env_data[f'dm_control{pd}'][ename]['demos']
        elif env_cfg['benchmark_name'] == 'atari':
            eid = env_cfg['atari_env_id']
            data_root = env_data[f'atari{pd}'][eid]['demos']
        else:
            raise NotImplementedError(
                f'cannot handle {env_cfg["benchmark_name"]}')

        out_file_path = os.path.join(data_root, 'demos.tgz')
    else:
        out_file_path = custom_out_file_path

    out_dir = os.path.dirname(out_file_path)
    if out_dir:
        logging.info(f"Creating output directory '{out_dir}'")
        os.makedirs(out_dir, exist_ok=True)

    # figure out what config keys to keep
    # (we remove keys for benchmarks other than the current one)
    all_env_names = {'magical', 'dm_control', 'atari'}
    other_env_names = all_env_names - {env_cfg['benchmark_name']}
    assert len(other_env_names) == len(all_env_names) - 1
    env_cfg_copy = sacred_copy(env_cfg)
    env_cfg_stripped = {
        k: v for k, v in env_cfg_copy.items()
        if not any(k.startswith(n) for n in other_env_names)
    }

    color_space = auto_env.load_color_space()
    venv = auto_env.load_vec_env()
    meta_dict = {
        'env_cfg': env_cfg_stripped,
        'action_space': serialize_gym_space(venv.action_space),
        'observation_space': serialize_gym_space(venv.observation_space),
        'color_space': color_space,
    }
    venv.close()

    logging.info(f"Will write {n_items} outputs to '{out_file_path}'")
    with webdataset.TarWriter(out_file_path, keep_meta=True, compress=True) \
         as writer:
        # first write _metadata.meta.pickle containing the benchmark config
        writer.dwrite(key='_metadata', meta_pickle=meta_dict)

        if shuffle_traj_order:
            item_order = np.random.permutation(n_items)
        else:
            item_order = np.arange(n_items)

        if os.isatty(sys.stdout.fileno()):
            range_iter = tqdm(item_order, desc='items')
        else:
            range_iter = tqdm(item_order)

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
                    # this, so I'm going to save things as pickles for now,
                    # even though that may be slower.
                    write_dict[key + '.pickle'] = array[i]
            writer.write(write_dict)


if __name__ == '__main__':
    # there are no FileObservers for this Sacred experiment; we are just using
    # Sacred to parse arguments
    mkdataset_demos_ex.run_commandline()
