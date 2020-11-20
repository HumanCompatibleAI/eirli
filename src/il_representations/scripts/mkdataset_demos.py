"""Make a repL WebDataset from demonstration data."""
import logging
import os
import sys

import numpy as np
import sacred
from sacred import Experiment
from tqdm import tqdm

from il_representations.algos.utils import set_global_seeds
from il_representations.data.write_dataset import (get_meta_dict,
                                                   get_out_file_map,
                                                   write_frames)
import il_representations.envs.auto as auto_env
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)

sacred.SETTINGS['CAPTURE_MODE'] = 'sys'  # workaround for sacred issue#740
mkdataset_demos_ex = Experiment('mkdataset_demos',
                                ingredients=[
                                    env_cfg_ingredient, env_data_ingredient,
                                    venv_opts_ingredient
                                ])


@mkdataset_demos_ex.config
def default_config():
    # shuffle order of different trajectories, while retaining order within
    # each single trajectory
    shuffle_traj_order = True
    # put an upper limit on number of trajectories to load
    n_traj_total = None
    # TODO(sam): support sharding

    _ = locals()
    del _


@mkdataset_demos_ex.main
def run(seed, env_data, env_cfg, shuffle_traj_order, n_traj_total):
    set_global_seeds(seed)
    # python built-in logging
    logging.basicConfig(level=logging.INFO)

    # load existing demo dictionary directly, w/ same code used to handle data
    # in il_train.py
    dataset_dict = auto_env.load_dataset(n_traj=n_traj_total)
    n_items = len(dataset_dict['obs'])
    # keys in dataset_dict: 'obs', 'next_obs', 'acts', 'infos', 'rews', 'dones'
    # numeric_types = (np.ndarray, numbers.Number, np.bool_)

    out_file_map = get_out_file_map()
    out_file_path = os.path.join(out_file_map['demos'], 'demos.tgz')

    meta_dict = get_meta_dict()

    if shuffle_traj_order:
        # write trajectories in random order
        item_order = np.random.permutation(n_items)
    else:
        # write trajectories in default order
        item_order = np.arange(n_items)

    def frame_gen():
        for idx in item_order:
            sub_dict = {}
            for key, arr in dataset_dict.items():
                sub_dict[key] = arr[idx]
            yield sub_dict

    frame_iter = frame_gen()
    if os.isatty(sys.stdout.fileno()):
        frame_iter = tqdm(frame_iter, desc='steps', total=n_items)

    logging.info(f"Will write {n_items} frames to '{out_file_path}'")
    write_frames(out_file_path, meta_dict, frame_iter)


if __name__ == '__main__':
    # there are no FileObservers for this Sacred experiment; we are just using
    # Sacred to parse arguments
    mkdataset_demos_ex.run_commandline()
