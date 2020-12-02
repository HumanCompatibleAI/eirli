"""Make a repL WebDataset from demonstration data."""
import logging
import os
import random
import sys
import warnings

import numpy as np
import sacred
from sacred import Experiment
from tqdm import tqdm

from il_representations.algos.utils import set_global_seeds
from il_representations.data.write_dataset import get_meta_dict, write_frames
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
    dataset_dict = auto_env.load_dict_dataset(n_traj=n_traj_total)
    n_samples = len(dataset_dict['obs'])
    # keys in dataset_dict: 'obs', 'next_obs', 'acts', 'infos', 'rews', 'dones'
    # numeric_types = (np.ndarray, numbers.Number, np.bool_)

    # split dataset into trajectories
    trajectories = []
    all_dones = dataset_dict['dones'].copy()
    added_final_done = False
    if not all_dones[-1]:
        # For, e.g., Atari, some of the data does not have a 'done' at the
        # final time step because the trajectory was truncated. For the purpose
        # of inferring trajectory boundaries, we insert a fake 'done' at the
        # end.
        warnings.warn("No 'done' at end of trajectories; inserting a fake one")
        all_dones[-1] = True
        added_final_done = True
    traj_ends, = np.nonzero(all_dones)
    # add one to ends so that when we index with array[start:end], we get the
    # full trajectory
    traj_ends = traj_ends + 1
    traj_starts = np.concatenate(([0], traj_ends[:-1]), axis=0)
    traj_start_end = np.stack((traj_starts, traj_ends), axis=1)
    for start_idx, end_idx in traj_start_end:
        trajectories.append({
            k: v[start_idx:end_idx] for k, v in dataset_dict.items()
        })
    n_traj = len(trajectories)

    # output path based on task name & benchmark name
    out_file_path = os.path.join(
        auto_env.get_data_dir(benchmark_name=env_cfg['benchmark_name'],
                              task_key=env_cfg['task_name'],
                              data_type='demos'), 'demos.tgz')

    # get metadata for the dataset
    meta_dict = get_meta_dict()

    if shuffle_traj_order:
        # write trajectories in random order
        random.shuffle(trajectories)

    def frame_gen():
        for traj_num, traj in enumerate(trajectories):
            traj_len = len(traj['obs'])

            # probably an indexing bug if this assert fails; we should not have
            # any zero-length trajectories
            assert traj_len > 0, 'somehow ended up with zero-length traj?'

            for idx in range(traj_len):
                sub_dict = {}
                for key, arr in traj.items():
                    sub_dict[key] = arr[idx]
                yield sub_dict

            # again, if this assert fails then there's probably an off-by-one
            is_last_traj = traj_num == len(trajectories) - 1
            assert sub_dict['dones'] or (is_last_traj and added_final_done), \
                "final step of trajectory was not 'done'"

    frame_iter = frame_gen()
    if os.isatty(sys.stdout.fileno()):
        frame_iter = tqdm(frame_iter, desc='steps', total=n_samples)

    logging.info(f"Will write {n_samples} frames ({n_traj} trajectories) to "
                 f"'{out_file_path}'")
    write_frames(out_file_path, meta_dict, frame_iter)


if __name__ == '__main__':
    # there are no FileObservers for this Sacred experiment; we are just using
    # Sacred to parse arguments
    mkdataset_demos_ex.run_commandline()
