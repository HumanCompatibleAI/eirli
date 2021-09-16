"""Utilities for working with Atari environments and demonstrations."""
import os
import random

import numpy as np

from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient)


@env_data_ingredient.capture
def _get_atari_data_opts(data_root, atari_demo_paths):
    # workaround for Sacred issue #206
    return data_root, atari_demo_paths


@env_cfg_ingredient.capture
def load_dataset_atari(task_name, n_traj=None, chans_first=True):
    data_root, atari_demo_paths = _get_atari_data_opts()

    # load trajectories from disk
    full_rollouts_path = os.path.join(data_root, atari_demo_paths[task_name])
    trajs_or_file = np.load(full_rollouts_path, allow_pickle=True)
    assert isinstance(trajs_or_file, np.lib.npyio.NpzFile)
    # handle .npz files (several arrays, maybe compressed, but we assume
    # there's only one)
    trajectories, = trajs_or_file.values()

    trajectories = list(trajectories)
    random.shuffle(trajectories)
    if n_traj is not None:
        trajectories = trajectories[:n_traj]

    # merge stats/actions/dones from all trajectories into one big dataset
    # (we use same naming convention as `imitation` here)
    merged_trajectories = {'obs': [], 'next_obs': [], 'acts': [], 'dones': []}
    for traj in trajectories:
        # we slice to :-1 so that we can have a meaningful next_obs
        merged_trajectories['obs'] += traj['states'][:-1]
        merged_trajectories['next_obs'] += traj['states'][1:]
        merged_trajectories['acts'] += traj['actions'][:-1]
        merged_trajectories['dones'] += traj['dones'][:-1]
    dataset_dict = {
        key: np.stack(values, axis=0)
        for key, values in merged_trajectories.items()
    }

    if chans_first:
        # In Gym Atari envs, channels are last; chans_first will transpose data
        # saved in that format so it's channels-first (making it compatible
        # with, e.g., Atari envs wrapped in a VecTransposeImage wrapper).
        for key in ('obs', 'next_obs'):
            dataset_dict[key] = np.transpose(dataset_dict[key], (0, 3, 1, 2))

    return dataset_dict
