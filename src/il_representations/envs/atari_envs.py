"""Utilities for working with Atari environments and demonstrations."""
import numpy as np

from stable_baselines3.common.vec_env import VecFrameStack
from il_representations.envs.config import benchmark_ingredient


@benchmark_ingredient.capture
def load_dataset_atari(atari_env_id, atari_demo_paths, chans_first=True):
    # load trajectories from disk
    full_rollouts_path = atari_demo_paths[atari_env_id]
    trajs_or_file = np.load(full_rollouts_path, allow_pickle=True)
    if isinstance(trajs_or_file, np.lib.npyio.NpzFile):
        # handle .npz files (several arrays, maybe compressed, but we assume
        # there's only one)
        trajectories, = trajs_or_file.values()
    else:
        # handle .npy files (one array)
        assert isinstance(trajectories, np.ndarray), type(trajectories)

    # merge stats/actions/dones from all trajectories into one big dataset
    # (we use same naming convention as `imitation` here)
    merged_trajectories = {'obs': [], 'acts': [], 'dones': []}
    for traj in trajectories:
        merged_trajectories['obs'] += traj['states']
        merged_trajectories['acts'] += traj['actions']
        merged_trajectories['dones'] += traj['dones']
    dataset_dict = {
        key: np.stack(values, axis=0)
        for key, values in merged_trajectories.items()
    }

    if chans_first:
        # by default, channels are last; chans_first transposes
        dataset_dict['obs'] = np.transpose(dataset_dict['obs'], (0, 3, 1, 2))

    return dataset_dict
