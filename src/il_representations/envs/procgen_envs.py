"""Utilities for working with Atari environments and demonstrations."""
import os
import random
import numpy as np

from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from procgen import ProcgenEnv

from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient)


@env_data_ingredient.capture
def _get_procgen_data_opts(data_root, procgen_demo_paths):
    # workaround for Sacred issue #206
    return data_root, procgen_demo_paths


@env_cfg_ingredient.capture
def load_dataset_procgen(task_name, n_traj=None, chans_first=True):
    data_root, procgen_demo_paths = _get_procgen_data_opts()

    # load trajectories from disk
    full_rollouts_path = os.path.join(data_root, procgen_demo_paths[task_name])
    trajectories = np.load(full_rollouts_path, allow_pickle=True)

    # do frame stacking on observations in each loaded trajectory sequence,
    # then concatenate the frame-stacked trajectories together to make one big
    # dataset
    cat_obs = np.concatenate(trajectories['obs'][:-1], axis=0)
    cat_nobs = np.concatenate(trajectories['obs'][1:], axis=0)
    # the remaining entries don't need any special stacking, so we just
    # concatenate them
    cat_acts = np.concatenate(trajectories['acts'], axis=0)
    cat_infos = np.concatenate(trajectories['infos'], axis=0)
    cat_rews = np.concatenate(trajectories['rews'], axis=0)
    cat_dones = np.concatenate(trajectories['dones'], axis=0)

    dataset_dict = {
        'obs': cat_obs,
        'next_obs': cat_nobs,
        'acts': cat_acts,
        'infos': cat_infos,
        'rews': cat_rews,
        'dones': cat_dones,
    }

    # TODO: Figure out whether we need chans first for procgen
    if chans_first:
        for key in ('obs', 'next_obs'):
            dataset_dict[key] = np.transpose(dataset_dict[key], (0, 3, 1, 2))

    return dataset_dict


@env_cfg_ingredient.capture
def ProcgenWrapper(task_name, num_envs=1, num_levels=0, start_level=0,
                   distribution_mode='easy'):
    # TODO: Check start level
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, start_level=start_level, distribution_mode=distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)



