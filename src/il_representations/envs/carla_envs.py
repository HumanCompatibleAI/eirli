import os
import numpy as np

from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient)
from il_representations.envs.utils import stack_obs_oldest_first


@env_data_ingredient.capture
def _get_carla_data_opts(data_root, carla_demo_paths):
    # workaround for Sacred issue #206
    return data_root, carla_demo_paths


@env_cfg_ingredient.capture
def load_dataset_carla(task_name, carla_frame_stack, chans_first=True,
                       n_traj=None):
    data_root, carla_demo_paths = _get_procgen_data_opts()

    # load trajectories from disk
    full_rollouts_path = os.path.join(data_root, carla_demo_paths[task_name])
    trajectories = np.load(full_rollouts_path, allow_pickle=True)
    breakpoint()

    # cat_obs = np.concatenate(trajectories['obs'], axis=0)[:-1]
    # cat_nobs = np.concatenate(trajectories['obs'], axis=0)[1:]
    # cat_acts = np.concatenate(trajectories['acts'], axis=0)
    # cat_rews = np.concatenate(trajectories['rews'], axis=0)
    # cat_dones = np.concatenate(trajectories['dones'], axis=0)
    # if n_traj is not None:
    #     nth_traj_end_idx = [i for i, n in enumerate(cat_dones) if n][n_traj-1] + 1
    #     cat_obs = cat_obs[:nth_traj_end_idx]
    #     cat_nobs = cat_nobs[:nth_traj_end_idx]
    #     cat_acts = cat_acts[:nth_traj_end_idx]
    #     cat_rews = cat_rews[:nth_traj_end_idx]
    #     cat_dones = cat_dones[:nth_traj_end_idx]

    # dataset_dict = {
    #     'obs': cat_obs,
    #     'next_obs': cat_nobs,
    #     'acts': cat_acts,
    #     'rews': cat_rews,
    #     'dones': cat_dones,
    # }

    # if chans_first:
    #     for key in ('obs', 'next_obs'):
    #         dataset_dict[key] = np.transpose(dataset_dict[key], (0, 3, 1, 2))
    # dataset_dict['obs'] = stack_obs_oldest_first(dataset_dict['obs'],
    #                                              procgen_frame_stack,
    #                                              use_zeroed_frames=True)
    # dataset_dict['next_obs'] = stack_obs_oldest_first(dataset_dict['next_obs'],
    #                                              procgen_frame_stack,
    #                                              use_zeroed_frames=True)

    # return dataset_dict
