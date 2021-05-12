import os
import random
import numpy as np

from procgen.gym_registration import make_env, register_environments

from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient)


@env_data_ingredient.capture
def _get_procgen_data_opts(data_root, procgen_demo_paths):
    # workaround for Sacred issue #206
    return data_root, procgen_demo_paths


@env_cfg_ingredient.capture
def load_dataset_procgen(task_name, procgen_frame_stack, n_traj=None,
                         chans_first=True):
    data_root, procgen_demo_paths = _get_procgen_data_opts()

    # load trajectories from disk
    full_rollouts_path = os.path.join(data_root, procgen_demo_paths[task_name])
    trajectories = np.load(full_rollouts_path, allow_pickle=True)

    cat_obs = np.concatenate(trajectories['obs'], axis=0)
    cat_acts = np.concatenate(trajectories['acts'], axis=0)
    cat_rews = np.concatenate(trajectories['rews'], axis=0)
    cat_dones = np.concatenate(trajectories['dones'], axis=0)

    dataset_dict = {
        'obs': cat_obs,
        'acts': cat_acts,
        'rews': cat_rews,
        'dones': cat_dones,
    }

    if chans_first:
        for key in ('obs', ):
            dataset_dict[key] = np.transpose(dataset_dict[key], (0, 3, 1, 2))
    dataset_dict['obs'] = _stack_obs_oldest_first(dataset_dict['obs'],
                                                  procgen_frame_stack)

    return dataset_dict


@env_cfg_ingredient.capture
def get_procgen_env_name(task_name):
    return f'procgen-{task_name}-v0'


@env_cfg_ingredient.capture
def _stack_obs_oldest_first(obs_arr, procgen_frame_stack):
    frame_accumulator = np.repeat([obs_arr[0]], procgen_frame_stack, axis=0)
    c, h, w = obs_arr.shape[1:]
    out_sequence = []
    for in_frame in obs_arr:
        frame_accumulator = np.concatenate(
            [frame_accumulator[1:], [in_frame]], axis=0)
        out_sequence.append(frame_accumulator.reshape(
            procgen_frame_stack * c, h, w))
    out_sequence = np.stack(out_sequence, axis=0)
    return out_sequence
