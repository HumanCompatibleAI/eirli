import logging
import os
import time

from gym import Wrapper, spaces
import numpy as np
from copy import deepcopy
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient )
from il_representations.envs.utils import wrap_env


def optional_observation_map(env, inner_obs):
    if hasattr(env, 'observation'):
        return env.observation(inner_obs)
    else:
        return inner_obs


def optional_action_map(env, inner_action):
    if hasattr(env, 'wrap_action'):
        return env.wrap_action(inner_action)
    else:
        return inner_action


def remove_iterator_dimension(dict_obs_or_act):
    output_dict = dict()
    for k in dict_obs_or_act.keys():
        if isinstance(dict_obs_or_act[k], dict):
            output_dict[k] = remove_iterator_dimension(dict_obs_or_act[k])
        else:
            output_dict[k] = dict_obs_or_act[k][0]
    return output_dict

@env_cfg_ingredient.capture
def get_env_name_minecraft(task_name):
    if task_name in ('FindCaves'):
        return f"{task_name}-v0"
    else:
        return f"MineRL{task_name}-v0"


@env_data_ingredient.capture
def _get_data_root(data_root):
    # workaround for Sacred issue #206
    return os.path.join(data_root, 'data')


# Note: this also uses configuration values from env_data_ingredient
# even though it can only be notated as a capture function for one
# ingredient at a time
@env_cfg_ingredient.capture
def load_dataset_minecraft(minecraft_wrappers, n_traj, frames_per_traj, chunk_length=100):
    import minerl  # lazy-load in case it is not installed
    import realistic_benchmarks.envs.envs # Registers new environments
    from realistic_benchmarks.utils import DummyEnv
    data_root = _get_data_root()
    env_name = get_env_name_minecraft()
    minecraft_data_root = os.path.join(data_root, 'minecraft')
    data_pipeline = minerl.data.make(environment=env_name,
                                     data_dir=minecraft_data_root)
    appended_trajectories = {'obs': [], 'acts': [], 'dones': [], 'next_obs': []}
    start_time = time.time()

    env_spec = deepcopy(data_pipeline.spec)
    dummy_env = DummyEnv(action_space=env_spec._action_space,
                         observation_space=env_spec._observation_space)
    wrapped_dummy_env = wrap_env(dummy_env, minecraft_wrappers)
    timesteps = 0

    trajectory_names = data_pipeline.get_trajectory_names()
    trajectory_subset = np.random.choice(trajectory_names, size=n_traj)
    for trajectory_name in trajectory_subset:
        data_loader = data_pipeline.load_data(trajectory_name)
        traj_frame_count = 0
        for current_obs, action, reward, next_obs, done in data_loader:
            wrapped_obs = optional_observation_map(wrapped_dummy_env, current_obs)
            wrapped_next_obs = optional_observation_map(wrapped_dummy_env, next_obs)
            wrapped_action = optional_action_map(wrapped_dummy_env, action)
            appended_trajectories['obs'].append(wrapped_obs)
            appended_trajectories['next_obs'].append(wrapped_next_obs)
            appended_trajectories['acts'].append(wrapped_action)
            appended_trajectories['dones'].append(done)
            traj_frame_count += 1
            timesteps += 1

            if timesteps % 1000 == 0:
                print(f"{timesteps} timesteps loaded")

            # if frames_per_traj is None, collect the whole trajectory
            if frames_per_traj is not None and traj_frame_count == frames_per_traj:
                appended_trajectories['dones'][-1] = True
                break
    end_time = time.time()
    for k in appended_trajectories:
        appended_trajectories[k] = np.array(appended_trajectories[k])
    logging.info(f"Minecraft trajectory collection took {round(end_time - start_time, 2)} seconds to complete")
    appended_trajectories['dones'][-1] = True
    return appended_trajectories


def construct_next_obs(trajectories_dict):
    # iterate over data
    # Figure out locations of dones/marking end of trajectory
    # For each trajectory, construct a next_obs vector that is obs[1:] + None
    dones_locations = np.where(trajectories_dict['dones'])[0]
    dones_locations = np.append(dones_locations, -1)
    prior_dones_loc = 0
    all_next_obs = []
    print(f"Done locations to process {dones_locations}")
    for done_loc in dones_locations:
        if done_loc == -1:
            trajectory_obs = trajectories_dict['obs'][prior_dones_loc:]
        else:
            trajectory_obs = trajectories_dict['obs'][prior_dones_loc:done_loc+1]
        next_obs = trajectory_obs[1:]
        expanded_thing = np.expand_dims(trajectory_obs[-1], axis=0)
        next_obs = np.append(next_obs, expanded_thing, axis=0) #duplicate final obs for final next_obs
        all_next_obs.append(next_obs)
        prior_dones_loc = done_loc

    del next_obs
    del trajectory_obs
    if len(all_next_obs) == 1:
        all_next_obs = all_next_obs[0]
    else:
        print("Concatenating")
        all_next_obs = np.concatenate(all_next_obs) #maybe this will make memory less horrible?
    trajectories_dict['next_obs'] = all_next_obs
    return trajectories_dict


def channels_first(el):
    if isinstance(el, np.ndarray):
        return np.moveaxis(el, -1, -3)

    elif isinstance(el, tuple):
        return (el[2], el[0], el[1])

    else:
        raise NotImplementedError("Input must be either array or tuple")