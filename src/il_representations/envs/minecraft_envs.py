import logging
import os
import time

from gym import Wrapper, spaces
import numpy as np

from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient )
from il_representations.envs.utils import wrap_env


def optional_observation_map(env, inner_obs):
    if hasattr(env, 'observation'):
        return env.observation(inner_obs)
    else:
        return inner_obs


def optional_action_map(env, inner_action):
    if hasattr(env, 'reverse_action'):
        return env.reverse_action(inner_action)
    else:
        return inner_action


def remove_iterator_dimension(dict_obs_or_act):
    output_dict = dict()
    for k in dict_obs_or_act.keys():
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
def load_dataset_minecraft(minecraft_wrappers, n_traj=None, chunk_length=100):
    import minerl  # lazy-load in case it is not installed
    import realistic_benchmarks.envs.envs # Registers new environments
    from realistic_benchmarks.utils import DummyEnv
    data_root = _get_data_root()
    env_name = get_env_name_minecraft()
    minecraft_data_root = os.path.join(data_root, 'minecraft')
    data_iterator = minerl.data.make(environment=env_name,
                                     data_dir=minecraft_data_root)
    appended_trajectories = {'obs': [], 'acts': [], 'dones': []}
    start_time = time.time()

    env_spec = data_iterator.spec

    # TODO undo this hack when nearbySmelt is no longer a string
    dummy_env = DummyEnv(action_space=env_spec._action_space,
                         observation_space=env_spec._observation_space)

    wrapped_dummy_env = wrap_env(dummy_env, minecraft_wrappers)

    for current_obs, action, reward, next_obs, done in data_iterator.batch_iter(batch_size=1,
                                                                                num_epochs=1,
                                                                                epoch_size=n_traj, # TODO does this make sense as a value of epoch_size?
                                                                                seq_len=chunk_length):
        # Data returned from the data_iterator is in batches of size `batch_size` x `chunk_size`
        # The zero-indexing is to remove the extra extraneous `batch_size` dimension,
        # which has been hardcoded to 1
        reshaped_obs = remove_iterator_dimension(current_obs)
        reshaped_action = remove_iterator_dimension(action)
        wrapped_obs = optional_observation_map(wrapped_dummy_env, reshaped_obs)
        wrapped_action = optional_action_map(wrapped_dummy_env, reshaped_action)
        appended_trajectories['obs'].append(wrapped_obs)
        appended_trajectories['acts'].append(wrapped_action)
        appended_trajectories['dones'].append(done[0])
    # Now, we need to go through and construct `next_obs` values, which aren't natively returned
    # by the environment
    merged_trajectories = {k: np.concatenate(v, axis=0) for k, v in appended_trajectories.items()}
    merged_trajectories = construct_next_obs(merged_trajectories)
    end_time = time.time()
    logging.info(f"Minecraft trajectory collection took {round(end_time - start_time, 2)} seconds to complete")
    merged_trajectories['dones'][-1] = True
    return merged_trajectories


def construct_next_obs(trajectories_dict):
    # iterate over data
    # Figure out locations of dones/marking end of trajectory
    # For each trajectory, construct a next_obs vector that is obs[1:] + None
    dones_locations = np.where(trajectories_dict['dones'])[0]
    dones_locations = np.append(dones_locations, -1)
    prior_dones_loc = 0
    all_next_obs = []
    for done_loc in dones_locations:
        if done_loc == -1:
            trajectory_obs = trajectories_dict['obs'][prior_dones_loc:]
        else:
            trajectory_obs = trajectories_dict['obs'][prior_dones_loc:done_loc+1]
        next_obs = trajectory_obs[1:]
        next_obs = np.append(next_obs, np.expand_dims(trajectory_obs[-1], axis=0), axis=0) #duplicate final obs for final next_obs
        all_next_obs.append(next_obs)
    if len(all_next_obs) == 1:
        merged_next_obs = all_next_obs[0]
    else:
        merged_next_obs = np.concatenate(all_next_obs)
    trajectories_dict['next_obs'] = merged_next_obs
    return trajectories_dict


def channels_first(el):
    if isinstance(el, np.ndarray):
        return np.moveaxis(el, -1, -3)

    elif isinstance(el, tuple):
        return (el[2], el[0], el[1])

    else:
        raise NotImplementedError("Input must be either array or tuple")