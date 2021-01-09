from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)
import os
import minerl
import numpy as np
from gym import Wrapper, spaces, Env, register
import time
import logging


@env_cfg_ingredient.capture
def get_env_name_minecraft(task_name):
    return f"MineRL{task_name}-v0"


@env_data_ingredient.capture
def _get_data_root(data_root):
    # workaround for Sacred issue #206
    return os.path.join(data_root, 'data')


# Note: this also uses configuration values from env_data_ingredient
# even though it can only be notated as a capture function for one
# ingredient at a time
@env_cfg_ingredient.capture
def load_dataset_minecraft(n_traj=None, chunk_length=100):
    data_root = _get_data_root()
    env_name = get_env_name_minecraft()
    minecraft_data_root = os.path.join(data_root, 'minecraft')
    data_iterator = minerl.data.make(environment=env_name,
                                     data_dir=minecraft_data_root,
                                     max_recordings=n_traj)
    appended_trajectories = {'obs': [], 'acts': [], 'dones': []}
    start_time = time.time()
    for current_state, action, reward, next_state, done in data_iterator.batch_iter(batch_size=1,
                                                                                    num_epochs=1,
                                                                                    seq_len=chunk_length):
        # Data returned from the data_iterator is in batches of size `batch_size` x `chunk_size`
        # The zero-indexing is to remove the extra extraneous `batch_size` dimension,
        # which has been hardcoded to 1
        appended_trajectories['obs'].append(MinecraftVectorWrapper.transform_obs(current_state)[0])
        appended_trajectories['acts'].append(MinecraftVectorWrapper.extract_action(action)[0])
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


class MinecraftVectorWrapper(Wrapper):
    """
    Currently, RepL code only works with pixel inputs, and imitation can only work with vector (rather than dict)
    action spaces. So, we currently (1) only allow VectorObfuscated environments (where the action dictionary
    has been processed into a vector), and (2) extract the observation space to only save the pixels, before we load
    the data in as a il_representations dataset
    """
    def __init__(self, env):
        super().__init__(env)
        assert 'vector' in env.action_space.spaces.keys(), "Wrapper is only implemented to work with Vector Obfuscated envs"
        self.action_space = env.action_space.spaces['vector']
        pov_space = env.observation_space.spaces['pov']
        transposed_pov_space = spaces.Box(low=channels_first(pov_space.low),
                                          high=channels_first(pov_space.high),
                                          shape=channels_first(pov_space.shape),
                                          dtype=np.uint8)
        self.observation_space = transposed_pov_space

    @staticmethod
    def transform_obs(obs):
        return channels_first(obs['pov']).astype(np.uint8)

    @staticmethod
    def extract_action(action):
        return action['vector']

    @staticmethod
    def dictify_action(action):
        return {'vector': action}

    def step(self, action):
        obs, rew, dones, infos = self.env.step(MinecraftVectorWrapper.dictify_action(action))
        transformed_obs = MinecraftVectorWrapper.transform_obs(obs)
        return transformed_obs, rew, dones, infos

    def reset(self):
        obs = self.env.reset()
        return MinecraftVectorWrapper.transform_obs(obs)

