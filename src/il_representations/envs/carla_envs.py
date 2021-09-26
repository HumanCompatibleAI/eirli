import os
import math
import numpy as np
import gym.spaces as spaces

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
    data_root, carla_demo_paths = _get_carla_data_opts()

    # load trajectories from disk
    full_rollouts_path = os.path.join(data_root, carla_demo_paths[task_name])
    trajectories = np.load(full_rollouts_path, allow_pickle=True)

    obs = trajectories['observations'].astype(np.uint8)
    nobs = trajectories['next_observations'].astype(np.uint8)
    acts = trajectories['actions'].astype(np.uint8)
    rews = trajectories['rewards'].astype(np.uint8)
    dones = trajectories['terminals'].astype(np.uint8)

    if n_traj is not None:
        nth_traj_end_idx = [i for i, n in enumerate(cat_dones) if n][n_traj-1] + 1
        obs = obs[:nth_traj_end_idx]
        nobs = nobs[:nth_traj_end_idx]
        acts = acts[:nth_traj_end_idx]
        rews = rews[:nth_traj_end_idx]
        dones = dones[:nth_traj_end_idx]

    dataset_dict = {
        'obs': obs,
        'next_obs': nobs,
        'acts': acts,
        'rews': rews,
        'dones': dones,
    }

    if chans_first:
        for key in ('obs', 'next_obs'):
            dataset_dict[key] = np.transpose(dataset_dict[key], (0, 3, 1, 2))

    dataset_dict['obs'] = stack_obs_oldest_first(dataset_dict['obs'],
                                                 carla_frame_stack,
                                                 use_zeroed_frames=True)
    dataset_dict['next_obs'] = stack_obs_oldest_first(dataset_dict['next_obs'],
                                                 carla_frame_stack,
                                                 use_zeroed_frames=True)

    return dataset_dict

class CarlaImageObsEnv:
    def __init__(self, wrapped_env):
        self.wrapped_env = wrapped_env
        self.action_space = self.wrapped_env.action_space
        width, height = math.sqrt(wrapped_env.observation_space.shape[0]/3), \
                        math.sqrt(wrapped_env.observation_space.shape[0]/3)
        assert width.is_integer() and height.is_integer(), \
            f'{width} and {height} are not valid integers.'
        low, high = 0, 255
        self.observation_space = spaces.Box(low=low,
            high=high,
            shape=(3, int(width), int(height)),
            dtype=np.float
        )
        self.num_envs = 1
        self.step_buffer = {'obs': [], 'rews': [], 'dones': [], 'infos': [],
                            'returned': True}

    def reset(self):
        obs = self.wrapped_env.reset()
        obs = obs.reshape([1] + list(self.observation_space.shape))
        return obs

    def step(self, action):
        next_obs, reward, done, info = self.wrapped_env.step(action)
        next_obs = next_obs.reshape([1] + list(self.observation_space.shape))
        return next_obs, reward, done, info

    def step_async(self, actions):
        assert self.step_buffer['returned'] is True, \
            'Previously called step_async without obtaining rollouts with step_wait.'
        self.step_buffer = {'obs': [], 'rews': [], 'dones': [], 'infos': [],
                            'returned': False}
        for action in actions:
            o, r, d, i = self.step(action)
            self.step_buffer['obs'].append(o)
            self.step_buffer['rews'].append(r)
            self.step_buffer['dones'].append(d)
            self.step_buffer['infos'].append(i)

    def step_wait(self):
        self.step_buffer['returned'] = True
        return (np.array(self.step_buffer['obs']).squeeze(axis=0),
                np.array(self.step_buffer['rews']),
                np.array(self.step_buffer['dones']),
                np.array(self.step_buffer['infos']))

    def close(self):
        self.wrapped_env.terminate()

    def __getattr__(self, attr):
        if attr == 'wrapped_env':
            raise AttributeError()
        return getattr(self.wrapped_env, attr)
