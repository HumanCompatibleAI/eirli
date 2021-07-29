import os
import numpy as np
import time
from collections import deque
from abc import abstractmethod

from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient)
from il_representations.envs.utils import stack_obs_oldest_first
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper


@env_data_ingredient.capture
def _get_procgen_data_opts(data_root, procgen_demo_paths):
    # workaround for Sacred issue #206
    return data_root, procgen_demo_paths


@env_cfg_ingredient.capture
def load_dataset_procgen(task_name, procgen_frame_stack, chans_first=True,
                         n_traj=None):
    data_root, procgen_demo_paths = _get_procgen_data_opts()

    # load trajectories from disk
    full_rollouts_path = os.path.join(data_root, procgen_demo_paths[task_name])
    trajectories = np.load(full_rollouts_path, allow_pickle=True)

    cat_obs = np.concatenate(trajectories['obs'], axis=0)
    cat_acts = np.concatenate(trajectories['acts'], axis=0)
    cat_rews = np.concatenate(trajectories['rews'], axis=0)
    cat_dones = np.concatenate(trajectories['dones'], axis=0)
    if n_traj is not None:
        nth_traj_end_idx = [i for i, n in enumerate(cat_dones) if n == True][n_traj-1] + 1
        cat_obs = cat_obs[:nth_traj_end_idx]
        cat_acts = cat_acts[:nth_traj_end_idx]
        cat_rews = cat_rews[:nth_traj_end_idx]
        cat_dones = cat_dones[:nth_traj_end_idx]

    dataset_dict = {
        'obs': cat_obs,
        'acts': cat_acts,
        'rews': cat_rews,
        'dones': cat_dones,
    }

    if chans_first:
        for key in ('obs', ):
            dataset_dict[key] = np.transpose(dataset_dict[key], (0, 3, 1, 2))
    dataset_dict['obs'] = stack_obs_oldest_first(dataset_dict['obs'],
                                                 procgen_frame_stack,
                                                 use_zeroed_frames=True)

    return dataset_dict


@env_cfg_ingredient.capture
def get_procgen_env_name(task_name):
    return f'procgen-{task_name}-v0'


class VecEnvObservationWrapper(VecEnvWrapper):
    @abstractmethod
    def process(self, obs):
        pass

    def reset(self):
        obs = self.venv.reset()
        return self.process(obs)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        return self.process(obs), rews, dones, infos


class VecExtractDictObs(VecEnvObservationWrapper):
    def __init__(self, venv, key):
        self.key = key
        super().__init__(venv=venv,
                         observation_space=venv.observation_space.spaces[self.key])

    def process(self, obs):
        return obs[self.key]


class VecMonitor(VecEnvWrapper):
    def __init__(self, venv, filename=None, keep_buf=0, info_keywords=()):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()
        self.info_keywords = info_keywords
        self.keep_buf = keep_buf
        if self.keep_buf:
            self.epret_buf = deque([], maxlen=keep_buf)
            self.eplen_buf = deque([], maxlen=keep_buf)

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
                for k in self.info_keywords:
                    epinfo[k] = info[k]
                info['episode'] = epinfo
                info['terminal_observation'] = obs[0]
                if self.keep_buf:
                    self.epret_buf.append(ret)
                    self.eplen_buf.append(eplen)
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos
