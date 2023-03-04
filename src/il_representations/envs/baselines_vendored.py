"""
This file contains functions copied from somewhere else, where our code base
requires some minimal modifications of the original function. This is a more
lightweight solution than some others, e.g., installing a full package just
for one supporting function, or submitting PRs to change the original package.
"""
from abc import ABC, abstractmethod
from collections import deque
import time

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper


class VecEnvObservationWrapper(VecEnvWrapper, ABC):
    """ Copied from openai/baselines/common/vec_env/vec_env.py. """
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
    """ Copied from openai/baselines/common/vec_env/vec_remove_dict_obs.py. """
    def __init__(self, venv, key):
        self.key = key
        super().__init__(venv=venv,
                         observation_space=venv.observation_space.spaces[self.key])

    def process(self, obs):
        return obs[self.key]


class VecMonitor(VecEnvWrapper):
    """
    Modified from openai/baselines/common/vec_env/vec_monitor.py. The imitation
    package requires the environment to return 'terminal_observation' as part
    of the step info, so we add it here.
    """
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
