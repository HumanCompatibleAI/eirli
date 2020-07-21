#!/usr/bin/env python3
"""Run an IL algorithm in some selected domain."""
import glob
import logging
import os

import gym
from magical.benchmarks import ChannelsFirst
from imitation.algorithms.bc import BC
from imitation.algorithms.adversarial import GAIL
from imitation.util.util import make_vec_env
import imitation.util.logger as imitation_logger
from sacred import Experiment
from sacred.observers import FileStorageObserver
import stable_baselines3.common.policies as sb3_pols
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.ppo import PPO

from il_representations.envs import magical_envs
from il_representations.envs.config import benchmark_ingredient
from il_representations.il.disc_rew_nets import ImageDiscrimNet

imitation_ex = Experiment('imitation', ingredients=[benchmark_ingredient])


@imitation_ex.config
def default_config():
    # FIXME(sam): come up with a more portable way of storing demonstration
    # paths. Maybe a per-user YAML file mapping {env name: data dir}?
    demo_pattern = '~/repos/magical/demos-ea/move-to-corner-2020-03-01/demo-*.pkl.gz'  # noqa: E501, F841
    algo = 'bc'  # noqa: F841


@imitation_ex.main
def run(algo, demo_pattern, _config):
    # python built-in logging
    logging.basicConfig(level=logging.INFO)
    # `imitation` logging
    # FIXME(sam): I used this hack from run_rep_learner.py, but I don't
    # actually know the right way to write continuous logs in sacred.
    imitation_logger.configure(imitation_ex.observers[0].dir,
                               ["stdout", "tensorboard"])

    logging.info(f"Loading trajectory data from '{demo_pattern}'")
    pickle_paths = glob.glob(os.path.expanduser(demo_pattern))
    gym_env_name_chans_last, dataset = magical_envs.load_data(
        pickle_paths, transpose_observations=True)

    logging.info(
        f"Loaded data for '{gym_env_name_chans_last}', setting up env")
    env_chans_last = gym.make(gym_env_name_chans_last)
    env_chans_first = ChannelsFirst(env_chans_last)

    logging.info(f"Setting up '{algo}' IL algorithm")

    if algo == 'bc':
        trainer = BC(observation_space=env_chans_first.observation_space,
                     action_space=env_chans_first.action_space,
                     policy_class=sb3_pols.ActorCriticCnnPolicy,
                     expert_data=dataset)

        logging.info("Beginning BC training")
        trainer.train(n_epochs=1)

    elif algo == 'gail':
        # Annoyingly, SB3 always adds a VecTransposeWrapper to the vec_env
        # that we pass in, so we have to build an un-transposed env first.
        vec_env_chans_last = make_vec_env(gym_env_name_chans_last,
                                          n_envs=2,
                                          seed=0,
                                          parallel=False)
        # vec_env_chans_first = VecTransposeImage(vec_env_chans_last)
        discrim_net = ImageDiscrimNet(
            observation_space=env_chans_first.observation_space,
            action_space=env_chans_first.action_space)
        ppo_algo = PPO(
            policy=sb3_pols.ActorCriticCnnPolicy,
            env=vec_env_chans_last,
            # verbose=1 and tensorboard_log=False is a hack to work around SB3
            # issue #109.
            verbose=1,
            tensorboard_log=None,
        )
        trainer = GAIL(
            vec_env_chans_last,
            dataset,
            ppo_algo,
            discrim_kwargs=dict(discrim_net=discrim_net),
        )
        # 2020-07-20: this doesn't work because SB3 tries to be clever about
        # automatically transposing images, but inadvertently causes images to
        # be transposed _twice_. The problematic line is "self._last_obs =
        # self.env.reset()" in BaseAlgorithm._setup_learn().

        # UPDATE: it happens b/c GAIL is overwriting the env inside PPO with a
        # wrapped environment that does _not_ apply VecTransposeImage. Hacky
        # solution is to make GAIL apply VecTranposeImage after wrapping the
        # environment. Non-hacky solution is to make SB3 algorithms *not wrap
        # environments by default*---that just seems like a brittle
        # anti-solution to the problem they're trying to solve.
        trainer.train(total_timesteps=2048)

    else:
        raise NotImplementedError(f"Can't handle algorithm '{algo}'")


if __name__ == '__main__':
    imitation_ex.observers.append(FileStorageObserver('imitation_runs'))
    imitation_ex.run_commandline()
