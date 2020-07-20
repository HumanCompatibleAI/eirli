#!/usr/bin/env python3
"""Run an IL algorithm in some selected domain."""
import glob
import logging
import os

import gym
from imitation.algorithms.bc import BC
from sacred import Experiment
from sacred.observers import FileStorageObserver
import stable_baselines3.common.policies as sb3_pols

from il_representations.envs import magical_envs

imitation_ex = Experiment('imitation')


@imitation_ex.config
def default_config():
    # FIXME(sam): come up with a more portable way of storing demonstration
    # paths. Maybe a per-user YAML file mapping {env name: data dir}?
    demo_pattern = '~/repos/magical/demos-ea/move-to-corner-2020-03-01/demo-*.pkl.gz'  # noqa: E501, F841


@imitation_ex.main
def run(demo_pattern, _config):
    logging.basicConfig(level=logging.INFO)

    logging.info(f"Loading trajectory data from '{demo_pattern}'")
    pickle_paths = glob.glob(os.path.expanduser(demo_pattern))
    gym_env_name, dataset = magical_envs.load_data(pickle_paths)

    logging.info(f"Loaded data for '{gym_env_name}', setting up env")
    env = gym.make(gym_env_name)

    logging.info("Setting up IL algorithm")
    trainer = BC(observation_space=env.observation_space,
                 action_space=env.action_space,
                 policy_class=sb3_pols.ActorCriticCnnPolicy,
                 expert_data=dataset)

    logging.info("Beginning training")
    trainer.train(n_epochs=1)


if __name__ == '__main__':
    imitation_ex.observers.append(FileStorageObserver('imitation_runs'))
    imitation_ex.run_commandline()
