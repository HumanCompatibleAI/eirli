"""Importing this file will automatically register all relevant DM-Control
environments with Gym."""
import glob
import gzip
import os

import cloudpickle
import dmc2gym
import gym
from imitation.data.rollout import flatten_trajectories
from imitation.data.datasets import TransitionsDictDatasetAdaptor

from il_representations.envs.config import benchmark_ingredient

IMAGE_SIZE = 100
_REGISTERED = False


def register_dmc_envs():
    # run once
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    common = dict(
        seed=0,
        visualize_reward=False,
        from_pixels=True,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        # we set this to "false" because SB3 does not yet play nicely with
        # channels-first observations
        channels_first=False)

    def entry_point(**kwargs):
        # add in common kwargs
        return dmc2gym.make(**kwargs, **common)

    # frame skip 2
    gym.register('DMC-Finger-Spin-v0',
                 entry_point=entry_point,
                 kwargs=dict(domain_name='finger',
                             task_name='spin',
                             frame_skip=2))

    # frame skip 4
    gym.register('DMC-Cheetah-Run-v0',
                 entry_point=entry_point,
                 kwargs=dict(domain_name='cheetah',
                             task_name='run',
                             frame_skip=4))

    # frame skip 8
    gym.register('DMC-Walker-Walk-v0',
                 entry_point=entry_point,
                 kwargs=dict(domain_name='walker',
                             task_name='walk',
                             frame_skip=8))
    gym.register('DMC-Cartpole-Swingup-v0',
                 entry_point=entry_point,
                 kwargs=dict(domain_name='cartpole',
                             task_name='swingup',
                             frame_skip=8))
    gym.register('DMC-Reacher-Easy-v0',
                 entry_point=entry_point,
                 kwargs=dict(domain_name='reacher',
                             task_name='easy',
                             frame_skip=8))
    gym.register('DMC-Ball-In-Cup-Catch-v0',
                 entry_point=entry_point,
                 kwargs=dict(domain_name='ball_in_cup',
                             task_name='catch',
                             frame_skip=8))


@benchmark_ingredient.capture
def load_dataset_dm_control(dm_control_env, dm_control_full_env_names,
                            dm_control_demo_patterns):
    # get real Gym env name
    gym_env_name_chans_last = dm_control_full_env_names[dm_control_env]

    # load data from all relevant paths
    data_pattern = dm_control_demo_patterns[dm_control_env]
    data_paths = glob.glob(os.path.expanduser(data_pattern))
    loaded_trajectories = []
    for data_path in data_paths:
        with gzip.GzipFile(data_path, 'rb') as fp:
            new_data = cloudpickle.load(fp)
        loaded_trajectories.extend(new_data)

    # join together all trajectories into a single dataset
    transitions = flatten_trajectories(loaded_trajectories)
    del loaded_trajectories
    dataset = TransitionsDictDatasetAdaptor(transitions)

    return gym_env_name_chans_last, dataset


register_dmc_envs()
