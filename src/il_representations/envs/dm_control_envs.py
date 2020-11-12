"""Importing this file will automatically register all relevant DM-Control
environments with Gym."""
import glob
import gzip
import os
import random

import cloudpickle
import dmc2gym
import gym
import numpy as np

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
        channels_first=True)

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
def _stack_obs_oldest_first(obs_arr, dm_control_frame_stack):
    """Takes an array of shape [T, C, H, W] and stacks the entries to produce a
    new array of shape [T, C*frame_stack, H, W] with frames stacked along the
    channels axis. Frames at stacked oldest-first, and the first frame_stack-1
    frames have zeros instead of older frames (because older frames don't
    exist). This is meant to be compatible with VecFrameStack in SB3. Notably,
    it is _not_ compatible with frame stacking in Gym, which repeats the first
    frame instead of using zeroed frames."""
    frame_accumulator = np.repeat(np.zeros_like(obs_arr[0][None]),
                                  dm_control_frame_stack,
                                  axis=0)
    c, h, w = obs_arr.shape[1:]
    out_sequence = []
    for in_frame in obs_arr:
        # drop the oldest frame, and append the new frame
        frame_accumulator = np.concatenate(
            [frame_accumulator[1:], in_frame[None]], axis=0)
        out_sequence.append(frame_accumulator.reshape(
            dm_control_frame_stack * c, h, w))
    out_sequence = np.stack(out_sequence, axis=0)
    return out_sequence


@benchmark_ingredient.capture
def load_dataset_dm_control(dm_control_env, dm_control_full_env_names,
                            dm_control_demo_patterns, dm_control_frame_stack,
                            n_traj, data_root):
    # load data from all relevant paths
    data_pattern = dm_control_demo_patterns[dm_control_env]
    user_pattern = os.path.expanduser(data_pattern)
    data_paths = glob.glob(os.path.join(data_root, user_pattern))
    loaded_trajs = []
    for data_path in data_paths:
        with gzip.GzipFile(data_path, 'rb') as fp:
            new_data = cloudpickle.load(fp)
        loaded_trajs.extend(new_data)

    loaded_trajs = list(loaded_trajs)
    random.shuffle(loaded_trajs)
    if n_traj is not None:
        loaded_trajs = loaded_trajs[:n_traj]

    # join together all trajectories into a single dataset
    dones_lists = [
        # for each trajectory of length T (not including final observation), we
        # create an array of `dones` consisting of T-1 False values and one
        # terminal True value
        np.array([False] * (len(t.acts) - 1) + [True], dtype='bool')
        for t in loaded_trajs
    ]

    # do frame stacking on observations in each loaded trajectory sequence,
    # then concatenate the frame-stacked trajectories together to make one big
    # dataset
    cat_obs = np.concatenate([
        _stack_obs_oldest_first(t.obs[:-1]) for t in loaded_trajs], axis=0)
    cat_nobs = np.concatenate([
        _stack_obs_oldest_first(t.obs[1:]) for t in loaded_trajs], axis=0)
    # the remaining entries don't need any special stacking, so we just
    # concatenate them
    cat_acts = np.concatenate([t.acts for t in loaded_trajs], axis=0)
    cat_infos = np.concatenate([t.infos for t in loaded_trajs], axis=0)
    cat_rews = np.concatenate([t.rews for t in loaded_trajs], axis=0)
    cat_dones = np.concatenate(dones_lists, axis=0)

    dataset_dict = {
        'obs': cat_obs,
        'next_obs': cat_nobs,
        'acts': cat_acts,
        'infos': cat_infos,
        'rews': cat_rews,
        'dones': cat_dones,
    }

    return dataset_dict


register_dmc_envs()
