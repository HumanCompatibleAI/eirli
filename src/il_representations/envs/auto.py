"""Code for automatically loading data, creating vecenvs, etc. based on
Sacred configuration."""

import logging

from imitation.util.util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
import numpy as np

from il_representations.algos.augmenters import ColorSpace
from il_representations.envs.atari_envs import load_dataset_atari
from il_representations.envs.config import benchmark_ingredient
from il_representations.envs.dm_control_envs import load_dataset_dm_control
from il_representations.envs.minecraft_envs import load_dataset_minecraft, MinecraftVectorWrapper, REAL_TO_MOCK_LOOKUP
from il_representations.envs.magical_envs import (get_env_name_magical,
                                                  load_dataset_magical)
import minerl

ERROR_MESSAGE = "no support for benchmark_name={benchmark['benchmark_name']!r}"

load_dataset_funcs = {
    'magical': load_dataset_magical,
    'dm_control': load_dataset_dm_control,
    'atari': load_dataset_atari,
    'minecraft': load_dataset_minecraft
}
@benchmark_ingredient.capture
def load_dataset(benchmark_name, n_traj, timesteps):
    if benchmark_name in load_dataset_funcs:
        dataset_dict = load_dataset_funcs[benchmark_name](n_traj=n_traj, timesteps=timesteps)
    else:
        raise NotImplementedError(ERROR_MESSAGE.format(**locals()))

    not_dones = np.logical_not(dataset_dict['dones'])
    num_trajectories = not_dones.shape[0]
    num_active_timesteps = not_dones.flatten().sum() + num_trajectories
    logging.info(f'Loaded dataset with {num_trajectories} trajectories and {num_active_timesteps} active timesteps')

    return dataset_dict


@benchmark_ingredient.capture
def get_gym_env_name(benchmark_name, atari_env_id, minecraft_env_id, dm_control_full_env_names,
                     dm_control_env, mock_minecraft):
    if benchmark_name == 'magical':
        return get_env_name_magical()
    elif benchmark_name == 'dm_control':
        return dm_control_full_env_names[dm_control_env]
    elif benchmark_name == 'atari':
        return atari_env_id
    elif benchmark_name == 'minecraft':
        if mock_minecraft:
            return REAL_TO_MOCK_LOOKUP[minecraft_env_id]
        else:
            return minecraft_env_id
    raise NotImplementedError(ERROR_MESSAGE.format(**locals()))


@benchmark_ingredient.capture
def load_vec_env(benchmark_name, atari_env_id, dm_control_full_env_names,
                 dm_control_env, venv_parallel, n_envs):
    """Create a vec env for the selected benchmark task and wrap it with any
    necessary wrappers."""
    gym_env_name = get_gym_env_name()
    if benchmark_name in ('magical', 'dm_control'):
        return make_vec_env(gym_env_name,
                            n_envs=n_envs,
                            parallel=venv_parallel)
    elif benchmark_name == 'minecraft':
        if venv_parallel:
            raise ValueError("MineRL environments can only be run with `venv_parallel`=False as a result of "
                             "issues with starting daemonic processes from SubprocVecEnv")
        return make_vec_env(gym_env_name,
                            n_envs=n_envs,
                            parallel=venv_parallel,
                            wrapper_class=MinecraftVectorWrapper)
    elif benchmark_name == 'atari':
        raw_atari_env = make_vec_env(gym_env_name,
                                     n_envs=n_envs,
                                     parallel=venv_parallel,
                                     wrapper_class=AtariWrapper)
        final_env = VecFrameStack(VecTransposeImage(raw_atari_env), 4)
        assert final_env.observation_space.shape == (4, 84, 84), \
            final_env.observation_space.shape
        return final_env
    raise NotImplementedError(ERROR_MESSAGE.format(**locals()))


@benchmark_ingredient.capture
def load_color_space(benchmark_name):
    color_spaces = {
        'magical': ColorSpace.RGB,
        'dm_control': ColorSpace.RGB,
        'atari': ColorSpace.GRAY,
        'minecraft': ColorSpace.RGB
    }
    try:
        return color_spaces[benchmark_name]
    except KeyError:
        raise NotImplementedError(ERROR_MESSAGE.format(**locals()))
