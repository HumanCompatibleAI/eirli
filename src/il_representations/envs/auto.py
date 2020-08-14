"""Code for automatically loading data, creating vecenvs, etc. based on
Sacred configuration."""

from imitation.util.util import make_vec_env
from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

from il_representations.algos.augmenters import ColorSpace
from il_representations.envs.atari_envs import load_dataset_atari
from il_representations.envs.config import benchmark_ingredient
from il_representations.envs.dm_control_envs import load_dataset_dm_control
from il_representations.envs.magical_envs import (get_env_name_magical,
                                                  load_dataset_magical)

ERROR_MESSAGE = "no support for benchmark_name={benchmark['benchmark_name']!r}"


@benchmark_ingredient.capture
def load_dataset(benchmark_name):
    if benchmark_name == 'magical':
        dataset_dict = load_dataset_magical()
    elif benchmark_name == 'dm_control':
        dataset_dict = load_dataset_dm_control()
    elif benchmark_name == 'atari':
        dataset_dict = load_dataset_atari()
    else:
        raise NotImplementedError(ERROR_MESSAGE.format(**locals()))
    return dataset_dict


@benchmark_ingredient.capture
def load_vec_env(benchmark_name,
                 atari_env_id,
                 dm_control_full_env_names,
                 dm_control_env,
                 parallel=False,
                 n_envs=1):
    if benchmark_name == 'magical':
        gym_env_name = get_env_name_magical()
        # FIXME(sam): I don't think that new versions of SB3 still use the
        # 'parallel' kwarg.
        return make_vec_env(gym_env_name, n_envs=n_envs, parallel=parallel)
    elif benchmark_name == 'dm_control':
        gym_env_name = dm_control_full_env_names[dm_control_env]
        return make_vec_env(gym_env_name, n_envs=n_envs, parallel=parallel)
    elif benchmark_name == 'atari':
        gym_env_name_hwc = atari_env_id
        assert not parallel, "currently does not support parallel kwarg"
        return VecTransposeImage(
            VecFrameStack(
                make_atari_env(gym_env_name_hwc,
                               n_envs=n_envs), 4))
    raise NotImplementedError(ERROR_MESSAGE.format(**locals()))


@benchmark_ingredient.capture
def load_color_space(benchmark_name):
    color_spaces = {
        'magical': ColorSpace.RGB,
        'dm_control': ColorSpace.RGB,
        'atari': ColorSpace.GRAY,
    }
    try:
        return color_spaces[benchmark_name]
    except KeyError:
        raise NotImplementedError(ERROR_MESSAGE.format(**locals()))
