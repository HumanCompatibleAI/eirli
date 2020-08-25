"""Code for automatically loading data, creating vecenvs, etc. based on
Sacred configuration."""

from imitation.util.util import make_vec_env
from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

from il_representations.algos.augmenters import ColorSpace
from il_representations.envs.atari_envs import load_dataset_atari
from il_representations.envs.config import benchmark_ingredient
from il_representations.envs.dm_control_envs import load_dataset_dm_control
from il_representations.envs.minecraft_envs import load_dataset_minecraft, MinecraftVectorWrapper
from il_representations.envs.magical_envs import (get_env_name_magical,
                                                  load_dataset_magical)
import minerl

ERROR_MESSAGE = "no support for benchmark_name={benchmark['benchmark_name']!r}"


@benchmark_ingredient.capture
def load_dataset(benchmark_name):
    if benchmark_name == 'magical':
        dataset_dict = load_dataset_magical()
    elif benchmark_name == 'dm_control':
        dataset_dict = load_dataset_dm_control()
    elif benchmark_name == 'atari':
        dataset_dict = load_dataset_atari()
    elif benchmark_name == 'minecraft':
        dataset_dict = load_dataset_minecraft()
    else:
        raise NotImplementedError(ERROR_MESSAGE.format(**locals()))
    return dataset_dict


@benchmark_ingredient.capture
def get_gym_env_name(benchmark_name, atari_env_id, minecraft_env_id, dm_control_full_env_names,
                     dm_control_env):
    if benchmark_name == 'magical':
        return get_env_name_magical()
    elif benchmark_name == 'dm_control':
        return dm_control_full_env_names[dm_control_env]
    elif benchmark_name == 'atari':
        return atari_env_id
    elif benchmark_name == 'minecraft':
        return minecraft_env_id
    raise NotImplementedError(ERROR_MESSAGE.format(**locals()))


@benchmark_ingredient.capture
def load_vec_env(benchmark_name,
                 atari_env_id,
                 dm_control_full_env_names,
                 dm_control_env,
                 parallel=False,
                 n_envs=1):
    """Create a vec env for the selected benchmark task and wrap it with any
    necessary wrappers."""
    gym_env_name = get_gym_env_name()
    if benchmark_name in ('magical', 'dm_control'):
        # FIXME(sam): I don't think that new versions of SB3 still use the
        # 'parallel' kwarg.
        return make_vec_env(gym_env_name, n_envs=n_envs, parallel=parallel)
    elif benchmark_name == 'minecraft':
        return make_vec_env(gym_env_name, n_envs=n_envs, parallel=parallel, wrapper_class=MinecraftVectorWrapper)
    elif benchmark_name == 'atari':
        assert not parallel, "currently does not support parallel kwarg"
        final_env = VecFrameStack(
            VecTransposeImage(make_atari_env(gym_env_name,
                                             n_envs=n_envs), ), 4)
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
