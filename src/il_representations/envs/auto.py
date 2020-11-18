"""Code for automatically loading data, creating vecenvs, etc. based on
Sacred configuration."""

import glob
import logging
import os

from imitation.util.util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

from il_representations.algos.augmenters import ColorSpace
from il_representations.data.read_dataset import load_ilr_datasets
from il_representations.envs.atari_envs import load_dataset_atari
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)
from il_representations.envs.dm_control_envs import load_dataset_dm_control
from il_representations.envs.magical_envs import (get_env_name_magical,
                                                  load_dataset_magical)
from il_representations.scripts.utils import update as dict_update

ERROR_MESSAGE = "no support for benchmark_name={benchmark_name!r}"


@env_cfg_ingredient.capture
def load_dataset(benchmark_name, n_traj=None):
    if benchmark_name == 'magical':
        dataset_dict = load_dataset_magical(n_traj=n_traj)
    elif benchmark_name == 'dm_control':
        dataset_dict = load_dataset_dm_control(n_traj=n_traj)
    elif benchmark_name == 'atari':
        dataset_dict = load_dataset_atari(n_traj=n_traj)
    else:
        raise NotImplementedError(ERROR_MESSAGE.format(**locals()))

    num_transitions = len(dataset_dict['dones'].flatten())
    num_dones = dataset_dict['dones'].flatten().sum()
    logging.info(f'Loaded dataset with {num_transitions} transitions. '
                 f'{num_dones} of these transitions have done == True')

    return dataset_dict


@env_cfg_ingredient.capture
def get_gym_env_name(benchmark_name, atari_env_id, dm_control_full_env_names,
                     dm_control_env_name):
    if benchmark_name == 'magical':
        return get_env_name_magical()
    elif benchmark_name == 'dm_control':
        return dm_control_full_env_names[dm_control_env_name]
    elif benchmark_name == 'atari':
        return atari_env_id
    raise NotImplementedError(ERROR_MESSAGE.format(**locals()))


@venv_opts_ingredient.capture
def _get_venv_opts(n_envs, venv_parallel):
    # helper to extract options from venv_opts, since we can't have two
    # captures on one function (see Sacred issue #206)
    return n_envs, venv_parallel


@env_cfg_ingredient.capture
def load_vec_env(benchmark_name, atari_env_id, dm_control_full_env_names,
                 dm_control_frame_stack):
    """Create a vec env for the selected benchmark task and wrap it with any
    necessary wrappers."""
    n_envs, venv_parallel = _get_venv_opts()
    gym_env_name = get_gym_env_name()
    if benchmark_name == 'magical':
        return make_vec_env(gym_env_name,
                            n_envs=n_envs,
                            parallel=venv_parallel)
    elif benchmark_name == 'dm_control':
        raw_dmc_env = make_vec_env(gym_env_name,
                                   n_envs=n_envs,
                                   parallel=venv_parallel)
        final_env = VecFrameStack(raw_dmc_env, n_stack=dm_control_frame_stack)
        dmc_chans = raw_dmc_env.observation_space.shape[0]

        # make sure raw env has 3 channels (should be RGB, IIRC)
        assert dmc_chans == 3

        # make sure stacked env has dmc_chans*frame_stack channels
        expected_shape = (dm_control_frame_stack * dmc_chans, ) \
            + raw_dmc_env.observation_space.shape[1:]
        assert final_env.observation_space.shape == expected_shape, \
            (final_env.observation_space.shape, expected_shape)

        # make sure images are square
        assert final_env.observation_space.shape[1:] \
            == final_env.observation_space.shape[1:][::-1]

        return final_env
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


@env_cfg_ingredient.capture
def _get_default_env_cfg(_config):
    return _config


@env_data_ingredient.capture
def load_new_style_ilr_dataset(configs,
                               dm_control_processed_data_dirs,
                               magical_processed_data_dirs,
                               atari_processed_data_dirs):
    """Load a new-style dataset for representation learning.

    Args:
        configs ([dict]): list of dicts with the following keys:
            - `type`: "random" or "demos", as appropriate.
            - `env_cfg`: subset of keys from `env_cfg_ingredient` specifying a
            particular environment name, etc.
          If any of the above keys are missing, they will be filled in with
          defaults: `type` defaults to "demos", and `env_cfg` keys are taken
          from `env_cfg_ingredient` by default. Only keys that differ from
          those defaults need to be overridden. For instance, if
          `env_cfg_ingredient` was configured with
          `benchmark_name="dm_control"`, then you could set `configs =
          [{"type": "random", "env_cfg": {"dm_control_env_name":
          "finger_spin"}}]` to use only rollouts from the `finger-spin`
          environment.

    (all other args are taken from env_data_ingredient)"""
    # by default we load demos for the configured base environment
    defaults = {
        'type': 'demos',
        'env_cfg': _get_default_env_cfg(),
    }
    all_tar_files = []

    if len(configs) == 0:
        raise ValueError("no dataset configurations supplied")

    for config in configs:
        # generate config dict, including defaults
        assert isinstance(config, dict) and config.keys() <= {
            'type', 'env_cfg'
        }
        orig_config = config
        config = dict_update(defaults, config)
        data_type = config['type']
        env_cfg = config['env_cfg']

        if env_cfg["benchmark_name"] == "magical":
            pfx = env_cfg['magical_env_prefix']
            data_root = magical_processed_data_dirs[pfx][data_type]
        elif env_cfg["benchmark_name"] == "dm_control":
            ename = env_cfg['dm_control_env_name']
            data_root = dm_control_processed_data_dirs[ename][data_type]
        elif env_cfg["benchmark_name"] == "atari":
            eid = env_cfg['atari_env_id']
            data_root = atari_processed_data_dirs[eid][data_type]
        else:
            raise NotImplementedError(
                f'cannot handle {env_cfg["benchmark_name"]}')

        tar_files = glob.glob(os.path.join(data_root, "*.tgz"))
        if len(tar_files) == 0:
            raise IOError(
                f"did not find any files in '{data_root}' (for dataset config "
                f"'{orig_config}')")
        all_tar_files.extend(tar_files)

    return load_ilr_datasets(all_tar_files)


@env_cfg_ingredient.capture
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
