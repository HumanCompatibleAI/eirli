"""Code for automatically loading data, creating vecenvs, etc. based on
Sacred configuration."""

from functools import partial
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
from il_representations.envs.minecraft_envs import (get_env_name_minecraft,
                                                    load_dataset_minecraft)
from il_representations.scripts.utils import update as dict_update
from il_representations.envs.utils import wrap_env

ERROR_MESSAGE = "no support for benchmark_name={benchmark_name!r}"


@env_cfg_ingredient.capture
def benchmark_is_available(benchmark_name):
    """Check whether the selected benchmark is actually available for use on
    this machine. Useful for skipping tests when deps are not installed.

    Returns a tuple of `(benchmark_available, message)`: if
    `benchmark_available` is `False`, then the `message` is a string explaining
    why the benchmark is not available; otherwise, `benchmark_available` is
    `True`, and `message` is `None`."""

    # 2020-01-04: for now this mostly a placeholder: we just assume
    # magical/dm_control/atari are installed by default, and only have logic
    # for skipping MineCraft (which is hard to install). In future, it would
    # make sense to extend this function so that magical and dm_control are
    # also optional (since those also have somewhat involved installation
    # steps).

    if benchmark_name == 'magical':
        return True, None
    elif benchmark_name == 'dm_control':
        return True, None
    elif benchmark_name == 'atari':
        return True, None
    elif benchmark_name == 'minecraft':
        # we check whether minecraft is installed by importing minerl
        try:
            import minerl  # noqa: F401
            import realistic_benchmarks
            return True, None
        except ImportError as ex:
            return False, "MineRL not installed, cannot use Minecraft " \
                f"envs (error: {ex})"
    else:
        raise NotImplementedError(ERROR_MESSAGE.format(**locals()))


@env_cfg_ingredient.capture
def load_dict_dataset(benchmark_name, n_traj=None):
    """Load a dict-type dataset. Also see load_wds_datasets, which instead
    lods a set of datasets that have been stored in a webdataset-compatible
    format."""
    if benchmark_name == 'magical':
        dataset_dict = load_dataset_magical(n_traj=n_traj)
    elif benchmark_name == 'dm_control':
        dataset_dict = load_dataset_dm_control(n_traj=n_traj)
    elif benchmark_name == 'atari':
        dataset_dict = load_dataset_atari(n_traj=n_traj)
    elif benchmark_name == 'minecraft':
        dataset_dict = load_dataset_minecraft(n_traj=n_traj)
    else:
        raise NotImplementedError(ERROR_MESSAGE.format(**locals()))

    num_transitions = len(dataset_dict['dones'].flatten())
    num_dones = dataset_dict['dones'].flatten().sum()
    logging.info(f'Loaded dataset with {num_transitions} transitions. '
                 f'{num_dones} of these transitions have done == True')
    if n_traj is not None and num_dones < n_traj:
        raise ValueError(
            f"Requested n_traj={n_traj}, but can only see {num_dones} dones")

    return dataset_dict


@env_cfg_ingredient.capture
def get_gym_env_name(benchmark_name, dm_control_full_env_names, task_name):
    """Get the name of the Gym environment corresponding to the current
    task."""
    if benchmark_name == 'magical':
        return get_env_name_magical()
    elif benchmark_name == 'dm_control':
        return dm_control_full_env_names[task_name]
    elif benchmark_name == 'atari':
        return task_name
    elif benchmark_name == 'minecraft':
        return get_env_name_minecraft()  # uses task_name implicitly through config param
    raise NotImplementedError(ERROR_MESSAGE.format(**locals()))


@venv_opts_ingredient.capture
def _get_venv_opts(n_envs, venv_parallel, parallel_workers):
    # helper to extract options from venv_opts, since we can't have two
    # captures on one function (see Sacred issue #206)
    return n_envs, venv_parallel, parallel_workers



@env_cfg_ingredient.capture
def load_vec_env(benchmark_name, dm_control_full_env_names,
                 dm_control_frame_stack, minecraft_max_env_steps, minecraft_wrappers):
    """Create a vec env for the selected benchmark task and wrap it with any
    necessary wrappers."""
    n_envs, venv_parallel, parallel_workers = _get_venv_opts()
    gym_env_name = get_gym_env_name()
    if benchmark_name == 'magical':
        return make_vec_env(gym_env_name,
                            n_envs=n_envs,
                            parallel=venv_parallel,
                            parallel_workers=parallel_workers)
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
                                     parallel_workers=parallel_workers,
                                     wrapper_class=AtariWrapper)
        final_env = VecFrameStack(VecTransposeImage(raw_atari_env), 4)
        assert final_env.observation_space.shape == (4, 84, 84), \
            final_env.observation_space.shape
        return final_env
    elif benchmark_name == 'minecraft':
        if venv_parallel:
            raise ValueError("MineRL environments can only be run with `venv_parallel`=False as a result of "
                             "issues with starting daemonic processes from SubprocVecEnv")
        return make_vec_env(gym_env_name,
                            n_envs=1,  # TODO fix this eventually; currently hitting error
                                       # noted here: https://github.com/minerllabs/minerl/issues/177
                            parallel=venv_parallel,
                            wrapper_class=partial(wrap_env, wrappers=minecraft_wrappers),
                            max_episode_steps=minecraft_max_env_steps)
    raise NotImplementedError(ERROR_MESSAGE.format(**locals()))


@env_cfg_ingredient.capture
def _get_default_env_cfg(_config):
    return _config


@env_data_ingredient.capture
def get_data_dir(benchmark_name, task_key, data_type, data_root):
    """Get the data directory for a given benchmark ("magical", "dm_control",
    etc.), task (e.g. "MoveToCorner", "finger-spin") and data type (e.g.
    "demos", "random")."""
    return os.path.join(data_root, 'data', 'processed',
                        data_type, benchmark_name, task_key)


def load_wds_datasets(configs):
    """Load datasets in webdataset (wds) format.

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
          [{"type": "random", "env_cfg": {"task_name": "finger_spin"}}]` to use
          only rollouts from the `finger-spin` dm_control environment.

    (all other args are taken from env_data_ingredient)"""
    # by default we load demos for the configured base environment
    defaults = {
        'type': 'demos',
        'env_cfg': _get_default_env_cfg(),
    }
    all_datasets = []

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
        benchmark_name = env_cfg["benchmark_name"]
        task_key = env_cfg['task_name']
        data_dir_for_config = get_data_dir(
            benchmark_name=benchmark_name, task_key=task_key,
            data_type=data_type)

        tar_files = glob.glob(os.path.join(data_dir_for_config, "*.tgz"))
        if len(tar_files) == 0:
            raise IOError(
                f"did not find any files in '{data_dir_for_config}' (for "
                f"dataset config '{orig_config}')")
        all_datasets.append(load_ilr_datasets(tar_files))

    # get combined metadata for all datasets
    color_space = all_datasets[0].meta['color_space']
    observation_space = all_datasets[0].meta['observation_space']
    action_space = all_datasets[0].meta['action_space']
    for sub_dataset in all_datasets:
        if sub_dataset.meta['color_space'] != color_space:
            raise ValueError(
                "was given datasets with mismatched color spaces: "
                f"'{sub_dataset.meta['color_space']}' != '{color_space}'")
        if sub_dataset.meta['observation_space'] != observation_space:
            raise ValueError(
                "was given datasets with mismatched observation spaces: "
                f"'{sub_dataset.meta['observation_space']}' != "
                f"'{observation_space}'")
        if sub_dataset.meta['action_space'] != action_space:
            raise ValueError(
                "was given datasets with mismatched action spaces: "
                f"'{sub_dataset.meta['action_space']}' != '{action_space}'")
    combined_meta = {
        'color_space': color_space,
        'observation_space': observation_space,
        'action_space': action_space,
    }

    return all_datasets, combined_meta


@env_cfg_ingredient.capture
def load_color_space(benchmark_name):
    """Determine which colour space is used for this benchmark. This is RGB for
    every task except Atari. Knowing the colour space is useful for determining
    dimensions of the frame stacks that get passed to CNNs."""
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
