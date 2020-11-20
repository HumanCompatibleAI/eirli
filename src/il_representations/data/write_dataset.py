"""Utilities for writing datasets in the new unified format."""

import os

import webdataset as wds

import il_representations.envs.auto as auto_env
from il_representations.envs.config import (ALL_BENCHMARK_NAMES,
                                            env_cfg_ingredient,
                                            env_data_ingredient)
from il_representations.envs.utils import serialize_gym_space
from il_representations.scripts.utils import sacred_copy


@env_data_ingredient.capture
def _get_env_data(_config):
    # workaround for Sacred issue #206
    return _config


@env_cfg_ingredient.capture
def get_out_file_map(benchmark_name, task_name):
    """Retrieve dictionary telling us where demonstrations for the current
    environment should go."""
    env_data = _get_env_data()
    return env_data['processed_data_dirs'][benchmark_name][task_name]


@env_cfg_ingredient.capture
def get_meta_dict(benchmark_name, _config):
    # figure out what config keys to keep
    # (we remove keys for benchmarks other than the current one)
    other_env_names = ALL_BENCHMARK_NAMES - {benchmark_name}
    assert len(other_env_names) == len(ALL_BENCHMARK_NAMES) - 1
    env_cfg_copy = sacred_copy(_config)
    env_cfg_stripped = {
        k: v
        for k, v in env_cfg_copy.items()
        if not any(k.startswith(n) for n in other_env_names)
    }

    color_space = auto_env.load_color_space()
    venv = auto_env.load_vec_env()
    meta_dict = {
        'env_cfg': env_cfg_stripped,
        'action_space': serialize_gym_space(venv.action_space),
        'observation_space': serialize_gym_space(venv.observation_space),
        'color_space': color_space,
    }
    venv.close()
    return meta_dict


def write_frames(out_file_path, meta_dict, frame_dicts, n_traj=None):
    out_dir = os.path.dirname(out_file_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with wds.TarWriter(out_file_path, keep_meta=True, compress=True) \
         as writer:
        # first write _metadata.meta.pickle containing the benchmark config
        writer.dwrite(key='_metadata', meta_pickle=meta_dict)
        # now write each frame in each trajectory
        for frame_num, frame_dict in enumerate(frame_dicts):
            write_dict = {
                '__key__': 'frame_%03d' % frame_num,
                'frame.pickle': frame_num
            }
            for key, array in frame_dict.items():
                write_dict[key + '.pickle'] = array
            writer.write(write_dict)
