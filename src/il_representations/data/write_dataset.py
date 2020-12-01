"""Utilities for writing datasets in the webdataset format."""

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
def get_meta_dict(benchmark_name, _config):
    """Generate a dictionary with metadata for the current task (as defined by
    `env_cfg`). When generating a webdataset, this dictionary will be written
    to the beginning of each shard. Having this metadata in the file makes it
    possible to determine how big the inputs to CNNs are, what sort of action
    space should be used for each task, etc."""

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
    """Write a series of frames to a webdataset shard. This function also makes
    sure to write the metadata dictionary `meta_dict` at the beginning of the
    shard, as expected by the data-loading utilities in `read_dataset.py`."""
    out_dir = os.path.dirname(out_file_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with wds.TarWriter(out_file_path, keep_meta=True, compress=True) \
          as writer:  # noqa: E207
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
