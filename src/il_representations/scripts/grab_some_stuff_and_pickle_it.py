#!/usr/bin/env python3
"""Grabs some random frames from a webdataset and pickles them. Also writes out
some metadata like trajectory number (in the webdataset) and also frame
number."""
import logging
import os
import pprint
from typing import Iterable, List

import numpy as np
import sacred
from sacred import Experiment
import torch
import webdataset as wds

from il_representations.algos.utils import set_global_seeds
from il_representations.envs import auto
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient)

sacred.SETTINGS['CAPTURE_MODE'] = 'no'  # workaround for sacred issue#740
grab_stuff_ex = Experiment(
    'grab_stuff',
    ingredients=[
        env_cfg_ingredient, env_data_ingredient
    ])


@grab_stuff_ex.config
def default_config():
    # number of frames to write out
    n_frames = None
    # oversample n_frames * oversample frames, then write a random
    # subset of n_frames different frames
    oversample = 2
    # where to write output?
    out_path = None
    # config to load
    dataset_config = {'type': 'demos'}

    _ = locals()
    del _


def trajectory_iter(dataset: wds.Dataset) -> Iterable[List[dict]]:
    """Yields one trajectory at a time from a webdataset."""
    traj = []
    for frame in dataset:
        traj.append(frame)
        if frame['dones']:
            yield traj
            traj = []


def frame_iter(dataset: wds.Dataset) -> Iterable[dict]:
    """Iterate over observations, along with """
    for traj_num, trajectory in enumerate(trajectory_iter(dataset)):
        for frame_num, frame in enumerate(trajectory):
            yield {
                'frame_num': frame_num,
                'traj_num': traj_num,
                **frame
            }


def sample_rand_frames(dataset: wds.Dataset, n_frames: int,
                       oversample: int) -> Iterable[dict]:
    """Sample n_frames randomly chosen frames from the first (oversample *
    n_frames) frames in the dataset."""
    upper_bound = n_frames * oversample
    indices_to_get = set(np.random.choice(
        upper_bound, replace=False, size=(n_frames, )))
    for frame_ind, frame in enumerate(frame_iter(dataset)):
        if frame_ind >= upper_bound:
            break
        if frame_ind in indices_to_get:
            yield frame


@grab_stuff_ex.main
def run(n_frames: int, out_path: str, dataset_config: dict, oversample: int,
        seed: int) -> None:
    set_global_seeds(seed)
    logging.getLogger().setLevel(logging.INFO)

    print(f'Supplied dataset config:')
    pprint.pprint(dataset_config)

    # we only support loading one dataset (hence the [dataset_config] thing)
    (webdataset, ), combined_meta = auto.load_wds_datasets(
        configs=[dataset_config])

    print(f"Collected metadata from loaded dataset:")
    pprint.pprint(combined_meta)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # now collect some stuff
    keys_to_keep = ['obs', 'frame_num', 'traj_num', 'next_obs', 'acts']
    frame_dicts = list(sample_rand_frames(webdataset, n_frames, oversample))
    out_dict = {
        k: np.stack(d[k] for d in frame_dicts) for k in keys_to_keep
    }
    out_dict['combined_meta'] = 'combined_meta'
    print(f'Writing a pickle to {out_path} (use torch.load to reload)')
    torch.save(out_dict, out_path)


if __name__ == '__main__':
    grab_stuff_ex.run_commandline()
