#!/usr/bin/env python3
"""Describe stats for a particular dataset, including number of timesteps,
number of trajectories, and mean/stddev of reward and/or score (if they are
included). """
import logging
import pprint

import numpy as np
import sacred
from sacred import Experiment

from il_representations.algos.utils import set_global_seeds
from il_representations.envs import auto
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient)
from il_representations.script_utils import trajectory_iter

sacred.SETTINGS['CAPTURE_MODE'] = 'no'  # workaround for sacred issue#740
dataset_stats_ex = Experiment(
    'dataset_stats',
    ingredients=[
        env_cfg_ingredient, env_data_ingredient
    ])


@dataset_stats_ex.config
def default_config():
    # config to load
    dataset_config = {'type': 'demos'}

    _ = locals()
    del _


@dataset_stats_ex.main
def run(dataset_config: dict, seed: int) -> None:
    set_global_seeds(seed)
    logging.getLogger().setLevel(logging.INFO)

    print('Supplied dataset config:')
    pprint.pprint(dataset_config)

    # we only support loading one dataset (hence the [dataset_config] thing)
    (webdataset, ), combined_meta = auto.load_wds_datasets(
        configs=[dataset_config])

    print("Collected metadata from loaded dataset:")
    pprint.pprint(combined_meta)

    # now write same trajectories to out_dir
    trajectories = trajectory_iter(webdataset)
    n_traj = 0
    n_ts = 0
    rets = None
    for idx, trajectory in enumerate(trajectories):
        n_traj += 1
        n_ts += len(trajectory)
        if 'rews' in trajectory[0]:
            if rets is None:
                rets = []
            rets.append(sum(frame['rews'] for frame in trajectory))
    print('Dataset stats')
    print('  #timesteps:', n_ts)
    print('  #trajectories:', n_traj)
    if rets is not None:
        ret_mean, ret_std = np.mean(rets), np.std(rets)
        print(f'  Mean return (±std): {ret_mean:.3g}±{ret_std:.3g}')


if __name__ == '__main__':
    dataset_stats_ex.run_commandline()
