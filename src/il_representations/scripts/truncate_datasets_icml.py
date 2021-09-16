#!/usr/bin/env python3
"""Truncate dict-style IL dataset and write it to a new data root"""
import logging

import sacred
from sacred import Experiment

from il_representations.algos.utils import set_global_seeds
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient)
from il_representations.envs.dm_control_envs import rewrite_dataset_dm_control
from il_representations.envs.magical_envs import rewrite_dataset_magical

sacred.SETTINGS['CAPTURE_MODE'] = 'no'  # workaround for sacred issue#740
truncate_datasets_icml_ex = Experiment(
    'truncate_datasts_icml',
    ingredients=[
        env_cfg_ingredient, env_data_ingredient
    ])


@truncate_datasets_icml_ex.config
def default_config():
    # this must be supplied in order to run the experiment
    new_data_root = None
    # this must also be supplied
    n_traj = None

    _ = locals()
    del _


@truncate_datasets_icml_ex.main
def run(new_data_root, n_traj, seed, env_cfg):
    set_global_seeds(seed)

    logging.getLogger().setLevel(logging.INFO)

    assert new_data_root is not None and isinstance(new_data_root, str), \
        "new_data_root must be a string, not None"
    assert n_traj is not None and isinstance(n_traj, int), \
        "n_traj must be an integer, not None"

    if env_cfg['benchmark_name'] == 'dm_control':
        rewrite_dataset_dm_control(n_traj=n_traj, new_data_root=new_data_root)
    elif env_cfg['benchmark_name'] == 'magical':
        rewrite_dataset_magical(n_traj=n_traj, new_data_root=new_data_root)
    else:
        raise NotImplementedError(
            f"Don't support benchmark {env_cfg['benchmark_name']}")


if __name__ == '__main__':
    truncate_datasets_icml_ex.run_commandline()
