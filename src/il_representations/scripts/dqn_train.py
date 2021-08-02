import sacred
import logging
import os
import torch
import numpy as np
from torch import nn
from torch.optim.adam import Adam
from sacred import Experiment, Ingredient
from sacred.observers import FileStorageObserver

from il_representations.algos.encoders import BaseEncoder
from il_representations.algos.utils import set_global_seeds
from il_representations.data.read_dataset import datasets_to_loader, SubdatasetExtractor
import il_representations.envs.auto as auto_env
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)
dqn_ex = Experiment(
    'dqn',
    ingredients=[
        env_cfg_ingredient,
        venv_opts_ingredient,
        env_data_ingredient,
    ])


@dqn_ex.config
def default_config():
    # TODO(Cynthia): Add comments for these variables, so future users know
    # what they mean.
    exp_ident = None
    device_name = 'auto'
    encoder_path = None
    final_pol_name = 'dqn_policy_final.pt'
    print_policy_summary = True
    dataset_configs = [{'type': 'demos'}]
    freeze_encoder = False
    encoder_kwargs = dict(
        obs_encoder_cls='MAGICALCNN',
        representation_dim=128,
        obs_encoder_cls_kwargs={}
    )


@dqn_ex.main
def train():
    pass


if __name__ == '__main__':
    dqn_ex.observers.append(FileStorageObserver('runs/dqn_runs'))
    dqn_ex.run_commandline()
