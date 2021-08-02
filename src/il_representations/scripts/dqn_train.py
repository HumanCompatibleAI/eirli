#!/usr/bin/env python3
"""Run offline DQN training, with trajectories loaded to the replay buffer."""
import sacred
import contextlib
import logging
import faulthandler
import os
# readline import is black magic to stop PDB from segfaulting; do not remove it
import readline  # noqa: F401
import signal
import torch as th
import numpy as np
from torch import nn
from torch.optim.adam import Adam
from sacred import Experiment, Ingredient
from sacred.observers import FileStorageObserver
from stable_baselines3.dqn import CnnPolicy, DQN

from il_representations.algos.encoders import BaseEncoder
from il_representations.algos.utils import set_global_seeds
from il_representations.data.read_dataset import datasets_to_loader, SubdatasetExtractor
import il_representations.envs.auto as auto_env
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)
from il_representations.scripts.policy_utils import make_policy
from il_representations.utils import augmenter_from_spec

sacred.SETTINGS['CAPTURE_MODE'] = 'no'  # workaround for sacred issue#740
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
    n_batches = 5000
    augs = 'translate,rotate,gaussian_blur,color_jitter_ex'
    save_every_n_batches = 50000
    n_trajs = None
    torch_num_threads = None
    postproc_arch = ()


@dqn_ex.capture
def do_training_dqn(venv_chans_first, demo_webdatasets, out_dir, augs,
                    device_name, final_pol_name, freeze_encoder, postproc_arch,
                    encoder_path, encoder_kwargs,):
    policy = make_policy(observation_space=venv_chans_first.observation_space,
                         action_space=venv_chans_first.action_space,
                         postproc_arch=postproc_arch,
                         freeze_pol_encoder=freeze_encoder,
                         policy_class=CnnPolicy,
                         encoder_path=encoder_path,
                         encoder_kwargs=encoder_kwargs)

    color_space = auto_env.load_color_space()
    augmenter = augmenter_from_spec(augs, color_space)

    # Build dataset in the format required by SB3's dqn training.

    # Initialize DQN agent.

    # Load data into DQN agent's memory.

    # Call DQN training.

    # Save policy.


@dqn_ex.main
def train(seed, torch_num_threads, dataset_configs, _config):
    faulthandler.register(signal.SIGUSR1)
    set_global_seeds(seed)
    # python built-in logging
    logging.basicConfig(level=logging.INFO)
    log_dir = os.path.abspath(dqn_ex.observers[0].dir)

    if torch_num_threads is not None:
        th.set_num_threads(torch_num_threads)

    with contextlib.closing(auto_env.load_vec_env()) as venv:
        demo_webdatasets, combined_meta = auto_env.load_wds_datasets(
            configs=dataset_configs)

        final_path = do_training_dqn(
            demo_webdatasets=demo_webdatasets,
            venv_chans_first=venv,
            out_dir=log_dir)

    return {'model_path': os.path.abspath(final_path)}


if __name__ == '__main__':
    dqn_ex.observers.append(FileStorageObserver('runs/dqn_runs'))
    dqn_ex.run_commandline()
