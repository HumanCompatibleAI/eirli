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
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.preprocessing import preprocess_obs

from il_representations.algos.encoders import BaseEncoder
from il_representations.algos.utils import set_global_seeds
from il_representations.data.read_dataset import datasets_to_loader, SubdatasetExtractor
import il_representations.envs.auto as auto_env
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)
from il_representations.scripts.policy_utils import make_policy, ModelSaver
from il_representations.utils import augmenter_from_spec

sacred.SETTINGS['CAPTURE_MODE'] = 'no'  # workaround for sacred issue#740
dqn_ex = Experiment(
    'dqn_train',
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
    n_trajs = None
    augs = 'translate,rotate,gaussian_blur,color_jitter_ex'
    save_every_n_batches = 50000
    n_traj = None
    torch_num_threads = None
    # the number of 'epochs' is used by the LR scheduler
    # (we still do `n_batches` total training, the scheduler just gets a chance
    # to update after every `n_batches / nominal_num_epochs` batches)
    nominal_num_epochs = 1
    postproc_arch = ()


@dqn_ex.capture
def do_training_dqn(venv_chans_first, dict_dataset, out_dir, augs, n_batches,
                    device_name, final_pol_name, freeze_encoder, postproc_arch,
                    encoder_path, encoder_kwargs, nominal_num_epochs,
                    save_every_n_batches):
    observation_space = venv_chans_first.observation_space
    device = get_device("auto" if device_name is None else device_name)
    policy = make_policy(observation_space=observation_space,
                         action_space=venv_chans_first.action_space,
                         postproc_arch=postproc_arch,
                         freeze_pol_encoder=freeze_encoder,
                         policy_class=CnnPolicy,
                         encoder_path=encoder_path,
                         encoder_kwargs=encoder_kwargs).to(device)

    color_space = auto_env.load_color_space()
    augmenter = augmenter_from_spec(augs, color_space)

    trainer = DQN(
        policy='CnnPolicy',
        env=venv_chans_first,
        device=device_name
    )
    trainer.policy = policy

    # Push data into DQN agent's memory.
    for idx in range(len(dict_dataset['obs'])):
        obs, next_obs, action, reward, done = dict_dataset['obs'][idx], \
                                              dict_dataset['next_obs'][idx], \
                                              dict_dataset['acts'][idx], \
                                              dict_dataset['rews'][idx], \
                                              dict_dataset['dones'][idx]
        obs = preprocess_obs(th.tensor(obs),
                             observation_space,
                             normalize_images=True)
        next_obs = preprocess_obs(th.tensor(next_obs),
                                  observation_space,
                                  normalize_images=True)
        if augmenter is not None:
            # Here we unsqueeze the obs first since the augmenter only takes
            # [N, C, H, W] inputs, so we need to fake a "batch size" here.
            obs = augmenter(th.unsqueeze(obs, dim=0)).squeeze()
            next_obs = augmenter(th.unsqueeze(next_obs, dim=0)).squeeze()

        trainer.replay_buffer.add(obs=obs,
                                  next_obs=next_obs,
                                  action=action,
                                  reward=reward,
                                  done=done)

    # Call DQN training.
    n_update_per_epoch = int(n_batches / nominal_num_epochs)
    model_saver = ModelSaver(trainer.policy,
                             save_dir=os.path.join(out_dir, 'snapshots'),
                             save_interval_batches=save_every_n_batches)
    for epoch in range(nominal_num_epochs):
        print(f'Training [{epoch}/{nominal_num_epochs}] epochs...')
        trainer.train(n_update_per_epoch)

        model_saver(n_update_per_epoch)

    model_saver.save(n_batches)
    return model_saver.last_save_path


@dqn_ex.main
def train(seed, torch_num_threads, dataset_configs, n_traj, _config):
    faulthandler.register(signal.SIGUSR1)
    set_global_seeds(seed)
    # python built-in logging
    logging.basicConfig(level=logging.INFO)
    log_dir = os.path.abspath(dqn_ex.observers[0].dir)

    if torch_num_threads is not None:
        th.set_num_threads(torch_num_threads)

    with contextlib.closing(auto_env.load_vec_env()) as venv:
        dict_dataset = auto_env.load_dict_dataset(n_traj=n_traj)

        final_path = do_training_dqn(
            dict_dataset=dict_dataset,
            venv_chans_first=venv,
            out_dir=log_dir)

    return {'model_path': os.path.abspath(final_path)}


if __name__ == '__main__':
    dqn_ex.observers.append(FileStorageObserver('runs/dqn_runs'))
    dqn_ex.run_commandline()
