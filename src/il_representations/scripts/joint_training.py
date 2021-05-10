#!/usr/bin/env python3
"""Jointly do repL and IL training."""
from contextlib import ExitStack, closing
import faulthandler
import logging
import os
# readline import is black magic to stop PDB from segfaulting; do not remove it
import readline  # noqa: F401
import signal

import imitation.util.logger as imitation_logger
import sacred
from sacred import Experiment, FileStorageObserver, Ingredient
from stable_baselines3.common.utils import get_device
import torch
import torch as th

from il_representations.algos.encoders import MAGICALCNN
from il_representations.algos.representation_learner import \
    RepresentationLearner
from il_representations.algos.utils import set_global_seeds
from il_representations.data.read_dataset import datasets_to_loader
from il_representations.envs import auto
import il_representations.envs.auto as auto_env
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)
from il_representations.il.bc import BC
from il_representations.il.utils import streaming_extract_keys
from il_representations.utils import augmenter_from_spec

sacred.SETTINGS['CAPTURE_MODE'] = 'no'  # workaround for sacred issue#740
repl_ingredient = Ingredient('repl')
bc_ingredient = Ingredient('bc')
train_ex = Experiment('train',
                      ingredients=[
                          repl_ingredient,
                          bc_ingredient,
                          env_cfg_ingredient,
                          env_data_ingredient,
                          venv_opts_ingredient,
                      ])


@repl_ingredient.config
def repl_defaults():
    dataset_configs = [{'type': 'demos'}]
    algo = "ActionConditionedTemporalCPC"
    algo_params = {
        'representation_dim': 128,
        'augmenter_kwargs': {
            # augmenter_spec is a comma-separated list of enabled
            # augmentations. Consult docstring for
            # imitation.augment.StandardAugmentations to see available
            # augmentations.
            "augmenter_spec": "translate,rotate,gaussian_blur,color_jitter_ex",
        },
    }
    # save input batches to the network in repL loop
    batch_save_interval = 1000

    _ = locals()
    del _


@bc_ingredient.config
def bc_defaults():
    dataset_configs = [{'type': 'demos'}]
    augs = 'translate,rotate,gaussian_blur,color_jitter_ex'
    batch_size = 32
    # regularisation
    ent_weight = 1e-3
    l2_weight = 1e-5

    _ = locals()
    del _


@train_ex.config
def default_config():
    # identifier for use in viskit & other analysis scripts
    exp_ident = None

    # how long to train for
    n_batches = 25000

    # size of shuffle buffers for data loaders
    shuffle_buffer_size = 1024

    # we use a shared optimiser for repL and IL
    optimizer_cls = th.optim.Adam
    optimizer_kwargs = dict(lr=1e-4)

    # we always construct the obs_encoder explicitly
    obs_encoder_cls = MAGICALCNN
    obs_encoder_kwargs = None

    # weight for repL term
    repl_weight = 1.0

    # TODO(sam): LR scheduler, if we think it will be useful

    # stop Torch taking up all cores needlessly
    torch_num_threads = 1

    device = "auto"

    _ = locals()
    del _


@train_ex.capture
def learn_repl_bc(repl_learner, repl_datasets, bc_learner, n_batches,
                  optimizer_cls, optimizer_kwargs, repl_weight):
    """Training loop for repL + BC."""
    # FIXME(sam): standardise the IL and repL data APIs. Either both should use
    # the .next_batch() approach (like BC), or both should have separate
    # methods for constructing data iterators from datasets (like repL).
    # (I kind of like the 'separate methods' approach, since that feels closest
    # to decoupling data loading entirely.)

    # dataset setup
    repl_data_iter = repl_learner.make_data_iter(datasets=repl_datasets,
                                                 batches_per_epoch=n_batches,
                                                 n_epochs=1)

    # optimizer and LR scheduler
    # FIXME(sam): get the trainable parameters for the policy too
    optimizer = optimizer_cls(
        repl_learner.all_trainable_params() +
        bc_learner.all_trainable_parameters, optimizer_kwargs)

    repl_learner.set_train(True)

    for batch_num in range(n_batches):
        bc_batch = bc_learner.next_batch()
        bc_loss, bc_stats = bc_learner.batch_forward(bc_batch)
        repl_batch = next(repl_data_iter)
        repl_loss, detached_debug_tensors = repl_learner.batch_forward(
            repl_batch)
        composite_loss = bc_loss + repl_weight * repl_loss
        optimizer.zero_grad()
        composite_loss.backward()
        optimizer.step()
        del composite_loss  # so we don't use again

        # TODO(sam): logging stuff should go here
        # TODO(sam): model-saving stuff should also go here
        # TODO(sam): consider making all this stuff into callbacks that can be
        # shared with dedicated repL and IL code too


@repl_ingredient.capture
def repl_setup(dataset_configs, obs_encoder, shuffle_buffer_size, algo, algo_params):
    # set up env/dataset/learner for repL
    repl_webdatasets, repl_combined_meta = auto.load_wds_datasets(
        configs=dataset_configs)
    color_space = repl_combined_meta['color_space']
    observation_space = repl_combined_meta['observation_space']
    action_space = repl_combined_meta['action_space']

    encoder_kwargs = algo_params.setdefault('encoder_kwargs', {})
    if encoder_kwargs.get('obs_encoder_cls') is not None \
       or encoder_kwargs.get('obs_encoder_kwargs'):
        raise ValueError(
            "Should not set repl.algo_params.obs_encoder* variables. Use the "
            "top-level config variables instead. ")

    # setting up repL algo
    assert issubclass(algo, RepresentationLearner)
    repl_algo_params = dict(algo_params)
    repl_algo_params['augmenter_kwargs'] = {
        'color_space': color_space,
        **repl_algo_params['augmenter_kwargs'],
    }
    repl_algo = algo
    logging.info(f"Running repL algo {repl_algo} with "
                 f"parameters {repl_algo_params}")
    repl_learner = repl_algo(observation_space=observation_space,
                             action_space=action_space,
                             shuffle_buffer_size=shuffle_buffer_size,
                             **repl_algo_params)
    return repl_learner, repl_webdatasets


@bc_ingredient.capture
def bc_setup(venv, obs_encoder, n_batches, shuffle_buffer_size, dataset_configs, batch_size,
             l2_weight, ent_weight, augs):
    il_demo_webdatasets, il_combined_meta = auto_env.load_wds_datasets(
        configs=dataset_configs)
    policy = make_policy(observation_space=venv.observation_space,
                         action_space=venv.action_space,
                         obs_encoder=obs_encoder,
                         freeze_pol_encoder=False)
    il_data_loader = datasets_to_loader(
        il_demo_webdatasets,
        batch_size=batch_size,
        nominal_length=batch_size * n_batches,
        shuffle=True,
        shuffle_buffer_size=shuffle_buffer_size,
        preprocessors=[streaming_extract_keys("obs", "acts")])
    color_space = il_combined_meta['color_space']
    bc_aug_fn = augmenter_from_spec(augs, color_space)
    bc_learner = BC(policy=policy,
                    expert_data_loader=il_data_loader,
                    l2_weight=l2_weight,
                    ent_weight=ent_weight,
                    augmentation_fn=bc_aug_fn)
    return bc_learner


@train_ex.main
def train(seed, torch_num_threads, device, repl, bc, n_batches,
          shuffle_buffer_size, _config):
    faulthandler.register(signal.SIGUSR1)
    set_global_seeds(seed)
    # python built-in logging
    logging.basicConfig(level=logging.INFO)
    # `imitation` logging
    log_dir = os.path.abspath(train_ex.observers[0].dir)
    imitation_logger.configure(log_dir, ["stdout", "csv", "tensorboard"])
    if torch_num_threads is not None:
        th.set_num_threads(torch_num_threads)
    device = get_device(device)

    # TODO: a few things:
    # - Create the obs encoder separately first so that it can be passed into
    #   both policy and repL learner (via encoder_kwargs['obs_encoder_cls']
    #   with encoder_kwargs['obs_encoder_kwargs']={}).
    # - Figure out where to put the BC data loader constructor. Arguably it
    #   should be in the BC code itself.
    # - Write a simple BC training loop
    # - Factor out repL training code and incorporate it in the loop
    # - Do a big refactor to tidy everything up. Keep in mind that you're going
    #   to have to do something similar for GAIL eventually.

    with ExitStack() as exit_stack:
        # set up env/dataset/learner for IL
        venv = auto_env.load_vec_env()
        exit_stack.push(closing(venv))
        bc_learner = bc_setup(venv=venv,
                              n_batches=n_batches,
                              shuffle_buffer_size=shuffle_buffer_size)
        repl_learner, repl_datasets = repl_setup(
            shuffle_buffer_size=shuffle_buffer_size)

        learn_repl_bc(repl_learner=repl_learner,
                      repl_datasets=repl_datasets,
                      bc_learner=bc_learner)


if __name__ == '__main__':
    train_ex.observers.append(FileStorageObserver('runs/joint_train_runs'))
    train_ex.run_commandline()
