#!/usr/bin/env python3
"""Jointly do repL and IL training."""
import collections
from contextlib import ExitStack, closing
import faulthandler
import logging
import os
import pathlib
# readline import is black magic to stop PDB from segfaulting; do not remove it
import readline  # noqa: F401
import signal

import imitation.data.rollout as il_rollout
import imitation.util.logger as im_log
import numpy as np
import sacred
from sacred import Experiment, Ingredient
from sacred.observers import FileStorageObserver
import stable_baselines3.common.policies as sb3_pols
from stable_baselines3.common.utils import get_device
import torch
import torch as th

from il_representations import algos
from il_representations.algos.encoders import MAGICALCNN, get_obs_encoder_cls
from il_representations.algos.representation_learner import \
    RepresentationLearner
from il_representations.algos.utils import set_global_seeds
from il_representations.envs import auto
import il_representations.envs.auto as auto_env
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)
from il_representations.il.bc import BC
from il_representations.policy_interfacing import ObsEncoderFeatureExtractor
from il_representations.utils import (augmenter_from_spec, save_repl_batches,
                                      Timers, weight_grad_norms)

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
    # postprocessor to apply between observation encoder and final output
    # (in this case we just have a single hidden layer before the final linear
    # layer)
    postproc_arch = [128]
    # evaluation interval
    short_eval_interval = 5000
    # number of trajectories for short intermediate evals
    # (not the final eval)
    short_eval_n_traj = 10

    _ = locals()
    del _


@train_ex.config
def default_config():
    # identifier for use in viskit & other analysis scripts
    exp_ident = None

    # how long to train for
    n_batches = 25000
    # how often to dump logs
    log_dump_interval = 250
    # how often to save items to log (without dumping); should be as high as
    # feasible given performance
    log_calc_interval = 10
    # how often to save (+ forced save at end)
    model_save_interval = 5000
    # how often to evaluate (TODO)

    # size of shuffle buffers for data loaders
    shuffle_buffer_size = 1024

    # we use a shared optimiser for repL and IL
    optimizer_cls = th.optim.Adam
    optimizer_kwargs = dict(lr=1e-4)

    # we always construct the obs_encoder explicitly
    obs_encoder_cls = MAGICALCNN
    obs_encoder_kwargs = {}

    # rep dim produced by shared encoder
    representation_dim = 128

    # weight for repL term
    repl_weight = 1.0

    # TODO(sam): LR scheduler, if we think it will be useful

    # stop Torch taking up all cores needlessly
    torch_num_threads = 1

    device = "auto"

    _ = locals()
    del _


def do_short_eval(*, policy, vec_env, n_rollouts, deterministic=False):
    trajectories = il_rollout.generate_trajectories(
        policy,
        vec_env,
        il_rollout.min_episodes(n_rollouts),
        rng=np.random,
        deterministic_policy=False)
    # make sure all the actions are finite
    for traj in trajectories:
        assert np.all(np.isfinite(traj.acts)), traj.acts

    # the "stats" dict has keys {return,len}_{min,max,mean,std}
    stats = il_rollout.rollout_stats(trajectories)
    stats = collections.OrderedDict([(key, stats[key])
                                    for key in sorted(stats)])

    # TODO(sam): try to get 'score' key out of MAGICAL
    return stats


@train_ex.capture
def learn_repl_bc(repl_learner, repl_datasets, bc_learner, bc_augmentation_fn,
                  bc_dataset, n_batches, optimizer_cls, optimizer_kwargs,
                  repl_weight, log_dump_interval, model_save_interval, repl, bc,
                  shuffle_buffer_size, log_dir, venv, log_calc_interval):
    """Training loop for repL + BC."""
    # dataset setup
    repl_data_iter = repl_learner.make_data_iter(datasets=repl_datasets,
                                                 batches_per_epoch=n_batches,
                                                 n_epochs=1)
    latest_eval_stats = None
    bc_data_iter = bc_learner.make_data_iter(
        il_dataset=bc_dataset,
        augmentation_fn=bc_augmentation_fn,
        batch_size=bc['batch_size'],
        n_batches=n_batches,
        shuffle_buffer_size=shuffle_buffer_size)

    # optimizer and LR scheduler
    # TODO(sam): check for duplicate params (does it matter if something is
    # included twice?)
    params = list(repl_learner.all_trainable_params()) \
        + list(bc_learner.all_trainable_params())
    optimizer = optimizer_cls(params, **optimizer_kwargs)

    timers = Timers()

    repl_learner.set_train(True)

    for batch_num in range(n_batches):
        with timers.time('forward_backward'):
            bc_batch = next(bc_data_iter)
            bc_loss, bc_stats = bc_learner.batch_forward(bc_batch)
            repl_batch = next(repl_data_iter)
            repl_loss, detached_debug_tensors = repl_learner.batch_forward(
                repl_batch)
            composite_loss = bc_loss + repl_weight * repl_loss
            optimizer.zero_grad()
            composite_loss.backward()
            optimizer.step()

        # TODO(sam): when you write the GAIL loop, decide which parts of this
        # can be factored out (probably most of it?)

        # model saving
        log_dir_path = pathlib.Path(log_dir)
        is_last_batch = batch_num == n_batches - 1
        if batch_num % model_save_interval == 0 or is_last_batch:
            with timers.time('model_save'):
                save_dir = log_dir_path / "checkpoints"
                logging.info(f"Saving model to '{save_dir}' (batch#={batch_num})")
                os.makedirs(save_dir, exist_ok=True)
                save_suffix = "_{batch_num:07d}_batches.ckpt"
                th.save(bc_learner, save_dir / ("bc" + save_suffix))
                th.save(repl_learner, save_dir / ("repl" + save_suffix))
                th.save(optimizer, save_dir / ("opt" + save_suffix))

        # repl batch saving
        if batch_num % repl['batch_save_interval'] == 0 or is_last_batch:
            with timers.time('repl_batch_save'):
                repl_batch_save_dir = log_dir_path / 'repl_batch_saves'
                logging.info(f"Saving repL batches to {repl_batch_save_dir} "
                             f"(batch#={batch_num})")
                save_repl_batches(
                    dest_dir=repl_batch_save_dir,
                    detached_debug_tensors=detached_debug_tensors,
                    batches_trained=batch_num,
                    color_space=auto.load_color_space(),
                    save_video=False)

        # occasional eval
        if batch_num % bc['short_eval_interval'] == 0:
            with timers.time('short_eval'):
                short_eval_n_traj = bc['short_eval_n_traj']
                logging.info(f"Evaluating {short_eval_n_traj} trajectories")
                latest_eval_stats = do_short_eval(
                    policy=bc_learner.policy, vec_env=venv,
                    n_rollouts=short_eval_n_traj)

        # logging
        if batch_num % log_calc_interval == 0:
            # TODO(sam): also log times taken for various parts of the training
            # loop.
            grad_norm, weight_norm = weight_grad_norms(params)
            im_log.sb_logger.record_mean('all_loss', composite_loss.item())
            im_log.sb_logger.record_mean('bc_loss', bc_loss.item())
            im_log.sb_logger.record_mean('repl_loss', repl_loss.item())
            im_log.sb_logger.record_mean('grad_norm', grad_norm.item())
            im_log.sb_logger.record_mean('weight_norm', weight_norm.item())
            for k, v in bc_stats.items():
                im_log.sb_logger.record_mean(k, float(v))
            # code above that computes eval stats should at least run on the
            # first step
            assert latest_eval_stats is not None, \
                "logging code rand before eval code for some reason"
            for k, v in latest_eval_stats.items():
                suffix = '_mean'
                if k.endswith(suffix):
                    im_log.sb_logger.record_mean(k[:-len(suffix)], v)
            for k, v in timers.dump_stats(reset=False).items():
                im_log.sb_logger.record('t_mean_' + k, v['mean'])
                im_log.sb_logger.record('t_max_' + k, v['max'])

        if batch_num % log_dump_interval == 0:
            im_log.dump(step=batch_num)


def init_policy(*,
                observation_space,
                action_space,
                postproc_arch,
                obs_encoder,
                log_std_init=0.0):
    # FIXME(sam): maybe this is wrong. We're not necessarily using the same
    # encoder that we use for repL, which seems bad---the repL encoder might
    # have some extra stuff on top of it to make it output distribution
    # parameters. It would be ideal if we were using samples from the
    # distribution or the distribution mean or something instead.
    policy_kwargs = {
        'features_extractor_class': ObsEncoderFeatureExtractor,
        'features_extractor_kwargs': {
            'obs_encoder': obs_encoder,
        },
        'net_arch': postproc_arch,
        'observation_space': observation_space,
        'action_space': action_space,
        # stupid LR so we get errors if we accidentally use the optimiser
        # attached to the policy
        'lr_schedule': (lambda _: 1e100),
        'ortho_init': False,
        'log_std_init': log_std_init,
    }
    policy = sb3_pols.ActorCriticCnnPolicy(**policy_kwargs)

    return policy


@repl_ingredient.capture
def repl_setup(dataset_configs, obs_encoder, shuffle_buffer_size, algo,
               algo_params):
    # set up env/dataset/learner for repL
    repl_webdatasets, repl_combined_meta = auto.load_wds_datasets(
        configs=dataset_configs)
    color_space = repl_combined_meta['color_space']
    observation_space = repl_combined_meta['observation_space']
    action_space = repl_combined_meta['action_space']

    algo_params = dict(algo_params)
    encoder_kwargs = algo_params.setdefault('encoder_kwargs', {})
    if encoder_kwargs.get('obs_encoder_cls') is not None \
       or encoder_kwargs.get('obs_encoder_kwargs'):
        raise ValueError(
            "Should not set repl.algo_params.obs_encoder* variables. Use the "
            "top-level config variables instead. ")

    # setting up repL algo
    if isinstance(algo, str):
        algo = getattr(algos, algo)
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
def bc_setup(venv, obs_encoder, n_batches, shuffle_buffer_size,
             dataset_configs, batch_size, l2_weight, ent_weight, augs,
             postproc_arch):
    il_demo_webdatasets, il_combined_meta = auto_env.load_wds_datasets(
        configs=dataset_configs)
    policy = init_policy(observation_space=venv.observation_space,
                         action_space=venv.action_space,
                         obs_encoder=obs_encoder,
                         postproc_arch=postproc_arch)
    color_space = il_combined_meta['color_space']
    bc_aug_fn = augmenter_from_spec(augs, color_space)
    bc_learner = BC(policy=policy,
                    l2_weight=l2_weight,
                    ent_weight=ent_weight)
    return bc_learner, bc_aug_fn, il_demo_webdatasets


@train_ex.main
def train(seed, torch_num_threads, device, repl, bc, n_batches,
          shuffle_buffer_size, obs_encoder_cls, obs_encoder_kwargs,
          model_save_interval, representation_dim, _config):
    # TODO(sam): consider factoring out this setup code. It is shared between
    # all our training and testing scripts at the moment (run_rep_learner,
    # il_train, il_test, etc.).
    faulthandler.register(signal.SIGUSR1)
    set_global_seeds(seed)
    # python built-in logging
    logging.basicConfig(level=logging.INFO)
    # `imitation` logging
    log_dir = os.path.abspath(train_ex.observers[0].dir)
    im_log.configure(log_dir, ["stdout", "csv", "tensorboard"])
    if torch_num_threads is not None:
        th.set_num_threads(torch_num_threads)
    device = get_device(device)

    with ExitStack() as exit_stack:
        # set up env
        venv = auto_env.load_vec_env()
        exit_stack.push(closing(venv))

        # set up obs encoder shared between IL and repL
        obs_encoder_cls = get_obs_encoder_cls(obs_encoder_cls,
                                              obs_encoder_kwargs)
        obs_encoder = obs_encoder_cls(observation_space=venv.observation_space,
                                      representation_dim=representation_dim,
                                      **obs_encoder_kwargs)

        # set up IL
        bc_learner, bc_augmentation_fn, bc_dataset = bc_setup(
            venv=venv,
            n_batches=n_batches,
            shuffle_buffer_size=shuffle_buffer_size,
            obs_encoder=obs_encoder)

        # setup for repL
        repl_learner, repl_datasets = repl_setup(
            shuffle_buffer_size=shuffle_buffer_size, obs_encoder=obs_encoder)

        # learning loop
        learn_repl_bc(repl_learner=repl_learner,
                      repl_datasets=repl_datasets,
                      bc_learner=bc_learner,
                      bc_augmentation_fn=bc_augmentation_fn,
                      bc_dataset=bc_dataset,
                      model_save_interval=model_save_interval,
                      log_dir=log_dir,
                      venv=venv)

    # final eval
    # TODO(sam): do final evaluation and write eval.json at the end of training


if __name__ == '__main__':
    train_ex.observers.append(FileStorageObserver('runs/joint_train_runs'))
    train_ex.run_commandline()
