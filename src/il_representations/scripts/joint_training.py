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
import imitation.util.logger as im_logger_module
import numpy as np
import sacred
from sacred import Experiment, Ingredient
from sacred.observers import FileStorageObserver
import stable_baselines3.common.policies as sb3_pols
from stable_baselines3.common.utils import get_device
import torch
from torch.optim.adam import Adam

from il_representations import algos
from il_representations.algos.encoders import MAGICALCNN, get_obs_encoder_cls
from il_representations.algos.representation_learner import \
    RepresentationLearner
from il_representations.algos.utils import set_global_seeds
from il_representations.configs.joint_training_configs import make_jt_configs
from il_representations.envs import auto
import il_representations.envs.auto as auto_env
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)
from il_representations.il.bc import BC
from il_representations.pol_eval import do_final_eval
from il_representations.policy_interfacing import ObsEncoderFeatureExtractor
from il_representations.utils import (Timers, augmenter_from_spec,
                                      save_repl_batches, weight_grad_norms)

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
make_jt_configs(train_ex)


@repl_ingredient.config
def repl_defaults():
    dataset_configs = [{'type': 'demos'}]
    algo = 'InverseDynamicsPrediction'
    algo_params = {
        'batch_size': 64,
        'augmenter_kwargs': {
            # augmenter_spec is a comma-separated list of enabled
            # augmentations. Consult docstring for
            # imitation.augment.StandardAugmentations to see available
            # augmentations.
            'augmenter_spec': 'translate,rotate,gaussian_blur,color_jitter',
        },
    }
    # save input batches to the network in repL loop
    batch_save_interval = 1000

    _ = locals()
    del _


@repl_ingredient.capture
def _repl_dummy(dataset_configs, algo_params):
    # DO NOT REMOVE THIS, REMOVING IT WILL BREAK SACRED
    # (see docstring for il_train._bc_dummy for explanation)
    pass


@bc_ingredient.config
def bc_defaults():
    dataset_configs = [{'type': 'demos'}]
    augs = 'translate,rotate,gaussian_blur,color_jitter'
    batch_size = 64
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


@bc_ingredient.capture
def _bc_dummy(dataset_configs, augs):
    # DO NOT REMOVE THIS, REMOVING IT WILL BREAK SACRED
    # (see docstring for il_train._bc_dummy for explanation)
    pass


@train_ex.config
def default_config():
    # identifier for use in viskit & other analysis scripts
    exp_ident = None

    # how long to train for
    n_batches = 25000
    # how often to dump logs
    log_dump_interval = 250
    # how often to save items to log (without dumping); should be as low as you
    # can make it without killing performance
    log_calc_interval = 10
    # how often to save (+ forced save at end)
    model_save_interval = 5000

    # number of trajectories to use in final eval
    final_eval_n_traj = 100

    # size of shuffle buffers for data loaders
    shuffle_buffer_size = 1024

    # we use a shared optimiser for repL and IL
    optimizer_cls = Adam
    optimizer_kwargs = dict(lr=1e-4)

    # we always construct the obs_encoder explicitly
    obs_encoder_cls = MAGICALCNN
    obs_encoder_kwargs = {}

    # rep dim produced by shared encoder
    representation_dim = 128

    # weight for repL term
    repl_weight = 1.0

    # stop Torch taking up all cores needlessly
    torch_num_threads = 1
    # max number of workers for Torch data loader to spool up
    dataset_max_workers = 1

    # will default to GPU if available, otherwise CPU
    device = "auto"

    # choose 'gail' or 'bc'
    il_algo = 'bc'

    _ = locals()
    del _


def do_short_eval(*, policy, vec_env, n_rollouts, deterministic=False):
    trajectories = il_rollout.generate_trajectories(
        policy,
        vec_env,
        il_rollout.make_min_episodes(n_rollouts),
        rng=np.random,
        deterministic_policy=False)
    # make sure all the actions are finite
    for traj in trajectories:
        assert np.all(np.isfinite(traj.acts)), traj.acts

    # the "stats" dict has keys {return,len}_{min,max,mean,std}
    stats = il_rollout.rollout_stats(trajectories)
    stats = collections.OrderedDict([(key, stats[key])
                                     for key in sorted(stats)])

    # also compute min/max/mean/std of eval_score, if present
    extra_stats = collections.OrderedDict()
    for traj in trajectories:
        assert traj.infos is not None
        last_info = traj.infos[-1]
        if 'eval_score' in last_info:
            stat_list = extra_stats.setdefault('eval_score', [])
            stat_list.append(last_info['eval_score'])
    for key, stat_list in extra_stats.items():
        for stat in ('min', 'mean', 'max', 'std'):
            stat_fn: np.generic = getattr(np, stat)
            stats[key + '_' + stat] = stat_fn(stat_list)

    return stats


def deduplicate_params(*params_iters, check_dedup=True):
    """Combines a series of iterators over model parameters into a single list
    of deduplicated parameters."""
    params_set = set()
    params_list_dedup = []
    n_params_dup = 0
    for params_iter in params_iters:
        for param in params_iter:
            n_params_dup += 1
            if param not in params_set:
                params_list_dedup.append(param)
                params_set.add(param)
    if check_dedup:
        assert len(params_list_dedup) < n_params_dup, \
            "After param deduplication, the number of parameters did not " \
            "drop. Is the encoder actually shared?"
    return params_list_dedup


@train_ex.capture
def learn_repl_bc(repl_learner, repl_datasets, bc_learner, bc_augmentation_fn,
                  bc_dataset, n_batches, optimizer_cls, optimizer_kwargs,
                  repl_weight, log_dump_interval, model_save_interval, repl,
                  bc, shuffle_buffer_size, log_dir, venv, log_calc_interval,
                  logger, dataset_max_workers):
    """Training loop for repL + BC."""
    # dataset setup
    with ExitStack() as train_exit_stack:
        repl_data_iter = repl_learner.make_data_iter(
            datasets=repl_datasets, batches_per_epoch=n_batches, n_epochs=1,
            max_workers=dataset_max_workers)
        # we need data iters to be explicitly closed when exiting; sometimes
        # waiting for implicit destruction via stops the data loader process
        # pool from closing properly
        train_exit_stack.push(closing(repl_data_iter))
        latest_eval_stats = None
        bc_data_iter = bc_learner.make_data_iter(
            il_dataset=bc_dataset,
            augmentation_fn=bc_augmentation_fn,
            batch_size=bc['batch_size'],
            n_batches=n_batches,
            shuffle_buffer_size=shuffle_buffer_size,
            max_workers=dataset_max_workers)
        train_exit_stack.push(closing(bc_data_iter))

        # optimizer and LR scheduler
        params_list_dedup = deduplicate_params(
            repl_learner.all_trainable_params(),
            bc_learner.all_trainable_params())
        optimizer = optimizer_cls(params_list_dedup, **optimizer_kwargs)

        timers = Timers()

        repl_learner.set_train(True)

        # some save paths
        log_dir_path = pathlib.Path(log_dir)
        save_dir = log_dir_path / "checkpoints"

        assert n_batches > 0

        for batch_num in range(n_batches):
            with timers.time('next_batch'):
                bc_batch = next(bc_data_iter)
                repl_batch = next(repl_data_iter)

            with timers.time('forward_back'):
                # this block is the actual forward/backward logic (everything
                # else is logging/checkpointing)
                bc_loss, bc_stats = bc_learner.batch_forward(bc_batch)
                repl_loss, repl_debug_tensors = repl_learner.batch_forward(
                    repl_batch)
                composite_loss = bc_loss + repl_weight * repl_loss
                optimizer.zero_grad()
                composite_loss.backward()
                optimizer.step()

            # model saving
            is_last_batch = batch_num == n_batches - 1
            should_save_model = model_save_interval \
                and (batch_num % model_save_interval == 0)
            if should_save_model or is_last_batch:
                with timers.time('model_save'):
                    logging.info(
                        f"Saving model to '{save_dir}' (batch#={batch_num})")
                    os.makedirs(save_dir, exist_ok=True)
                    save_suffix = f"_{batch_num:07d}_batches.ckpt"
                    bc_path = save_dir / ("bc" + save_suffix)
                    repl_path = save_dir / ("repl" + save_suffix)
                    opt_path = save_dir / ("opt" + save_suffix)
                    torch.save(bc_learner, bc_path)
                    torch.save(repl_learner, repl_path)
                    torch.save(optimizer, opt_path)

            # repl batch saving
            batch_save_interval = repl['batch_save_interval']
            should_save_repl_batch = batch_save_interval \
                and batch_num % batch_save_interval == 0
            if should_save_repl_batch or is_last_batch:
                with timers.time('repl_batch_save'):
                    repl_batch_save_dir = log_dir_path / 'repl_batch_saves'
                    logging.info(
                        f"Saving repL batches to {repl_batch_save_dir} "
                        f"(batch#={batch_num})")
                    save_repl_batches(
                        dest_dir=repl_batch_save_dir,
                        detached_debug_tensors=repl_debug_tensors,
                        batches_trained=batch_num,
                        color_space=auto.load_color_space(),
                        save_video=False)

            # occasional eval
            short_eval_interval = bc['short_eval_interval']
            should_do_short_eval = short_eval_interval \
                and batch_num % short_eval_interval == 0
            if should_do_short_eval:
                with timers.time('short_eval'):
                    short_eval_n_traj = bc['short_eval_n_traj']
                    logging.info(
                        f"Evaluating {short_eval_n_traj} trajectories "
                        f"(batch#={batch_num})")
                    latest_eval_stats = do_short_eval(
                        policy=bc_learner.policy, vec_env=venv,
                        n_rollouts=short_eval_n_traj)

            # logging
            should_log_values = log_calc_interval \
                and (batch_num % log_calc_interval == 0)
            if should_log_values:
                grad_norm, weight_norm = weight_grad_norms(params_list_dedup)
                logger.record_mean('all_loss', composite_loss.item())
                logger.record_mean('bc_loss', bc_loss.item())
                logger.record_mean('repl_loss', repl_loss.item())
                logger.record_mean('grad_norm', grad_norm.item())
                logger.record_mean('weight_norm', weight_norm.item())
                logger.record_mean('batch_num', batch_num)
                for k, v in bc_stats.items():
                    logger.record_mean('bc_' + k, float(v))
                # code above that computes eval stats should at least run on
                # the first step
                if latest_eval_stats is not None:
                    for k, v in latest_eval_stats.items():
                        suffix = '_mean'
                        if k.endswith(suffix):
                            logger.record_mean('eval_' + k[:-len(suffix)], v)
                else:
                    assert not short_eval_interval, \
                        "logging code ran before eval code, even though " \
                        "short eval is enabled"
                for k, v in timers.dump_stats(reset=False).items():
                    logger.record('t_mean_' + k, v['mean'])
                    logger.record('t_max_' + k, v['max'])

            should_dump_logs = batch_num % log_dump_interval == 0
            if should_dump_logs:
                logger.dump(step=batch_num)

    # return final saved policy path
    pol_path = save_dir / "policy_final.ckpt"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(bc_learner.policy, pol_path)
    return pol_path


def init_policy(*,
                observation_space,
                action_space,
                postproc_arch,
                obs_encoder,
                log_std_init=0.0):
    policy_kwargs = {
        'features_extractor_class': ObsEncoderFeatureExtractor,
        'features_extractor_kwargs': {
            'obs_encoder': obs_encoder,
        },
        'net_arch': list(postproc_arch),
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
               algo_params, representation_dim):
    # set up env/dataset/learner for repL
    repl_webdatasets, repl_combined_meta = auto.load_wds_datasets(
        configs=dataset_configs)
    color_space = repl_combined_meta['color_space']
    observation_space = repl_combined_meta['observation_space']
    action_space = repl_combined_meta['action_space']

    algo_params = dict(algo_params)
    encoder_kwargs = algo_params.setdefault('encoder_kwargs', {})
    if encoder_kwargs.get('obs_encoder_cls') is not None \
       or encoder_kwargs.get('obs_encoder_cls_kwargs') is not None \
       or encoder_kwargs.get('latent_dim') is not None:
        raise ValueError(
            "Should not set repl.algo_params.obs_encoder* variables. Use the "
            "top-level config variables instead. ")
    encoder_kwargs.update({
        'obs_encoder_cls': lambda obs_space, latent_dim: obs_encoder,
        'latent_dim': representation_dim,
        'obs_encoder_cls_kwargs': None,
    })

    assert 'representation_dim' not in algo_params
    algo_params['representation_dim'] = representation_dim

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
             postproc_arch, device):
    il_demo_webdatasets, il_combined_meta = auto_env.load_wds_datasets(
        configs=dataset_configs)
    policy = init_policy(observation_space=venv.observation_space,
                         action_space=venv.action_space,
                         obs_encoder=obs_encoder,
                         postproc_arch=postproc_arch)
    policy = policy.to(device)
    color_space = il_combined_meta['color_space']
    bc_aug_fn = augmenter_from_spec(augs, color_space, device)
    bc_learner = BC(policy=policy, l2_weight=l2_weight, ent_weight=ent_weight)
    return bc_learner, bc_aug_fn, il_demo_webdatasets


@train_ex.main
def train(seed, torch_num_threads, device, repl, bc, n_batches,
          shuffle_buffer_size, obs_encoder_cls, obs_encoder_kwargs,
          model_save_interval, representation_dim, exp_ident,
          final_eval_n_traj, il_algo, _config):
    faulthandler.register(signal.SIGUSR1)
    set_global_seeds(seed)
    # python built-in logging
    logging.basicConfig(level=logging.INFO)
    # `imitation` logging
    log_dir = os.path.abspath(train_ex.observers[0].dir)
    if torch_num_threads is not None:
        torch.set_num_threads(torch_num_threads)
    device = get_device(device)

    with ExitStack() as exit_stack:
        # set up env
        venv = auto_env.load_vec_env()
        exit_stack.push(closing(venv))

        logger = im_logger_module.configure(
            log_dir, ["stdout", "csv", "tensorboard"])
        exit_stack.push(closing(logger))

        # set up obs encoder shared between IL and repL
        obs_encoder_cls = get_obs_encoder_cls(obs_encoder_cls,
                                              obs_encoder_kwargs)
        obs_encoder = obs_encoder_cls(observation_space=venv.observation_space,
                                      representation_dim=representation_dim,
                                      **obs_encoder_kwargs)
        orig_oe_params = list(obs_encoder.named_parameters())

        # setup for repL
        repl_learner, repl_datasets = repl_setup(
            shuffle_buffer_size=shuffle_buffer_size,
            obs_encoder=obs_encoder,
            representation_dim=representation_dim)
        repl_oe_params = list(repl_learner.encoder.network.named_parameters())

        # learning loop
        if il_algo == 'bc':
            # set up IL
            bc_learner, bc_augmentation_fn, bc_dataset = bc_setup(
                venv=venv,
                n_batches=n_batches,
                shuffle_buffer_size=shuffle_buffer_size,
                obs_encoder=obs_encoder,
                device=device)
            bc_oe_params = list(
                bc_learner.policy.features_extractor.obs_encoder
                .named_parameters())
            # are params actually shared?
            assert orig_oe_params == bc_oe_params
            assert orig_oe_params == repl_oe_params
            final_pol_path = learn_repl_bc(
                repl_learner=repl_learner,
                repl_datasets=repl_datasets,
                bc_learner=bc_learner,
                bc_augmentation_fn=bc_augmentation_fn,
                bc_dataset=bc_dataset,
                model_save_interval=model_save_interval,
                log_dir=log_dir,
                venv=venv,
                logger=logger)
        else:
            raise ValueError(f"do not know how to handle il_algo={il_algo!r}")

    # final eval
    # (note that this is pulling in a bunch of params from env_cfg_ingredient
    # and venv_opts_ingredient)
    final_pol_path = os.path.abspath(final_pol_path)
    do_final_eval(
        policy_path=final_pol_path,
        out_dir=log_dir,
        n_rollouts=final_eval_n_traj,
        # xoring with something arbitrary so we're not using same
        # seed as rest of code (might cause weird correlations)
        seed=seed ^ 0x4c0a,
        device=device,
        deterministic_policy=False,
        write_video=False,
        video_file_name=None,
        run_id=exp_ident)


_has_fso = False


def add_fso():
    # joint_training_cluster.py will sometimes (accidentally) call this twice
    # when running on Ray. I suspect that's because Ray is not properly
    # resetting the global Python namespace between runs, or something of that
    # nature.
    global _has_fso
    if not _has_fso:
        train_ex.observers.append(FileStorageObserver('runs/joint_train_runs'))
        _has_fso = True


if __name__ == '__main__':
    add_fso()
    train_ex.run_commandline()
