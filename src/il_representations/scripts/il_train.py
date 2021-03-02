#!/usr/bin/env python3
"""Run an IL algorithm in some selected domain."""
import faulthandler
import logging
import os
# readline import is black magic to stop PDB from segfaulting; do not remove it
import readline  # noqa: F401
import signal

from imitation.algorithms.adversarial import GAIL
from imitation.algorithms.bc import BC
import imitation.data.types as il_types
import imitation.util.logger as imitation_logger
import numpy as np
import sacred
from sacred import Experiment, Ingredient
from sacred.observers import FileStorageObserver
import stable_baselines3.common.policies as sb3_pols
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo import PPO
import torch as th
from torch import nn

from il_representations.algos.encoders import BaseEncoder
from il_representations.algos.utils import set_global_seeds
from il_representations.data.read_dataset import datasets_to_loader
import il_representations.envs.auto as auto_env
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)
from il_representations.il.bc_support import BCModelSaver
from il_representations.il.disc_rew_nets import ImageDiscrimNet
from il_representations.il.gail_pol_save import GAILSavePolicyCallback
from il_representations.il.score_logging import SB3ScoreLoggingCallback
from il_representations.policy_interfacing import EncoderFeatureExtractor
from il_representations.utils import (augmenter_from_spec, freeze_params,
                                      print_policy_info)

bc_ingredient = Ingredient('bc')


@bc_ingredient.config
def bc_defaults():
    # number of passes to make through dataset
    n_batches = 5000
    augs = 'translate,rotate,gaussian_blur,color_jitter_ex'
    log_interval = 500
    batch_size = 32
    save_every_n_batches = None
    optimizer_cls = th.optim.Adam
    optimizer_kwargs = dict(lr=1e-4)
    lr_scheduler_cls = None
    lr_scheduler_kwargs = None
    # the number of 'epochs' is used by the LR scheduler
    # (we still do `n_batches` total training, the scheduler just gets a chance
    # to update after every `n_batches / nominal_num_epochs` batches)
    nominal_num_epochs = 10

    _ = locals()
    del _


gail_ingredient = Ingredient('gail')


@gail_ingredient.config
def gail_defaults():
    # These default settings are copied from
    # https://arxiv.org/pdf/2011.00401.pdf (page 19). They should work for
    # MAGICAL, but not sure about the other tasks.
    # WARNING: the number of parallel vec envs is actually an important
    # hyperparameter. I set this to 32 in the MAGICAL paper, but 24 should work
    # too.

    ppo_n_steps = 8
    ppo_n_epochs = 12
    # "batch size" is actually the size of a _minibatch_. The amount of data
    # used for each training update is ppo_n_steps*n_envs.
    ppo_batch_size = 64
    ppo_init_learning_rate = 6e-5
    ppo_final_learning_rate = 0.0
    ppo_gamma = 0.8
    ppo_gae_lambda = 0.8
    ppo_ent = 1e-5
    ppo_adv_clip = 0.01
    ppo_max_grad_norm = 1.0
    # normalisation + clipping is experimental; previously I just did
    # normalisation (to stddev of 0.1) with no clipping
    ppo_norm_reward = True
    ppo_clip_reward = float('inf')
    # target standard deviation for rewards
    ppo_reward_std = 0.01

    disc_n_updates_per_round = 12
    disc_batch_size = 48
    disc_lr = 2.5e-5
    disc_augs = "rotate,translate,noise"

    # number of env time steps to perform during reinforcement learning
    total_timesteps = int(1e6)
    # save intermediate snapshot after this many environment time steps
    save_every_n_steps = 5e4
    # dump logs every <this many> steps (at most)
    # (5000 is about right for MAGICAL; something like 25000 is probably better
    # for DMC)
    log_interval_steps = 5e3

    _ = locals()
    del _


@gail_ingredient.capture
def _gail_dummy(disc_augs):
    """Similar to _bc_dummy above, but for GAIL augmentations."""
    raise NotImplementedError("this function is not meant to be called")


sacred.SETTINGS['CAPTURE_MODE'] = 'no'  # workaround for sacred issue#740
il_train_ex = Experiment(
    'il_train',
    ingredients=[
        # We need env_cfg_ingredient to determine which environment to train
        # on, venv_opts_ingredient to construct a vecenv for the environment,
        # and env_data_ingredient to load training data. bc_ingredient and
        # gail_ingredient are used for BC and GAIL, respectively (otherwise
        # ignored).
        env_cfg_ingredient,
        venv_opts_ingredient,
        env_data_ingredient,
        bc_ingredient,
        gail_ingredient,
    ])


@il_train_ex.config
def default_config():
    # exp_ident is an arbitrary string. Set it to a meaningful value to help
    # you identify runs in viskit.
    exp_ident = None
    # manually set number of Torch threads
    torch_num_threads = 1
    # device to place all computations on
    device_name = 'auto'
    # choose between 'bc'/'gail'
    algo = 'bc'
    # place to load pretrained encoder from (if not given, it will be
    # re-intialised from scratch)
    encoder_path = None
    # file name for final policy
    final_pol_name = 'policy_final.pt'
    # Should we print a summary of the policy on init? This will show the
    # architecture of the policy.
    print_policy_summary = True
    # dataset configurations for webdataset code
    # (you probably don't want to change this)
    dataset_configs = [{'type': 'demos'}]
    # size of the buffer used for intermediate shuffling
    # (smaller = lower memory usage, but less effective shuffling)
    shuffle_buffer_size = 1024
    # should we freeze weights of the encoder?
    freeze_encoder = False
    # these defaults are mostly optimised for GAIL, but should be fine for BC
    # too (it only uses the venv for evaluation)
    venv_opts = dict(
        venv_parallel=True,
        n_envs=16,
    )
    encoder_kwargs = dict(
        obs_encoder_cls='MAGICALCNN',
        representation_dim=128,
        obs_encoder_cls_kwargs={}
    )
    ortho_init = False
    log_std_init = 0.0
    # This is the mlp architecture applied _after_ the encoder; after this MLP,
    # Stable Baselines will apply a linear layer to ensure outputs (policy,
    # value function) are of the right shape. By default this is empty, so the
    # encoder output is just piped straight into the final linear layers for
    # the policy and value function, respectively.
    postproc_arch = ()

    _ = locals()
    del _


@il_train_ex.capture
def load_encoder(*,
                 encoder_path,
                 freeze_encoder,
                 encoder_kwargs,
                 observation_space):
    if encoder_path is not None:
        encoder = th.load(encoder_path)
        assert isinstance(encoder, nn.Module)
    else:
        encoder = BaseEncoder(observation_space, **encoder_kwargs)
    if freeze_encoder:
        freeze_params(encoder)
        assert len(list(encoder.parameters())) == 0
    return encoder


@il_train_ex.capture
def make_policy(*,
                observation_space,
                action_space,
                ortho_init,
                log_std_init,
                postproc_arch,
                freeze_encoder,
                lr_schedule=None,
                print_policy_summary=True):
    # TODO(sam): this should be unified with the representation learning code
    # so that it can be configured in the same way, with the same default
    # encoder architecture & kwargs.
    encoder = load_encoder(observation_space=observation_space)
    policy_kwargs = {
        'features_extractor_class': EncoderFeatureExtractor,
        'features_extractor_kwargs': {
            "encoder": encoder,
        },
        'net_arch': postproc_arch,
        'observation_space': observation_space,
        'action_space': action_space,
        # SB3 policies require a learning rate for the embedded optimiser. BC
        # should not use that optimiser, though, so we set the LR to some
        # insane value that is guaranteed to cause problems if the optimiser
        # accidentally is used for something (using infinite or non-numeric
        # values fails initial validation, so we need an insane-but-finite
        # number).
        'lr_schedule':
        (lambda _: 1e100) if lr_schedule is None else lr_schedule,
        'ortho_init': ortho_init,
        'log_std_init': log_std_init
    }
    policy = sb3_pols.ActorCriticCnnPolicy(**policy_kwargs)

    if print_policy_summary:
        # print policy info in case it is useful for the caller
        print("Policy info:")
        print_policy_info(policy, observation_space)

    return policy


def streaming_extract_keys(*keys_to_keep):
    """Filter a generator of dicts to keep only the specified keys."""
    def gen(data_iter):
        for data_dict in data_iter:
            yield {k: data_dict[k] for k in keys_to_keep}
    return gen


def add_infos(data_iter):
    """Add a dummy 'infos' value to each dict in a data stream."""
    for data_dict in data_iter:
        yield {'infos': {}, **data_dict}


@il_train_ex.capture
def do_training_bc(venv_chans_first, demo_webdatasets, out_dir, bc,
                   device_name, final_pol_name, shuffle_buffer_size):
    policy = make_policy(observation_space=venv_chans_first.observation_space,
                         action_space=venv_chans_first.action_space)
    color_space = auto_env.load_color_space()
    augmenter = augmenter_from_spec(bc['augs'], color_space)

    # build dataset in the format required by imitation
    data_loader = datasets_to_loader(
        demo_webdatasets,
        batch_size=bc['batch_size'],
        nominal_length=int(np.ceil(
            bc['batch_size'] * bc['n_batches']
            / bc['nominal_num_epochs'])),
        shuffle=True,
        shuffle_buffer_size=shuffle_buffer_size,
        preprocessors=[streaming_extract_keys("obs", "acts")])

    trainer = BC(
        observation_space=venv_chans_first.observation_space,
        action_space=venv_chans_first.action_space,
        policy_class=lambda **kwargs: policy,
        policy_kwargs=None,
        expert_data=data_loader,
        device=device_name,
        augmentation_fn=augmenter,
        optimizer_cls=bc['optimizer_cls'],
        optimizer_kwargs=bc['optimizer_kwargs'],
        lr_scheduler_cls=bc['lr_scheduler_cls'],
        lr_scheduler_kwargs=bc['lr_scheduler_kwargs'],
        ent_weight=1e-3,
        l2_weight=1e-5,
    )

    save_interval = bc['save_every_n_batches']
    if save_interval is not None:
        epoch_end_callbacks = [BCModelSaver(policy,
                                            os.path.join(out_dir, 'snapshots'),
                                            save_interval)]
    else:
        epoch_end_callbacks = []

    logging.info("Beginning BC training")
    trainer.train(n_epochs=None,
                  n_batches=bc['n_batches'],
                  log_interval=bc['log_interval'],
                  epoch_end_callbacks=epoch_end_callbacks)

    final_path = os.path.join(out_dir, final_pol_name)
    logging.info(f"Saving final BC policy to {final_path}")
    trainer.save_policy(final_path)
    return final_path


@il_train_ex.capture
def do_training_gail(
    *,
    venv_chans_first,
    demo_webdatasets,
    device_name,
    out_dir,
    final_pol_name,
    gail,
    shuffle_buffer_size,
    encoder_path,
):
    device = get_device(device_name)
    discrim_net = ImageDiscrimNet(
        observation_space=venv_chans_first.observation_space,
        action_space=venv_chans_first.action_space,
        encoder=load_encoder(
            observation_space=venv_chans_first.observation_space),
    )

    def policy_constructor(observation_space,
                           action_space,
                           lr_schedule,
                           use_sde=False):
        """Construct a policy with the right LR schedule (since PPO will
        actually use it, unlike BC)."""
        assert not use_sde
        return make_policy(observation_space=observation_space,
                           action_space=action_space,
                           lr_schedule=lr_schedule)

    def linear_lr_schedule(prog_remaining):
        """Linearly anneal LR from `init` to `final` (both taken from context).

        This is called by SB3. `prog_remaining` falls from 1.0 (at the start)
        to 0.0 (at the end)."""
        init = gail['ppo_init_learning_rate']
        final = gail['ppo_final_learning_rate']
        alpha = prog_remaining
        return alpha * init + (1 - alpha) * final

    ppo_algo = PPO(
        policy=policy_constructor,
        env=venv_chans_first,
        # verbose=1 and tensorboard_log=False is a hack to work around SB3
        # issue #109.
        verbose=1,
        tensorboard_log=None,
        device=device,
        n_steps=gail['ppo_n_steps'],
        batch_size=gail['ppo_batch_size'],
        n_epochs=gail['ppo_n_epochs'],
        ent_coef=gail['ppo_ent'],
        gamma=gail['ppo_gamma'],
        gae_lambda=gail['ppo_gae_lambda'],
        clip_range=gail['ppo_adv_clip'],
        learning_rate=linear_lr_schedule,
        max_grad_norm=gail['ppo_max_grad_norm'],
    )
    color_space = auto_env.load_color_space()
    augmenter = augmenter_from_spec(gail['disc_augs'], color_space)

    data_loader = datasets_to_loader(
        demo_webdatasets,
        batch_size=gail['disc_batch_size'],
        # nominal_length is arbitrary; we could make it basically anything b/c
        # nothing in GAIL depends on the 'length' of the expert dataset
        nominal_length=int(1e6),
        shuffle=True,
        shuffle_buffer_size=shuffle_buffer_size,
        preprocessors=[streaming_extract_keys(
            "obs", "acts", "next_obs", "dones"), add_infos],
        drop_last=True,
        collate_fn=il_types.transitions_collate_fn)

    trainer = GAIL(
        venv=venv_chans_first,
        expert_data=data_loader,
        gen_algo=ppo_algo,
        n_disc_updates_per_round=gail['disc_n_updates_per_round'],
        expert_batch_size=gail['disc_batch_size'],
        discrim_kwargs=dict(discrim_net=discrim_net, normalize_images=True),
        normalize_obs=False,
        normalize_reward=gail['ppo_norm_reward'],
        normalize_reward_std=gail['ppo_reward_std'],
        clip_reward=gail['ppo_clip_reward'],
        disc_opt_kwargs=dict(lr=gail['disc_lr']),
        disc_augmentation_fn=augmenter,
        gen_callbacks=[SB3ScoreLoggingCallback()],
    )
    save_callback = GAILSavePolicyCallback(
        ppo_algo=ppo_algo, save_every_n_steps=gail['save_every_n_steps'],
        save_dir=out_dir)
    trainer.train(
        total_timesteps=gail['total_timesteps'], callback=save_callback,
        log_interval_timesteps=gail['log_interval_steps'])

    final_path = os.path.join(out_dir, final_pol_name)
    logging.info(f"Saving final GAIL policy to {final_path}")
    th.save(ppo_algo.policy, final_path)
    return final_path


@il_train_ex.main
def train(seed, algo, encoder_path, freeze_encoder, torch_num_threads,
          dataset_configs, _config):
    faulthandler.register(signal.SIGUSR1)
    set_global_seeds(seed)
    # python built-in logging
    logging.basicConfig(level=logging.INFO)
    # `imitation` logging
    # FIXME(sam): I used this hack from run_rep_learner.py, but I don't
    # actually know the right way to write log files continuously in Sacred.
    log_dir = os.path.abspath(il_train_ex.observers[0].dir)
    imitation_logger.configure(log_dir, ["stdout", "csv", "tensorboard"])
    if torch_num_threads is not None:
        th.set_num_threads(torch_num_threads)

    venv = auto_env.load_vec_env()
    demo_webdatasets, combined_meta = auto_env.load_wds_datasets(
        configs=dataset_configs)
    if encoder_path:
        logging.info(f"Loading pretrained encoder from '{encoder_path}'")
        encoder = th.load(encoder_path)
        if freeze_encoder:
            freeze_params(encoder)
            assert len(list(encoder.parameters())) == 0
    else:
        logging.info("No encoder provided, will init from scratch")
        encoder = None

    logging.info(f"Setting up '{algo}' IL algorithm")

    if algo == 'bc':
        final_path = do_training_bc(
            demo_webdatasets=demo_webdatasets,
            venv_chans_first=venv,
            out_dir=log_dir)

    elif algo == 'gail':
        final_path = do_training_gail(
            demo_webdatasets=demo_webdatasets,
            venv_chans_first=venv,
            out_dir=log_dir)

    else:
        raise NotImplementedError(f"Can't handle algorithm '{algo}'")

    # FIXME(sam): make sure this always closes correctly, even when there's an
    # exception after creating it (could use try/catch or a context manager)
    venv.close()

    return {'model_path': os.path.abspath(final_path)}


if __name__ == '__main__':
    il_train_ex.observers.append(FileStorageObserver('runs/il_train_runs'))
    il_train_ex.run_commandline()
