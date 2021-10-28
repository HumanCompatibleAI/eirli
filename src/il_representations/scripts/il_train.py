#!/usr/bin/env python3
"""Run an IL algorithm in some selected domain."""
import faulthandler
import contextlib
import logging
import os
# readline import is black magic to stop PDB from segfaulting; do not remove it
import readline  # noqa: F401
import signal

from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.gail import GAIL
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
from torch.optim.adam import Adam

from il_representations.algos.encoders import BaseEncoder
from il_representations.algos.utils import set_global_seeds
from il_representations.data.read_dataset import datasets_to_loader
import il_representations.envs.auto as auto_env
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)
from il_representations.il.bc_support import BCModelSaver
from il_representations.il.disc_rew_nets import ImageDiscrimNet, ImageRewardNet
from il_representations.il.gail_pol_save import GAILSavePolicyCallback
from il_representations.il.score_logging import SB3ScoreLoggingCallback
from il_representations.il.utils import add_infos, streaming_extract_keys
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
    # The interval to save BC policy networks. If it's set to None,
    # intermediate policies will not be saved.
    save_every_n_batches = 50000
    if save_every_n_batches is not None:
        assert isinstance(save_every_n_batches, int) and \
                save_every_n_batches > 0
    optimizer_cls = Adam
    optimizer_kwargs = dict(lr=1e-4)
    lr_scheduler_cls = None
    lr_scheduler_kwargs = None
    # the number of 'epochs' is used by the LR scheduler
    # (we still do `n_batches` total training, the scheduler just gets a chance
    # to update after every `n_batches / nominal_num_epochs` batches)
    nominal_num_epochs = 10
    # regularisation
    ent_weight = 1e-3
    l2_weight = 1e-5

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

    ppo_n_steps = 7
    ppo_n_epochs = 7
    # "batch size" is actually the size of a _minibatch_. The amount of data
    # used for each training update is ppo_n_steps*n_envs.
    ppo_batch_size = 48
    ppo_init_learning_rate = 0.00025
    ppo_final_learning_rate = 0.0
    ppo_gamma = 0.985
    ppo_gae_lambda = 0.76
    ppo_ent = 4.5e-8
    ppo_adv_clip = 0.006
    ppo_max_grad_norm = 1.0
    # normalisation + clipping is experimental; previously I just did
    # normalisation (to stddev of 0.1) with no clipping
    ppo_norm_reward = True
    ppo_clip_reward = float('inf')
    # target standard deviation for rewards
    ppo_reward_std = 0.01
    # set these to True/False (or non-None, tru-ish/false-ish values) in order
    # to override the root freeze_encoder setting (they will produce warnings)
    freeze_pol_encoder = None
    freeze_disc_encoder = None

    disc_n_updates_per_round = 2
    disc_batch_size = 48
    disc_lr = 0.0006
    disc_augs = "color_jitter_mid,erase,flip_lr,gaussian_blur,noise,rotate"

    # number of env time steps to perform during reinforcement learning
    total_timesteps = 500000
    # save intermediate snapshot after this many environment time steps
    save_every_n_steps = 5e4
    # dump logs every <this many> steps (at most)
    # (5000 is about right for MAGICAL; something like 25000 is probably better
    # for DMC)
    log_interval_steps = 10000

    # use AIRL objective instead of GAIL objective
    # TODO(sam): remove this if AIRL doesn't work; if AIRL does work, rename
    # `gail_ingredient` to `adv_il_ingredient` or similar
    use_airl = False

    _ = locals()
    del _


@bc_ingredient.capture
def _bc_dummy(augs, optimizer_kwargs, lr_scheduler_kwargs):
    """Dummy function to indicate to sacred that the given arguments are
    actually used somewhere.

    (Sacred has a bug in ingredient parsing where it fails to correctly detect
    that config options for sub-ingredients of a command are actually used.
    This only happens when you try to set an attribute of such an option that
    was not initially declared in the config, like when you set
    `bc.optimizer_kwargs.some_thing=42` for instance.)

    PLEASE DO NOT REMOVE THIS, IT WILL BREAK SACRED."""
    raise NotImplementedError("this function is not meant to be called")


@gail_ingredient.capture
def _gail_dummy(disc_augs):
    """Similar to _bc_dummy above, but for GAIL augmentations.

    PLEASE DO NOT REMOVE THIS, IT WILL BREAK SACRED."""
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
    # In case we want to continue training a policy from a previously failed
    # run, we provide the saved policy path here.
    policy_continue_path = None
    num_path_provided = sum(x is not None for x in [encoder_path,
                                                    policy_continue_path])
    # Either a pretrained encoder or a trained policy can be provided,
    # but not both.
    assert num_path_provided <= 1, 'Detected multiple paths for policy.'
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
    # TODO(sam): remove this global setting entirely & replace it with BC and
    # GAIL-specific settings so that we can control which models get frozen
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
    # Sometimes we reload trained policies from a previously failed run.
    # log_start_batch stands for the actual n_update the policy previously gets
    # trained, so when saving policies we use its actual batch update number as
    # its identifier.
    log_start_batch = 0
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
def load_encoder_or_policy(*,
                           encoder_path,
                           policy_continue_path,
                           algo,
                           freeze,
                           encoder_kwargs,
                           observation_space):
    encoder_or_policy = None
    # Load a previously saved policy.
    if policy_continue_path is not None:
        assert algo == 'bc', 'Currently only support policy reload for BC.'
        encoder_or_policy = th.load(policy_continue_path)
        assert isinstance(encoder_or_policy, sb3_pols.ActorCriticCnnPolicy)
    else:  # Load an existing encoder, or initialize a new one.
        if encoder_path is not None:
            encoder_or_policy = th.load(encoder_path)
            assert isinstance(encoder_or_policy, nn.Module)
        else:
            encoder_or_policy = BaseEncoder(observation_space,
                                            **encoder_kwargs)
    if freeze:
        freeze_params(encoder_or_policy)
        assert len(list(encoder_or_policy.parameters())) == 0
    return encoder_or_policy


@il_train_ex.capture
def make_policy(*,
                observation_space,
                action_space,
                ortho_init,
                log_std_init,
                postproc_arch,
                freeze_pol_encoder,
                lr_schedule=None,
                print_policy_summary=True,
                modified_encoder_path=None,
                encoder_kwargs=None):
    # This check is to ensure that some old interpret.py code raises an
    # exception when run (apparently it relied on modified_encoder_path before,
    # but then support for modified_encoder_path was removed). This should be
    # fixed in a later PR.
    if modified_encoder_path is not None:
        raise TypeError('make_policy no longer takes a modified_encoder_path')

    # TODO(sam): this should be unified with the representation learning code
    # so that it can be configured in the same way, with the same default
    # encoder architecture & kwargs.
    encoder_or_policy = load_encoder_or_policy(
        observation_space=observation_space,
        freeze=freeze_pol_encoder)

    if isinstance(encoder_or_policy, sb3_pols.ActorCriticCnnPolicy):
        policy = encoder_or_policy
    else:
        encoder = encoder_or_policy

        # Normally the last layer of an encoder is a linear layer, but in
        # some special cases like Jigsaw, we only train the convolution
        # layers (with linearity handled by the decoder). In BC
        # training we still need the full encoder (linear layers included),
        # so here we load the weights for conv layers, and leave linear
        # layers randomly initialized.
        if hasattr(encoder, 'network') and \
           not isinstance(encoder.network.shared_network[-1], th.nn.Linear):
            full_encoder = BaseEncoder(observation_space,
                                       **encoder_kwargs)

            partial_encoder_dict = encoder.state_dict()
            full_encoder_dict = full_encoder.state_dict()

            # pretrained_dict contains weights & bias for conv layers only.
            pretrained_dict = {k: v for k, v in partial_encoder_dict.items() if
                               k in full_encoder_dict}
            full_encoder_dict.update(pretrained_dict)
            full_encoder.load_state_dict(full_encoder_dict)

            encoder = full_encoder

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


@il_train_ex.capture
def do_training_bc(venv_chans_first, demo_webdatasets, out_dir, bc,
                   device_name, final_pol_name, shuffle_buffer_size,
                   log_start_batch, freeze_encoder):
    policy = make_policy(observation_space=venv_chans_first.observation_space,
                         action_space=venv_chans_first.action_space,
                         freeze_pol_encoder=freeze_encoder)
    color_space = auto_env.load_color_space()
    augmenter = augmenter_from_spec(bc['augs'], color_space)

    # build dataset in the format required by imitation
    nom_num_epochs = bc['nominal_num_epochs']
    nom_num_batches = max(1, int(np.ceil(bc['n_batches'] / nom_num_epochs)))
    data_loader = datasets_to_loader(
        demo_webdatasets,
        batch_size=bc['batch_size'],
        # we make nominal_length large enough that we don't have to re-init the
        # dataset, and also make it a multiple of the batch size so that we
        # don't have to care about the size of the last batch (so
        # drop_last=True doesn't matter in theory)
        nominal_length=bc['batch_size'] * nom_num_batches,
        shuffle=True,
        shuffle_buffer_size=shuffle_buffer_size,
        preprocessors=(streaming_extract_keys("obs", "acts"), ),
        drop_last=True)

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
        ent_weight=bc['ent_weight'],
        l2_weight=bc['l2_weight'],
    )

    save_interval = bc['save_every_n_batches']
    model_save_dir = os.path.join(out_dir, 'snapshots')
    os.makedirs(model_save_dir, exist_ok=True)
    model_saver = BCModelSaver(policy,
                               model_save_dir,
                               save_interval,
                               start_nupdate=log_start_batch)

    epoch_end_callbacks = [model_saver] if save_interval else []

    bc_batches = bc['n_batches']

    logging.info("Beginning BC training")
    trainer.train(n_epochs=None,
                  n_batches=bc_batches,
                  log_interval=bc['log_interval'],
                  epoch_end_callbacks=epoch_end_callbacks)

    model_saver.save(bc_batches)
    final_path = model_saver.last_save_path
    return final_path


@il_train_ex.capture
def _gail_should_freeze(pol_or_disc, *, freeze_encoder, gail):
    """Determine whether we should freeze policy/discriminator. There's a
    root-level freeze_encoder setting in il_train, but we can override it with
    gail.freeze_{pol,disc}_encoder (if those are not None)."""
    assert pol_or_disc in ('pol', 'disc')
    specific_freeze_name = f'freeze_{pol_or_disc}_encoder'
    specific_freeze = gail[specific_freeze_name]
    if specific_freeze is not None and specific_freeze != freeze_encoder:
        logging.warning(f"Overriding global freeze_encoder={freeze_encoder} "
                        f"with {specific_freeze_name}={specific_freeze} for "
                        f"{pol_or_disc}")
        return specific_freeze
    return freeze_encoder


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

    def policy_constructor(observation_space,
                           action_space,
                           lr_schedule,
                           use_sde=False):
        """Construct a policy with the right LR schedule (since PPO will
        actually use it, unlike BC)."""
        assert not use_sde
        return make_policy(observation_space=observation_space,
                           action_space=action_space,
                           lr_schedule=lr_schedule,
                           freeze_pol_encoder=_gail_should_freeze('pol'))

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
        # (we are not currently using an LR scheduler for the discriminator,
        # so we do not bother allowing a configurable length like the one used
        # in BC)
        nominal_length=int(1e6),
        shuffle=True,
        shuffle_buffer_size=shuffle_buffer_size,
        preprocessors=[streaming_extract_keys(
                           "obs", "acts", "next_obs", "dones"), add_infos],
        drop_last=True,
        collate_fn=il_types.transitions_collate_fn)

    common_adv_il_kwargs = dict(
            venv=venv_chans_first,
            expert_data=data_loader,
            gen_algo=ppo_algo,
            n_disc_updates_per_round=gail['disc_n_updates_per_round'],
            expert_batch_size=gail['disc_batch_size'],
            normalize_obs=False,
            normalize_reward=gail['ppo_norm_reward'],
            normalize_reward_std=gail['ppo_reward_std'],
            clip_reward=gail['ppo_clip_reward'],
            disc_opt_kwargs=dict(lr=gail['disc_lr']),
            disc_augmentation_fn=augmenter,
            gen_callbacks=[SB3ScoreLoggingCallback()],
    )
    if gail['use_airl']:
        trainer = AIRL(
            **common_adv_il_kwargs,
            reward_net_cls=ImageRewardNet,
            reward_net_kwargs=dict(
                encoder=load_encoder_or_policy(
                    observation_space=venv_chans_first.observation_space,
                    freeze=_gail_should_freeze('disc'))
            )
        )
    else:
        discrim_net = ImageDiscrimNet(
            observation_space=venv_chans_first.observation_space,
            action_space=venv_chans_first.action_space,
            encoder=load_encoder_or_policy(
                observation_space=venv_chans_first.observation_space,
                freeze=_gail_should_freeze('disc')))
        trainer = GAIL(
            discrim_kwargs=dict(discrim_net=discrim_net,
                                normalize_images=True),
            **common_adv_il_kwargs,
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

    with contextlib.closing(auto_env.load_vec_env()) as venv:
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

    return {'model_path': os.path.abspath(final_path)}


if __name__ == '__main__':
    il_train_ex.observers.append(FileStorageObserver('runs/il_train_runs'))
    il_train_ex.run_commandline()
