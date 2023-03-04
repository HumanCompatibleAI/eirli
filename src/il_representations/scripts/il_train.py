#!/usr/bin/env python3
"""Run an IL algorithm in some selected domain."""
import contextlib
import faulthandler
import logging
import os
# readline import is black magic to stop PDB from segfaulting; do not remove it
import readline  # noqa: F401
import signal

from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.bc import BC
import imitation.data.types as il_types
import imitation.util.logger as im_logger_module
import numpy as np
import sacred
from sacred import Experiment, Ingredient
from sacred.observers import FileStorageObserver
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper
import torch as th
from torch.optim.adam import Adam

from il_representations.algos.utils import set_global_seeds
from il_representations.data.read_dataset import datasets_to_loader
import il_representations.envs.auto as auto_env
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)
from il_representations.il.disc_rew_nets import ImageRewardNet
from il_representations.il.gail_pol_save import GAILSavePolicyCallback
from il_representations.il.score_logging import SB3ScoreLoggingCallback
from il_representations.il.utils import add_infos, streaming_extract_keys
from il_representations.scripts.policy_utils import (ModelSaver,
                                                     load_encoder_or_policy,
                                                     make_policy)
from il_representations.utils import augmenter_from_spec, freeze_params

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


class _DecorrelateEnvsDefault:
    """A smart default setting for `gail.decorrelate_envs`.

    Evaluates as false-ish when procgen is selected, and true-ish otherwise.
    procgen uses a gym3 interface, and so does not support our decorrelation
    strategy of taking a random number of actions in each environment."""
    @property
    @env_cfg_ingredient.capture
    def value(self, benchmark_name):
        return benchmark_name != 'procgen'

    def __bool__(self):
        return bool(self.value)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.value})'

    def __str__(self):
        return f'{self.__class__.__name__}={self.value}'


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
    # should discriminator augs be temporally consistent?
    disc_augs_consistent = False

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

    # if true, this takes a random number of actions in each environment at the
    # beginning of GAIL training to decorrelate episode completion times (not
    # supported by procgen)
    decorrelate_envs = _DecorrelateEnvsDefault()

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
def do_training_bc(venv_chans_first, demo_webdatasets, out_dir, bc,
                   device_name, final_pol_name, logger, shuffle_buffer_size,
                   log_start_batch, freeze_encoder, ortho_init, log_std_init,
                   postproc_arch, encoder_path, policy_continue_path, algo,
                   encoder_kwargs, print_policy_summary):
    policy = make_policy(observation_space=venv_chans_first.observation_space,
                         action_space=venv_chans_first.action_space,
                         ortho_init=ortho_init,
                         log_std_init=log_std_init,
                         postproc_arch=postproc_arch,
                         freeze_pol_encoder=freeze_encoder,
                         encoder_path=encoder_path,
                         policy_continue_path=policy_continue_path,
                         algo=algo,
                         encoder_kwargs=encoder_kwargs,
                         print_policy_summary=print_policy_summary)
    color_space = auto_env.load_color_space()
    device = get_device(device_name)
    augmenter = augmenter_from_spec(bc['augs'], color_space, device)

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
        policy=policy,
        demonstrations=data_loader,
        device=device,
        augmentation_fn=augmenter,
        optimizer_cls=bc['optimizer_cls'],
        optimizer_kwargs=bc['optimizer_kwargs'],
        ent_weight=bc['ent_weight'],
        l2_weight=bc['l2_weight'],
        custom_logger=logger,
        batch_size=bc['batch_size'],
    )

    save_interval = bc['save_every_n_batches']
    model_save_dir = os.path.join(out_dir, 'snapshots')
    os.makedirs(model_save_dir, exist_ok=True)
    model_saver = ModelSaver(policy,
                             model_save_dir,
                             save_interval,
                             start_nupdate=log_start_batch)

    on_epoch_end = model_saver if save_interval else None

    bc_batches = bc['n_batches']

    logging.info("Beginning BC training")
    trainer.train(n_epochs=None,
                  n_batches=bc_batches,
                  log_interval=bc['log_interval'],
                  on_epoch_end=on_epoch_end)

    model_saver.save(bc_batches)
    final_path = model_saver.last_save_path
    model_saver.save_by_name(final_pol_name)
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


class KludgyResetVecEnv(VecEnvWrapper):
    """Kludgy vecenv that takes a random number of steps in each
    sub-environment when reset() is called. This desynchronises the
    environments. This is a good idea to do at the beginning of RL training
    with actor-critic methods."""
    def __init__(self, *args, max_steps, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps
        self.has_reset = False

    def reset(self):
        self.has_reset = True
        for i in range(self.num_envs):
            num_steps = np.random.randint(self.max_steps)
            for _ in range(num_steps):
                action = self.action_space.sample()
                (_, _, d, _),  = self.env_method(
                    'step', action, indices=[i])
                if d:
                    self.env_method('reset', indices=[i])
        final_actions = [
            self.action_space.sample() for _ in range(self.num_envs)
        ]
        # Here we take one final step to get observations for all environments.
        # We cannot store observations using the env_method('step') calls
        # above, since they do not pass through any of the veceenv's wrappers
        # (e.g. for stacking).
        obses, _, _, _ = self.venv.step(final_actions)
        return obses

    def step_wait(self):
        assert self.has_reset, \
            ".reset() was never called on this function, so sub-envs have " \
            "not been randomly advanced"
        return self.venv.step_wait()


@il_train_ex.capture
def do_training_gail(
    *,
    venv_chans_first,
    demo_webdatasets,
    device_name,
    out_dir,
    final_pol_name,
    logger,
    gail,
    shuffle_buffer_size,
    encoder_path,
    encoder_kwargs,
    ortho_init,
    log_std_init,
    postproc_arch,
    freeze_encoder,
    policy_continue_path,
    algo,
    print_policy_summary,
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
                           ortho_init=ortho_init,
                           log_std_init=log_std_init,
                           postproc_arch=postproc_arch,
                           freeze_pol_encoder=freeze_encoder,
                           encoder_path=encoder_path,
                           policy_continue_path=policy_continue_path,
                           algo=algo,
                           encoder_kwargs=encoder_kwargs,
                           lr_schedule=lr_schedule,
                           print_policy_summary=print_policy_summary)

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
    ppo_algo.set_logger(logger)
    color_space = auto_env.load_color_space()
    augmenter = augmenter_from_spec(
        gail['disc_augs'],
        color_space,
        device,
        temporally_consistent=gail['disc_augs_consistent'])

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

    reward_net = ImageRewardNet(
            observation_space=venv_chans_first.observation_space,
            action_space=venv_chans_first.action_space,
            encoder=load_encoder_or_policy(
                encoder_path=encoder_path,
                algo=algo,
                encoder_kwargs=encoder_kwargs,
                observation_space=venv_chans_first.observation_space,
                freeze=_gail_should_freeze('disc')))
    common_adv_il_kwargs = dict(
        venv=venv_chans_first,
        demonstrations=data_loader,
        gen_algo=ppo_algo,
        n_disc_updates_per_round=gail['disc_n_updates_per_round'],
        demo_batch_size=gail['disc_batch_size'],
        normalize_obs=False,
        normalize_reward=gail['ppo_norm_reward'],
        normalize_reward_kwargs=dict(
            norm_reward_std=gail['ppo_reward_std'],
            clip_reward=gail['ppo_clip_reward'],
        ),
        disc_opt_kwargs=dict(lr=gail['disc_lr']),
        disc_augmentation_fn=augmenter,
        gen_callbacks=[SB3ScoreLoggingCallback()],
        custom_logger=logger,
        reward_net=reward_net,
        # The heuristics in imitation yield a false positive for variable
        # horizons if we do the reset hack below. This setting prevents that
        # from happening.
        allow_variable_horizon=True,
    )
    if gail['use_airl']:
        trainer = AIRL(**common_adv_il_kwargs)
    else:
        trainer = GAIL(**common_adv_il_kwargs)

    save_callback = GAILSavePolicyCallback(
        ppo_algo=ppo_algo, save_every_n_steps=gail['save_every_n_steps'],
        save_dir=out_dir)

    # apply a wrapper which advances each sub-environment by some random number
    # of steps on reset in order to decorrelate them (we cannot do this
    # beforehand; SB3 internals call reset() at the beginning of training, and
    # it is impossible to turn that off without changing
    # algorithm._setup_learn() somehow)
    if gail['decorrelate_envs']:
        trainer.venv_train = KludgyResetVecEnv(trainer.venv_train,
                                               max_steps=500)
    trainer.gen_algo.set_env(trainer.venv_train)
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
    logger = im_logger_module.configure(log_dir, ["stdout", "csv", "tensorboard"])
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
                out_dir=log_dir,
                logger=logger)

        elif algo == 'gail':
            final_path = do_training_gail(
                demo_webdatasets=demo_webdatasets,
                venv_chans_first=venv,
                out_dir=log_dir,
                logger=logger)

        else:
            raise NotImplementedError(f"Can't handle algorithm '{algo}'")

    return {'model_path': os.path.abspath(final_path)}


if __name__ == '__main__':
    il_train_ex.observers.append(FileStorageObserver('runs/il_train_runs'))
    il_train_ex.run_commandline()
