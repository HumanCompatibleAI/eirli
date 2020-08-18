#!/usr/bin/env python3
"""Run an IL algorithm in some selected domain."""
import logging
import os

from imitation.algorithms.adversarial import GAIL
from imitation.algorithms.bc import BC
from imitation.augment import StandardAugmentations
import imitation.util.logger as imitation_logger
from sacred import Experiment
from sacred.observers import FileStorageObserver
import stable_baselines3.common.policies as sb3_pols
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo import PPO
import torch as th

from il_representations.data import TransitionsMinimalDataset
import il_representations.envs.auto as auto_env
from il_representations.envs.config import benchmark_ingredient
from il_representations.il.disc_rew_nets import ImageDiscrimNet
from il_representations.policy_interfacing import EncoderFeatureExtractor

il_train_ex = Experiment('il_train', ingredients=[benchmark_ingredient])


@il_train_ex.config
def default_config():
    # ##################
    # Common config vars
    # ##################

    # device to place all computations on
    device_name = 'auto'  # noqa: F841
    # choose between 'bc'/'gail'
    algo = 'bc'  # noqa: F841
    # place to load pretrained encoder from (if not given, it will be
    # re-intialised from scratch)
    encoder_path = None  # noqa: F841

    # ##############
    # BC config vars
    # ##############

    # number of passes to make through dataset
    bc_n_epochs = 250  # noqa: F841
    bc_augs = 'rotate,translate,noise'  # noqa: F841

    # #####################
    # GAIL config variables
    # #####################

    # number of env time steps to perform during reinforcement learning
    gail_total_timesteps = int(1e6)  # noqa: F841
    # "gail_disc_batch_size" is how many samples we take from the expert and
    # novice buffers to do a round of discriminator optimisation.
    # "gail_disc_minibatch_size" controls the size of the minibatches that we
    # divide that into. Thus, we do batch_size/minibatch_size minibatches of
    # optimisation at each discriminator update. gail_disc_batch_size = 256.
    # (this is a different naming convention to SB3 PPO)
    gail_disc_batch_size = 256  # noqa: F841
    gail_disc_minibatch_size = 32  # noqa: F841
    gail_disc_lr = 1e-4  # noqa: F841
    gail_disc_augs = "rotate,translate,noise"  # noqa: F841
    gail_ppo_n_steps = 16  # noqa: F841
    # "batch size" is actually the size of a _minibatch_. The amount of data
    # used for each training update is gail_ppo_n_steps*n_envs.
    gail_ppo_batch_size = 32  # noqa: F841
    gail_ppo_n_epochs = 4  # noqa: F841
    gail_ppo_learning_rate = 2.5e-4  # noqa: F841
    gail_ppo_gamma = 0.95  # noqa: F841
    gail_ppo_gae_lambda = 0.95  # noqa: F841
    gail_ppo_ent = 1e-5  # noqa: F841
    gail_ppo_adv_clip = 0.05  # noqa: F841
    # these defaults are mostly optimised for GAIL, but should be fine for BC
    # too (it only uses the venv for evaluation)
    benchmark = dict(  # noqa: F841
        venv_parallel=True,
        n_envs=16,
    )


@il_train_ex.capture
def make_policy(venv, encoder_or_path):
    # TODO(sam): this should be unified with the representation learning code
    # so that it can be configured in the same way, with the same defaults,
    # etc.
    common_policy_kwargs = {
        'observation_space': venv.observation_space,
        'action_space': venv.action_space,
        # SB3 policies require a learning rate for the embedded optimiser. BC
        # should not use that optimiser, though, so we set the LR to some
        # insane value that is guaranteed to cause problems if the optimiser
        # accidentally is used for something (using infinite or non-numeric
        # values fails initial validation, so we need an insane-but-finite
        # number).
        'lr_schedule': lambda _: 1e100,
        'ortho_init': False,
    }
    if encoder_or_path is not None:
        if isinstance(encoder_or_path, str):
            encoder_key = 'encoder_path'
        else:
            encoder_key = 'encoder'
        policy_kwargs = {
            'features_extractor_class': EncoderFeatureExtractor,
            'features_extractor_kwargs': {
                encoder_key: encoder_or_path
            },
            **common_policy_kwargs,
        }
    else:
        policy_kwargs = {
            # don't pass a representation learner; we'll just use the default
            **common_policy_kwargs,
        }
    policy = sb3_pols.ActorCriticCnnPolicy(**policy_kwargs)
    return policy


@il_train_ex.capture
def do_training_bc(venv_chans_first, dataset, out_dir, bc_n_epochs, encoder,
                   bc_augs, device_name):
    policy = make_policy(venv_chans_first, encoder)
    color_space = auto_env.load_color_space()
    augmenter = StandardAugmentations.from_string_spec(
        bc_augs, stack_color_space=color_space)
    trainer = BC(
        observation_space=venv_chans_first.observation_space,
        action_space=venv_chans_first.action_space,
        policy_class=lambda **kwargs: policy,
        policy_kwargs=None,
        expert_data=dataset,
        device=device_name,
        augmentation_fn=augmenter,
    )

    logging.info("Beginning BC training")
    trainer.train(n_epochs=bc_n_epochs)

    final_path = os.path.join(out_dir, "policy_final.pt")
    logging.info(f"Saving final BC policy to {final_path}")
    trainer.save_policy(final_path)


@il_train_ex.capture
def do_training_gail(
    venv_chans_first,
    dataset,
    device_name,
    encoder,
    out_dir,
    gail_total_timesteps,
    gail_disc_batch_size,
    gail_disc_minibatch_size,
    gail_disc_lr,
    gail_disc_augs,
    gail_ppo_n_steps,
    gail_ppo_batch_size,
    gail_ppo_ent,
    gail_ppo_adv_clip,
    gail_ppo_n_epochs,
    gail_ppo_gamma,
    gail_ppo_gae_lambda,
    gail_ppo_learning_rate,
):
    # Supporting encoder init requires:
    # - Thinking more about how to handle LR of the optimiser stuffed inside
    #   the policy (at the moment we just set an insane default LR because BC
    #   doesn't use it, but PPO actually will use it).
    # - Thinking about how to init the discriminator as well (GAIL
    #   discriminators are incredibly finicky, so that's probably where most
    #   of the value of representation learning will come from in GAIL).
    assert encoder is None, "encoder not yet supported"

    device = get_device(device_name)
    discrim_net = ImageDiscrimNet(
        observation_space=venv_chans_first.observation_space,
        action_space=venv_chans_first.action_space,
    )
    ppo_algo = PPO(
        policy=sb3_pols.ActorCriticCnnPolicy,
        env=venv_chans_first,
        # verbose=1 and tensorboard_log=False is a hack to work around SB3
        # issue #109.
        verbose=1,
        tensorboard_log=None,
        device=device,
        n_steps=gail_ppo_n_steps,
        batch_size=gail_ppo_batch_size,
        n_epochs=gail_ppo_n_epochs,
        ent_coef=gail_ppo_ent,
        gamma=gail_ppo_gamma,
        gae_lambda=gail_ppo_gae_lambda,
        clip_range=gail_ppo_adv_clip,
        learning_rate=gail_ppo_learning_rate,
    )
    color_space = auto_env.load_color_space()
    augmenter = StandardAugmentations.from_string_spec(
        gail_disc_augs, stack_color_space=color_space)
    trainer = GAIL(
        venv_chans_first,
        dataset,
        ppo_algo,
        disc_batch_size=gail_disc_batch_size,
        disc_minibatch_size=gail_disc_minibatch_size,
        discrim_kwargs=dict(discrim_net=discrim_net),
        obs_norm=False,
        rew_norm=True,
        disc_opt_kwargs=dict(lr=gail_disc_lr),
        disc_augmentation_fn=augmenter,
    )

    trainer.train(total_timesteps=gail_total_timesteps)

    final_path = os.path.join(out_dir, 'policy_final.pt')
    logging.info(f"Saving final GAIL policy to {final_path}")
    th.save(ppo_algo.policy, final_path)


@il_train_ex.main
def train(algo, bc_n_epochs, benchmark, encoder_path, _config):
    # python built-in logging
    logging.basicConfig(level=logging.INFO)
    # `imitation` logging
    # FIXME(sam): I used this hack from run_rep_learner.py, but I don't
    # actually know the right way to write log files continuously in Sacred.
    log_dir = il_train_ex.observers[0].dir
    imitation_logger.configure(log_dir, ["stdout", "tensorboard"])

    venv = auto_env.load_vec_env()
    dataset_dict = auto_env.load_dataset()
    dataset = TransitionsMinimalDataset(dataset_dict)

    if encoder_path:
        logging.info(f"Loading pretrained encoder from '{encoder_path}'")
        encoder = th.load(encoder_path)
    else:
        logging.info("No encoder provided, will init from scratch")
        encoder = None

    logging.info(f"Setting up '{algo}' IL algorithm")

    if algo == 'bc':
        do_training_bc(dataset=dataset,
                       venv_chans_first=venv,
                       out_dir=log_dir,
                       encoder=encoder)

    elif algo == 'gail':
        do_training_gail(dataset=dataset,
                         venv_chans_first=venv,
                         out_dir=log_dir,
                         encoder=encoder)

    else:
        raise NotImplementedError(f"Can't handle algorithm '{algo}'")

    # FIXME(sam): make sure this always closes correctly, even when there's an
    # exception after creating it (could use try/catch or a context manager)
    venv.close()

    return log_dir


if __name__ == '__main__':
    il_train_ex.observers.append(FileStorageObserver('il_train_runs'))
    il_train_ex.run_commandline()
