import os
import torch as th
from torch import nn

import stable_baselines3.common.policies as sb3_pols
from stable_baselines3.dqn import CnnPolicy

from il_representations.algos.encoders import BaseEncoder
from il_representations.utils import freeze_params, print_policy_info
from il_representations.policy_interfacing import EncoderFeatureExtractor


def load_encoder(*,
                 encoder_path,
                 freeze,
                 encoder_kwargs,
                 observation_space):
    if encoder_path is not None:
        encoder = th.load(encoder_path)
        assert isinstance(encoder, nn.Module)
    else:
        encoder = BaseEncoder(observation_space, **encoder_kwargs)
    if freeze:
        freeze_params(encoder)
        assert len(list(encoder.parameters())) == 0
    return encoder


def make_policy(*,
                observation_space,
                action_space,
                postproc_arch,
                freeze_pol_encoder,
                encoder_path,
                encoder_kwargs,
                ortho_init=False,
                log_std_init=0.0,
                lr_schedule=None,
                policy_class=sb3_pols.ActorCriticCnnPolicy,
                optimizer_class=th.optim.Adam,
                optimizer_kwargs=None,
                print_policy_summary=True):
    # TODO(sam): this should be unified with the representation learning code
    # so that it can be configured in the same way, with the same default
    # encoder architecture & kwargs.
    encoder = load_encoder(encoder_path=encoder_path,
                           encoder_kwargs=encoder_kwargs,
                           observation_space=observation_space,
                           freeze=freeze_pol_encoder)
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
    }

    if isinstance(encoder, policy_class):
        policy = encoder
    elif policy_class == sb3_pols.ActorCriticCnnPolicy:
        policy_kwargs.update({'ortho_init': ortho_init,
                              'log_std_init': log_std_init})

        policy = policy_class(**policy_kwargs)
    elif policy_class == CnnPolicy:
        policy_kwargs.update({'optimizer_class': optimizer_class,
                              'optimizer_kwargs': optimizer_kwargs,
                              })

        policy = policy_class(**policy_kwargs)
    else:
        raise NotImplementedError

    if print_policy_summary:
        # print policy info in case it is useful for the caller
        print("Policy info:")
        print_policy_info(policy, observation_space)

    return policy


class ModelSaver:
    """Callback that saves the policy every N epochs."""
    def __init__(self, policy, save_dir, save_interval_batches,
                 start_nupdate=0):
        self.policy = policy
        self.save_dir = save_dir
        self.last_save_batches = start_nupdate
        self.save_interval_batches = save_interval_batches

        # Sometimes the loaded policy is already trained for some time
        # (e.g., when we load a policy from a failed run). Here we note
        # the n_update number it has been trained for, and save the policy
        # file using its actual batch update number.
        self.start_nupdate = start_nupdate
        self.last_save_path = None
        os.makedirs(self.save_dir, exist_ok=True)

    def __call__(self, batch_num, **kwargs):
        """It is assumed that this is called on epoch end."""
        batch_num = batch_num + self.start_nupdate
        if batch_num >= self.last_save_batches + self.save_interval_batches:
            self.save(batch_num)

    def save(self, batch_num):
        """Save policy."""
        save_fn = f'policy_{batch_num:08d}_batches.pt'
        save_path = os.path.join(self.save_dir, save_fn)
        th.save(self.policy, save_path)
        print(f"Saved policy to {save_path}!")
        self.last_save_batches = batch_num
        self.last_save_path = save_path
        assert os.path.isfile(self.last_save_path)
