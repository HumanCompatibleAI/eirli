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
        policy = policy_class(**policy_kwargs)
    else:
        raise NotImplementedError

    if print_policy_summary:
        # print policy info in case it is useful for the caller
        print("Policy info:")
        print_policy_info(policy, observation_space)

    return policy

