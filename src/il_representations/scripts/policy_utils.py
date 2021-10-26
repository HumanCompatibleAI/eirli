import os
import torch as th
from torch import nn

import stable_baselines3.common.policies as sb3_pols

from il_representations.algos.encoders import BaseEncoder
from il_representations.utils import freeze_params, print_policy_info
from il_representations.policy_interfacing import EncoderFeatureExtractor


def load_encoder_or_policy(*,
                           encoder_path,
                           algo,
                           encoder_kwargs,
                           observation_space,
                           freeze=False,
                           policy_continue_path=None):
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


def make_policy(*,
                observation_space,
                action_space,
                postproc_arch,
                freeze_pol_encoder,
                encoder_path,
                encoder_kwargs,
                log_std_init=0.0,
                algo=None,
                ortho_init=False,
                lr_schedule=None,
                policy_class=sb3_pols.ActorCriticCnnPolicy,
                extra_policy_kwargs=None,
                policy_continue_path=None,
                print_policy_summary=True):
    # TODO(sam): this should be unified with the representation learning code
    # so that it can be configured in the same way, with the same default
    # encoder architecture & kwargs.
    encoder_or_policy = load_encoder_or_policy(
        encoder_path=encoder_path,
        algo=algo,
        encoder_kwargs=encoder_kwargs,
        observation_space=observation_space,
        freeze=freeze_pol_encoder,
        policy_continue_path=policy_continue_path)

    if isinstance(encoder_or_policy, policy_class):
        # Loading an existing policy
        policy = encoder_or_policy
    else:
        # Loading a repl pretrained encoder
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
        }
        if policy_class == sb3_pols.ActorCriticCnnPolicy:
            policy_kwargs.update({'ortho_init': ortho_init,
                                  'log_std_init': log_std_init})
        if extra_policy_kwargs is not None:
            policy_kwargs.update(extra_policy_kwargs)

        policy = policy_class(**policy_kwargs)

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
            self.save(batch_num, **kwargs)

    def save(self, batch_num, policy=None, **kwargs):
        """Save policy."""
        save_fn = f'policy_{batch_num:08d}_batches.pt'
        save_path = os.path.join(self.save_dir, save_fn)
        if policy is None:
            th.save(self.policy, save_path)
        else:
            th.save(policy, save_path)
        print(f"Saved policy to {save_path}!")
        self.last_save_batches = batch_num
        self.last_save_path = save_path
        assert os.path.isfile(self.last_save_path)
