"""Custom discriminator/reward networks for `imitation`."""

import torch as th
from torch import nn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from il_representations.algos.encoders import compute_rep_shape_encoder, DeterministicEncoder


class ImageDiscrimNet(nn.Module):
    """Image-based discriminator network. This is intended to be passed as a
    `discrim_net` argument to `DiscrimNetGAIL` in
    `imitation.rewards.discrim_net`."""

    # TODO(sam): make this take an `Encoder` (either `StochasticEncoder` or
    # `DeterministicEncoder`) instead.

    def __init__(self,
                 observation_space,
                 action_space,
                 encoder=None,
                 encoder_cls=None,
                 encoder_kwargs=None,
                 fc_dim=256):
        super().__init__()

        if encoder is not None:
            assert encoder_cls is None, "cannot supply both encoder and encoder_cls"
            self.obs_encoder = encoder
            obs_out_dim, = compute_rep_shape_encoder(observation_space, self.obs_encoder)
        else:
            if encoder_cls is None:
                encoder_cls = DeterministicEncoder
            if encoder_kwargs is None:
                encoder_kwargs = {}
            self.obs_encoder = encoder_cls(obs_space=observation_space, representation_dim=fc_dim,
                                           **encoder_kwargs)
            obs_out_dim = fc_dim

        # postprocess_mlp takes both the raw action and the image features
        action_dim = get_flattened_obs_dim(action_space)
        mlp_in = obs_out_dim + action_dim
        self.postprocess_mlp = nn.Sequential(
            nn.Linear(mlp_in, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, observation, action, traj_info=None):
        obs_dist = self.obs_encoder(observation, traj_info=traj_info)
        assert isinstance(obs_dist, th.distributions.Distribution)
        obs_feats = obs_dist.mean
        obs_feats_and_act = th.cat((obs_feats, action), dim=1)
        final_result = self.postprocess_mlp(obs_feats_and_act)

        # make sure that we don't have a trailing unit dimension (that can mess
        # up DiscrimNetGAIL)
        final_result = final_result.squeeze(1)
        assert final_result.shape == (observation.size(0), )

        return final_result
