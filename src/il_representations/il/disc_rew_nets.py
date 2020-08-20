"""Custom discriminator/reward networks for `imitation`."""

import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.preprocessing import get_flattened_obs_dim


class ImageDiscrimNet(nn.Module):
    """Image-based discriminator network. This is intended to be passed as a
    `discrim_net` argument to `DiscrimNetGAIL` in
    `imitation.rewards.discrim_net`."""

    def __init__(self,
                 observation_space,
                 action_space,
                 image_feature_extractor=NatureCNN,
                 features_dim=256):
        super().__init__()

        # image_feature_extractor produces features for the image, but not the
        # action
        self.image_feature_extractor = image_feature_extractor(
            observation_space=observation_space, features_dim=features_dim)

        # postprocess_mlp takes both the raw action and the image features
        action_dim = get_flattened_obs_dim(action_space)
        mlp_in = features_dim + action_dim
        self.postprocess_mlp = nn.Sequential(
            nn.Linear(mlp_in, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, 1),
        )

    def forward(self, observation, action):
        obs_feats = self.image_feature_extractor(observation)
        obs_feats_and_act = th.cat((obs_feats, action), dim=1)
        final_result = self.postprocess_mlp(obs_feats_and_act)

        # make sure that we don't have a trailing unit dimension (that can mess
        # up DiscrimNetGAIL)
        final_result = final_result.squeeze(1)
        assert final_result.shape == (observation.size(0), )

        return final_result
