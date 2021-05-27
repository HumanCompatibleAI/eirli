from stable_baselines3.common.policies import BaseFeaturesExtractor
import torch

from il_representations.algos.encoders import (compute_output_shape,
                                               compute_rep_shape_encoder)


class EncoderFeatureExtractor(BaseFeaturesExtractor):
    """Wraps a BaseEncoder so that it can be used as a policy feature
    extractor."""
    def __init__(self, observation_space, features_dim=None, encoder=None,
                 encoder_path=None, finetune=True):
        # Allow user to either pass in an existing encoder, or a path from
        # which to load a pickled encoder.
        assert encoder is not None or encoder_path is not None, \
            "You must pass in either an encoder object or a path to an encoder"
        assert not (encoder is not None and encoder_path is not None), \
            "Please pass in only one of `encoder` and `encoder_path`"
        if encoder is not None:
            representation_encoder = encoder
        else:
            representation_encoder = torch.load(encoder_path)

        # do forward prop to infer the feature dim
        if features_dim is None:
            features_dim, = compute_rep_shape_encoder(observation_space,
                                                      representation_encoder)

        super().__init__(observation_space, features_dim)

        self.representation_encoder = representation_encoder

        if not finetune:
            # Set requires_grad to false if we want to not further train
            # weights
            for param in self.representation_encoder.parameters():
                param.requires_grad = False

    def forward(self, observations):
        features_dist = self.representation_encoder(observations,
                                                    traj_info=None)
        mean = features_dist.mean
        # make sure we're not getting NaN actions
        # TODO(sam): this seems inefficient, since we need to sync with GPU.
        # Might be better to run as a check against IL loss or something.
        assert torch.all(torch.isfinite(mean)), mean
        return mean


class ObsEncoderFeatureExtractor(BaseFeaturesExtractor):
    """Wraps an observation encoder so that it can be used as a policy feature
    extractor."""
    def __init__(self, observation_space, obs_encoder=None):
        features_dim, = compute_output_shape(observation_space, [obs_encoder])
        super().__init__(observation_space, features_dim)
        self.obs_encoder = obs_encoder

    def forward(self, obs):
        return self.obs_encoder(obs)
