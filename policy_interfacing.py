import torch
from stable_baselines3.common.policies import BaseFeaturesExtractor
# FeatureExtractor path
# feature extractors need to take in pre-processed images
# They need to have a `feature_dim`
#


class EncoderFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, encoder, trained_weights_path=None):
        super().__init__(observation_space, features_dim)
        self.representation_encoder = encoder
        self.trained_weights_path = trained_weights_path
        if self.trained_weights_path is not None:
            self._load_encoder_weights()

    def _load_encoder_weights(self):
        pretrained_dict = torch.load(self.trained_weights_path, device=self.device)
        self.representation_encoder.load_state_dict(pretrained_dict, map_location=self.device)

    def forward(self, observations):
        features_dist = self.representation_encoder(observations, traj_info=None)
        return features_dist.loc


class EncoderPolicyHead(torch.nn.Module):
    def __init__(self, encoder, trained_weights_path, action_size, device):
        super().__init__()
        self.representation_encoder = encoder
        self.trained_weights_path = trained_weights_path
        self.device = device
        self.action_layer = torch.nn.Linear(encoder.representation_dim, action_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def _load_encoder_weights(self):
        pretrained_dict = torch.load(self.trained_weights_path, device=self.device)
        self.representation_encoder.load_state_dict(pretrained_dict, map_location=self.device)

    def forward(self, x):
        representation_dist = self.representation_encoder(x)
        representation = representation_dist.loc
