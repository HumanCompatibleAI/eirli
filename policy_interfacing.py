import torch
from stable_baselines3.common.policies import BaseFeaturesExtractor


class EncoderFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, encoder=None, encoder_path=None):
        super().__init__(observation_space, features_dim)
        assert encoder is not None or encoder_path is not None, "You must pass in either an encoder object or a path to an encoder"
        assert not (encoder is not None and encoder_path is not None), "Please pass in only one of `encoder` and `encoder_path`"
        if encoder is not None:
            self.representation_encoder = encoder
        else:
            self.representation_encoder = torch.load(encoder_path)

    def forward(self, observations):
        features_dist = self.representation_encoder(observations, traj_info=None)
        return features_dist.loc


class EncoderPolicyHead(torch.nn.Module):
    def __init__(self, encoder, action_size, device, trained_weights_path):
        super().__init__()
        self.representation_encoder = encoder
        self.trained_weights_path = trained_weights_path
        if self.trained_weights_path is not None:
            self._load_encoder_weights()
        self.device = device
        self.action_layer = torch.nn.Linear(encoder.representation_dim, action_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def _load_encoder_weights(self):
        pretrained_dict = torch.load(self.trained_weights_path, device=self.device)
        self.representation_encoder.load_state_dict(pretrained_dict, map_location=self.device)

    def forward(self, x):
        representation_dist = self.representation_encoder(x)
        representation = representation_dist.loc
        action_probas = self.softmax(self.action_layer(representation))
        return action_probas
