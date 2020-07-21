import torch.nn as nn
import copy
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import itertools

"""
LossDecoders are meant to be mappings between the representation being learned, 
and the representation or tensor that is fed directly into the loss. In many cases, these are the 
same, and this will just be a NoOp. 

Some cases where it is different: 
- When you are using a Projection Head in your contrastive loss, and comparing similarities of vectors that are 
k >=1 nonlinear layers downstream from the actual representation you'll use in later tasks 
- When you're learning a VAE, and the loss is determined by how effectively you can reconstruct the image 
from a representation vector, the LossDecoder will handle that representation -> image mapping 
- When you're predicting actions given current and next state, you'll want to predict those actions given 
both the representation of the current state, and also information about the next state. This occasional
need for extra information beyond the central context state is why we have `extra_context` as an optional 
bit of data that pair constructors can return, to be passed forward for use here 
"""


class LossDecoder(nn.Module):
    def __init__(self, representation_dim, projection_shape, sample=False):
        super().__init__()
        self.representation_dim = representation_dim
        self.projection_dim = projection_shape
        self.sample = sample

    def forward(self, z, traj_info, extra_context=None):
        pass

    # Calls to self() will call self.forward()
    def decode_target(self, z, traj_info, extra_context=None):
        return self(z, traj_info, extra_context=extra_context)

    def decode_context(self, z, traj_info, extra_context=None):
        return self(z, traj_info, extra_context=extra_context)

    def get_vector(self, z_dist):
        if self.sample:
            return z_dist.sample()
        else:
            return z_dist.loc


class NoOp(LossDecoder):
    def forward(self, z, traj_info, extra_context=None):
        return z


class ProjectionHead(LossDecoder):
    def __init__(self, representation_dim, projection_shape, sample=False, learn_scale=False):
        super(ProjectionHead, self).__init__(representation_dim, projection_shape, sample)

        self.shared_mlp = nn.Sequential(nn.Linear(self.representation_dim, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 256),
                                      nn.ReLU())
        self.mean_layer = nn.Linear(256, self.projection_dim)

        if learn_scale:
            self.scale_layer = nn.Linear(256, self.projection_dim)
        else:
            self.scale_layer = lambda x: torch.ones(self.projection_dim, device=x.device)

    def forward(self, z_dist, traj_info, extra_context=None):
        z = self.get_vector(z_dist)
        shared_repr = self.shared_mlp(z)
        return Normal(loc=self.mean_layer(shared_repr), scale=torch.exp(self.scale_layer(shared_repr)))


class MomentumProjectionHead(LossDecoder):
    def __init__(self, representation_dim, projection_shape, sample=False, momentum_weight=0.99, learn_scale=False):
        super(MomentumProjectionHead, self).__init__(representation_dim, projection_shape, sample=sample)
        self.context_decoder = ProjectionHead(representation_dim, projection_shape,
                                              sample=sample, learn_scale=learn_scale)
        self.target_decoder = copy.deepcopy(self.context_decoder)
        for param in self.target_decoder.parameters():
            param.requires_grad = False
        self.momentum_weight = momentum_weight

    def decode_context(self, z_dist, traj_info, extra_context=None):
        return self.context_decoder(z_dist, traj_info, extra_context=extra_context)

    def decode_target(self, z_dist, traj_info, extra_context=None):
        """
        Encoder target/keys using momentum-updated key encoder. Had some thought of making _momentum_update_key_encoder
        a backwards hook, but seemed overly complex for an initial POC
        :param x:
        :return:
        """
        with torch.no_grad():
            self._momentum_update_key_encoder()
            decoded_z_dist = self.target_decoder(z_dist, traj_info, extra_context=extra_context)
            return Normal(loc=decoded_z_dist.loc.detach(), scale=decoded_z_dist.scale.detach())

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.context_decoder.parameters(), self.target_decoder.parameters()):
            param_k.data = param_k.data * self.momentum_weight + param_q.data * (1. - self.momentum_weight)


class BYOLProjectionHead(MomentumProjectionHead):
    def __init__(self, representation_dim, projection_shape, momentum_weight=0.99):
        super(BYOLProjectionHead, self).__init__(representation_dim, projection_shape, momentum_weight=momentum_weight)
        self.context_predictor = ProjectionHead(projection_shape, projection_shape)

    def forward(self, z_dist, traj_info, extra_context=None):
        internal_dist = super().forward(z_dist, traj_info, extra_context=extra_context)
        prediction_dist = self.context_predictor(internal_dist, traj_info, extra_context=None)
        return Normal(loc=F.normalize(prediction_dist.loc, dim=1), scale=prediction_dist.scale)

    def decode_target(self, z_dist, traj_info, extra_context=None):
        with torch.no_grad():
            prediction_dist = super().decode_target(z_dist, traj_info, extra_context=extra_context)
            return Normal(loc=F.normalize(prediction_dist.loc, dim=1), scale=prediction_dist.scale)