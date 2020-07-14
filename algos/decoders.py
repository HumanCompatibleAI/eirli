import torch.nn as nn
import copy
import torch
import torch.nn.functional as F
from torch.distributions import Normal

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
        super(LossDecoder, self).__init__()
        self.representation_dim = representation_dim
        self.projection_dim = projection_shape
        self.sample = sample

    def forward(self, z, traj_info, extra_context=None):
        pass

    def decode_target(self, z, traj_info, extra_context=None):
        return self.forward(z, traj_info, extra_context=extra_context)

    def decode_context(self, z, traj_info, extra_context=None):
        return self.forward(z, traj_info, extra_context=extra_context)

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

        self.g1 = nn.Linear(self.representation_dim, 256)
        self.g2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, self.projection_dim)
        if learn_scale:
            self.scale = nn.Linear(256, self.projection_dim)
        else:
            self.scale = lambda x: torch.ones(self.projection_dim)
        self.relu = nn.ReLU()

    def forward(self, z_dist, traj_info, extra_context=None):
        z = self.get_vector(z_dist)
        z = self.relu(self.g1(z))
        z = self.relu(self.g2(z))
        mean = self.mean(z)
        scale = torch.exp(self.scale(z))
        return Normal(loc=mean, scale=scale)


class MomentumProjectionHead(LossDecoder):
    def __init__(self, representation_dim, projection_shape, sample=False, momentum_weight=0.99, learn_scale=False):
        super(MomentumProjectionHead, self).__init__(representation_dim, projection_shape, sample=sample)
        self.context_decoder = ProjectionHead(representation_dim, projection_shape,
                                              sample=sample, learn_scale=learn_scale)
        self.target_decoder = copy.deepcopy(self.context_decoder)
        self.momentum_weight = momentum_weight

    def parameters(self, recurse=True):
        return self.context_decoder.parameters(recurse=recurse)

    def forward(self, z_dist, traj_info, extra_context=None):
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
            return self.target_decoder(z_dist, traj_info, extra_context=extra_context)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.context_decoder.parameters(), self.target_decoder.parameters()):
            param_k.data = param_k.data * self.momentum_weight + param_q.data * (1. - self.momentum_weight)


class BYOLProjectionHead(MomentumProjectionHead):
    def __init__(self, representation_dim, projection_shape, momentum_weight=0.99):
        super(BYOLProjectionHead, self).__init__(representation_dim, projection_shape, momentum_weight=momentum_weight)
        self.context_predictor = ProjectionHead(projection_shape, projection_shape)

    def parameters(self, recurse=True):
        # In BYOL, the loss is given by MSE(predict(project(z_context)), stop_gradient(project(z_target)))
        # So, for the parameters, we want to include everything in predict and project for the context,
        # regardless of whether recurse is True
        # The projection is handled by the superclass, so we use its parameters method
        return self.context_predictor.parameters(recurse=recurse) + super().parameters(recurse=recurse)

    def forward(self, z_dist, traj_info, extra_context=None):
        internal_dist = super().forward(z_dist, traj_info, extra_context=extra_context)
        prediction_dist = self.context_predictor(internal_dist, traj_info, extra_context=None)
        return Normal(loc=F.normalize(prediction_dist.loc, dim=1), scale=prediction_dist.scale)

    def decode_target(self, z_dist, traj_info, extra_context=None):
        with torch.no_grad():
            prediction_dist = super().decode_target(z_dist, traj_info, extra_context=extra_context)
            return Normal(loc=F.normalize(prediction_dist.loc, dim=1), scale=prediction_dist.scale)


# Currently WIP: Implement more of this once I figure out the details of how a VAE works
class VAEDecoder(LossDecoder):
    # https://github.com/ldeecke/vae-torch/blob/master/architecture/nn.py
    def __init__(self, representation_dim, projection_shape, sample=False):
        super(VAEDecoder, self).__init__(representation_dim, projection_shape, sample)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(self.representation_dim, 64)
        self.dense2 = nn.Linear(64, 64)
        self.deconv1

    def forward(self, z, traj_info, extra_context=None):
        batch_size = z.shape[0]
        z = self.relu(self.dense1(z))
        z = self.relu(self.dense2(z))
        z = z.resize(batch_size, 8, 8)


#
# # Maybe shouldn't ultimately live in decoders, but stuck it here for now
# class PolicyHead(nn.Module):
#     # Goes from a representation to a policy
#     def __init__(self, action_size, representation_dim, sample=False):
#         super(PolicyHead, self).__init__()
#         self.action_size = action_size
#         self.representation_dim = representation_dim
#         self.fc_pi = nn.Linear(representation_dim, action_size)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, z):
#         return self.softmax(self.fc_pi(z))
