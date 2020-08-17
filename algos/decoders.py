import torch.nn as nn
import copy
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from .utils import independent_multivariate_normal
import gym.spaces as spaces
import numpy as np


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

#TODO change shape to dim throughout this file and the code

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

    def ones_like_projection_dim(self, x):
        return torch.ones(size=(x.shape[0], self.projection_dim,), device=x.device)

class NoOp(LossDecoder):
    def forward(self, z, traj_info, extra_context=None):
        return z

class ProjectionHead(LossDecoder):
    def __init__(self, representation_dim, projection_shape, sample=False, learn_scale=False):
        super(ProjectionHead, self).__init__(representation_dim, projection_shape, sample)

        dim = self.representation_dim
        # TODO(rohinmshah): Make the architecture configurable rather than reusing dim everywhere
        self.shared_mlp = nn.Sequential(nn.Linear(dim, dim),
                                        nn.ReLU(),
                                        nn.Linear(dim, dim),
                                        nn.ReLU())
        self.mean_layer = nn.Linear(dim, self.projection_dim, bias=False)

        if learn_scale:
            self.scale_layer = nn.Linear(256, self.projection_dim)
        else:
            self.scale_layer = self.ones_like_projection_dim

    def forward(self, z_dist, traj_info, extra_context=None):
        z = self.get_vector(z_dist)
        shared_repr = self.shared_mlp(z)
        return independent_multivariate_normal(loc=self.mean_layer(shared_repr), scale=torch.exp(self.scale_layer(shared_repr)))


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
            return MultivariateNormal(loc=decoded_z_dist.loc.detach(), covariance_matrix=decoded_z_dist.covariance_matrix.detach())

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.context_decoder.parameters(), self.target_decoder.parameters()):
            param_k.data = param_k.data * self.momentum_weight + param_q.data * (1. - self.momentum_weight)


class BYOLProjectionHead(MomentumProjectionHead):
    def __init__(self, representation_dim, projection_shape, momentum_weight=0.99, sample=False):
        super(BYOLProjectionHead, self).__init__(representation_dim, projection_shape,
                                                 sample=sample, momentum_weight=momentum_weight)
        self.context_predictor = ProjectionHead(projection_shape, projection_shape)

    def forward(self, z_dist, traj_info, extra_context=None):
        internal_dist = super().forward(z_dist, traj_info, extra_context=extra_context)
        prediction_dist = self.context_predictor(internal_dist, traj_info, extra_context=None)
        return independent_multivariate_normal(loc=F.normalize(prediction_dist.loc, dim=1), scale=prediction_dist.scale)

    def decode_target(self, z_dist, traj_info, extra_context=None):
        with torch.no_grad():
            prediction_dist = super().decode_target(z_dist, traj_info, extra_context=extra_context)
            return MultivariateNormal(loc=F.normalize(prediction_dist.loc, dim=1),
                                      covariance_matrix=prediction_dist.covariance_matrix)



class ActionConditionedVectorDecoder(LossDecoder):
    def __init__(self, representation_dim, projection_shape, action_space, sample=False, action_encoding_dim=128,
                 action_encoder_layers=1, learn_scale=False, action_embedding_dim=5, use_lstm=False):
        super(ActionConditionedVectorDecoder, self).__init__(representation_dim, projection_shape, sample=sample)
        self.learn_scale = learn_scale

        # Machinery for turning raw actions into vectors. If actions are discrete, this is done via an Embedding.
        # If actions are continuous/box, this is done via a simple flattening.
        if isinstance(action_space, spaces.Discrete):
            self.action_processer= nn.Embedding(num_embeddings=action_space.n, embedding_dim=action_embedding_dim)
            processed_action_dim = action_embedding_dim
        elif isinstance(action_space, spaces.Box):
            self.action_processer = torch.flatten
            processed_action_dim = np.prod(action_space.shape)
        else:
            raise NotImplementedError("Action conditioning is only currently implemented for Discrete and Box action spaces")

        # Machinery for aggregating information from an arbitrary number of actions into a single vector,
        # either through a LSTM, or by simply averaging the vector representations of the k states together
        if use_lstm:
            self.action_encoder = nn.LSTM(processed_action_dim, action_encoding_dim, action_encoder_layers, batch_first=True)
        else:
            self.action_encoder = None
            action_encoding_dim = processed_action_dim

        # Machinery for mapping a concatenated (context representation, action representation) into a projection
        self.action_conditioned_projection = nn.Linear(representation_dim + action_encoding_dim, projection_shape)

        # If learning scale/std deviation parameter, declare a layer for that, otherwise, return a unit-constant vector
        if self.learn_scale:
            self.scale_projection = nn.Linear(representation_dim + action_encoding_dim, projection_shape)
        else:
            self.scale_projection = self.ones_like_projection_dim

    def decode_target(self, z_dist, traj_info, extra_context=None):
        return z_dist

    def decode_context(self, z_dist, traj_info, extra_context=None):
        # Get a single vector out of the the distribution object passed in by the
        # encoder (either via sampling or taking the mean)
        z = self.get_vector(z_dist)
        actions = extra_context
        # Process each batch-vectorized set of actions, and then stack
        # processed_actions shape - [Batch-dim, Seq-dim, Processed-Action-Dim]
        processed_actions = torch.stack([self.action_processer(action) for action in actions], dim=1)

        # Encode multiple actions into a single action vector (format based on LSTM)
        if self.action_encoder is not None:
            output, (hidden, cell) = self.action_encoder(processed_actions)
        else:
            hidden = torch.mean(processed_actions, dim=1)

        action_encoding_vector = torch.squeeze(hidden)

        # Concatenate context representation and action representation and map to a merged representation
        merged_vector = torch.cat([z, action_encoding_vector], dim=1)
        mean_projection = self.action_conditioned_projection(merged_vector)
        scale = self.scale_projection(merged_vector)
        return independent_multivariate_normal(loc=mean_projection, scale=scale)

