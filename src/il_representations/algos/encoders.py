import torch
import torch.nn as nn
import copy
from torch.distributions import MultivariateNormal
from functools import reduce
import numpy as np
from stable_baselines3.common.policies import NatureCNN
from gym.spaces import Box
from il_representations.algos.utils import independent_multivariate_normal


"""
Encoders conceptually serve as the bit of the representation learning architecture that learns the representation itself
(except in RNN cases, where encoders only learn the per-frame representation). 

The only real complex thing to note here is the MomentumEncoder architecture, which creates two CNNEncoders, 
and updates weights of one as a slowly moving average of the other. Note that this bit of momentum is separated 
from the creation and filling of a queue of representations, which is handled by the BatchExtender module 
"""

DEFAULT_CNN_ARCHITECTURE = {
    'CONV': [
                {'out_dim': 32, 'kernel_size': 8, 'stride': 4},
                {'out_dim': 64, 'kernel_size': 4, 'stride': 2},
                {'out_dim': 64, 'kernel_size': 3, 'stride': 1},
            ],
    'DENSE': [
                # this value works for Atari, but will be ovewritten for other envs
                {'in_dim': 64*7*7},
             ]
}


def sb_conv_arch_output_size(image_shape, conv_arch):
    """
    Given the input image's shape of (H, W) and a convolution network's architecture, compute the output feature
    map CONV(image)'s shape (C, H, W)
    """
    def compute_out_hw(h_in, w_in, layer_conf):
        import math
        padding = layer_conf['padding']
        dilation = layer_conf['dilation']
        kernel_size = layer_conf['kernel_size']
        stride = layer_conf['stride']
        h_out = math.floor((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
        w_out = math.floor((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
        return h_out, w_out

    h, w = image_shape
    default_config = {
        'padding': [0, 0],
        'dilation': [1, 1],
        'kernel_size': [],
        'stride': [1, 1],
    }

    for layer_config in conv_arch:
        for param, default in default_config.items():
            param_value = layer_config.get(param, default)
            if isinstance(param_value, int):
                param_value = [param_value, param_value]
            layer_config[param] = param_value
            assert len(layer_config[param]) == 2
        h, w = compute_out_hw(h, w, layer_config)

    out_channel = conv_arch[-1]['out_dim']
    return [out_channel, h, w]


class DefaultStochasticCNN(nn.Module):
    def __init__(self, obs_space, representation_dim):
        super().__init__()
        self.input_channel = obs_space.shape[0]
        self.representation_dim = representation_dim
        shared_network_layers = []

        # figure out how big the convolution output will be
        conv_arch = DEFAULT_CNN_ARCHITECTURE['CONV']
        dense_arch = DEFAULT_CNN_ARCHITECTURE['DENSE'].copy()  # copy to mutate
        dense_in_dim = np.prod(sb_conv_arch_output_size(obs_space.shape[1:],
                                                        conv_arch))
        dense_arch[0]['in_dim'] = dense_in_dim

        for layer_spec in conv_arch:
            shared_network_layers.append(nn.Conv2d(self.input_channel, layer_spec['out_dim'],
                                                   kernel_size=layer_spec['kernel_size'], stride=layer_spec['stride']))
            shared_network_layers.append(nn.ReLU())
            self.input_channel = layer_spec['out_dim']

        shared_network_layers.append(nn.Flatten())
        for ind, layer_spec in enumerate(dense_arch[:-1]):
            in_dim, out_dim = layer_spec.get('in_dim'), layer_spec.get('out_dim')
            shared_network_layers.append(nn.Linear(in_dim, out_dim))
            shared_network_layers.append(nn.ReLU())

        self.shared_network = nn.Sequential(*shared_network_layers)
        self.mean_layer = nn.Linear(dense_arch[-1]['in_dim'], self.representation_dim)
        self.scale_layer = nn.Linear(dense_arch[-1]['in_dim'], self.representation_dim)

    def forward(self, x):
        shared_repr = self.shared_network(x)
        mean = self.mean_layer(shared_repr)
        scale = torch.exp(self.scale_layer(shared_repr))
        return mean, scale


class Encoder(nn.Module):
    # Calls to self() will call self.forward()
    def __init__(self):
        super().__init__()

    @property
    def device(self):
        first_param = next(self.parameters())
        return first_param.device

    def encode_target(self, x, traj_info):
        return self(x, traj_info)

    def encode_context(self, x, traj_info):
        return self(x, traj_info)

    def encode_extra_context(self, x, traj_info):
        return x


class DeterministicEncoder(Encoder):
    def __init__(self, obs_space, representation_dim, architecture_module_cls=None, scale_constant=1, **kwargs):
        """
        :param obs_space: The observation space that this Encoder will be used on
        :param representation_dim: The number of dimensions of the representation that will be learned
        :param architecture_module_cls: An internal architecture implementing `forward` to return a single vector
        representing the mean representation z of a fixed-variance representation distribution
        """
        super().__init__(**kwargs)
        if architecture_module_cls is None:
            architecture_module_cls = NatureCNN
        self.network = architecture_module_cls(obs_space, representation_dim)
        self.scale_constant = scale_constant

    def forward(self, x, traj_info):
        features = self.network(x)
        return independent_multivariate_normal(loc=features, scale=self.scale_constant)


class StochasticEncoder(Encoder):
    def __init__(self, obs_space, representation_dim, architecture_model_cls=None, **kwargs):
        """
        :param obs_space: The observation space that this Encoder will be used on
        :param representation_dim: The number of dimensions of the representation that will be learned
        :param architecture_module_cls: An internal architecture implementing `forward` to return
        two vectors, representing the mean AND learned standard deviation of a representation distribution
        """
        super().__init__(**kwargs)
        if architecture_model_cls is None:
            architecture_model_cls = DefaultStochasticCNN
        self.network = architecture_model_cls(obs_space, representation_dim)

    def forward(self, x, traj_info):
        features, scale = self.network(x)
        return independent_multivariate_normal(loc=features, scale=scale)


class DynamicsEncoder(DeterministicEncoder):
    # For the Dynamics encoder we want to keep the ground truth pixels as unencoded pixels
    def encode_target(self, x, traj_info):
        return independent_multivariate_normal(loc=x, scale=0)


class InverseDynamicsEncoder(DeterministicEncoder):
    def encode_extra_context(self, x, traj_info):
        return self.forward(x, traj_info)


class MomentumEncoder(Encoder):
    def __init__(self, obs_shape, representation_dim, learn_scale=False,
                 momentum_weight=0.999, inner_encoder_architecture_cls=None, **kwargs):
        super().__init__(**kwargs)
        if learn_scale:
            inner_encoder_cls = StochasticEncoder
        else:
            inner_encoder_cls = DeterministicEncoder
        self.query_encoder = inner_encoder_cls(obs_shape, representation_dim, inner_encoder_architecture_cls)
        self.momentum_weight = momentum_weight
        self.key_encoder = copy.deepcopy(self.query_encoder)
        for param in self.key_encoder.parameters():
            param.requires_grad = False

    def forward(self, x, traj_info):
        return self.query_encoder(x, traj_info)

    def encode_target(self, x, traj_info):
        """
        Encoder target/keys using momentum-updated key encoder. Had some thought of making _momentum_update_key_encoder
        a backwards hook, but seemed overly complex for an initial proof of concept
        :param x:
        :return:
        """
        with torch.no_grad():
            self._momentum_update_key_encoder()
            z_dist = self.key_encoder(x, traj_info)
            return MultivariateNormal(loc=z_dist.loc.detach(), covariance_matrix=z_dist.covariance_matrix.detach())

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = param_k.data * self.momentum_weight + param_q.data * (1. - self.momentum_weight)


class RecurrentEncoder(Encoder):
    def __init__(self, obs_shape, representation_dim, learn_scale=False, num_recurrent_layers=2,
                 single_frame_repr_dim=None, min_traj_size=5, inner_encoder_architecture_cls=None, rnn_output_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.num_recurrent_layers = num_recurrent_layers
        self.min_traj_size = min_traj_size
        self.representation_dim = representation_dim
        self.single_frame_repr_dim = representation_dim if single_frame_repr_dim is None else single_frame_repr_dim
        self.single_frame_encoder = DeterministicEncoder(obs_shape, self.single_frame_repr_dim,
                                                         inner_encoder_architecture_cls)
        self.context_rnn = nn.LSTM(self.single_frame_repr_dim, rnn_output_dim,
                                   self.num_recurrent_layers, batch_first=True)
        self.mean_layer = nn.Linear(rnn_output_dim, self.representation_dim)
        if learn_scale:
            self.scale_layer = nn.Linear(rnn_output_dim, self.representation_dim)
        else:
            self.scale_layer = self.ones_like_representation_dim

    def ones_like_representation_dim(self, x):
        return torch.ones(size=(x.shape[0], self.representation_dim,), device=x.device)

    def _reshape_and_stack(self, z, traj_info):
        batch_size = z.shape[0]
        input_shape = z.shape[1:]
        trajectory_id, timesteps = traj_info
        # We should have trajectory_id values for every element in the batch z
        assert len(z) == len(trajectory_id), "Every element in z must have a trajectory ID in a RecurrentEncoder"
        # A set of all distinct trajectory IDs
        trajectories = torch.unique(trajectory_id)
        padded_trajectories = []
        mask_lengths = []
        for trajectory in trajectories:
            traj_timesteps = timesteps[trajectory_id == trajectory]
            assert list(traj_timesteps) == sorted(list(traj_timesteps)), "Batches must be sorted to use a RecurrentEncoder"
            # Get all Z vectors associated with a trajectory, which have now been confirmed to be sorted timestep-wise
            traj_z = z[trajectory_id == trajectory]
            # Keep track of how many actual unpadded values were in the trajectory
            mask_lengths.append(traj_z.shape[0])
            pad_size = batch_size - traj_z.shape[0]
            padding = torch.zeros((pad_size,) + input_shape).to(self.device)
            padded_z = torch.cat([traj_z, padding])
            padded_trajectories.append(padded_z)
        assert np.mean(mask_lengths) > self.min_traj_size, f"Batches must contain trajectories with an average " \
                                                           f"length above {self.min_traj_size}. Trajectories found: {traj_info}"
        stacked_trajectories = torch.stack(padded_trajectories, dim=0)
        return stacked_trajectories, mask_lengths

    def encode_target(self, x, traj_info):
        return self.single_frame_encoder(x, traj_info)

    def forward(self, x, traj_info):
        # Reshape the input z to be (some number of) batch_size-length trajectories
        z = self.single_frame_encoder(x, traj_info).loc
        stacked_trajectories, mask_lengths = self._reshape_and_stack(z, traj_info)
        hiddens, final = self.context_rnn(stacked_trajectories)
        # Pull out only the hidden states corresponding to actual non-padding inputs, and concat together
        masked_hiddens = []
        for i, trajectory_length in enumerate(mask_lengths):
            masked_hiddens.append(hiddens[i][:trajectory_length])
        flattened_hiddens = torch.cat(masked_hiddens, dim=0)

        mean = self.mean_layer(flattened_hiddens)
        scale = self.scale_layer(flattened_hiddens)
        return independent_multivariate_normal(loc=mean, scale=scale)

