import torch
import torch.nn as nn
import copy
from torch.distributions import Normal

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
                {'in_dim': 64*7*7}
             ]
}


class CNNEncoder(nn.Module):
    def __init__(self, obs_shape, representation_dim, architecture=DEFAULT_CNN_ARCHITECTURE, learn_stddev=False):
        super(CNNEncoder, self).__init__()

        self.input_channel = obs_shape[2]
        self.conv_layers = []
        self.dense_layers = []
        for layer_spec in architecture['CONV']:
            self.conv_layers.append(nn.Conv2d(self.input_channel, layer_spec['out_dim'],
                                              kernel_size=layer_spec['kernel_size'], stride=layer_spec['stride']))
            self.input_channel = layer_spec['out_dim']
        # Needs to be a ModuleList rather than just a list for the parameters of the listed layers
        # to be visible as part of the module .parameters() return
        self.conv_layers = nn.ModuleList(self.conv_layers)

        for ind, layer_spec in enumerate(architecture['DENSE'][:-1]):
            in_dim, out_dim = layer_spec.get('in_dim'), layer_spec.get('out_dim')
            self.dense_layers.append(nn.Linear(in_dim, out_dim))
        self.mean_layer = nn.Linear(architecture['DENSE'][-1]['in_dim'], representation_dim)
        if learn_stddev:
            self.stddev_layer = nn.Linear(architecture['DENSE'][-1]['in_dim'], representation_dim)
        else:
            self.stddev_layer = None

        self.dense_layers = nn.ModuleList(self.dense_layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x /= 255
        for conv_layer in self.conv_layers:
            x = self.relu(conv_layer(x))
        x = torch.flatten(x, 1)
        for dense_layer in self.dense_layers:
            x = self.relu(dense_layer(x))

        # TODO should there be a ReLU here?
        mean = self.relu(self.mean_layer(x))
        if self.stddev_layer is not None:
            stddev = self.relu(self.stddev_layer(x))
            # TODO are scale and stdev the same thing?
            return Normal(loc=mean, scale=stddev)
        else:
            return Normal(loc=mean, scale=1)


class DynamicsEncoder(CNNEncoder):
    def __init__(self, obs_shape, representation_dim, architecture=DEFAULT_CNN_ARCHITECTURE):
        super(DynamicsEncoder, self).__init__(obs_shape, representation_dim, architecture)
        # For the Dynamics encoder we want to keep the ground truth pixels as unencoded pixels
        self.encode_target = lambda x: Normal(loc=x, scale=0)


class InverseDynamicsEncoder(CNNEncoder):
    def __init__(self, obs_shape, representation_dim, architecture=DEFAULT_CNN_ARCHITECTURE):
        super(InverseDynamicsEncoder, self).__init__(obs_shape, representation_dim, architecture)
        self.encode_extra_context = self.forward


class MomentumEncoder(nn.Module):
    # TODO have some way to pass in optional momentum_weight param
    def __init__(self, obs_shape, representation_dim, momentum_weight=0.999):
        super(MomentumEncoder, self).__init__()
        self.query_encoder = CNNEncoder(obs_shape, representation_dim)
        self.key_encoder = copy.deepcopy(self.query_encoder)
        self.momentum_weight = momentum_weight

    def parameters(self, recurse=True):
        return self.query_encoder.parameters()

    def forward(self, x):
        return self.query_encoder(x)

    def encode_target(self, x):
        """
        Encoder target/keys using momentum-updated key encoder. Had some thought of making _momentum_update_key_encoder
        a backwards hook, but seemed overly complex for an initial POC
        :param x:
        :return:
        """
        with torch.no_grad():
            self._momentum_update_key_encoder()
            return self.key_encoder(x)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = param_k.data * self.momentum_weight + param_q.data * (1. - self.momentum_weight)
