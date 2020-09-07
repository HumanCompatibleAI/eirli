import copy
import os
import traceback
import warnings

from torch.distributions import MultivariateNormal
import numpy as np
from stable_baselines3.common.preprocessing import preprocess_obs
import torch
from torch import nn

from il_representations.algos.utils import independent_multivariate_normal


"""
Encoders conceptually serve as the bit of the representation learning architecture that learns the representation itself
(except in RNN cases, where encoders only learn the per-frame representation). 

The only real complex thing to note here is the MomentumEncoder architecture, which creates two CNNEncoders, 
and updates weights of one as a slowly moving average of the other. Note that this bit of momentum is separated 
from the creation and filling of a queue of representations, which is handled by the BatchExtender module 
"""


def compute_output_shape(observation_space, layers):
    """Compute the size of the output after passing an observation from
    `observation_space` through the given `layers`."""
    # [None] adds a batch dimension to the random observation
    torch_obs = torch.tensor(observation_space.sample()[None])
    with torch.no_grad():
        sample = preprocess_obs(torch_obs, observation_space, normalize_images=True)
        for layer in layers:
            # forward prop to compute the right size
            sample = layer(sample)

    # make sure batch axis still matches
    assert sample.shape[0] == torch_obs.shape[0]

    # return everything else
    return sample.shape[1:]


def compute_rep_shape_encoder(observation_space, encoder):
    """Compute representation shape for an entire Encoder."""
    sample_obs = torch.FloatTensor(observation_space.sample()[None])
    sample_obs = preprocess_obs(sample_obs, observation_space,
                                normalize_images=True)
    device_encoder = encoder.to(sample_obs.device)
    with torch.no_grad():
        sample_dist = device_encoder(sample_obs, traj_info=None)
        sample_out = sample_dist.sample()

    # batch dim check
    assert sample_out.shape[0] == sample_obs.shape[0]

    return sample_out.shape[1:]


def warn_on_non_image_tensor(x):
    """Do some basic checks to make sure the input image tensor looks like a
    batch of stacked square frames. Good sanity check to make sure that
    preprocessing is not being messed up somehow."""
    stack_str = None

    def do_warning(message):
        # issue a warning, but annotate it with some information about the
        # stack (specifically, basenames of code files and line number at the
        # time of exception for each stack frame except this one)
        nonlocal stack_str
        if stack_str is None:
            frames = traceback.extract_stack()
            stack_str = '/'.join(
                f'{os.path.basename(frame.filename)}:{frame.lineno}'
                # [:-1] skips the current frame
                for frame in frames[:-1])
        warnings.warn(message + f" (stack: {stack_str})")

    # check that image has rank 4
    if x.ndim != 4:
        do_warning(f"Image tensor has rank {x.ndim}, not rank 4")

    # check that H=W
    if x.shape[2] != x.shape[3]:
        do_warning(
            f"Image tensor shape {x.shape} doesn't have square images")

    # check that image is in [0,1] (approximately)
    # this is the range that SB uses
    v_min = torch.min(x).item()
    v_max = torch.max(x).item()
    if v_min < -0.01 or v_max > 1.01:
        do_warning(
            f"Input image tensor has values in range [{v_min}, {v_max}], "
            "not expected range [0, 1]")

    std = torch.std(x).item()
    if std < 0.05:
        do_warning(
            f"Input image tensor values have low stddev {std} (range "
            f"[{v_min}, {v_max}])")


class BasicCNN(nn.Module):
    """Similar to the CNN from the Nature DQN paper."""
    def __init__(self, observation_space, representation_dim):
        super().__init__()

        self.input_channel = observation_space.shape[0]
        shared_network_layers = []

        # first apply convolution layers + flattening
        conv_arch = [
            {'out_dim': 32, 'kernel_size': 8, 'stride': 4},
            {'out_dim': 64, 'kernel_size': 4, 'stride': 2},
            {'out_dim': 64, 'kernel_size': 3, 'stride': 1},
        ]
        for layer_spec in conv_arch:
            shared_network_layers.append(nn.Conv2d(self.input_channel, layer_spec['out_dim'],
                                                   kernel_size=layer_spec['kernel_size'], stride=layer_spec['stride']))
            shared_network_layers.append(nn.ReLU())
            self.input_channel = layer_spec['out_dim']
        shared_network_layers.append(nn.Flatten())

        # now customise the dense layers to handle an appropriate-sized conv output
        dense_in_dim, = compute_output_shape(observation_space, shared_network_layers)
        dense_arch = [
            # this input size is accurate for Atari, but will be ovewritten for other envs
            {'in_dim': 64*7*7},
        ]
        dense_arch[0]['in_dim'] = dense_in_dim
        dense_arch[-1]['out_dim'] = representation_dim

        # apply the dense layers
        for ind, layer_spec in enumerate(dense_arch[:-1]):
            shared_network_layers.append(nn.Linear(layer_spec['in_dim'], layer_spec['out_dim']))
            shared_network_layers.append(nn.ReLU())
        # no ReLU after last layer
        last_layer_spec = dense_arch[-1]
        shared_network_layers.append(
            nn.Linear(last_layer_spec['in_dim'], last_layer_spec['out_dim']))

        self.shared_network = nn.Sequential(*shared_network_layers)

    def forward(self, x):
        warn_on_non_image_tensor(x)
        return self.shared_network(x)


class MAGICALCNN(nn.Module):
    """The CNN from the MAGICAL paper."""
    def __init__(self,
                 observation_space,
                 representation_dim,
                 use_bn=True,
                 use_ln=False,
                 dropout=None,
                 use_sn=False,
                 width=2,
                 ActivationCls=torch.nn.ReLU):
        super().__init__()

        def conv_block(in_chans, out_chans, kernel_size, stride, padding):
            # We sometimes disable bias because batch norm has its own bias.
            conv_layer = nn.Conv2d(
                in_chans,
                out_chans,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_bn,
                padding_mode='zeros')

            if use_sn:
                # apply spectral norm if necessary
                conv_layer = nn.utils.spectral_norm(conv_layer)

            layers = [conv_layer]

            if dropout:
                # dropout after conv, but before activation
                # (doesn't matter for ReLU)
                layers.append(nn.Dropout2d(dropout))

            layers.append(ActivationCls())

            if use_bn:
                # Insert BN layer after convolution (and optionally after
                # dropout). I doubt order matters much, but see here for
                # CONTROVERSY:
                # https://github.com/keras-team/keras/issues/1802#issuecomment-187966878
                layers.append(nn.BatchNorm2d(out_chans))

            return layers

        w = width
        conv_out_dim = 64 * w
        conv_layers = [
            # at input: (96, 96) (assuming MAGICAL; for other domains it will
            # be 84x84)
            *conv_block(observation_space.shape[0], 32 * w, kernel_size=5, stride=1, padding=2),
            # now: (96, 96)
            *conv_block(32 * w, 64 * w, kernel_size=3, stride=2, padding=1),
            # now: (48, 48)
            *conv_block(64 * w, 64 * w, kernel_size=3, stride=2, padding=1),
            # now: (24, 24)
            *conv_block(64 * w, 64 * w, kernel_size=3, stride=2, padding=1),
            # now: (12, 12)
            *conv_block(64 * w, conv_out_dim, kernel_size=3, stride=2, padding=1),
            # now (Trained network,Trained network)
            nn.Flatten()
        ]

        # another FC layer to make feature maps the right size
        fc_in_size, = compute_output_shape(observation_space, conv_layers)
        fc_layers = [
            nn.Linear(fc_in_size, 128 * w),
            ActivationCls(),
            nn.Linear(128 * w, representation_dim),
        ]
        if use_sn:
            # apply SN to linear layers too
            fc_layers = [
                nn.utils.spectral_norm(layer) if isinstance(layer, nn.Linear) else layer
                for layer in fc_layers
            ]

        all_layers = [*conv_layers, *fc_layers]
        self.shared_network = nn.Sequential(*all_layers)

    def forward(self, x):
        warn_on_non_image_tensor(x)
        return self.shared_network(x)


# string names for convolutional networks; this makes it easier to choose
# between them from the command line
NETWORK_SHORT_NAMES = {
    'BasicCNN': BasicCNN,
    'MAGICALCNN': MAGICALCNN,
}


def get_obs_encoder_cls(obs_encoder_cls):
    if obs_encoder_cls is None:
        return BasicCNN
    if isinstance(obs_encoder_cls, str):
        try:
            return NETWORK_SHORT_NAMES[obs_encoder_cls]
        except KeyError:
            raise ValueError(f"Unknown encoder name '{obs_encoder_cls}'")
    return obs_encoder_cls


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
    def __init__(self, obs_space, representation_dim, obs_encoder_cls=None, scale_constant=1, **kwargs):
        """
        :param obs_space: The observation space that this Encoder will be used on
        :param representation_dim: The number of dimensions of the representation that will be learned
        :param obs_encoder_cls: An internal architecture implementing `forward` to return a single vector
        representing the mean representation z of a fixed-variance representation distribution
        """
        super().__init__(**kwargs)
        obs_encoder_cls = get_obs_encoder_cls(obs_encoder_cls)
        self.network = obs_encoder_cls(obs_space, representation_dim)
        self.scale_constant = scale_constant

    def forward(self, x, traj_info):
        features = self.network(x)
        return independent_multivariate_normal(loc=features, scale=self.scale_constant)


class StochasticEncoder(Encoder):
    def __init__(self, obs_space, representation_dim, obs_encoder_cls=None, latent_dim=None, **kwargs):
        """
        :param obs_space: The observation space that this Encoder will be used on
        :param representation_dim: The number of dimensions of the representation that will be learned
        :param obs_encoder_cls: An internal architecture implementing `forward` to return a single
            vector. This is expected NOT to end in a ReLU (i.e. final layer should be linear).
        :param latent_dim: Dimension of the latents that feed into mean and std networks (default:
            representation_dim * 2).
        two vectors, representing the mean AND learned standard deviation of a representation distribution
        """
        super().__init__(**kwargs)
        obs_encoder_cls = get_obs_encoder_cls(obs_encoder_cls)
        if latent_dim is None:
            latent_dim = representation_dim * 2
        self.network = obs_encoder_cls(obs_space, latent_dim)
        self.mean_layer = nn.Linear(latent_dim, representation_dim)
        self.scale_layer = nn.Linear(latent_dim, representation_dim)

    def forward(self, x, traj_info):
        shared_repr = self.network(x)
        mean = self.mean_layer(shared_repr)
        scale = torch.exp(self.scale_layer(shared_repr))
        return independent_multivariate_normal(loc=mean, scale=scale)


class DynamicsEncoder(DeterministicEncoder):
    # For the Dynamics encoder we want to keep the ground truth pixels as unencoded pixels
    def encode_target(self, x, traj_info):
        return independent_multivariate_normal(loc=x, scale=0)


class InverseDynamicsEncoder(DeterministicEncoder):
    def encode_extra_context(self, x, traj_info):
        return self.forward(x, traj_info)


class MomentumEncoder(Encoder):
    def __init__(self, obs_shape, representation_dim, learn_scale=False,
                 momentum_weight=0.999, obs_encoder_cls=None, **kwargs):
        super().__init__(**kwargs)
        obs_encoder_cls = get_obs_encoder_cls(obs_encoder_cls)
        if learn_scale:
            inner_encoder_cls = StochasticEncoder
        else:
            inner_encoder_cls = DeterministicEncoder
        self.query_encoder = inner_encoder_cls(obs_shape, representation_dim, obs_encoder_cls)
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
                 single_frame_repr_dim=None, min_traj_size=5, obs_encoder_cls=None, rnn_output_dim=64, **kwargs):
        super().__init__(**kwargs)
        obs_encoder_cls = get_obs_encoder_cls(obs_encoder_cls)
        self.num_recurrent_layers = num_recurrent_layers
        self.min_traj_size = min_traj_size
        self.representation_dim = representation_dim
        self.single_frame_repr_dim = representation_dim if single_frame_repr_dim is None else single_frame_repr_dim
        self.single_frame_encoder = DeterministicEncoder(obs_shape, self.single_frame_repr_dim,
                                                         obs_encoder_cls)
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

