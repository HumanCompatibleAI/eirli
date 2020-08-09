import os
import torch
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .batch_extenders import IdentityBatchExtender
from .base_learner import BaseEnvironmentLearner
from .utils import AverageMeter, LinearWarmupCosine, save_model, Logger
from .augmenters import AugmentContextOnly
import itertools
from gym.spaces import Box

def to_dict(kwargs_element):
    # To get around not being able to have empty dicts as default values
    if kwargs_element is None:
        return {}
    else:
        return kwargs_element



class RepresentationLearner(BaseEnvironmentLearner):
    def __init__(self, env, log_dir, encoder, decoder, loss_calculator, target_pair_constructor,
                 augmenter=AugmentContextOnly,
                 batch_extender=IdentityBatchExtender,
                 optimizer=torch.optim.Adam,
                 scheduler=None,
                 representation_dim=512,
                 projection_dim=None,
                 device=None,
                 shuffle_batches=True,
                 batch_size=256,
                 preprocess_extra_context=True,
                 save_interval=1,
                 optimizer_kwargs=None,
                 target_pair_constructor_kwargs=None,
                 augmenter_kwargs=None,
                 encoder_kwargs=None,
                 decoder_kwargs=None,
                 batch_extender_kwargs=None,
                 loss_calculator_kwargs=None,
                 scheduler_kwargs=None):

        super(RepresentationLearner, self).__init__(env)
        # TODO clean up this kwarg parsing at some point
        self.log_dir = log_dir
        self.logger = Logger(log_dir)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.shuffle_batches = shuffle_batches
        self.batch_size = batch_size
        self.preprocess_extra_context = preprocess_extra_context
        self.save_interval = save_interval

        self._make_channels_first()

        if projection_dim is None:
            # If no projection_dim is specified, it will be assumed to be the same as representation_dim
            # This doesn't have any meaningful effect unless you specify a projection head.
            projection_dim = representation_dim

        self.augmenter = augmenter(**to_dict(augmenter_kwargs))
        self.target_pair_constructor = target_pair_constructor(**to_dict(target_pair_constructor_kwargs))

        self.encoder = encoder(self.observation_space, representation_dim, **to_dict(encoder_kwargs)).to(self.device)
        self.decoder = decoder(representation_dim, projection_dim, **to_dict(decoder_kwargs)).to(self.device)

        if batch_extender_kwargs is None:
            # Doing this to avoid having batch_extender() take an optional kwargs dict
            self.batch_extender = batch_extender()
        else:
            if batch_extender_kwargs.get('queue_size') is not None:
                # Doing this slightly awkward updating of kwargs to avoid having
                # the superclass of BatchExtender accept queue_dim as an argument
                batch_extender_kwargs['queue_dim'] = projection_dim

            self.batch_extender = batch_extender(**batch_extender_kwargs)

        self.loss_calculator = loss_calculator(device=self.device, **to_dict(loss_calculator_kwargs))

        trainable_encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        trainable_decoder_params = [p for p in self.decoder.parameters() if p.requires_grad]
        self.optimizer = optimizer(trainable_encoder_params + trainable_decoder_params,
                                   **to_dict(optimizer_kwargs))

        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer, **to_dict(scheduler_kwargs))
        else:
            self.scheduler = None
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'contrastive_tf_logs'), flush_secs=15)
        self.encoder_checkpoints_path = os.path.join(self.log_dir, 'checkpoints', 'representation_encoder')
        os.makedirs(self.encoder_checkpoints_path, exist_ok=True)
        self.decoder_checkpoints_path = os.path.join(self.log_dir, 'checkpoints', 'loss_decoder')
        os.makedirs(self.decoder_checkpoints_path, exist_ok=True)

    def log_info(self, loss, step, epoch_ind, training_epochs):
        self.writer.add_scalar('loss', loss, step)
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, step)
        self.logger.log(f"Pretrain Epoch [{epoch_ind+1}/{training_epochs}], step {step}, "
                        f"lr {lr}, "
                        f"loss {loss}")

    def _make_channels_first(self):
        assert isinstance(self.observation_space, Box) and len(self.observation_shape) == 3, \
            "Only image observations are supported (Box observation spaces with shapes of length 3)"

        # Assumes an image in form (C, H, W) or (H, W, C) with H = W != C
        x, y, z = self.observation_shape
        if x != y and y == z:
            self.permutation_tuple = None
        else:
            assert x == y and x != z, "Can only handle square images in format (C, H, W) or (H, W, C)"
            self.observation_shape = (z, x, y)
            low = self.observation_space.low.reshape(self.observation_shape)
            high = self.observation_space.high.reshape(self.observation_shape)
            self.observation_space = Box(shape=self.observation_shape, low=low, high=high, dtype=np.uint8)
            self.permutation_tuple = (0, 3, 1, 2)

    def _tensorize(self, arr):
        """
        :param arr: A numpy array
        :return: A torch tensor moved to the device associated with this learner
        """
        return torch.FloatTensor(arr).to(self.device)

    def _preprocess(self, input_data):
        # Make channels first for image inputs
        if self.permutation_tuple is not None:
            assert isinstance(input_data, torch.Tensor) and len(input_data.shape) == 4, "Expected 4D-tensor to permute"
            input_data = input_data.permute(self.permutation_tuple)

        # Normalization to range [-1, 1]
        if isinstance(self.observation_space, Box):
            low, high = self.observation_space.low, self.observation_space.high
            low_min, low_max, high_min, high_max = low.min(), low.max(), high.min(), high.max()
            assert low_min == low_max and high_min == high_max
            low, high = low_min, high_max
            mid = (low + high) / 2
            delta = high - mid
            input_data = (input_data - mid) / delta
        return input_data

    def _preprocess_extra_context(self, extra_context):
        if extra_context is None or not self.preprocess_extra_context:
            return extra_context
        return self._preprocess(extra_context)

    # TODO maybe make static?
    def unpack_batch(self, batch):
        """
        :param batch: A batch that may contain a numpy array of extra context, but may also simply have an
        empty list as a placeholder value for the `extra_context` key. If the latter, return None for extra_context,
        rather than an empty list (Torch data loaders can only work with lists and arrays, not None types)
        :return:
        """
        if len(batch['extra_context']) == 0:
            return batch['context'].data.numpy(), batch['target'].data.numpy(), batch['traj_ts_ids'], None
        else:
            return batch['context'].data.numpy(), batch['target'].data.numpy(), batch['traj_ts_ids'], batch['extra_context']


    def learn(self, dataset, training_epochs):
        """
        :param dataset:
        :return:
        """
        # Construct representation learning dataset of correctly paired (context, target) pairs
        dataset = self.target_pair_constructor(dataset)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle_batches)
        # Set encoder and decoder to be in training mode
        self.encoder.train(True)
        self.decoder.train(True)

        loss_record = []
        for epoch in range(training_epochs):
            loss_meter = AverageMeter()
            dataiter = iter(dataloader)
            for step, batch in enumerate(dataloader, start=1):

                # Construct batch (currently just using Torch's default batch-creator)
                batch = next(dataiter)
                contexts, targets, traj_ts_info, extra_context = self.unpack_batch(batch)

                # Use an algorithm-specific augmentation strategy to augment either
                # just context, or both context and targets
                contexts, targets = self.augmenter(contexts, targets)
                contexts, targets = self._tensorize(contexts), self._tensorize(targets)
                # Note: preprocessing might be better to do on CPU if, in future, we can parallelize doing so
                contexts, targets = self._preprocess(contexts), self._preprocess(targets)
                extra_context = self._preprocess_extra_context(extra_context)

                # These will typically just use the forward() function for the encoder, but can optionally
                # use a specific encode_context and encode_target if one is implemented
                encoded_contexts = self.encoder.encode_context(contexts, traj_ts_info)
                encoded_targets = self.encoder.encode_target(targets, traj_ts_info)
                # Typically the identity function
                extra_context = self.encoder.encode_extra_context(extra_context, traj_ts_info)


                # Use an algorithm-specific decoder to "decode" the representations into a loss-compatible tensor
                # As with encode, these will typically just use forward()
                decoded_contexts = self.decoder.decode_context(encoded_contexts, traj_ts_info, extra_context)
                decoded_targets = self.decoder.decode_target(encoded_targets, traj_ts_info, extra_context)


                # Optionally add to the batch before loss. By default, this is an identity operation, but
                # can also implement momentum queue logic
                decoded_contexts, decoded_targets = self.batch_extender(decoded_contexts, decoded_targets)

                # Use an algorithm-specific loss function. Typically this only requires decoded_contexts and
                # decoded_targets, but VAE requires encoded_contexts, so we pass it in here
                loss = self.loss_calculator(decoded_contexts, decoded_targets, encoded_contexts)
                loss_meter.update(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.log_info(loss, step, epoch, training_epochs)

            if self.scheduler is not None:
                self.scheduler.step()
            loss_record.append(loss_meter.avg.cpu().item())
            self.encoder.train(False)
            self.decoder.train(False)
            if epoch % self.save_interval == 0:
                torch.save(self.encoder, os.path.join(self.encoder_checkpoints_path, f'{epoch}_epochs.ckpt'))
                torch.save(self.decoder, os.path.join(self.decoder_checkpoints_path, f'{epoch}_epochs.ckpt'))
