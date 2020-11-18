import inspect
import logging
import os

import imitation.util.logger as logger
import numpy as np
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.utils import get_device
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.dataloader import DataLoader, default_collate
from torch.utils.tensorboard import SummaryWriter
import webdataset.filters as wds_filters

from il_representations.algos.base_learner import BaseEnvironmentLearner
from il_representations.algos.batch_extenders import QueueBatchExtender
from il_representations.algos.utils import AverageMeter, LinearWarmupCosine

DEFAULT_HARDCODED_PARAMS = [
    'encoder', 'decoder', 'loss_calculator', 'augmenter',
    'target_pair_constructor', 'batch_extender'
]


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def to_dict(kwargs_element):
    # To get around not being able to have empty dicts as default values
    if kwargs_element is None:
        return {}
    else:
        return kwargs_element


class RepresentationLearner(BaseEnvironmentLearner):
    def __init__(self, *,
                 observation_space,
                 action_space,
                 log_dir,
                 encoder=None,
                 decoder=None,
                 loss_calculator=None,
                 target_pair_constructor=None,
                 augmenter=None,
                 batch_extender=None,
                 optimizer=torch.optim.Adam,
                 scheduler=None,
                 representation_dim=512,
                 projection_dim=None,
                 device=None,
                 shuffle_batches=True,
                 shuffle_buffer_size=1024,
                 batch_size=256,
                 preprocess_extra_context=True,
                 preprocess_target=True,
                 save_interval=100,
                 optimizer_kwargs=None,
                 target_pair_constructor_kwargs=None,
                 augmenter_kwargs,
                 encoder_kwargs=None,
                 decoder_kwargs=None,
                 batch_extender_kwargs=None,
                 loss_calculator_kwargs=None,
                 dataset_max_workers=min(4, os.cpu_count()),
                 scheduler_kwargs=None):

        super(RepresentationLearner, self).__init__(
            observation_space=observation_space, action_space=action_space)
        for el in (encoder, decoder, loss_calculator, target_pair_constructor):
            assert el is not None
        # TODO clean up this kwarg parsing at some point
        self.log_dir = log_dir
        logger.configure(log_dir, ["stdout", "csv", "tensorboard"])

        self.encoder_checkpoints_path = os.path.join(self.log_dir, 'checkpoints', 'representation_encoder')
        os.makedirs(self.encoder_checkpoints_path, exist_ok=True)
        self.decoder_checkpoints_path = os.path.join(self.log_dir, 'checkpoints', 'loss_decoder')
        os.makedirs(self.decoder_checkpoints_path, exist_ok=True)

        self.device = get_device("auto" if device is None else device)
        self.shuffle_batches = shuffle_batches
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = batch_size
        self.preprocess_extra_context = preprocess_extra_context
        self.preprocess_target = preprocess_target
        self.save_interval = save_interval
        self.dataset_max_workers = dataset_max_workers

        if projection_dim is None:
            # If no projection_dim is specified, it will be assumed to be the same as representation_dim
            # This doesn't have any meaningful effect unless you specify a projection head.
            projection_dim = representation_dim

        self.augmenter = augmenter(**augmenter_kwargs)
        self.target_pair_constructor = target_pair_constructor(**to_dict(target_pair_constructor_kwargs))

        encoder_kwargs = to_dict(encoder_kwargs)
        self.encoder = encoder(self.observation_space, representation_dim, **encoder_kwargs).to(self.device)
        self.decoder = decoder(representation_dim, projection_dim, **to_dict(decoder_kwargs)).to(self.device)

        if batch_extender is QueueBatchExtender:
            # TODO maybe clean this up?
            batch_extender_kwargs = batch_extender_kwargs or {}
            batch_extender_kwargs['queue_dim'] = projection_dim
            batch_extender_kwargs['device'] = self.device

        if batch_extender_kwargs is None:
            # Doing this to avoid having batch_extender() take an optional kwargs dict
            self.batch_extender = batch_extender()
        else:
            self.batch_extender = batch_extender(**batch_extender_kwargs)

        self.loss_calculator = loss_calculator(self.device, **to_dict(loss_calculator_kwargs))

        trainable_encoder_params, trainable_decoder_params = self._get_trainable_parameters()
        self.optimizer = optimizer(trainable_encoder_params + trainable_decoder_params,
                                   **to_dict(optimizer_kwargs))

        self.scheduler_cls = scheduler
        self.scheduler = None
        self.scheduler_kwargs = scheduler_kwargs or {}

        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'contrastive_tf_logs'), flush_secs=15)


    def _calculate_norms(self, norm_type=2):
        """
        :param norm_type: the order of the norm
        :return: the norm of the gradient and the norm of the weights
        """
        norm_type = float(norm_type)

        encoder_params, decoder_params = self._get_trainable_parameters()
        trainable_params = encoder_params + decoder_params
        stacked_gradient_norms = torch.stack([torch.norm(p.grad.detach(), norm_type).to(self.device) for p in trainable_params])
        stacked_weight_norms = torch.stack([torch.norm(p.detach(), norm_type).to(self.device) for p in trainable_params])

        gradient_norm = torch.norm(stacked_gradient_norms, norm_type)
        weight_norm = torch.norm(stacked_weight_norms, norm_type)

        return gradient_norm, weight_norm

    def _get_trainable_parameters(self):
        """
        :return: the trainable encoder parameters and the trainable decoder parameters
        """
        trainable_encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        trainable_decoder_params = [p for p in self.decoder.parameters() if p.requires_grad]
        return trainable_encoder_params, trainable_decoder_params

    def _prep_tensors(self, tensors_or_arrays):
        """
        :param tensors_or_arrays: A list of Torch tensors or numpy arrays (or None)
        :return: A torch tensor moved to the device associated with this
            learner, and converted to float
        """
        if tensors_or_arrays is None:
            # sometimes we get passed optional arguments with default value
            # None; we can ignore them & return None in response
            return
        if not torch.is_tensor(tensors_or_arrays):
            tensor_list = [torch.as_tensor(tens) for tens in tensors_or_arrays]
            batch_tensor = torch.stack(tensor_list, dim=0)
        else:
            batch_tensor = tensors_or_arrays
        if batch_tensor.ndim == 4:
            # if the batch_tensor looks like images, we check that it's also NCHW
            is_nchw_heuristic = \
                batch_tensor.shape[1] < batch_tensor.shape[2] \
                and batch_tensor.shape[1] < batch_tensor.shape[3]
            if not is_nchw_heuristic:
                raise ValueError(
                    f"Batch tensor axes {batch_tensor.shape} do not look "
                    "like they're in NCHW order. Did you accidentally pass in "
                    "a channels-last tensor?")
        if torch.is_floating_point(batch_tensor):
            # cast double to float for perf reasons (also drops half-precision)
            dtype = torch.float
        else:
            # otherwise use whatever the input type was (typically uint8 or
            # int64, but presumably original dtype was fine whatever it was)
            dtype = None
        return batch_tensor.to(self.device, dtype=dtype)

    def _preprocess(self, input_data):
        # SB will normalize to [0,1]
        return preprocess_obs(input_data, self.observation_space,
                              normalize_images=True)

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
            return batch['context'], batch['target'], batch['traj_ts_ids'], None
        else:
            return batch['context'], batch['target'], batch['traj_ts_ids'], batch['extra_context']

    def learn(self, dataset, training_epochs=None, training_batches=None):
        """
        :param dataset:
        :return:
        """
        assert training_epochs is not None or training_batches is not None, \
            "One of training_epochs or training_batches must be specified"
        assert not (training_epochs is not None and training_batches is not None), \
            "Only one of training_epochs or training_batches can be specified"
        # Construct representation learning dataset of correctly paired
        # (context, target) pairs
        dataset.pipe(self.target_pair_constructor)
        if self.shuffle_batches:
            dataset.shuffle(self.shuffle_buffer_size)
        dataset.pipe(wds_filters.batched(
            batchsize=int(self.batch_size), partial=True,
            collation_fn=default_collate))

        # do not start more workers than the number of shards
        wds_workers = min(len(dataset.urls), self.dataset_max_workers)
        assert wds_workers > 0
        # TODO(sam): this isn't guaranteed to interleave data from different
        # shards randomly. The optimal solution is to write a new
        # IterableDataset that joins several individual datasets by taking a
        # random number of samples from each. (or we could just have tiny,
        # one-trajectory shards)
        dataloader = DataLoader(dataset, num_workers=wds_workers,
                                batch_size=None)

        loss_record = []

        if self.scheduler_cls is not None:
            if self.scheduler_cls in [CosineAnnealingLR, LinearWarmupCosine]:
                self.scheduler = self.scheduler_cls(self.optimizer, training_epochs, **to_dict(self.scheduler_kwargs))
            else:
                self.scheduler = self.scheduler_cls(self.optimizer, **to_dict(self.scheduler_kwargs))

        self.encoder.train(True)
        self.decoder.train(True)
        batches_trained = 0
        epochs_trained = 0
        training_complete = False
        logging.debug(f"Training with {training_epochs} epochs and {training_batches} batches")
        logging.debug(f"Batch size is {self.batch_size}")

        while not training_complete:
            loss_meter = AverageMeter()
            # Set encoder and decoder to be in training mode

            for step, batch in enumerate(dataloader):
                # Construct batch (currently just using Torch's default batch-creator)
                contexts, targets, traj_ts_info, extra_context = self.unpack_batch(batch)

                # Use an algorithm-specific augmentation strategy to augment either
                # just context, or both context and targets
                contexts, targets = self._prep_tensors(contexts), self._prep_tensors(targets)
                extra_context = self._prep_tensors(extra_context)
                traj_ts_info = self._prep_tensors(traj_ts_info)
                # Note: preprocessing might be better to do on CPU if, in future, we can parallelize doing so
                contexts = self._preprocess(contexts)
                if self.preprocess_target:
                    targets = self._preprocess(targets)
                contexts, targets = self.augmenter(contexts, targets)
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
                loss_item = loss.item()
                assert not np.isnan(loss_item), "Loss is not NAN"
                loss_meter.update(loss_item)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                del loss  # so we don't use again

                gradient_norm, weight_norm = self._calculate_norms()

                # FIXME(sam): don't plot this every batch, plot it every k
                # batches (will make log files much smaller & let us stream
                # them to head node on Ray cluster)
                loss_meter.update(loss_item)
                logger.record('loss', loss_item)
                logger.record('gradient_norm', gradient_norm.item())
                logger.record('weight_norm', weight_norm.item())
                logger.record('epoch', epochs_trained)
                logger.record('within_epoch_step', step)
                logger.record('batches_trained', batches_trained)
                logger.dump(step=batches_trained)
                batches_trained += 1
                if (training_batches is not None and batches_trained >= training_batches):
                    logging.info(f"Breaking out of training in epoch {epochs_trained} because max batches "
                                 f"value of {training_batches} has been reached")
                    training_complete = True
                    break

            assert batches_trained > 0, \
                "went through training loop with no batches---empty dataset?"
            if epochs_trained == 0:
                # we infer the size of the dataset from the number of
                # iterations through the loop the first time
                logging.debug(f"Dataset size is {batches_trained}")

            if self.scheduler is not None:
                self.scheduler.step()
            loss_record.append(loss_meter.avg)

            epochs_trained += 1
            if (training_epochs is not None and epochs_trained >= training_epochs):
                training_complete = True

            should_save_checkpoint = (training_complete or
                                      epochs_trained % self.save_interval == 0)
            if should_save_checkpoint:
                most_recent_encoder_checkpoint_path = os.path.join(self.encoder_checkpoints_path,
                                                                   f'{epochs_trained}_epochs.ckpt')
                torch.save(self.encoder, most_recent_encoder_checkpoint_path)
                torch.save(self.decoder, os.path.join(self.decoder_checkpoints_path,
                                                      f'{epochs_trained}_epochs.ckpt'))

        return loss_record, most_recent_encoder_checkpoint_path
