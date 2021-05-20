import inspect
import logging
import os

import imitation.util.logger as im_log
import numpy as np
import stable_baselines3.common.distributions as sb3_dists
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.utils import get_device
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from il_representations.algos.base_learner import BaseEnvironmentLearner
from il_representations.algos.batch_extenders import QueueBatchExtender
from il_representations.algos.utils import AverageMeter, LinearWarmupCosine
from il_representations.data.read_dataset import (SubdatasetExtractor,
                                                  datasets_to_loader)
from il_representations.utils import (Timers, repeat_chain_non_empty,
                                      weight_grad_norms)

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
                 encoder=None,
                 decoder=None,
                 loss_calculator=None,
                 target_pair_constructor=None,
                 augmenter=None,
                 batch_extender=None,
                 representation_dim=512,
                 projection_dim=None,
                 device=None,
                 shuffle_batches=True,
                 shuffle_buffer_size=1024,
                 batch_size=384,
                 preprocess_extra_context=True,
                 preprocess_target=True,
                 target_pair_constructor_kwargs=None,
                 augmenter_kwargs,
                 encoder_kwargs=None,
                 decoder_kwargs=None,
                 batch_extender_kwargs=None,
                 loss_calculator_kwargs=None,
                 dataset_max_workers=0):

        super(RepresentationLearner, self).__init__(
            observation_space=observation_space, action_space=action_space)
        for el in (encoder, decoder, loss_calculator, target_pair_constructor):
            assert el is not None

        self.device = get_device("auto" if device is None else device)
        self.shuffle_batches = shuffle_batches
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = batch_size
        self.preprocess_extra_context = preprocess_extra_context
        self.preprocess_target = preprocess_target
        self.dataset_max_workers = dataset_max_workers

        if projection_dim is None:
            # If no projection_dim is specified, it will be assumed to be the
            # same as representation_dim.
            # This doesn't have any meaningful effect unless you specify a
            # projection head.
            projection_dim = representation_dim

        self.augmenter = augmenter(**augmenter_kwargs)
        self.target_pair_constructor = target_pair_constructor(
            **to_dict(target_pair_constructor_kwargs))

        encoder_kwargs = to_dict(encoder_kwargs)
        decoder_kwargs = to_dict(decoder_kwargs)
        duplicate_learn_scale = (encoder_kwargs.get('learn_scale', False) and
                                 decoder_kwargs.get('learn_scale', False))
        assert not duplicate_learn_scale, \
            "learn_scale shouldn't be set on encoder and decoder at same time"

        self.encoder = encoder(self.observation_space, representation_dim,
                               **encoder_kwargs).to(self.device)
        self.decoder = decoder(representation_dim, projection_dim,
                               **decoder_kwargs).to(self.device)

        if batch_extender is QueueBatchExtender:
            # TODO maybe clean this up?
            if batch_extender_kwargs is None:
                batch_extender_kwargs = {}
            else:
                batch_extender_kwargs = dict(batch_extender_kwargs)
            batch_extender_kwargs['queue_dim'] = projection_dim
            batch_extender_kwargs['device'] = self.device

        if batch_extender_kwargs is None:
            # Doing this to avoid having batch_extender() take an optional kwargs dict
            self.batch_extender = batch_extender()
        else:
            self.batch_extender = batch_extender(**batch_extender_kwargs)

        self.loss_calculator = loss_calculator(self.device, **to_dict(loss_calculator_kwargs))

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
            # if the batch_tensor looks like images, we check that it's also
            # NCHW
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

    @staticmethod
    def _unpack_batch(batch):
        """
        :param batch: A batch that may contain a numpy array of extra context,
            but may also simply have an empty list as a placeholder value for
            the `extra_context` key. If the latter, return None for
            extra_context, rather than an empty list (Torch data loaders can
            only work with lists and arrays, not None types)
        :return:
        """
        if len(batch['extra_context']) == 0:
            return batch['context'], batch['target'], batch['traj_ts_ids'], None
        else:
            return batch['context'], batch['target'], batch['traj_ts_ids'], batch['extra_context']

    def all_trainable_params(self):
        """
        :return: the trainable encoder parameters and the trainable decoder
            parameters.
        """
        trainable_encoder_params = [
            p for p in self.encoder.parameters() if p.requires_grad
        ]
        trainable_decoder_params = [
            p for p in self.decoder.parameters() if p.requires_grad
        ]
        return trainable_encoder_params + trainable_decoder_params

    def batch_forward(self, batch):
        """Do a forward pass on the given batch (from iterator generated by
        the .make_data_iter() method). Returns (1) the computed loss, and (2)
        a dictionary of detached tensors that are useful for debugging and
        sanity checks."""
        # Construct batch (currently just using Torch's default batch-creator)
        raw_contexts, raw_targets, traj_ts_info, extra_context = self.unpack_batch(batch)

        # Use an algorithm-specific augmentation strategy to augment either
        # just context, or both context and targets
        raw_contexts, raw_targets = self._prep_tensors(raw_contexts), self._prep_tensors(raw_targets)
        extra_context = self._prep_tensors(extra_context)
        traj_ts_info = self._prep_tensors(traj_ts_info)
        # Note: preprocessing might be better to do on CPU if, in future, we
        # can parallelize doing so
        raw_contexts = self._preprocess(raw_contexts)
        if self.preprocess_target:
            raw_targets = self._preprocess(raw_targets)
        contexts, targets = self.augmenter(raw_contexts, raw_targets)
        extra_context = self._preprocess_extra_context(extra_context)
        # This is typically a noop, but sometimes we also augment the extra
        # context
        extra_context = self.augmenter.augment_extra_context(extra_context)

        # These will typically just use the forward() function for the encoder,
        # but can optionally use a specific encode_context and encode_target if
        # one is implemented
        encoded_contexts = self.encoder.encode_context(contexts, traj_ts_info)
        encoded_targets = self.encoder.encode_target(targets, traj_ts_info)
        # Typically the identity function
        encoded_extra_context = self.encoder.encode_extra_context(extra_context, traj_ts_info)

        # Use an algorithm-specific decoder to "decode" the representations
        # into a loss-compatible tensor As with encode, these will typically
        # just use forward()
        decoded_contexts = self.decoder.decode_context(encoded_contexts, traj_ts_info, encoded_extra_context)
        decoded_targets = self.decoder.decode_target(encoded_targets, traj_ts_info, encoded_extra_context)

        # Optionally add to the batch before loss. By default, this is an
        # identity operation, but can also implement momentum queue logic
        decoded_contexts, decoded_targets = self.batch_extender(decoded_contexts, decoded_targets)

        # Use an algorithm-specific loss function. Typically this only requires
        # decoded_contexts and decoded_targets, but VAE requires
        # encoded_contexts, so we pass it in here
        loss = self.loss_calculator(decoded_contexts, decoded_targets, encoded_contexts)

        # things that get returned to calling function for (optional)
        # saving/analysis (we shouldn't do much to them---this code needs to be
        # fast)
        debug_objects = dict(
            contexts=contexts,
            targets=targets,
            extra_context=extra_context,
            encoded_contexts=encoded_contexts,
            encoded_targets=encoded_targets,
            encoded_extra_context=encoded_extra_context,
            decoded_contexts=decoded_contexts,
            decoded_targets=decoded_targets,
            traj_ts_info=traj_ts_info,
            raw_contexts=raw_contexts,
            raw_targets=raw_targets,
        )
        # we go through the debug objects and convert them all to tensors
        detached_debug_tensors = {}
        for key, value in debug_objects.items():
            if value is None:
                # skip things that we don't have access to
                continue

            if isinstance(value, (torch.distributions.Distribution,
                                  sb3_dists.Distribution)):
                # TODO(sam): profile this. If it's slow, then make it lazy so
                # that .sample() is only called when we actually want to save a
                # sample (I don't know how to make it lazy without keeping the
                # graph alive though ugh).
                det_sample = value.sample().detach()
                detached_debug_tensors[key + '_sample'] = det_sample
            elif torch.is_tensor(value):
                detached_debug_tensors[key] = value.detach()
            else:
                raise TypeError(f"Do not know how to detach {key}={value!r}")

        return loss, detached_debug_tensors

    def set_train(self, val=True):
        """Put modules in train mode (val=True) or test mode (val=False)."""
        self.encoder.train(val)
        self.decoder.train(val)

    def make_data_iter(self, datasets, batches_per_epoch, n_epochs, n_trajs):
        subdataset_extractor = SubdatasetExtractor(n_trajs=n_trajs)
        dataloader = datasets_to_loader(
            datasets, batch_size=self.batch_size,
            nominal_length=n_epochs * batches_per_epoch * self.batch_size,
            max_workers=self.dataset_max_workers,
            shuffle_buffer_size=self.shuffle_buffer_size,
            shuffle=self.shuffle_batches,
            preprocessors=(subdataset_extractor, self.target_pair_constructor))
        data_iter = repeat_chain_non_empty(dataloader)
        return data_iter

    def learn(self, datasets, batches_per_epoch, n_epochs, *, callbacks=(),
              end_callbacks=(), log_dir, log_interval=100,
              calc_log_interval=10, save_interval=1000, scheduler_cls=None,
              scheduler_kwargs=None, optimizer_cls=Adam,
              optimizer_kwargs=None, n_trajs=None):
        """Run repL training loop.

        Args:
            datasets ([wds.Dataset]): list of webdataset datasets which we will
                sample from. Each one likely represents data from different
                tasks, or different policies.
            batches_per_epoch (int): each 'epoch' simply consists of a fixed
                number of batches, with data drawn equally from the given
                datasets (as opposed to each epoch consisting of a complete
                cycle through all datasets).
            n_epochs (int): the total number of 'epochs' of optimisation to
                perform. Total number of updates will be `batches_per_epoch *
                n_epochs`.
            n_trajs (int): the total number of trajectories we want to use
                for training. The default 'None' will use the whole dataset.
            callbacks ([dict -> None]): list of functions to call at the
                end of each batch, after computing the loss and updating the
                network but before dumping logs. Will be provided with all
                local variables.
            end_callbacks ([dict -> None]): these callbacks will only be
                called once, at the end of training.
            log_dir (str): directory to store checkpoints etc. to.
            log_interval (int): num batches between log dumps.
            calc_log_interval (int): how often to record statistics which will
                be averaged in log dumps. Should generally be less than
                log_interval.
            save_interval (int): how often to save encoder/decoder snapshots.
            scheduler_cls (type): learning rate scheduler class.
            scheduler_kwargs (dict): keyword args to pass to scheduler_cls.
            optimizer_cls (type): optimizer class.
            optimizer_kwargs (dict): kwargs to pass to optimizer_cls.

        Returns: tuple of `(loss_record, most_recent_encoder_checkpoint_path)`.
            `loss_record` is a list of average loss values encountered at each
            epoch. `most_recent_encoder_checkpoint_path` is self-explanatory.
        """
        loss_record = []

        # dataset setup
        data_iter = self.make_data_iter(
            datasets=datasets, batches_per_epoch=batches_per_epoch,
            n_epochs=n_epochs, n_trajs=n_trajs)

        # optimizer and LR scheduler
        optimizer = optimizer_cls(self.all_trainable_params(),
                                  **to_dict(optimizer_kwargs))
        if scheduler_cls is not None:
            scheduler_kwargs = scheduler_kwargs or {}
            if scheduler_cls in [CosineAnnealingLR, LinearWarmupCosine]:
                scheduler = scheduler_cls(
                    optimizer, n_epochs, **to_dict(scheduler_kwargs))
            else:
                scheduler = scheduler_cls(
                    optimizer, **to_dict(scheduler_kwargs))
        else:
            scheduler = None

        self.encoder.train(True)
        self.decoder.train(True)
        batches_trained = 0
        logging.debug(
            f"Training for {n_epochs} epochs, each of {batches_per_epoch} "
            f"batches (batch size {self.batch_size})")

        for epoch_num in range(1, n_epochs + 1):
            loss_meter = AverageMeter()
            # Set encoder and decoder to be in training mode

            samples_seen = 0
            timers = Timers()
            timers.start('batch')
            for step in range(batches_per_epoch):
                batch = next(data_iter)
                # detached_debug_tensors isn't used directly in this function,
                # but might be used by callbacks that exploit locals()
                loss, detached_debug_tensors = self.batch_forward(batch)

                if batches_trained % calc_log_interval == 0:
                    # TODO(sam): I suspect we can get the same effect by just
                    # detaching loss & waiting until after optimizer.step()
                    # call to do .item(). Benchmark later & decide.
                    loss_item = loss.item()
                    assert not np.isnan(loss_item), "Loss is NaN"
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                del loss  # so we don't use again

                for callback in callbacks:
                    callback(locals())

                # restart time-per-batch counter
                timers.stop('batch')
                timers.start('batch')

                if batches_trained % calc_log_interval == 0:
                    gradient_norm, weight_norm = weight_grad_norms(
                        self.all_trainable_params())
                    loss_meter.update(loss_item)
                    im_log.sb_logger.record_mean('loss', loss_item)
                    im_log.sb_logger.record_mean(
                        'gradient_norm', gradient_norm.item())
                    im_log.sb_logger.record_mean(
                        'weight_norm', weight_norm.item())
                    im_log.record('epoch', epoch_num)
                    im_log.record('within_epoch_step', step)
                    im_log.record('batches_trained', batches_trained)
                    timer_stats = timers.dump_stats(check_running=False)
                    time_per_batch = timer_stats['batch']['mean']
                    im_log.record('time_per_batch', time_per_batch)
                    im_log.record(
                        'time_per_ksample',
                        1000 * time_per_batch / self.batch_size)

                if batches_trained % log_interval == 0:
                    im_log.dump(step=batches_trained)

                batches_trained += 1
                samples_seen += len(batch['context'])

            assert batches_trained > 0, \
                "went through training loop with no batches---empty dataset?"
            if epoch_num == 0:
                logging.debug(f"Epoch yielded {samples_seen} data points "
                              f"({step + 1} batches)")

            if scheduler is not None:
                scheduler.step()
            loss_record.append(loss_meter.avg)

            # save checkpoint on last epoch, or at regular interval
            is_last_epoch = epoch_num == n_epochs
            should_save_checkpoint = (is_last_epoch or
                                      epoch_num % save_interval == 0)
            if should_save_checkpoint:
                # save encoder
                encoder_checkpoints_path = os.path.join(
                    log_dir, 'checkpoints', 'representation_encoder')
                os.makedirs(encoder_checkpoints_path, exist_ok=True)
                most_recent_encoder_checkpoint_path = os.path.join(
                    encoder_checkpoints_path, f'{epoch_num}_epochs.ckpt')
                torch.save(self.encoder, most_recent_encoder_checkpoint_path)

                # save decoder
                decoder_checkpoints_path = os.path.join(
                    log_dir, 'checkpoints', 'loss_decoder')
                os.makedirs(decoder_checkpoints_path, exist_ok=True)
                torch.save(self.decoder, os.path.join(
                    decoder_checkpoints_path, f'{epoch_num}_epochs.ckpt'))

        for callback in end_callbacks:
            callback(locals())

        # if we were not scheduled to dump on the last batch we trained on,
        # then do one last log dump to make sure everything is there
        if not (batches_trained % log_interval == 0):
            im_log.dump(step=batches_trained)

        assert is_last_epoch, "did not make it to last epoch"
        assert should_save_checkpoint, "did not save checkpoint on last epoch"

        return loss_record, most_recent_encoder_checkpoint_path
