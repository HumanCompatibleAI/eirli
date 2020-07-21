import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .batch_extenders import IdentityBatchExtender
from .base_learner import BaseEnvironmentLearner
from .utils import AverageMeter, LinearWarmupCosine, save_model, Logger
from .augmenters import AugmentContextOnly

DEFAULT_HYPERPARAMS = {
            'optimizer': torch.optim.SGD,
            'optimizer_kwargs': {'lr': 0.03, 'momentum': 0.9, 'weight_decay': 1e-4},
            'pretrain_epochs': 200,
            'max_grad_norm': 0.5,
            'batch_size': 256,
            'warmup_epochs': 10,
            'representation_dim': 512,
            'projection_dim': None,
            'seed': 0,
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'recurrent': False,
            'target_pair_constructor_kwargs': {},
            'augmenter_kwargs': {},
            'encoder_kwargs': {},
            'decoder_kwargs': {},
            'batch_extender_kwargs': {},
            'loss_calculator_kwargs': {},
        }

class RepresentationLearner(BaseEnvironmentLearner):
    def __init__(self, env, log_dir, encoder, decoder, loss_calculator, target_pair_constructor,
                 augmenter=AugmentContextOnly, batch_extender=IdentityBatchExtender, **kwargs):
        super(RepresentationLearner, self).__init__(env)
        self.env = env
        # TODO clean up this kwarg parsing at some point
        self.log_dir = log_dir
        self.logger = Logger(log_dir)
        for hyperparam, default in DEFAULT_HYPERPARAMS.items():
            setattr(self, hyperparam, kwargs.get(hyperparam, default))

        if self.projection_dim is None:
            # If no projection_dim is specified, it will be assumed to be the same as representation_dim
            # This doesn't have any meaningful effect unless you specify a projection head.
            self.projection_dim = self.representation_dim
        self.augmenter = augmenter(**self.augmenter_kwargs)
        self.target_pair_constructor = target_pair_constructor(**self.target_pair_constructor_kwargs)

        self.encoder = encoder(self.observation_shape, self.representation_dim, **self.encoder_kwargs).to(self.device)
        self.decoder = decoder(self.representation_dim, self.projection_dim, **self.decoder_kwargs).to(self.device)

        self.batch_extender = batch_extender(queue_dim=self.projection_dim, **self.batch_extender_kwargs)
        self.loss_calculator = loss_calculator(device=self.device, **self.loss_calculator_kwargs)

        self.optimizer = self.optimizer(list(self.encoder.parameters()) + list(self.decoder.parameters()), **self.optimizer_kwargs)

        # TODO make the scheduler parameterizable
        self.scheduler = LinearWarmupCosine(self.optimizer, self.warmup_epochs, self.pretrain_epochs)
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'contrastive_tf_logs'), flush_secs=15)

    def log_info(self, loss, step, epoch_ind):
        self.writer.add_scalar('loss', loss, step)
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, step)
        self.logger.log(f"Pretrain Epoch [{epoch_ind+1}/{self.pretrain_epochs}], step {step}, "
                        f"lr {lr}, "
                        f"loss {loss}")

    def tensorize(self, arr):
        """
        :param arr: A numpy array
        :return: A torch tensor moved to the device associated with this learner
        """
        return torch.FloatTensor(arr).to(self.device)

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
            return batch['context'].data.numpy(), batch['target'].data.numpy(), batch['traj_ts_ids'], batch['extra_context'].data.numpy()

    def learn(self, dataset):
        """

        :param dataset:
        :return:
        """
        # Construct representation learning dataset of correctly paired (context, target) pairs
        dataset = self.target_pair_constructor(dataset)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False if self.recurrent else True)

        # Set encoder and decoder to be in training mode
        self.encoder.train(True)
        self.decoder.train(True)

        loss_record = []
        for epoch in range(self.pretrain_epochs):
            loss_meter = AverageMeter()
            dataiter = iter(dataloader)
            for step in range(1, len(dataloader) + 1):

                # Construct batch (currently just using Torch's default batch-creator)
                batch = next(dataiter)
                contexts, targets, traj_ts_info, extra_context = self.unpack_batch(batch)

                # Use an algorithm-specific augmentation strategy to augment either
                # just context, or both context and targets
                contexts, targets = self.augmenter(contexts, targets)
                contexts, targets = self.tensorize(contexts), self.tensorize(targets)

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
                self.log_info(loss, step, epoch)

            self.scheduler.step()
            loss_record.append(loss_meter.avg.cpu().item())
            self.encoder.train(False)
            self.decoder.train(False)
            save_model(self.encoder, 'representation_encoder_network', os.path.join(self.log_dir, 'checkpoints'))
            save_model(self.decoder, 'representation_decoder_network', os.path.join(self.log_dir, 'checkpoints'))
