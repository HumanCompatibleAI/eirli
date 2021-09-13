"""
BatchExtenders are used in situations where you want to pass a batch forward
for loss that is different than the batch seen by your encoder. The currently
implemented situation where this is the case is Momentum, where you want to
pass forward a bunch of negatives from prior encoding runs to increase the
difficulty of your prediction task. One might also imagine this being useful
for doing trajectory-mixing in a RNN case where batches naturally need to be
all from a small number of trajectories, but this isn't yet implemented.
"""
from abc import ABC, abstractmethod
import torch
from torch.distributions import Normal
from il_representations.algos.utils import independent_multivariate_normal


class BatchExtender(ABC):
    @abstractmethod
    def __call__(self, context_dist, target_dist):
        pass


class IdentityBatchExtender(BatchExtender):
    def __call__(self, contexts, targets):
        return contexts, targets


class QueueBatchExtender(BatchExtender):
    def __init__(self, queue_dim, device, queue_size=8192, sample=False):
        super(QueueBatchExtender, self).__init__()
        self.queue_size = queue_size
        self.representation_dim = queue_dim
        self.sample = sample
        self.device = device
        self.queue_loc = torch.randn(self.queue_size, self.representation_dim)
        self.queue_scale = torch.ones(self.queue_size, self.representation_dim)
        self.queue_ptr = 0

    def __call__(self, context_dist, target_dist):
        # Call up current contents of the queue, duplicate. Add targets to the queue,
        # potentially overriding old information in the process. Return targets concatenated to contents of queue
        targets_mean = target_dist.mean
        targets_stddev = target_dist.stddev

        # Pull out the diagonals of our MultivariateNormal covariance matrices, so we don't store all the extra 0s
        batch_size = targets_mean.shape[0]
        queue_targets_scale = (self.queue_scale.clone().detach()).to(self.device)
        queue_targets_loc = (self.queue_loc.clone().detach()).to(self.device)

        # Insert all the targets into the queue, wrapping around at the end.
        # insert_ptr is a pointer into targets_{loc,scale}.
        insert_ptr = 0
        while insert_ptr < batch_size:
            # number of elements we'll insert on this round
            n_inserted = min(
                # don't insert more than we have in the targets array
                batch_size - insert_ptr,
                # don't insert beyond the end of the queue
                self.queue_size - self.queue_ptr)
            assert n_inserted > 0, \
                (insert_ptr, batch_size, self.queue_ptr, self.queue_size)

            # now overwrite the relevant elements using fresh data from the
            # target_* tensors
            self.queue_loc[self.queue_ptr:self.queue_ptr + n_inserted] \
                = targets_mean[insert_ptr:insert_ptr + n_inserted]
            self.queue_scale[self.queue_ptr:self.queue_ptr + n_inserted] \
                = targets_stddev[insert_ptr:insert_ptr + n_inserted]

            # advance pointers
            insert_ptr += n_inserted
            self.queue_ptr = (self.queue_ptr + n_inserted) % self.queue_size

        merged_mean = torch.cat([targets_mean, queue_targets_loc], dim=0)
        merged_stddev = torch.cat([targets_stddev, queue_targets_scale], dim=0)
        merged_target_dist = independent_multivariate_normal(mean=merged_mean,
                                                             stddev=merged_stddev)
        return context_dist, merged_target_dist
