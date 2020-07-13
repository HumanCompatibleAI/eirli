from abc import ABC
import torch

"""
BatchExtenders are used in situations where you want to pass a batch forward for loss that is different than the 
batch seen by your encoder. The currently implemented situation where this is the case is Momentum, where you want 
to pass forward a bunch of negatives from prior encoding runs to increase the difficulty of your prediction task. 
One might also imagine this being useful for doing trajectory-mixing in a RNN case where batches naturally need 
to be all from a small number of trajectories, but this isn't yet implemented. 
"""


class BatchExtender(ABC):
    # TODO Is there a better way to optionally allow for more arguments ignored by the children?
    def __init__(self, **kwargs):
        pass

    def __call__(self, contexts, targets):
        pass


class IdentityBatchExtender(BatchExtender):
    def __call__(self, contexts, targets):
        return contexts, targets


class QueueBatchExtender(BatchExtender):
    def __init__(self, queue_size, queue_dim):
        super(QueueBatchExtender, self).__init__()
        self.queue_size = queue_size
        self.representation_dim = queue_dim
        self.queue = torch.randn(self.queue_size, self.representation_dim)
        self.queue_ptr = 0

    def __call__(self, contexts, targets):
        # Call up current contents of the queue, duplicate. Add targets to the queue,
        # potentially overriding old information in the process. Return targets concatenated to contents of queue
        batch_size = targets.shape[0]
        queue_targets = self.queue.clone().detach()
        self.queue[self.queue_ptr:self.queue_ptr + batch_size] = targets
        self.queue_ptr = (self.queue_ptr + batch_size) % self.queue_size
        merged_targets = torch.cat([targets, queue_targets], dim=0)

        return contexts, merged_targets
