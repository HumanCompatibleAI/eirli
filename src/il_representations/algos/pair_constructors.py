"""
Pair Constructors turn an iterator of observations/actions/etc. into a dataset of `context`, `target`, and
`extra_context` data elements, along with a metadata tensor containing the trajectory ID and timestep ID for each
element in the dataset. The `context` element is conceptually thought of as the element you're using to do prediction,
whereas the `target` is the ground truth or "positive" we're trying to predict from the context, though this prediction
framework is admittedly a somewhat fuzzy match to the actual variety of techniques.

- In temporal contrastive loss settings, context is generally the element at position (t), and target the element at
position (t+k)
- In pure-augmentation contrastive loss settings, context and target are the same element (which will be augmented
in different ways)
- In a VAE, context and target are also the same element. Context will be mapped into a representation and then decoded
back out, whereas the target will "tag along" as ground truth pixels needed to calculate the loss.
- In Dynamics modeling, context is the current state at time (t), target is the state at time (t+1) and extra context
is the action taken at time (t)
"""

from abc import ABC, abstractmethod
import itertools

import numpy as np


class TargetPairConstructor(ABC):

    @abstractmethod
    def __call__(self, data_iter):
        pass


class IdentityPairConstructor:
    def __call__(self, data_iter):
        timestep = 0
        traj_ind = 0
        for step_dict in data_iter:
            yield {
                'context': step_dict['obs'],
                'target': step_dict['obs'],
                'extra_context': [],
                'traj_ts_ids': [traj_ind, timestep],
            }
            if step_dict['dones']:
                traj_ind += 1
                timestep = 0
            else:
                timestep += 1


class _CircularBuffer:
    """Circular buffer for storing observations, actions, etc. etc."""
    def __init__(self, capacity, example_item):
        self.cap = capacity
        self.buf = np.zeros_like(
            example_item, shape=(self.cap, ) + example_item.shape)
        self.next_free = 0
        self.full = False

    def append(self, item):
        assert item.shape == self.buf.shape[1:]
        self.buf[self.next_free] = item
        self.next_free = (self.next_free + 1) % self.cap
        # we fill up once we wrap around
        self.full = self.full or self.next_free == 0

    def get_oldest(self):
        """Get the oldest item that was inserted into in a full circular
        buffer."""
        # in TemporalOffsetPairConstructor we only use this method when the
        # queue is already full
        assert self.full, "have only implemented this for full queues"
        last_idx = (self.next_free + 1) % self.cap
        return self.buf[last_idx]

    def concat_all(self):
        """Concatenate all the items in a full circular buffer, in the order
        they were added."""
        # again TemporalOffsetPairConstructor only calls this on full queues
        assert self.full, "have only implemented this for full queues"
        return np.concatenate(
            (self.buf[self.next_free:], self.buf[:self.next_free]))

    def reset(self):
        """Reset circular buffer."""
        self.next_free = 0
        self.full = False
        self.buf[:] = 0  # defensive


class TemporalOffsetPairConstructor(TargetPairConstructor):
    def __init__(self, mode=None, temporal_offset=1):
        assert mode in (None, 'dynamics', 'inverse_dynamics')
        self.mode = mode
        self.k = temporal_offset

    def __call__(self, data_iter):
        trajectory_ind = timestep = 0

        # peek at first item in data iterator to create circular buffer, then
        # 'put it back' so we will encounter it again
        first_dict = next(data_iter)
        obs_queue = _CircularBuffer(self.k, first_dict['obs'])
        act_queue = _CircularBuffer(self.k, first_dict['acts'])
        data_iter = itertools.chain([first_dict], data_iter)

        for step_dict in data_iter:
            assert obs_queue.full == act_queue.full
            if obs_queue.full:
                if self.mode is None:
                    yield {
                        'context': obs_queue.get_oldest(),
                        'target': step_dict['obs'],
                        'extra_context': [],
                        'traj_ts_ids': [trajectory_ind, timestep]
                    }
                elif self.mode == 'dynamics':
                    yield {
                        'context': obs_queue.get_oldest(),
                        'target': step_dict['obs'],
                        'extra_context': act_queue.concat_all(),
                        'traj_ts_ids': [trajectory_ind, timestep]
                    }
                elif self.mode == 'inverse_dynamics':
                    yield {
                        'context': obs_queue.get_oldest(),
                        'target': act_queue.concat_all(),
                        'extra_context': step_dict['obs'],
                        'traj_ts_ids': [trajectory_ind, timestep]
                    }
                else:
                    assert False, f"mode {self.mode} not recognised"

            if step_dict['dones']:
                timestep = 0
                trajectory_ind += 1
                obs_queue.reset()
                act_queue.reset()
            else:
                timestep += 1

            obs_queue.append(step_dict['obs'])
            act_queue.append(step_dict['acts'])
