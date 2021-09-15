"""
Pair Constructors turn an iterator of temporally-ordered
observations/actions/etc. into a dataset of `context`, `target`, and
`extra_context` data elements, along with a metadata tensor containing the
trajectory ID and timestep ID for each element in the dataset. The iterator
should satisfy two invariants:

- First, it should yield observations and actions in the order they were
  encountered in the environment (i.e. it should not interleave trajectories).
- Second, each value it yields should be a dictionary with keys 'obs',
  'acts', and 'dones'. The corresponding values should each be numpy arrays
  representing the relevant object at a single timestep (e.g. 'acts' should be
  just a single action, not a sequence of actions).

The `context` element is conceptually thought of as the element you're using to
do prediction, whereas the `target` is the ground truth or "positive" we're
trying to predict from the context, though this prediction framework is
admittedly a somewhat fuzzy match to the actual variety of techniques.

- In temporal contrastive loss settings, context is generally the element at
  position (t), and target the element at position (t+k)
- In pure-augmentation contrastive loss settings, context and target are the
  same element (which will be augmented in different ways)
- In a VAE, context and target are also the same element. Context will be
  mapped into a representation and then decoded back out, whereas the target
  will "tag along" as ground truth pixels needed to calculate the loss.
- In Dynamics modeling, context is the current state at time (t), target is the
  state at time (t+1) and extra context is the action taken at time (t)
"""

from abc import ABC, abstractmethod
import itertools
import math
import random
import warnings

import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF


class TargetPairConstructor(ABC):

    @abstractmethod
    def __call__(self, data_iter):
        """Generator that transforms environment step data in data_iter.

        Args:
            data_iter: iterable yielding dictionaries with 'obs', 'acts', and
                'dones' keys. Each dictionary represents a single time step.

        Yields: dictionaries with keys 'context', 'target', 'extra_context',
            and 'traj_ts_ids'. Types of values vary depending on the algorithm.
        """


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
        self.next_insert = 0
        self.full = False

    def append(self, item):
        assert item.shape == self.buf.shape[1:]
        self.buf[self.next_insert] = item
        self.next_insert = (self.next_insert + 1) % self.cap
        # we fill up once we wrap around
        self.full = self.full or self.next_insert == 0

    def get_oldest(self):
        """Get the oldest item that was inserted into in a full circular
        buffer."""
        # in TemporalOffsetPairConstructor we only use this method when the
        # queue is already full
        assert self.full, "have only implemented this for full queues"
        assert 0 <= self.next_insert < self.cap
        # return copy instead of view
        oldest = self.buf[self.next_insert].copy()
        return oldest

    def concat_all(self):
        """Concatenate all the items in a full circular buffer, in the order
        they were added."""
        # again TemporalOffsetPairConstructor only calls this on full queues
        assert self.full, "have only implemented this for full queues"
        result = np.concatenate(
            (self.buf[self.next_insert:], self.buf[:self.next_insert]))
        # make sure we haven't somehow ended up with a view
        # (I'm concerned np.concatenate might be doing optimisations to avoid
        # copies in some cases)
        result = np.require(result, requirements=('OWNDATA', ))
        return result

    def reset(self):
        """Reset circular buffer."""
        self.next_insert = 0
        self.full = False
        self.buf[:] = 0  # defensive, better to catch errors


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

            obs_queue.append(step_dict['obs'])
            act_queue.append(step_dict['acts'])

            if step_dict['dones']:
                timestep = 0
                trajectory_ind += 1
                obs_queue.reset()
                act_queue.reset()
            else:
                timestep += 1


class JigsawPairConstructor(TargetPairConstructor):
    def __init__(self, permutation_path, n_tiles=9):
        self.permutation_path = permutation_path
        self.permutation = np.load(self.permutation_path)
        self.n_tiles = n_tiles
        unit_size = math.sqrt(self.n_tiles)
        assert unit_size.is_integer(), 'self.n_tiles is not a square number.'
        self.unit_size = int(unit_size)


    def __call__(self, data_iter):
        timestep = 0
        traj_ind = 0

        # Get tile dim
        first_dict = next(data_iter)
        original_h, original_w = first_dict['obs'][0].shape[-2], \
                                 first_dict['obs'][0].shape[-1]

        # We -2 after division here because, as the original paper indicated,
        # preserving the whole image tile may let the network exploit edge information
        # and not really learn useful representations. Therefore, here we are using
        # the central-cropped version of these image tiles.
        tile_h, tile_w = int(original_h / math.sqrt(self.n_tiles)) - 2, \
                         int(original_w / math.sqrt(self.n_tiles)) - 2

        permutation_class = np.random.permutation(len(self.permutation))
        permute_idx = 0

        for step_dict in data_iter:
            # Process images into image tiles
            obs = step_dict['obs']
            target_list = []
            processed_obs = self.make_jigsaw_puzzle(obs,
                                                    permutation_class[permute_idx],
                                                    tile_h,
                                                    tile_w)
            target_list.append(np.random.randint(len(self.permutation)))

            target_list = np.array(target_list)
            yield {
                'context': processed_obs,
                'target': target_list,
                'extra_context': [],
                'traj_ts_ids': [traj_ind, timestep],
            }
            if step_dict['dones']:
                traj_ind += 1
                timestep = 0
            else:
                timestep += 1

    def make_jigsaw_puzzle(self, image, permutation_type, tile_h, tile_w):
        permute = self.permutation[permutation_type]

        _, h, w = image.shape
        image = torch.tensor(image)
        h_unit = h / self.unit_size
        w_unit = w / self.unit_size

        if not h_unit.is_integer() or not w_unit.is_integer():
            warnings.warn(f'Input images can not be evenly divided into '+
                          f'{self.n_tiles} {self.unit_size}*{self.unit_size} images.')

        h_unit, w_unit = int(h_unit), int(w_unit)

        # Split the image into tiles of [C, tile_h, tile_w] in its
        # permutation sequence.
        tiles = []
        for pos in permute:
            pos_h = int(pos // self.unit_size) * h_unit
            pos_w = int(pos % self.unit_size) * w_unit

            tile = TF.crop(image, top=pos_h, left=pos_w, height=tile_h,
                            width=tile_w)
            tiles.append(tile)

        # Concat tiles.
        concat_tiles = []
        for i in range(self.unit_size):
            # Shape: [C, tile_h, W]
            tile_w = torch.cat(tiles[i * self.unit_size : (i+1) * self.unit_size], dim=2)
            concat_tiles.append(tile_w)
        tiles = torch.cat(concat_tiles, dim=1) # Shape: [C, H, W]
        return tiles


