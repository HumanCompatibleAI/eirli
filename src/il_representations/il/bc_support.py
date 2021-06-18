"""Support code for using imitation's BC implementation."""
import os

import torch as th


class BCModelSaver:
    """Callback that saves BC policy every N epochs."""
    def __init__(self, policy, save_dir, epoch_length, save_interval_batches,
                 start_batch=0):
        self.policy = policy
        self.save_dir = save_dir
        self.last_save_batches = 0
        self.save_interval_batches = save_interval_batches
        self.batch_count = start_batch
        self.epoch_length = epoch_length

    def __call__(self, batch_num, **kwargs):
        """It is assumed that this is called on epoch end."""
        if batch_num >= self.last_save_batches + self.save_interval_batches:
            os.makedirs(self.save_dir, exist_ok=True)
            save_fn = f'policy_{batch_num:08d}_batches.pt'
            save_path = os.path.join(self.save_dir, save_fn)
            th.save(self.policy, save_path)
            print(f"Saved policy to {save_path}!")
            self.last_save_batches = batch_num
