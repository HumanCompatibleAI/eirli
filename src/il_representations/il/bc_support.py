"""Support code for using imitation's BC implementation."""
import os

import torch as th


class BCModelSaver:
    """Callback that saves BC policy every N epochs."""
    def __init__(self, policy, save_dir, save_interval_batches):
        self.policy = policy
        self.save_dir = save_dir
        self.last_save_batches = 0
        self.save_interval_batches = save_interval_batches

    def __call__(self, *, batch_num, **kwargs):
        """It is assumed that this is called on epoch end."""
        if batch_num >= self.last_save_batches + self.save_interval_batches:
            os.makedirs(self.save_dir, exist_ok=True)
            save_fn = f'policy_{batch_num:07d}_batches.pt'
            save_path = os.path.join(self.save_dir, save_fn)
            th.save(self.policy, save_path)
            self.last_save_batches = batch_num
