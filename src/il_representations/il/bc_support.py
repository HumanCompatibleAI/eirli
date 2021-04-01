"""Support code for using imitation's BC implementation."""
import os

import torch as th


class BCModelSaver:
    """Callback that saves BC policy every N epochs."""
    def __init__(self, policy, save_dir, save_interval_epochs):
        self.policy = policy
        self.save_dir = save_dir
        self.last_save_epochs = 0
        self.save_interval_epochs = save_interval_epochs
        self.epoch_count = 0

    def __call__(self, **kwargs):
        """It is assumed that this is called on epoch end."""
        self.epoch_count += 1
        if self.epoch_count >= self.last_save_epochs + self.save_interval_epochs:
            os.makedirs(self.save_dir, exist_ok=True)
            save_fn = f'policy_{self.epoch_count:04d}_epochs.pt'
            save_path = os.path.join(self.save_dir, save_fn)
            th.save(self.policy, save_path)
            print(f"Saved policy to {save_path}!")
            self.last_save_epochs = self.epoch_count
