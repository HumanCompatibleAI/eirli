import os

import torch as th


class GAILSavePolicyCallback:
    """This callback can be passed to AdversarialTrainer.train() to save a
    policy snapshot every `save_every_n_steps` time steps."""
    def __init__(self,
                 ppo_algo,
                 save_every_n_steps,
                 save_dir,
                 *,
                 save_template='policy_{timesteps:08d}_steps.pt'):
        # I don't think this callback is actually kept around by GAIL or
        # PPO, so we shouldn't need a weakref
        self.ppo_algo = ppo_algo
        self.save_every_n_steps = save_every_n_steps
        self.save_dir = save_dir
        self.last_save_num_steps = None
        self.save_template = save_template

    def __call__(self, rounds):
        """This gets called after each 'round' (consisting of some
        generator updates followed by some discriminator updates)."""
        num_timesteps = self.ppo_algo.num_timesteps
        if num_timesteps is None:
            num_timesteps = 0

        if (self.last_save_num_steps is not None and num_timesteps <
                self.last_save_num_steps + self.save_every_n_steps):
            return
        self.last_save_num_steps = num_timesteps

        intermediate_pol_name = self.save_template.format(
            timesteps=num_timesteps)
        save_path = os.path.join(self.save_dir, intermediate_pol_name)
        th.save(self.ppo_algo.policy, save_path)
