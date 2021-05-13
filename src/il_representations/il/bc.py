#!/usr/bin/env python3
"""BC implementation that makes it easy to use auxiliary losses."""
import itertools as it

import gym
from stable_baselines3.common import preprocessing
import torch

from il_representations.data.read_dataset import datasets_to_loader
from il_representations.il.utils import streaming_extract_keys


def _prep_batch_bc(batch, observation_space, augmentation_fn, device):
    """Take a batch from the data loader and prepare it for BC."""
    acts_tensor = (
        torch.as_tensor(batch["acts"]).contiguous().to(device)
    )
    obs_tensor = torch.as_tensor(batch["obs"]).contiguous().to(device)
    obs_tensor = preprocessing.preprocess_obs(
        obs_tensor,
        observation_space,
        normalize_images=True,
    )
    # we always apply augmentations to observations
    if augmentation_fn is not None:
        obs_tensor = augmentation_fn(obs_tensor)
    # FIXME(sam): SB policies *always* apply preprocessing, so we
    # need to undo the preprocessing we did before applying
    # augmentations. The code below is the inverse of SB's
    # preprocessing.preprocess_obs, but only for Box spaces.
    if isinstance(observation_space, gym.spaces.Box):
        if preprocessing.is_image_space(observation_space):
            obs_tensor = obs_tensor * 255.0
    return obs_tensor, acts_tensor


class BC:
    """Bare-bones BC implementation without outer loop etc.

    Adapted from imitation's `bc.py`."""
    def __init__(
        self,
        *,
        policy,
        l2_weight,
        ent_weight,
    ):
        self.policy = policy
        self.l2_weight = l2_weight
        self.ent_weight = ent_weight
        self.observation_space = self.policy.observation_space

    def all_trainable_params(self):
        """Trainable parameters for BC (only includes policy parameters)."""
        return self.policy.parameters()

    def make_data_iter(self, il_dataset, augmentation_fn, batch_size,
                       n_batches, shuffle_buffer_size):
        expert_data_loader = datasets_to_loader(
            il_dataset,
            batch_size=batch_size,
            nominal_length=batch_size * n_batches,
            shuffle=True,
            shuffle_buffer_size=shuffle_buffer_size,
            preprocessors=[streaming_extract_keys("obs", "acts")])
        data_iter = it.chain.from_iterable(it.repeat(expert_data_loader))
        for batch in data_iter:
            yield _prep_batch_bc(
                batch=batch, observation_space=self.observation_space,
                augmentation_fn=augmentation_fn,
                device=self.policy.device)

    def batch_forward(self, obs_acts):
        """Do a forward pass of the network, given a set of observations and
        actions."""
        obs, acts = obs_acts

        # first return value is state value (which we don't use)
        _, log_prob, entropy = self.policy.evaluate_actions(obs, acts)
        prob_true_act = torch.exp(log_prob).mean()
        log_prob = log_prob.mean()
        ent_loss = entropy = entropy.mean()

        l2_norms = [
            torch.sum(torch.square(w)) for w in self.policy.parameters()
        ]
        l2_loss_raw = sum(l2_norms) / 2

        ent_term = -self.ent_weight * ent_loss
        neglogp = -log_prob
        l2_term = self.l2_weight * l2_loss_raw
        loss = neglogp + ent_term + l2_term

        # FIXME(sam): I don't think the .item() calls here are JIT-able, and
        # they tend to be very slow. Would be nice to split it out and/or
        # minimize the number of calls that are necessary. Possibly just
        # delaying the .item() calls as long as possible would help---instead
        # of doing .item(), I could just .detach(). Once all follow-up
        # operations have been done (e.g. backwards pass), I could call .item()
        # again. (IDK if that will work, but worth benchmarking it)
        stats_dict = dict(
            neglogp=neglogp.item(),
            loss=loss.item(),
            prob_true_act=prob_true_act.item(),
            ent_loss_raw=entropy.item(),
            ent_loss_term=ent_term.item(),
            l2_loss_raw=l2_loss_raw.item(),
            l2_loss_term=l2_term.item(),
        )

        return loss, stats_dict
