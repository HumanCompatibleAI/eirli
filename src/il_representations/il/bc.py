#!/usr/bin/env python3
"""BC implementation that makes it easy to use auxiliary losses."""
import weakref

import gym
from stable_baselines3.common import preprocessing
import torch

from il_representations.data.read_dataset import datasets_to_loader
from il_representations.il.utils import streaming_extract_keys


def _prep_batch_bc(batch, observation_space, augmentation_fn, device):
    """Take a batch from the data loader and prepare it for BC."""
    acts_tensor = (torch.as_tensor(batch["acts"]).contiguous().to(device))
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
    # preprocessing.preprocess_obs, but only for Box spaces. Should make sure
    # this doesn't break silently elsewhere.
    if isinstance(observation_space, gym.spaces.Box):
        if preprocessing.is_image_space(observation_space):
            obs_tensor = obs_tensor * 255.0
    return obs_tensor, acts_tensor


def _prep_batch_bc_preproc(observation_space, augmentation_fn):
    """Take a single sample from the webdataset pipeline and prepare it for BC
    (everything except batching and moving to GPU)."""
    def inner_preproc(samples):
        for sample in samples:
            act_tensor = torch.as_tensor(sample["acts"]).contiguous()
            obs_tensor_4d = torch.as_tensor(sample["obs"])[None].contiguous()
            assert obs_tensor_4d.ndim == 4  # NCHW, N=1 due to [None] above
            obs_tensor_4d = preprocessing.preprocess_obs(
                obs_tensor_4d,
                observation_space,
                normalize_images=True,
            )
            # we always apply augmentations to observations
            if augmentation_fn is not None:
                obs_tensor_4d = augmentation_fn(obs_tensor_4d)
            # FIXME(sam): SB policies *always* apply preprocessing, so we need
            # to undo the preprocessing we did before applying augmentations.
            # The code below is the inverse of SB's
            # preprocessing.preprocess_obs, but only for Box spaces. Should
            # make sure this doesn't break silently elsewhere.
            if isinstance(observation_space, gym.spaces.Box):
                if preprocessing.is_image_space(observation_space):
                    obs_tensor_4d = obs_tensor_4d * 255.0
            final_obs_tensor = obs_tensor_4d.squeeze(0)
            yield {'obs': final_obs_tensor, 'acts': act_tensor}
    return inner_preproc


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

    def make_data_iter(self,
                       il_dataset,
                       augmentation_fn,
                       batch_size,
                       n_batches,
                       shuffle_buffer_size,
                       **ds_to_loader_kwargs):
        expert_data_loader = datasets_to_loader(
            il_dataset,
            batch_size=batch_size,
            nominal_length=batch_size * n_batches,
            shuffle=True,
            shuffle_buffer_size=shuffle_buffer_size,
            preprocessors=[
                streaming_extract_keys("obs", "acts"),
                # CHANGED 2022-01-08: moved augmentations to subprocess
                # FIXME(sam): make the '.to("cpu")' call unnecessary
                _prep_batch_bc_preproc(
                    observation_space=self.observation_space,
                    augmentation_fn=augmentation_fn.to('cpu')),
            ],
            **ds_to_loader_kwargs)
        data_iter = None
        try:
            while True:
                data_iter = iter(expert_data_loader)
                for batch in data_iter:
                    # CHANGED 2022-01-08: _prep_batch_bc is no longer necessary
                    # now that _prep_batch_bc_preproc is part of webdataset
                    # preprocessing pipeline.
                    # yield _prep_batch_bc(batch=batch,
                    #                      observation_space=self.observation_space,
                    #                      augmentation_fn=augmentation_fn,
                    #                      device=self.policy.device)
                    dev = self.policy.device
                    yield batch['obs'].to(dev), batch['acts'].to(dev)
        finally:
            if data_iter is not None:
                # Explicit __del__ call is a hack to ensure that data_iter's
                # worker pool gets shut down.

                # (Sometimes the shutdown does not happen in the course of
                # normal garbage collection; possibly there is a reference
                # cycle in multiprocess iterator code for Torch's data loader.
                # Torch does not provide us with an _explicit_ mechanism to
                # close the process pool, so we are stuck doing this.)
                data_iter.__del__()

    def batch_forward(self, obs_acts):
        """Do a forward pass of the network, given a set of observations and
        actions."""
        obs, acts = obs_acts
        obs = obs.to(self.policy.device)
        acts = acts.to(self.policy.device)

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

        # TODO(sam): I don't think the .item() calls here are JIT-able, and
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
