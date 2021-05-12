#!/usr/bin/env python3
"""BC implementation that makes it easy to use auxiliary losses."""
import itertools as it

import gym
from stable_baselines3.common import preprocessing
import torch

from il_representations.data.read_dataset import datasets_to_loader
from il_representations.il.utils import streaming_extract_keys

# Some considerations, after a bit of thinking:
#
# - We probably want a shared optimiser so that we can trade off repL loss and
#   IL loss.
# - It's likely easier to give the repL and BC learners entirely independent
#   forward passes. The encoder part could be shared in theory
# - Separate forward passes probably means separate augmentation and batch
#   construction phases.
# - I don't really want to get tied to this specific factorisation. If I can
#   minimise coupling between components then I should do that.
# - I may need some algorithm-dependent outer loop. e.g. if I do PPO + GAIL, I
#   probably want an outer loop that is "aware" of the idea of PPO's
#   epoch/minibatch structure, of alternating GAIL updates, and of repL.
# - For GAIL, I probably want separate repL objectives both for discriminator
#   learning and for policy learning. I don't know whether I'll share a
#   representation between the two yet (realistically, probably not).
# - There's also complication that PPO probably ought to have access to the
#   GAIL discriminator in order to evaluate rewards, but it shouldn't "train"
#   the discriminator. Hmm, alternatively, I could make reward evaluation the
#   province of the higher-level code, so that PPO doesn't have to be aware
#   that it's even part of GAIL (this wouldn't work for Q-learning, but
#   Q-learning also has a different structure & will probably need different
#   high-level code anyway).

# SKETCH: How I expect caller to use BC + repL together?
#
# # this should track (but deduplicate) parameters
# combined_model = JointModel(*bc.models(), *repl.models())
# combined_model.train()
# for batches_elapseed in range(n_batches):
#     # data
#     bc_batch = bc.next_batch()
#     repl_batch = repl.next_batch()
#
#     # forward + back
#     combined_model.zero_grad()
#     bc_loss, bc_stats = bc_forward(bc_batch)
#     repl_loss, repl_stats = repl.forward(repl_batch)
#     total_loss = bc_loss + repl_loss
#     total_loss.backward()
#
#     # opt step
#     combined_model.assert_has_grads()
#     joint_optimizer.step()
#
#     # logging
#     log_stats({
#         **bc_stats, **repl_stats,
#         'total_loss': total_loss.item(),
#         'batches_elapsed': batches_elapsed,
#     })

# Main confusion after writing this sketch:
#
# - How do we keep track of all the models? Some of them will be shared between
#   BC and repL, some of them will not be. Don't want to double-pass parameters
#   to the optimiser, but perhaps it is easier to do that and then deduplicate
#   them after the fact (this is intrinsically heuristic though).
# - I know how to check that every parameter in the combined model has grads.
#   How do I check that I didn't omit any leaves, though?

# SKETCH: How I expect caller to use PPO + GAIL disc. + repL together
# # this should track (but deduplicate) parameters
# combined_model = JointModel(*bc.models(), *repl.models())
# combined_model.train()
# steps_elapsed = 0
# while steps_elapsed < n_timesteps:
#     # data
#     # FIXME(sam): need to figure out how data flows between modules for GAIL.
#     # That's more complicated, because PPO always needs "fresh" samples, and
#     # GAIL benefits from fresh samples too. They probably also need/want
#     # replay buffers to work properly. I may want to use the replay buffer as
#     # additional repL data too. Messy :(
#     bc_batch = bc.next_batch()
#     repl_batch = repl.next_batch()
#
#     # forward + back
#     combined_model.zero_grad()
#     bc_loss, bc_stats = bc_forward(bc_batch)
#     repl_loss, repl_stats = repl.forward(repl_batch)
#     total_loss = bc_loss + repl_loss
#     total_loss.backward()
#
#     # opt step
#     assert_has_grads(combined_opt)
#     joint_optimizer.step()
#
#     # logging
#     log_stats({
#         **disc_stats, **ppo_stats, **repl_stats,
#         'total_pol_loss': total_pol_loss.item(),
#         'total_disc_loss': total_disc_loss.item(),
#         'batches_elapsed': batches_elapsed,
#     })

# Rough steps:
# - Sample some rollouts
# - PPO+repL step:
#    - Forward pass on PPO with sampled rollouts.
#    - Sample repL data from repL dataset (and/or rollouts)
#    - Forward pass on repL
#    - Joint backwards pass + opt step
# - Disc.+repL steps:
#    - Add fresh rollouts to GAIL replay buffer.
#    - Sample novice data from replay buffer, expert data from dataset
#    - Forward pass on discriminator
#    - Sample repL data from repL dataset (maybe we can have a replay buffer
#      for repL too, we'll see)
#    - Forward pass on repL
#    - Joint backwards pass + opt step
# - Stats


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
