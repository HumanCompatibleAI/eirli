#!/usr/bin/env python3
"""WIP PPO + GAIL implementation that makes it easy to use auxiliary losses.
Forked from Ilya Kostrikov's implementation on Github:

https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py

As of 2021-05-19 this is not yet finished (will probably work on it once I have
BC results).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def batch_forward(self, sample):
        obs_batch, recurrent_hidden_states_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, \
            old_action_log_probs_batch, adv_targ = sample

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy, _ \
            = self.actor_critic.evaluate_actions(
                obs_batch, recurrent_hidden_states_batch, masks_batch,
                actions_batch)

        ratio = torch.exp(action_log_probs -
                          old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch) \
                .clamp(-self.clip_param, self.clip_param)
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (
                value_pred_clipped - return_batch).pow(2)
            value_loss = 0.5 * torch.max(
                value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (return_batch - values).pow(2).mean()

        total_loss = value_loss * self.value_loss_coef + action_loss \
            - dist_entropy * self.entropy_coef
        stats_dict = dict(
            value_loss=value_loss.item(),
            action_loss=action_loss.item(),
            dist_entropy=dist_entropy.item(),
        )

        return total_loss, stats_dict

    def update_generator(self, rollouts):
        # TODO: maybe refactor this so that instead of passing in a `rollouts`
        # object, I'm passing in something that's legible to SB3 etc.
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)

            for sample in data_generator:
                ppo_loss, stats_dict = self.batch_forward(sample)
                total_loss = yield ppo_loss

                # opt step + grad clip
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()


class DiscriminatorDataset(IterableDataset):
    """Dataset for training discriminator. Capable of mixing both expert and
    novice datasets, and of adding more novice samples at execution time.

    Uses label convention that 1 = real and 0 = fake (so discriminator
    estimates p(fake|data))."""
    def __init__(self, expert_dataset, novice_circ_buffer_size=4096):
        pass


class Discriminator:
    """Implements only the GAIL discriminator."""
    def __init__(
        self,
        *,
        discriminator,
    ):
        self.discriminator = discriminator

    def all_trainable_params(self):
        """Trainable parameters for discriminator."""
        return self.discriminator.parameters()

    def make_data_iter(self, expert_dataset, novice_dataset, augmentation_fn,
                       batch_size, n_batches, shuffle_buffer_size):
        # Label convention is inherited from reference implementation of GAIL:
        # expert/demo/real = 1/high; generator/novice/fake = 0/low. Learning
        # binary "is real?" classifier.

        expert_data_loader = datasets_to_loader(
            expert_dataset,
            batch_size=batch_size,
            nominal_length=batch_size * n_batches,
            shuffle=True,
            shuffle_buffer_size=shuffle_buffer_size,
            preprocessors=[streaming_extract_keys("obs", "acts")])
        data_iter = repeat_chain_non_empty(expert_data_loader)
        for batch in data_iter:
            yield _prep_batch_bc(
                batch=batch, observation_space=self.observation_space,
                augmentation_fn=augmentation_fn,
                device=self.policy.device)

    def batch_forward(self, obs_reals):
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
