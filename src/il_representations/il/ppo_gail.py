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

    def update(self, rollouts):
        # TODO: maybe refactor this so that instead of passing in a `rollouts`
        # object, I'm passing in something that's legible to SB3 etc.
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                total_loss, stats_dict = self.batch_forward(sample)

                # opt step + grad clip
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += stats_dict['value_loss']
                action_loss_epoch += stats_dict['action_loss']
                dist_entropy_epoch += stats_dict['dist_entropy']

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
