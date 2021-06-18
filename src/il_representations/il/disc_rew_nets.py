"""Custom discriminator/reward networks for `imitation`."""

from imitation.rewards.reward_nets import RewardNet
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
import torch as th
from torch import nn

from il_representations.algos.encoders import (BaseEncoder,
                                               compute_rep_shape_encoder)


class ImageDiscrimNet(nn.Module):
    """Image-based discriminator network. This is intended to be passed as a
    `discrim_net` argument to `DiscrimNetGAIL` in
    `imitation.rewards.discrim_net`."""
    def __init__(self,
                 observation_space,
                 action_space,
                 encoder=None,
                 encoder_cls=None,
                 encoder_kwargs=None,
                 fc_dim=256):
        super().__init__()

        if encoder is not None:
            assert encoder_cls is None, \
                "cannot supply both encoder and encoder_cls"
            self.obs_encoder = encoder
            obs_out_dim, = compute_rep_shape_encoder(observation_space,
                                                     self.obs_encoder)
        else:
            if encoder_cls is None:
                encoder_cls = BaseEncoder
            if encoder_kwargs is None:
                encoder_kwargs = {}
            self.obs_encoder = encoder_cls(obs_space=observation_space,
                                           representation_dim=fc_dim,
                                           **encoder_kwargs)
            obs_out_dim = fc_dim

        # postprocess_mlp takes both the raw action and the image features
        action_dim = get_flattened_obs_dim(action_space)
        mlp_in = obs_out_dim + action_dim
        self.postprocess_mlp = nn.Sequential(
            nn.Linear(mlp_in, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, observation, action, traj_info=None):
        obs_dist = self.obs_encoder(observation, traj_info=traj_info)
        assert isinstance(obs_dist, th.distributions.Distribution)
        obs_feats = obs_dist.mean
        obs_feats_and_act = th.cat((obs_feats, action), dim=1)
        final_result = self.postprocess_mlp(obs_feats_and_act)

        # make sure that we don't have a trailing unit dimension (that can mess
        # up DiscrimNetGAIL)
        final_result = final_result.squeeze(1)
        assert final_result.shape == (observation.size(0), )

        return final_result


class _IDNWithoutNextStateOrAction(ImageDiscrimNet):
    def forward(self, state, action, next_state, done, traj_info=None):
        return super().forward(state, action, traj_info=traj_info)


class ImageRewardNet(RewardNet):
    """Reward net for AIRL that wraps ImageDiscrimNet (which basically does the
    thing we want anyway)."""
    def __init__(self, observation_space, action_space, **idn_kwargs):
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         use_state=True,
                         use_action=True,
                         use_next_state=False,
                         use_done=False,
                         normalize_images=True)
        self._base_reward_net = _IDNWithoutNextStateOrAction(
            observation_space=observation_space,
            action_space=action_space,
            **idn_kwargs)

    @property
    def base_reward_net(self):
        return self._base_reward_net

    def reward_train(self, state, action, next_state, done):
        # no shaping network (what does that even achieve?)
        rew = self.base_reward_net(state, action, next_state, done)
        assert rew.shape == state.shape[:1]
        return rew
