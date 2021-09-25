"""Environments from D4RL."""
from gym import ObservationWrapper, spaces
import numpy as np

from il_representations.envs.config import env_cfg_ingredient


@env_cfg_ingredient.capture
def load_dataset_d4rl(task_name, n_traj=None):
    # lazy-load D4RL to register envs
    import d4rl  # noqa: F401
    raise NotImplementedError()


class D4RLVectorWrapper(ObservationWrapper):
    """Wrapper for D4RL+CARLA that reshapes observations to be (48,48,3) and
    then transposes to channels-first."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=self.observation(env.observation_space.low),
            high=self.observation(env.observation_space.high))

    @staticmethod
    def observation(obs):
        assert obs.shape == (48 * 48 * 3, ), obs.shape
        obs = obs.reshape(48, 48, 3)
        obs = np.transpose(obs, (2, 0, 1))
        assert obs.shape == (3, 48, 48)
        return obs
