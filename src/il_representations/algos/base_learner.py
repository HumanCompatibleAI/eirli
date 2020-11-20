import gym

from il_representations.algos.utils import set_global_seeds


class BaseEnvironmentLearner:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.observation_shape = observation_space.shape
        self.action_space = action_space
        # FIXME(sam): action_size is only needed for EncoderSimplePolicyHead,
        # which (arguably) should infer action size from a vecenv. Remove this
        # if EncoderSimplePolicyHead is refactored.
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.action_size = action_space.n
        elif (isinstance(self.action_space, gym.spaces.Box)
              and len(self.action_space.shape) == 1):
            self.action_size, = self.action_space.shape
        else:
            raise NotImplementedError(
                f"can't handle action space {self.action_space}")

    def set_random_seed(self, seed):
        if seed is None:
            return
        # Seed python, numpy and torch random generator
        set_global_seeds(seed)
