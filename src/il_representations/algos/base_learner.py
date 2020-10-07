import gym

from il_representations.algos.utils import set_global_seeds


class BaseEnvironmentLearner:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.observation_shape = self.env.observation_space.shape
        self.action_space = env.action_space
        # FIXME(sam): action_size is only needed for EncoderSimplePolicyHead,
        # which (arguably) should infer action size from a vecenv. Remove this
        # if EncoderSimplePolicyHead is refactored.
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.action_size = env.action_space.n
        elif (isinstance(self.action_space, gym.spaces.Box) and len(self.action_space.shape) == 1):
            self.action_size, = self.action_space.shape
        else:
            raise NotImplementedError(f"can't handle action space {self.action_space}")

    def set_random_seed(self, seed):
        if seed is None:
            return
        # Seed python, numpy and torch random generator
        set_global_seeds(seed)
        if self.env is not None:
            self.env.seed(seed)
            # Seed the action space. Useful when selecting random actions
            self.env.action_space.seed(seed)
        self.action_space.seed(seed)
