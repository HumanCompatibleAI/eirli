from .utils import set_global_seeds


class BaseEnvironmentLearner:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.observation_shape = self.env.observation_space.shape
        self.action_space = env.action_space
        self.action_size = env.action_space.n

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
