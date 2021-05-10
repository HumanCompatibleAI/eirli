from il_representations.algos.utils import set_global_seeds


class BaseEnvironmentLearner:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.observation_shape = observation_space.shape
        self.action_space = action_space

    def set_random_seed(self, seed):
        if seed is None:
            return
        # Seed python, numpy and torch random generator
        set_global_seeds(seed)
