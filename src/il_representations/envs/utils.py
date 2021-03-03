import gym
import numpy as np

try:
    from realistic_benchmarks.wrappers import ObservationWrapper

    class MinecraftPOVWrapper(ObservationWrapper):
        def __init__(self, env):
            super(ObservationWrapper, self).__init__(env)
            non_transposed_shape = self.env.observation_space['pov'].shape
            self.high = np.max(self.env.observation_space['pov'].high)
            transposed_shape = (non_transposed_shape[2],
                                non_transposed_shape[0],
                                non_transposed_shape[1])
            # Note: this assumes the Box is of the form where low/high values are vector but need to be scalar
            transposed_obs_space = gym.spaces.Box(low=0,
                                                  high=1,
                                                  shape=transposed_shape)
            self.observation_space = transposed_obs_space

        def inner_to_outer_observation_map(self, obs):
            # Minecraft returns shapes in NHWC by default, and with unnormalized pixel ranges
            return obs['pov'].transpose(0,3,1,2)/self.high

except ImportError:
    raise Warning("Realistic Benchmarks is not installed; as a result much Minecraft functionality will not work")

def wrap_env(env, wrappers):
    for wrapper in wrappers:
        env = wrapper(env)
    return env

def serialize_gym_space(space):
    """Convert Gym space to a format suitable for long-term pickle storage
    (i.e. for pickles that will be transferred between machines running
    different versions of Gym)."""
    if isinstance(space, gym.spaces.Box):
        if not np.isscalar(space.low):
            # This is to fix a weird issue where Box requires the shape to not be a vector if the
            # low and high values also are
            space_shape = None
        else:
            space_shape = space.shape
        return _KwargSerialisableObject(gym.spaces.Box, {
            'low': space.low,
            'high': space.high,
            'shape': space_shape,
            'dtype': space.dtype,
        })
    elif isinstance(space, gym.spaces.Discrete):
        return _KwargSerialisableObject(gym.spaces.Discrete, {
            'n': space.n,
        })
    else:
        raise NotImplementedError(f"don't know how to pickle space '{space}'")


def _inflate_object(object_type, kwargs):
    return object_type(**kwargs)


class _KwargSerialisableObject:
    """A container object that unpickles to `object_type(**kwargs)`. Used to
    pickle Gym observation spaces for storage in webdataset archives.
    Reconstructing the space from scratch during unpickling makes it more
    likely that it will be compatible with the current version of Gym, since
    the Gym internal API changes regularly."""
    def __init__(self, object_type, kwargs):
        self.object_type = object_type
        self.kwargs = kwargs

    def __reduce__(self):
        return (_inflate_object, (self.object_type, self.kwargs))
