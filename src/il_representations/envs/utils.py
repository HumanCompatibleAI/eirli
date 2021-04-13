import gym


def serialize_gym_space(space):
    """Convert Gym space to a format suitable for long-term pickle storage
    (i.e. for pickles that will be transferred between machines running
    different versions of Gym)."""
    if isinstance(space, gym.spaces.Box):
        return _KwargSerialisableObject(gym.spaces.Box, {
            'low': space.low,
            'high': space.high,
            'shape': space.shape,
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
