import gym
import numpy as np


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


def stack_obs_oldest_first(obs_arr, frame_stack, use_zeroed_frames=True):
    """Takes an array of shape [T, C, H, W] and stacks the entries to produce a
    new array of shape [T, C*frame_stack, H, W] with frames stacked along the
    channels axis. Frames at stacked oldest-first, and the first frame_stack-1
    frames have zeros instead of older frames (because older frames don't
    exist). This is meant to be compatible with VecFrameStack in SB3."""

    if use_zeroed_frames:  # Typically for dmc environments.
        frame_accumulator = np.repeat(np.zeros_like(obs_arr[0][None]),
                                      frame_stack,
                                      axis=0)
    else:  # Repeat the first frame. Typically used for Gym and Procgen.
        frame_accumulator = np.repeat([obs_arr[0]], frame_stack, axis=0)

    c, h, w = obs_arr.shape[1:]
    out_sequence = []
    for in_frame in obs_arr:
        frame_accumulator = np.concatenate(
            [frame_accumulator[1:], [in_frame]], axis=0)
        out_sequence.append(frame_accumulator.reshape(
            frame_stack * c, h, w))
    out_sequence = np.stack(out_sequence, axis=0)
    return out_sequence

