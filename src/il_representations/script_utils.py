import collections
import copy
import enum
import logging
from typing import TypeVar
import urllib

import numpy as np

from il_representations.utils import NUM_CHANS


class StagesToRun(str, enum.Enum):
    """These enum flags are used to control whether pretrain_n_adapt tunes RepL, or
    IL, or both."""
    REPL_AND_IL = "REPL_AND_IL"
    REPL_ONLY = "REPL_ONLY"
    IL_ONLY = "IL_ONLY"


class ReuseRepl(str, enum.Enum):
    """These enum flags are used to control whether
    pretrain_n_adapt reuses repl or not """
    YES = "YES"
    NO = "NO"
    IF_AVAILABLE = "IF_AVAILABLE"


def update(d, *updates):
    """Recursive dictionary update (pure)."""
    d = copy.copy(d)  # to make this pure
    for u in updates:
        for k, v in u.items():
            if isinstance(d.get(k), collections.Mapping):
                # recursive insert into a mapping
                d[k] = update(d[k], v)
            else:
                # if the existing value is not a mapping, then overwrite it
                d[k] = v
    return d


T = TypeVar('T')


def sacred_copy(o: T) -> T:
    """Perform a deep copy on nested dictionaries and lists.

    If `d` is an instance of dict or list, copies `d` to a dict or list
    where the values are recursively copied using `sacred_copy`. Otherwise, `d`
    is copied using `copy.deepcopy`. Note this intentionally loses subclasses.
    This is useful if e.g. `d` is a Sacred read-only dict. However, it can be
    undesirable if e.g. `d` is an OrderedDict.

    Args:
        o (object): if dict, copy recursively; otherwise, use `copy.deepcopy`.

    Returns: A deep copy of d."""
    if isinstance(o, dict):
        return {k: sacred_copy(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [sacred_copy(v) for v in o]
    else:
        return copy.deepcopy(o)


def detect_ec2():
    """Auto-detect if we are running on EC2."""
    try:
        EC2_ID_URL = 'http://169.254.169.254/latest/dynamic/instance-identity/document'
        with urllib.request.urlopen(EC2_ID_URL, timeout=3) as f:
            response = f.read().decode()
            if 'availabilityZone' in response:
                return True
            else:
                raise ValueError(f"Received unexpected response from '{EC2_ID_URL}'")
    except urllib.error.URLError:
        return False


def get_n_chans() -> int:
    # FIXME(sam): this is here to avoid a recursive import. The issue is that
    # il_representations.envs wants to use update() from this file. Ideally we
    # should move update() into a separate file, or move get_n_chans into
    # auto.py.
    from il_representations.envs import auto
    return NUM_CHANS[auto.load_color_space()]


def simplify_stacks(obs_vec: np.ndarray, keep_only_latest: bool) -> np.ndarray:
    """Turn an image frame stack into a single image. If
    `keep_only_latest=True`, then it uses only the most recent image in each
    stack. If `keep_only_test=False`, then it concatenates the images in the
    stack horizontally."""
    # simple sanity checks to make sure frames are N*(C*H)*W
    assert obs_vec.ndim == 4, f"obs_vec.shape={obs_vec.shape}, so ndim != 4"
    if obs_vec.shape[-1] != obs_vec.shape[-2]:
        logging.warning(
            f"obs_vec.shape={obs_vec.shape} does not look N(C*F)HW, "
            "since H!=W")
    n_chans = get_n_chans()
    stack_len = obs_vec.shape[1] // n_chans
    assert stack_len * n_chans == obs_vec.shape[1], \
        f"obs_vec.shape={obs_vec.shape} should be N(C*F)HW, " \
        f"but first dim is not divisible by n_chans={n_chans}"
    new_shape = obs_vec.shape[:1] + (stack_len, n_chans) + obs_vec.shape[2:]
    destacked = np.reshape(obs_vec, new_shape)
    # put stack dimension first
    transposed = np.transpose(destacked, (1, 0, 2, 3, 4))
    if keep_only_latest:
        final_obs_vec = transposed[-1]
    else:
        final_obs_vec = np.concatenate(transposed, axis=3)
    # now it's actually N*C*H*W', where W' has absorbed all the stacked frames
    # from before
    return final_obs_vec


def trajectory_iter(dataset):
    """Yields one trajectory at a time from a webdataset."""
    traj = []
    for frame in dataset:
        traj.append(frame)
        if frame['dones']:
            yield traj
            traj = []
