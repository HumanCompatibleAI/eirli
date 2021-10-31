import copy
import enum
import logging
import os
import weakref
from typing import TypeVar
import urllib

import numpy as np
from ray.tune.schedulers import FIFOScheduler

from il_representations.envs.auto import get_n_chans


class StagesToRun(str, enum.Enum):
    """These enum flags are used to control whether pretrain_n_adapt tunes RepL, or
    IL, or both."""
    REPL_AND_IL = "REPL_AND_IL"
    REPL_ONLY = "REPL_ONLY"
    IL_ONLY = "IL_ONLY"

    REPL_AND_RL = "REPL_AND_RL"
    RL_ONLY = "RL_ONLY"


class ReuseRepl(str, enum.Enum):
    """These enum flags are used to control whether
    pretrain_n_adapt reuses repl or not """
    YES = "YES"
    NO = "NO"
    IF_AVAILABLE = "IF_AVAILABLE"


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


class CheckpointFIFOScheduler(FIFOScheduler):
    """Variant of FIFOScheduler that periodically saves the given search
    algorithm. Useful for, e.g., SkOptSearch, where it is helpful to be able to
    re-instantiate the search object later on."""

    # FIXME: this is a stupid hack, inherited from another project. There
    # should be a better way of saving skopt internals as part of Ray Tune.
    # Perhaps defining a custom trainable would do the trick?
    def __init__(self, search_alg):
        self.search_alg = weakref.proxy(search_alg)

    def on_trial_complete(self, trial_runner, trial, result):
        rv = super().on_trial_complete(trial_runner, trial, result)
        # references to _local_checkpoint_dir and _session_dir are a bit hacky
        checkpoint_path = os.path.join(
            trial_runner._local_checkpoint_dir,
            f'search-alg-{trial_runner._session_str}.pkl')
        self.search_alg.save(checkpoint_path + '.tmp')
        os.rename(checkpoint_path + '.tmp', checkpoint_path)
        return rv


def relative_symlink(src, dst):
    link_dir_abs, link_fn = os.path.split(os.path.abspath(dst))
    if not link_fn:
        raise ValueError(f"path dst='{dst}' has empty basename")
    # absolute path to src, and path relative to link_dir
    src_abspath = os.path.abspath(src)
    src_relpath = os.path.relpath(src_abspath, start=link_dir_abs)

    os.makedirs(link_dir_abs, exist_ok=True)
    link_dir_fd = os.open(link_dir_abs, os.O_RDONLY)
    try:
        # both src_relpath and link_fn are relative to link_dir, which is
        # represented by the file descriptor link_dir_fd
        os.symlink(src_relpath, link_fn, dir_fd=link_dir_fd)
    finally:
        os.close(link_dir_fd)
