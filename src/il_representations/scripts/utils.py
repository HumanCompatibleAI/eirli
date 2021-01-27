import collections
import copy
import enum
import urllib
import torch
from torchsummary import summary
from functools import partialmethod

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


def sacred_copy(o):
    """Perform a deep copy on nested dictionaries and lists.

    If `d` is an instance of dict or list, copies `d` to a dict or list
    where the values are recursively copied using `sacred_copy`. Otherwise, `d`
    is copied using `copy.deepcopy`. Note this intentionally loses subclasses.
    This is useful if e.g. `d` is a Sacred read-only dict. However, it can be
    undesirable if e.g. `d` is an OrderedDict.

    :param o: (object) if dict, copy recursively; otherwise, use `copy.deepcopy`.
    :return A deep copy of d."""
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


def print_policy_info(policy, obs_space):
    """Print model information of the policy"""
    print(policy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    summary(policy, (obs_space.shape[0], obs_space.shape[1], obs_space.shape[2]))


# copied from https://stackoverflow.com/a/38911383
def partial_repl_class(cls, new_class_name, *args, **kwargs):
    class MC(type):
        # metaclass nonsense from here: https://stackoverflow.com/a/55053439
        def __repr__(self):
            return new_class_name

    class PartialedRepLearner(cls, metaclass=MC):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return PartialedRepLearner
