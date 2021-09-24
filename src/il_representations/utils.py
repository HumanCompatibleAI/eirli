"""Miscellaneous tools that don't fit elsewhere."""
import collections
from collections.abc import Iterable, Mapping, Sequence
import contextlib
import functools
import hashlib
import json
import math
import os
import pdb
import pickle
import re
import sys
import time
from typing import Dict, List

from PIL import Image
from imitation.augment.color import ColorSpace
from imitation.augment.convenience import StandardAugmentations
import numpy as np
from skvideo.io import FFmpegWriter
import torch as th
from torchsummary import summary
import torchvision.utils as vutils
import webdataset as wds

WEBDATASET_SAVE_KEY = "obs.pyd"

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def convert_to_simple_webdataset(dataset, file_out_name, file_out_path):
    full_wds_url = os.path.join(file_out_path, f"{file_out_name}.tar")
    dirname = os.path.dirname(full_wds_url)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with wds.TarWriter(full_wds_url) as sink:
        for index in range(len(dataset)):
            print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)
            sink.write({
                    "__key__": "sample%06d" % index,
                    WEBDATASET_SAVE_KEY: dataset[index]
                })
    return full_wds_url


def _unpickle_data(sample):
    result = pickle.loads(sample[WEBDATASET_SAVE_KEY])
    return result


def load_simple_webdataset(wds_url):
    return wds.Dataset(wds_url).map(_unpickle_data)


def recursively_sort(element):
    """Ensures that any dicts in nested dict/list object
    collection are converted to OrderedDicts"""
    if isinstance(element, collections.Mapping):
        sorted_dict = collections.OrderedDict()
        for k in sorted(element.keys()):
            sorted_dict[k] = recursively_sort(element[k])
        return sorted_dict
    elif isinstance(element, Sequence) and not isinstance(element, str):
        return [recursively_sort(inner_el) for inner_el in element]
    else:
        return str(element)


def hash_configs(merged_config):
    """MD5 hash of a dictionary."""
    sorted_dict = recursively_sort(merged_config)
    # Needs to be double-encoded because result of jsonpickle is Unicode
    encoded = json.dumps(sorted_dict).encode('utf-8')
    digest = hashlib.md5(encoded).hexdigest()
    return digest


def freeze_params(module):
    """Modifies Torch module in-place to convert all its parameters to buffers,
    and give them require_grad=False. This is a slightly hacky way of
    "freezing" the module."""

    # We maintain this stack so that we can traverse the module tree
    # depth-first. We'll terminate once we've traversed all modules.
    module_stack = [module]

    while module_stack:
        # get next module from end of the stack
        next_module = module_stack.pop()

        # sanity check to ensure we only have named params
        param_list = list(next_module.parameters(recurse=False))
        named_param_list = list(next_module.named_parameters(recurse=False))
        assert len(param_list) == len(named_param_list), \
            f"cannot handle module '{next_module}' with unnamed parameters"

        # now remove each param (delattr) and replace it with a buffer
        # (register_buffer)
        for param_name, param_var in named_param_list:
            param_tensor = param_var.data.clone().detach()
            assert not param_tensor.requires_grad
            delattr(next_module, param_name)
            next_module.register_buffer(param_name, param_tensor)

        # do the same for child modules
        module_stack.extend(next_module.children())

    # sanity check to make sure we have no params on the root module
    remaining_params = list(module.parameters())
    assert len(remaining_params) == 0, \
        f"module '{module}' has params remaining: {remaining_params}"


NUM_CHANS = {
    ColorSpace.RGB: 3,
    ColorSpace.GRAY: 1,
}


def image_tensor_to_rgb_grid(image_tensor, color_space):
    """Converts an image tensor to a montage of images.

    Args:
        image_tensor (Tensor): tensor containing (possibly stacked) frames.
            Tensor values should be in [0, 1], and tensor shape should be […,
            n_frames*chans_per_frame, H, W]; the last three dimensions are
            essential, but the trailing dimensions do not matter.
         color_space (ColorSpace): color space for the images. This is needed
            to infer how many frames are in each frame stack.

    Returns:
         grid (Tensor): a [3*H*W] RGB image containing all the stacked frames
            passed in as input, arranged in a (roughly square) grid.
    """
    assert isinstance(image_tensor, th.Tensor)
    image_tensor = image_tensor.detach().cpu()

    # make sure shape is correct & data is in the right range
    assert image_tensor.ndim >= 3, image_tensor.shape
    assert th.all((-0.01 <= image_tensor) & (image_tensor <= 1.01)), \
        f"this only takes intensity values in [0,1], but range is " \
        f"[{image_tensor.min()}, {image_tensor.max()}]"
    n_chans = NUM_CHANS[color_space]
    assert (image_tensor.shape[-3] % n_chans) == 0, \
        f"expected image to be stack of frames with {n_chans} channels " \
        f"each, but image tensor is of shape {image_tensor.shape}"

    # Reshape into [N,3,H,W] or [N,1,H,W], depending on how many channels there
    # are per frame.
    nchw_tensor = image_tensor.reshape((-1, n_chans) + image_tensor.shape[-2:])

    if n_chans == 1:
        # tile grayscale to RGB
        nchw_tensor = th.cat((nchw_tensor, ) * 3, dim=-3)

    # make sure it really is RGB
    assert nchw_tensor.ndim == 4 and nchw_tensor.shape[1] == 3

    # clamp to right value range
    clamp_tensor = th.clamp(nchw_tensor, 0, 1.)

    # number of rows scales with sqrt(num frames)
    # (this keeps image roughly square)
    nrow = max(1, int(math.sqrt(clamp_tensor.shape[0])))

    # now convert to an image grid
    grid = vutils.make_grid(clamp_tensor,
                            nrow=nrow,
                            normalize=False,
                            scale_each=False,
                            range=(0, 1))
    assert grid.ndim == 3 and grid.shape[0] == 3, grid.shape

    return grid


def save_rgb_tensor(rgb_tensor, file_path):
    """Save an RGB Torch tensor to a file. It is assumed that rgb_tensor is of
    shape [3,H,W] (channels-first), and that it has values in [0,1]."""
    assert isinstance(rgb_tensor, th.Tensor)
    assert rgb_tensor.ndim == 3 and rgb_tensor.shape[0] == 3, rgb_tensor.shape
    detached = rgb_tensor.detach()
    rgb_tensor_255 = (detached.clamp(0, 1) * 255).round()
    chans_last = rgb_tensor_255.permute((1, 2, 0))
    np_array = chans_last.detach().byte().cpu().numpy()
    pil_image = Image.fromarray(np_array)
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    pil_image.save(file_path)


class TensorFrameWriter:
    """Writes N*(F*C)*H*W tensor frames to a video file."""
    def __init__(self, out_path, color_space, fps=25, config=None, adjust_axis=True, make_grid=True):
        self.out_path = out_path
        self.color_space = color_space
        ffmpeg_out_config = {
            '-r': str(fps),
            '-vcodec': 'libx264',
            '-pix_fmt': 'yuv420p',
        }
        if config is not None:
            ffmpeg_out_config.update(config)

        self.writer = FFmpegWriter(out_path, outputdict=ffmpeg_out_config)
        self.adjust_axis = adjust_axis
        self.make_grid = make_grid

    def add_tensor(self, tensor):
        """Add a tensor of shape [..., C, H, W] representing the frame stacks
        for a single time step. Call this repeatedly for each time step you
        want to add."""
        if self.writer is None:
            raise RuntimeError("Cannot run add_tensor() again after closing!")
        grid = tensor
        if self.make_grid:
            grid = image_tensor_to_rgb_grid(tensor, self.color_space)
        np_grid = grid.numpy()
        if self.adjust_axis:
            # convert to (H, W, 3) numpy array
            np_grid = np_grid.transpose((1, 2, 0))
        byte_grid = (np_grid * 255).round().astype('uint8')
        self.writer.writeFrame(byte_grid)

    def __enter__(self):
        assert self.writer is not None, \
            "cannot __enter__ this again once it is closed"
        return self

    def __exit__(self, *args):
        # this fn receives args exc_type, exc_val, exc_tb (but all are unused)
        self.close()

    def close(self):
        if self.writer is None:
            return
        self.writer.close()
        self.writer = None

    def __del__(self):
        self.close()


class SaneDict(dict):
    # used in SacredUnpickler
    pass


class SaneList(list):
    # used in SacredUnpickler
    pass


class SacredUnpickler(pickle.Unpickler):
    """Unpickler that replaces Sacred's ReadOnlyDict/ReadOnlyList with
    dict/list."""
    overrides = {
        # for some reason we need to replace dict with a custom class, or
        # else we get an AttributeError complaining that 'dict' has no
        # attribute '__dict__' (I don't know why this hapens)
        ('sacred.config.custom_containers', 'ReadOnlyDict'): SaneDict,
        ('sacred.config.custom_containers', 'ReadOnlyList'): SaneList,
    }

    def find_class(self, module, name):
        key = (module, name)
        if key in self.overrides:
            return self.overrides[key]
        return super().find_class(module, name)


def load_sacred_pickle(fp, **kwargs):
    """Unpickle an object that may contain Sacred ReadOnlyDict and ReadOnlyList
    objects. It will convert those objects to plain dicts/lists."""
    return SacredUnpickler(fp, **kwargs).load()


def save_repl_batches(*, dest_dir, detached_debug_tensors, batches_trained,
                      color_space, save_video=False):
    """Save batches of data produced by the innards of a
    `RepresentationLearner`. Tries to save in the easiest-to-open format (e.g.
    image files for things that look like images, pickles for 1D tensors,
    etc.)."""
    os.makedirs(dest_dir, exist_ok=True)

    # now loop over items and save using appropriate format
    for save_name, save_value in (detached_debug_tensors.items()):
        if isinstance(save_value, th.distributions.Distribution):
            # take sample instead of mean so that we can see noise
            save_value = save_value.sample()
        if th.is_tensor(save_value):
            save_value = save_value.detach().cpu()

        # heuristic to check if this is an image
        probably_an_image = th.is_tensor(save_value) \
            and save_value.ndim == 4 \
            and save_value.shape[-2] == save_value.shape[-1]
        clean_save_name = re.sub(r'[^\w_ \-]', '-', save_name)
        save_prefix = f'{clean_save_name}_{batches_trained:06d}'
        save_path_no_suffix = os.path.join(dest_dir, save_prefix)

        if probably_an_image:
            # probably an image
            save_path = save_path_no_suffix + '.png'
            # save as image
            save_image = save_value.float().clamp(0, 1)

            # Save decoded contexts as videos
            if save_video:
                video_out_path = save_path_no_suffix + '.mp4'
                video_writer = TensorFrameWriter(
                    video_out_path, color_space=color_space)
                for image in save_image:
                    image = image_tensor_to_rgb_grid(image, color_space)
                    video_writer.add_tensor(image)
                video_writer.close()

            as_rgb = image_tensor_to_rgb_grid(save_image, color_space)
            save_rgb_tensor(as_rgb, save_path)
        else:
            # probably not an image
            save_path = save_path_no_suffix + '.pt'
            # will save with Torch's generic serialisation code
            th.save(save_value, save_path)


class RepLSaveExampleBatchesCallback:
    """Save (possibly image-based) contexts, targets, and encoded/decoded
    contexts/targets."""
    def __init__(self,
                 save_interval_batches,
                 dest_dir,
                 color_space,
                 save_video=False):
        self.save_interval_batches = save_interval_batches
        self.dest_dir = dest_dir
        self.last_save = None
        self.color_space = color_space
        self.save_video = save_video

    def __call__(self, repl_locals):
        batches_trained = repl_locals['batches_trained']

        # check whether we should save anything
        should_save = self.last_save is None \
            or self.last_save + self.save_interval_batches <= batches_trained
        if not should_save:
            return
        self.last_save = batches_trained

        save_repl_batches(
            dest_dir=self.dest_dir,
            detached_debug_tensors=repl_locals['detached_debug_tensors'],
            batches_trained=batches_trained,
            color_space=self.color_space,
            save_video=self.save_video)


class SigmoidRescale(th.nn.Module):
    """Rescales input to be in [min_val, max_val]; useful for pixel decoder."""
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.val_range = max_val - min_val

    def forward(self, x):
        return th.sigmoid(x) * self.val_range + self.min_val


def up(p):
    """Return the path *above* whatever object the path `p` points to.
    Examples:

        up("/foo/bar") == "/foo"
        up("/foo/bar/") == "/foo
        up(up(up("foo/bar"))) == ".."
    """
    return os.path.normpath(os.path.join(p, ".."))


class IdentityModule(th.nn.Module):
    """Parameter-free Torch module which passes through input unchanged."""
    def forward(self, x):
        return x


def augmenter_from_spec(spec, color_space):
    """Construct an image augmentation module from an augmenter spec, expressed
    as either a string of comma-separated augmenter names, or a dict of kwargs
    for StandardAugmentations."""
    if isinstance(spec, str):
        return StandardAugmentations.from_string_spec(spec, color_space)
    elif isinstance(spec, dict):
        return StandardAugmentations(**spec, stack_color_space=color_space)
    elif spec is None:
        # FIXME(sam): really this should return None, and I should fix callers
        # to reflect that. Right now the repL code does not handle the case
        # where this returns None.
        return IdentityModule()
    raise TypeError(
        f"don't know how to handle spec of type '{type(spec)}': '{spec}'")


def pyhash_mutable_types(mutable):
    """A Python hash() implementation for nested mutable types."""
    # note that we do hash(tuple(sorted(...))) to deal with nondeterministic
    # iteration order
    try:
        return hash(mutable)
    except TypeError:
        if isinstance(mutable, Mapping):
            return hash(tuple(sorted(
                pyhash_mutable_types(t) for t in mutable.items())))
        elif isinstance(mutable, Iterable):
            return hash(tuple(sorted((pyhash_mutable_types(t) for t in mutable))))
        raise


class WrappedConfig:
    """Dumb wrapper class used in pretrain_n_adapt to hide things from skopt.
    It's in a separate module so that we can pickle it when pretrain_n_adapt is
    __main__."""
    def __init__(self, config_dict):
        self.config_dict = config_dict

    def __eq__(self, other):
        if not isinstance(other, WrappedConfig):
            return NotImplemented
        return other.config_dict == self.config_dict

    def __hash__(self):
        return pyhash_mutable_types(self.config_dict)

    def __repr__(self):
        """Shorter repr in case this object gets printed."""
        return f'WrappedConfig@{hex(id(self))}'


def print_policy_info(policy, obs_space):
    """Print model information of the policy"""
    print(policy)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    policy = policy.to(device)
    obs_shape = (obs_space.shape[0], obs_space.shape[1], obs_space.shape[2])
    summary(policy, obs_shape)


@functools.total_ordering
class SacredProofTuple(Sequence):
    """Pseudo-tuple that can be passed through Sacred without
    being silently cast to a list."""
    def __init__(self, *elems):
        self._elems = elems

    def __eq__(self, other):
        if not isinstance(other, SacredProofTuple):
            return NotImplemented
        return other._elems == self._elems

    def __hash__(self, other):
        return hash(self._elems)

    def __lt__(self, other):
        if not isinstance(other, SacredProofTuple):
            return NotImplemented
        return other._elems < self._elems

    # __len__ and __getitem__ are required by Sequence (__iter__(),
    # __reversed__(), count(), and index() are provided automatically)
    def __len__(self):
        return len(self._elems)

    def __getitem__(self, idx):
        return self._elems[idx]

    def __repr__(self):
        return 'NotATuple' + repr(self._elems)


def weight_grad_norms(params, *, norm_type=2):
    """Calculate the gradient norm and the weight norm of the policy network.

    Adapted from `BC._calculate_policy_norms` in imitation.

    Args:
        params: list of Torch parameters to compute norm of
        norm_type: order of the norm (1, 2, etc.).

    Returns: Tuple of `(gradient_norm, weight_norm)`, where:
        - gradient_norm is the norm of the gradient of the policy network
          (stored in each parameter's .grad attribute)
        - weight_norm is the norm of the weights of the policy network
    """
    norm_type = float(norm_type)

    gradient_parameters = [p for p in params if p.grad is not None]
    stacked_gradient_norms = th.stack(
        [th.norm(p.grad.detach(), norm_type) for p in gradient_parameters])
    stacked_weight_norms = th.stack(
        [th.norm(p.detach(), norm_type) for p in params])

    gradient_norm = th.norm(stacked_gradient_norms, norm_type).cpu().numpy()
    weight_norm = th.norm(stacked_weight_norms, norm_type).cpu().numpy()

    return gradient_norm, weight_norm


class Timers:
    """Wrapper for a collection of timers.

    Usage: call `.start("some_name_for_op")` before doing the operation you
    want to time, then `.stop("some_name_for_op")` immediately afterward. Doing
    `.dump_stats()` will compute statistics for recorded times under all names
    (e.g. "some_name_for_op" and any other names you use)."""
    def __init__(self):
        self.last_start: Dict[str, float] = {}
        self.records: Dict[str, List[float]] = {}

    @contextlib.contextmanager
    def time(self, timer_name):
        """Time 'with' block body."""
        try:
            self.start(timer_name)
            yield self
        finally:
            self.stop(timer_name)

    def start(self, name: str) -> None:
        """Start a timer."""
        if name in self.last_start:
            raise ValueError(
                f"Tried to do .start({name!r}), but {name!r} is still "
                "running; should .stop() it first.")

        self.last_start[name] = time.monotonic()

    def stop(self, name: str, *, check_running=True) -> None:
        """Stop a timer."""
        if name not in self.last_start:
            if check_running:
                raise ValueError(
                    f"Tried to .stop({name!r}), but {name!r} is not running")
            else:
                return

        elapsed = time.monotonic() - self.last_start[name]

        # clear running timer
        del self.last_start[name]

        self.records.setdefault(name, []).append(elapsed)

    def dump_stats(self, *, check_running=True, reset=True) \
            -> Dict[str, Dict[str, float]]:
        """Clear all timer records and return a nested dictionary of stats.
        Keys at top level of dict are timer names, keys at second level are
        stat names (min/mean/max/std), and values at second level are just
        floats."""
        if len(self.last_start) > 0 and check_running:
            raise ValueError(
                "Tried to .dump_stats() with timers still running: "
                f"{list(self.last_start.keys())}")

        rv = collections.OrderedDict()
        for name, values in sorted(self.records.items()):
            rv[name] = collections.OrderedDict()
            for stat, stat_fn in [('min', np.min), ('max', np.max),
                                  ('mean', np.mean), ('std', np.std)]:
                rv[name][stat] = stat_fn(values)

        if reset:
            # clear saved times, but not running timers
            self.records = {}

        return rv

    def reset(self):
        """Clear all running timers and all saved times."""
        self.records = {}
        self.last_start = {}


class EmptyIteratorException(Exception):
    """Raised when a function is incorrectly passed an empty iterator."""


def repeat_chain_non_empty(iterable):
    """Equivalent to itertools.chain.from_iterable(itertools.repeat(iterator)),
    but checks that iterator is non-empty."""
    while True:
        yielded_item = False
        for item in iterable:
            yield item
            yielded_item = True
        if not yielded_item:
            raise EmptyIteratorException(f"iterable {iterable} was empty")


def get_policy_nupdate(policy_path):
    match_result = re.match(r".*policy_(?P<n_update>\d+)_batches.pt",
                            policy_path)
    assert match_result is not None, r'policy_path does not fit pattern' \
                                     r'.*policy_(?P<n_update>\d+)_batches.pt'
    return match_result.group('n_update')
