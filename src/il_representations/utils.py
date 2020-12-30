"""Miscellaneous tools that don't fit elsewhere."""
import collections
import hashlib
import jsonpickle
import math
import os
import pdb
import pickle
import sys

from PIL import Image
from imitation.augment.color import ColorSpace
from skvideo.io import FFmpegWriter
import torch as th
import torchvision.utils as vutils


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


def hash_config(config_dict):
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    sorted_dict = collections.OrderedDict({k:config_dict[k] for k in sorted(config_dict.keys())})
    encoded = jsonpickle.encode(sorted_dict).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


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
    def __init__(self, out_path, color_space, fps=25, config=None):
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

    def add_tensor(self, tensor):
        """Add a tensor of shape [..., C, H, W] representing the frame stacks
        for a single time step. Call this repeatedly for each time step you
        want to add."""
        if self.writer is None:
            raise RuntimeError("Cannot run add_tensor() again after closing!")
        grid = image_tensor_to_rgb_grid(tensor, self.color_space)
        # convert to (H, W, 3) numpy array
        np_grid = grid.numpy().transpose((1, 2, 0))
        byte_grid = (np_grid * 255).round().astype('uint8')
        self.writer.writeFrame(byte_grid)

    def __enter__(self):
        assert self.writer is not None, \
            "cannot __enter__ this again once it is closed"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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
