#!/usr/bin/env python3
# test harness to benchmark different Gaussian blur implementations
import os
import re
import tempfile
import time
import urllib.request

import numpy as np
from PIL import Image
import torch as th
from kornia.filters.gaussian import GaussianBlur2d
from torch import nn
from kornia.filters.filter import filter2d_separable
from kornia.filters.kernels import get_gaussian_kernel1d

USE_GPU = True
IMAGE_BATCH_SIZE = (64, 3, 64, 64)
WARMUP_ITERS = 15
TIMING_ITERS = 1000
EXAMPLE_IMAGE_URL = \
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Duisburg" \
    "%2C_Landschaftspark_Duisburg-Nord_--_2020_--_7824-6.jpg/640px-Duisburg" \
    "%2C_Landschaftspark_Duisburg-Nord_--_2020_--_7824-6.jpg"


def get_example_image():
    """Load example image as Torch array (CHW, float, [0,1])."""
    with tempfile.NamedTemporaryFile(mode="rb") as fp:
        urllib.request.urlretrieve(EXAMPLE_IMAGE_URL, fp.name)
        image = Image.open(fp)
        as_np = np.asarray(image)
    # convert to CHW float image
    return (th.from_numpy(as_np).float() / 255).permute((2, 0, 1))


def save_image(image_tensor, out_path):
    byte_image = (image_tensor.permute((1, 2, 0)) * 255).byte()
    as_numpy = byte_image.numpy()
    image = Image.fromarray(as_numpy, 'RGB')
    out_path = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    image.save(out_path)


class OriginalGaussianBlur(nn.Module):
    """Original Gaussian blur implementation."""

    def __init__(self, kernel_hw=5, sigma=1, p=0.5):
        super().__init__()
        assert kernel_hw >= 1 and (kernel_hw % 2) == 1
        sigma = float(sigma)
        assert sigma > 0
        self.blur_op = GaussianBlur2d((kernel_hw, kernel_hw), (sigma, sigma))
        self.p = p

    def forward(self, images):
        batch_size = images.size(0)
        should_blur = th.rand(batch_size) < self.p
        blur_elems = []
        for batch_idx in range(batch_size):
            if should_blur[batch_idx]:
                rotated_image = self.blur_op(images[batch_idx:batch_idx + 1])
                blur_elems.append(rotated_image)
            else:
                blur_elems.append(images[batch_idx:batch_idx + 1])
        blur_images = th.cat(blur_elems, dim=0)
        return blur_images


class _JITGaussianBlur(nn.Module):
    """Like original implementation, but JITs everything."""

    def __init__(self, kernel_hw: int = 5, sigma: float = 1.0, p: float = 0.5):
        super().__init__()
        self.blur_op = GaussianBlur2d((kernel_hw, kernel_hw), (sigma, sigma))
        self.p = p

    def forward(self, images):
        batch_size = images.size(0)
        should_blur = th.rand(batch_size) < self.p
        blur_elems = []
        for batch_idx in range(batch_size):
            if should_blur[batch_idx]:
                rotated_image = self.blur_op(images[batch_idx:batch_idx + 1])
                blur_elems.append(rotated_image)
            else:
                blur_elems.append(images[batch_idx:batch_idx + 1])
        blur_images = th.cat(blur_elems, dim=0)
        return blur_images


class JITGaussianBlur:
    def __init__(self, **kwargs):
        self.gaussian_blur = _JITGaussianBlur(**kwargs)
        self._jit_blur = None

    def to(self, *args, **kwargs):
        self.gaussian_blur = self.gaussian_blur.to(*args, **kwargs)
        return self

    @property
    def jit_blur(self):
        if self._jit_blur is None:
            self._jit_blur = th.jit.script(self.gaussian_blur)
        return self._jit_blur

    def __call__(self, images):
        return self.jit_blur(images)


class _BatchedJITGaussianBlur(nn.Module):
    """Like original implementation, but JITs everything."""

    def __init__(self, kernel_hw: int = 5, sigma: float = 1.0, p: float = 0.5):
        super().__init__()
        self.blur_op = GaussianBlur2d((kernel_hw, kernel_hw), (sigma, sigma))
        self.p = p

    def forward(self, images):
        batch_size = images.size(0)
        should_blur = th.rand(batch_size) < self.p
        out_images = images.clone()
        out_images[should_blur] = self.blur_op(images[should_blur])
        return out_images


class BatchedJITGaussianBlur:
    def __init__(self, **kwargs):
        self.gaussian_blur = _JITGaussianBlur(**kwargs)
        self._jit_blur = None

    def to(self, *args, **kwargs):
        self.gaussian_blur = self.gaussian_blur.to(*args, **kwargs)
        return self

    @property
    def jit_blur(self):
        if self._jit_blur is None:
            self._jit_blur = th.jit.script(self.gaussian_blur)
        return self._jit_blur

    def __call__(self, images):
        return self.jit_blur(images)


@th.jit.script
def _blur(images: th.Tensor, kernel: th.Tensor, p: float):
    batch_size = images.size(0)
    blur_mask = th.rand(batch_size) < p
    out_images = images.clone()
    extracted = images[blur_mask]
    blurred = filter2d_separable(extracted, kernel, kernel,
                                 "reflect")
    out_images[blur_mask] = blurred
    return out_images


class FastGaussianBlur(nn.Module):
    """Gaussian blur that is faster than Kornia default. Main time saving is
    not recomputing blur kernel on each forward pass."""
    def __init__(self, kernel_hw: int = 5, sigma: float = 1.0, p: float = 0.5):
        super().__init__()
        assert isinstance(kernel_hw, int) and kernel_hw > 0, kernel_hw
        assert isinstance(sigma, float) and sigma > 0, sigma
        assert isinstance(p, float) and 0 <= p <= 1, p
        # kernel must be of size [1, kernel_hw] for filter2d_separable
        self.kernel: th.Tensor = get_gaussian_kernel1d(kernel_hw, sigma)[None]
        assert self.kernel.shape == (1, kernel_hw), self.kernel.shape
        self.p = p

    def forward(self, images):
        return _blur(images, self.kernel, self.p)


def main():
    blur_params = dict(kernel_hw=5, sigma=1.0, p=0.5)
    impls = [
        ("Original", OriginalGaussianBlur),
        ("JIT-ed original", JITGaussianBlur),
        ("Batched+JIT-ed original", BatchedJITGaussianBlur),
        ("Custom", FastGaussianBlur),
    ]
    name_width = max(len(name) for name, _ in impls)
    device = th.device('cuda' if USE_GPU else 'cpu')
    fake_batch = th.randn(*IMAGE_BATCH_SIZE).to(device)
    example_image = get_example_image()
    example_image_batch = th.tile(example_image[None], (16, 1, 1, 1))
    for name, impl in impls:
        fn = impl(**blur_params)
        if hasattr(fn, 'to'):
            fn = fn.to(device)

        # warmup/debug iterations, with assertions
        for _ in range(WARMUP_ITERS):
            new_batch = fn(fake_batch)
            assert new_batch.shape == fake_batch.shape

        # save an example batch
        out_fn = "results-" + re.sub(r"[^a-z0-9]", "-", name.lower()) \
            + ".png"
        out_examples = fn(example_image_batch)
        out_examples_stacked = th.cat(tuple(out_examples), dim=2)
        save_image(out_examples_stacked, out_fn)

        # now actually benchmark
        start_time = time.time()
        results = []
        for _ in range(TIMING_ITERS):
            results.append(fn(fake_batch).sum())
        # sum().item() pushes results to CPU, thwarting lazy eval
        sum(results).item()
        elapsed = time.time() - start_time

        # display results
        ms_per_iter = elapsed / TIMING_ITERS * 1000
        print(f"Time per iteration for {name:<{name_width}}: "
              f"{ms_per_iter:8.3f}ms")


if __name__ == '__main__':
    main()
