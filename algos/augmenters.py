import enum
from torchvision import transforms
from .utils import gaussian_blur
import torch
from abc import ABC, abstractmethod
import PIL
"""
These are pretty basic: when constructed, they take in a list of augmentations, and
either augment just the context, or both the context and the target, depending on the algorithm.
"""

DEFAULT_AUGMENTATIONS = (
    transforms.ToPILImage(),
    transforms.RandomAffine(  # could add rotation etc., but would be costly
        degrees=0, translate=(0.05, 0.05), resample=PIL.Image.NEAREST),
    transforms.Lambda(gaussian_blur),
    transforms.ToTensor(),
)


class ColorSpace(str, enum.Enum):
    RGB = 'RGB'
    GRAY = 'GRAY'


class Augmenter(ABC):
    def __init__(self, color_space=ColorSpace.GRAY, augmentations=DEFAULT_AUGMENTATIONS):
        self.augment_op = transforms.Compose(augmentations)
        self.color_space = color_space

    def _apply(self, frames):
        """Apply augmentations to an N*C*H*W stack of frames."""
        # FIXME(sam): this should really be done concurrently on CPU, or shoved
        # over to the GPU. Realistically the augmentations we have right now
        # are probably fastest to do in Torch, on the GPU.
        frames_out = []
        for frame in frames:
            # split the frame into separate sub-frames
            if self.color_space == ColorSpace.RGB:
                # frames are RGB, so frame stack must be of size 3
                assert (frames.size(1) % 3) == 0
                sub_frames = torch.split(frame, 3)
            elif self.color_space == ColorSpace.GRAY:
                sub_frames = torch.split(frame, 1)
            else:
                raise NotImplementedError("no support for color space",
                                          self.color_space)
            stack_frame = torch.cat(
                [self.augment_op(sub_frame) for sub_frame in sub_frames],
                dim=0)
            frames_out.append(stack_frame)
        return torch.stack(frames_out, dim=0)

    @abstractmethod
    def __call__(self, contexts, targets):
        pass


class NoAugmentation(Augmenter):
    def __call__(self, contexts, targets):
        return contexts, targets


class AugmentContextAndTarget(Augmenter):
    def __call__(self, contexts, targets):
        return self._apply(contexts), self._apply(targets)


class AugmentContextOnly(Augmenter):
    def __call__(self, contexts, targets):
        return self._apply(contexts), targets