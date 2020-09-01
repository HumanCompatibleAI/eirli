import enum
from torchvision import transforms
from imitation.augment.color import ColorSpace  # noqa: F401
from imitation.augment.convenience import StandardAugmentations
from il_representations.algos.utils import gaussian_blur
import torch
from abc import ABC, abstractmethod
import PIL
"""
These are pretty basic: when constructed, they take in a list of augmentations, and
either augment just the context, or both the context and the target, depending on the algorithm.
"""


class Augmenter(ABC):
    def __init__(self, augmenter_spec, color_space):
        augment_op = StandardAugmentations.from_string_spec(
            augmenter_spec, color_space)
        self.augment_op = augment_op

    @abstractmethod
    def __call__(self, contexts, targets):
        pass


class NoAugmentation(Augmenter):
    def __call__(self, contexts, targets):
        return contexts, targets


class AugmentContextAndTarget(Augmenter):
    def __call__(self, contexts, targets):
        return self.augment_op(contexts), self.augment_op(targets)


class AugmentContextOnly(Augmenter):
    def __call__(self, contexts, targets):
        return self.augment_op(contexts), targets
