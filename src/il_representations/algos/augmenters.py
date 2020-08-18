import enum
from torchvision import transforms
from imitation.augment.color import ColorSpace  # noqa: F401
from imitation.augment.convenience import KorniaAugmentations
from il_representations.algos.utils import gaussian_blur
import torch
from abc import ABC, abstractmethod
import PIL
"""
These are pretty basic: when constructed, they take in a list of augmentations, and
either augment just the context, or both the context and the target, depending on the algorithm.
"""


class RepLearnAugmenter(ABC):
    def __init__(self, augment_op):
        assert isinstance(augment_op, KorniaAugmentations)
        self.augment_op = augment_op

    @abstractmethod
    def __call__(self, contexts, targets):
        pass


class NoAugmentation(RepLearnAugmenter):
    def __call__(self, contexts, targets):
        return contexts, targets


class AugmentContextAndTarget(RepLearnAugmenter):
    def __call__(self, contexts, targets):
        return self.augment_op(contexts), self.augment_op(targets)


class AugmentContextOnly(RepLearnAugmenter):
    def __call__(self, contexts, targets):
        return self.augment_op(contexts), targets
