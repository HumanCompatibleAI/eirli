from torchvision import transforms
from .utils import gaussian_blur
import numpy as np
import torch
from abc import ABC, abstractmethod

"""
These are pretty basic: when constructed, they take in a list of augmentations, and 
either augment just the context, or both the context and the target, depending on the algorithm. 
"""

DEFAULT_AUGMENTATIONS = (transforms.ToPILImage(),
                         transforms.Pad(4),
                         transforms.RandomCrop(84),
                         transforms.Lambda(gaussian_blur),)
class Augmenter(ABC):
    def __init__(self, multi_logger, augmentations=DEFAULT_AUGMENTATIONS):
        # TODO at some point check if I need to convert this to list or if it can stay a tuple
        self.augment_op = transforms.Compose(list(augmentations))
        self.multi_logger = multi_logger

    @abstractmethod
    def __call__(self, contexts, targets):
        pass

class NoAugmentation(Augmenter):
    def __call__(self, contexts, targets):
        return contexts, targets

class AugmentContextAndTarget(Augmenter):
    def __call__(self, contexts, targets):
        return [np.array(self.augment_op(el)) for el in contexts], [np.array(self.augment_op(el)) for el in targets]


class AugmentContextOnly(Augmenter):
    def __call__(self, contexts, targets):
        return [np.array(self.augment_op(el)) for el in contexts], targets

