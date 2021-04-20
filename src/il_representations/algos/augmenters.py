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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Augmenter(ABC):
    def __init__(self, augmenter_spec, color_space, augment_func=None):
        self.augment_func = augment_func
        if augment_func:
            self.augment_op = augment_func
        else:
            augment_op = StandardAugmentations.from_string_spec(
                augmenter_spec, color_space)
            self.augment_op = augment_op

    @abstractmethod
    def __call__(self, contexts, targets):
        pass

    def augment_extra_context(self, extra_contexts):
        return extra_contexts


class NoAugmentation(Augmenter):
    def __call__(self, contexts, targets):
        return contexts, targets


class AugmentContextAndTarget(Augmenter):
    def __call__(self, contexts, targets):
        if self.augment_func:
            context_ret, target_ret = [], []
            for context, target in zip(contexts, targets):
                context_ret.append(self.augment_op(context))
                target_ret.append(self.augment_op(target))
            return torch.stack(context_ret, dim=0).to(device), \
                   torch.stack(target_ret, dim=0).to(device)
        return self.augment_op(contexts), self.augment_op(targets)


class AugmentContextOnly(Augmenter):
    def __call__(self, contexts, targets):
        return self.augment_op(contexts), targets


class AugmentContextAndExtraContext(AugmentContextOnly):
    def augment_extra_context(self, extra_contexts):
        return self.augment_op(extra_contexts)
