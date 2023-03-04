"""Augmenters take in a list of augmentations, and either augment just the
context, or both the context and the target, depending on the algorithm."""
from abc import ABC, abstractmethod

from imitation.augment.color import ColorSpace  # noqa: F401

from il_representations.utils import augmenter_from_spec


class Augmenter(ABC):
    """Abstract base class for all augmenters."""
    def __init__(self, augmenter_spec=None, color_space=None, device=None):
        if augmenter_spec is not None:
            assert device is not None
            self.augment_op = augmenter_from_spec(augmenter_spec, color_space,
                                                  device)

    @abstractmethod
    def __call__(self, contexts, targets):
        pass

    def augment_extra_context(self, extra_contexts):
        return extra_contexts


class NoAugmentation(Augmenter):
    """Noop augmenter; does not augment anything."""
    def __call__(self, contexts, targets):
        return contexts, targets


class AugmentContextAndTarget(Augmenter):
    """Applies independent augmentations to contexts and targets, but not
    extra_contexts."""
    def __call__(self, contexts, targets):
        return self.augment_op(contexts), self.augment_op(targets)


class AugmentContextOnly(Augmenter):
    """Applies augmentations to only contexts; targets and extra_contexts are
    left as-is."""
    def __call__(self, contexts, targets):
        return self.augment_op(contexts), targets


class AugmentContextAndExtraContext(AugmentContextOnly):
    """Independently augments both contexts and extra_contexts, but not
    targets."""
    def augment_extra_context(self, extra_contexts):
        return self.augment_op(extra_contexts)
