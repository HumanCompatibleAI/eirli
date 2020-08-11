from torchvision import transforms
from .utils import gaussian_blur
import numpy as np
from abc import ABC, abstractmethod
from math import ceil
"""
These are pretty basic: when constructed, they take in a list of augmentations, and 
either augment just the context, or both the context and the target, depending on the algorithm. 
"""





DEFAULT_AUGMENTATIONS = (transforms.ToPILImage(),
                         transforms.Pad(4),
                         transforms.RandomCrop(84),
                         transforms.Lambda(gaussian_blur),)


class Augmenter(ABC):
    def __init__(self, augmentations=DEFAULT_AUGMENTATIONS, batch_augmentation_size=2048):
        # TODO at some point check if I need to convert this to list or if it can stay a tuple
        self.augment_op = transforms.Compose(list(augmentations))
        self.batch_augmentation_size = batch_augmentation_size

    @abstractmethod
    def __call__(self, dataset):
        pass

    def dataset_to_aug_batches(self, dataset):
        contexts, targets = np.stack([el['context'] for el in dataset]), np.stack([el['target'] for el in dataset])
        num_splits = ceil(len(dataset)/self.batch_augmentation_size)
        # array_split allows splits to not be of equal size
        contexts, targets = np.array_split(contexts, num_splits), np.array_split(contexts, num_splits)
        return contexts, targets

    def to_dataset(self, contexts, targets, dataset):
        for i in range(contexts.shape[0]):
            dataset[i]['context'] = contexts[i]
            dataset[i]['target'] = targets[i]
        return dataset

    def augment(self, data_batch):
        return [np.array(self.augment_op(el)) for el in data_batch]

class AugmentContextAndTarget(Augmenter):
    def __call__(self, dataset):
        contexts, targets = self.dataset_to_aug_batches(dataset)
        augmented_context = np.concatenate([self.augment(batch) for batch in contexts])
        augmented_targets = np.concatenate([self.augment(batch) for batch in targets])
        return self.to_dataset(augmented_context, augmented_targets, dataset)


class AugmentContextOnly(Augmenter):
    def __call__(self, dataset):
        contexts, targets = self.dataset_to_aug_batches(dataset)
        augmented_context = np.concatenate([self.augment(batch) for batch in contexts])
        return self.to_dataset(augmented_context, targets, dataset)
