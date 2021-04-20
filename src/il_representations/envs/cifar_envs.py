import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

from imitation.augment.color import ColorSpace
from gym.spaces import Discrete, Box


def load_dataset_cifar():
    """Return a dataset dict"""
    dataset = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True,
                                           transform=transforms.ToTensor())

    obs, acts = [], []
    for i in range(len(dataset)):
        img, label = dataset[i]
        obs.append(img.cpu().numpy())
        acts.append(label)

    obs = np.stack([o for o in obs], axis=0)
    acts = np.array(acts)

    data_dict = {
        'obs': obs,
        'acts': acts,
        'dones': np.array([False] * len(dataset)),
    }

    return data_dict


class MockGymEnv(object):
    """A mock Gym env for a supervised learning dataset pretending to be an RL
    task. Action space is set to Discrete(1), observation space corresponds to
    the original supervised learning task.
    """
    def __init__(self):
        self.observation_space = Box(low=0.0, high=1.0, shape=(3, 32, 32), dtype=np.float32)
        self.action_space = Discrete(1)
        self.color_space = ColorSpace.RGB

    def seed(self, seed):
        pass

    def close(self):
        pass