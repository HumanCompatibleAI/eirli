"""Miscellaneous utilities for our representation learning code."""
import matplotlib
matplotlib.use('agg')
import random
import gym
import itertools
import math
import os
import struct
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from PIL import Image
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import _LRScheduler


def independent_multivariate_normal(mean, stddev):
    # Create a normal distribution, which by default will assume all dimensions but one are a batch dimension
    dist = torch.distributions.Normal(mean, stddev, validate_args=True)
    # Wrap the distribution in an Independent wrapper, which reclassifies all but one dimension as part of the actual
    # sample shape, but keeps variances defined only on the diagonal elements of what would be the MultivariateNormal
    multivariate_mimicking_dist = torch.distributions.Independent(dist, len(mean.shape) - 1)
    return multivariate_mimicking_dist


def add_noise(state, noise_std_dev):
    noise = np.random.normal(0, noise_std_dev, state.shape[0])
    noise_state = state + noise
    return noise_state


def show_image(image):
    im = Image.fromarray(image, 'RGB')
    im.show()


def gaussian_blur(img):
    image = np.array(img)
    image_blur = cv2.GaussianBlur(image, (3, 3), 0)
    new_image = image_blur
    return new_image


def show_plt_image(img):
    if(img.shape[0]) == 4:
        img = img.permute(1, 2, 0)
    plt.imshow(img)
    plt.show()


# TODO: Have the calls to savefig below save to the log directory (or at least make the output directory in case it doesn't exist)
def plot(arr, env_id, gap=1):
    fig = plt.figure()
    x = np.arange(len(arr.shape[1])) * gap
    plt.plot(x, arr[0], marker='', color='steelblue', linewidth=0.8, alpha=0.9, label='Reward')
    plt.plot(x, arr[1], marker='', color='Green', linewidth=0.8, alpha=0.9, label='Lossx40')

    plt.legend(loc='lower right')
    plt.title(f"{env_id}", fontsize=14)
    plt.xlabel("episode", fontsize=12)
    plt.ylabel("score", fontsize=12)

    plt.savefig(os.path.abspath('../') + f'/output/[{time_now(datetime.now())}]{env_id}.png')
    plt.close(fig)


def plot_single(arr, label, msg):
    fig = plt.figure()
    x = np.array(list(range(len(arr))))
    plt.plot(x, arr, marker='', color='steelblue', linewidth=0.8, alpha=0.9, label=label)
    plt.legend(loc='upper right')
    plt.xlabel("episode", fontsize=12)

    plt.savefig(os.path.abspath('../') + f'/output/[{time_now(datetime.now())}]{msg}.png')
    plt.close(fig)


def save_model(model, env_id, save_path):
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f'[{time_now(datetime.now())}]{env_id}.pth'))


def time_now(n):
    date_time = n.strftime("%m-%d-%Y-%H-%M-%S")
    return date_time


class Logger:
    def __init__(self, log_dir):
        self.file = os.path.join(log_dir, f'train_log.txt')

    def log(self, msg):
        t = datetime.now()
        message = f"[{time_now(t)}] {msg}"
        print(message)
        f = open(self.file, "a+", buffering=1)
        f.write(message + '\n')
        f.close()


class LinearWarmupCosine(_LRScheduler):
    def __init__(self, optimizer, T_max, warmup_epoch=30, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epoch = warmup_epoch
        super(LinearWarmupCosine, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.warmup_epoch > 0:
            if self.last_epoch <= self.warmup_epoch:
                return [base_lr / self.warmup_epoch * self.last_epoch for base_lr in self.base_lrs]
        if ((self.last_epoch - self.warmup_epoch) - 1 - (self.T_max - self.warmup_epoch)) % (2 * (self.T_max - self.warmup_epoch)) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / (self.T_max - self.warmup_epoch))) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        else:
            return [(1 + math.cos(math.pi * (self.last_epoch - self.warmup_epoch) / (self.T_max - self.warmup_epoch))) /
                    (1 + math.cos(math.pi * ((self.last_epoch - self.warmup_epoch) - 1) / (self.T_max - self.warmup_epoch))) *
                    (group['lr'] - self.eta_min) + self.eta_min
                    for group in self.optimizer.param_groups]


def set_global_seeds(seed):
    """
    set the seed for python random, tensorflow, numpy and gym spaces

    :param seed: (int or None) the seed
    """
    if seed is None:
        # seed from os.urandom if no seed given
        seed, = struct.unpack('<I', os.urandom(4))

    # we use this rng to create a separate seed for each random stream
    rng = np.random.RandomState(seed)
    torch.manual_seed(rng.randint((1 << 31) - 1))
    np.random.seed(rng.randint((1 << 31) - 1))
    random.seed(rng.randint((1 << 31) - 1))
    # prng was removed in latest gym version
    if hasattr(gym.spaces, 'prng'):
        gym.spaces.prng.seed(rng.randint((1 << 31) - 1))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def generate_jigsaw_permutations(n_tiles=9, n_perms=1000):
    """Generate n_perms permutations from list(range(n_tiles)), and select
    permutations with large hamming distance. Modified from
    bbrattoli/JigsawPuzzlePytorch.

    Args:
        n_tiles (int): The number of tiles to split an image into. Default: 9.
        n_perms (int): The number of permutations needed. Default: 1000.
    """

    all_perms = np.array(list(itertools.permutations(list(range(n_tiles)),
                                                     n_tiles)))

    perms = []
    j = np.random.randint(len(all_perms))
    for i in range(n_perms):
        if i == 0:
            perms = np.array(all_perms[j]).reshape([1, -1])
        else:
            perms = np.concatenate([perms, all_perms[j].reshape([1, -1])], axis=0)

        all_perms = np.delete(all_perms, j, axis=0)

        # Calculate selected permutations' distance with all possible
        # permutations, and pick the one that are most different from
        # selected ones. This is to make sure the generated jigsaw puzzles are
        # sufficiently different from each other.
        D = cdist(perms, all_perms, metric='hamming').mean(axis=0).flatten()
        j = D.argmax()

    return perms
