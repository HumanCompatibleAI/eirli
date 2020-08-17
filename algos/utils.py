import torch
import numpy as np
import random
import gym
import math
from torch.optim.lr_scheduler import _LRScheduler
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from numbers import Number
import cv2


def independent_multivariate_normal(loc, scale):
    batch_dim = loc.shape[0]
    if isinstance(scale, Number):
        # If the scale was passed in as a scalar, convert it to the same shape as loc
        scale = torch.ones(loc.shape)*scale

    # Turn each <feature-dim>-length vector in the batch into a diagonal matrix, because we want an
    # independent multivariate normal
    merged_covariance_matrix = torch.stack([torch.diag(scale[i]) for i in range(batch_dim)])
    return torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=merged_covariance_matrix)


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
    def __init__(self, optimizer, warmup_epoch, total_epochs, eta_min=0.0, last_epoch=-1):
        assert warmup_epoch >= 0
        self.eta_min = eta_min
        self.warmup_epoch = warmup_epoch
        self.cosine_epochs = total_epochs - warmup_epoch
        super(LinearWarmupCosine, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Linear scaling if we are in the warmup stage
        use_linear_scaling = self.warmup_epoch > 0 and self.last_epoch < self.warmup_epoch
        result = []
        for base_lr in self.base_lrs:
            delta = base_lr - self.eta_min
            if use_linear_scaling:
                fraction = (self.last_epoch + 1) / self.warmup_epoch
            else:
                rescaled_epoch = self.last_epoch - self.warmup_epoch
                fraction = 0.5 * (1 + math.cos(math.pi * rescaled_epoch / self.cosine_epochs))
            result.append(self.eta_min + fraction * delta)
        return result


def set_global_seeds(seed):
    """
    set the seed for python random, tensorflow, numpy and gym spaces

    :param seed: (int) the seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # prng was removed in latest gym version
    if hasattr(gym.spaces, 'prng'):
        gym.spaces.prng.seed(seed)


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
