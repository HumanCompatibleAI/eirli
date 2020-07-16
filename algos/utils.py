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
import cv2
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.cmd_util import make_atari_env
now = datetime.now()


def add_noise(state, noise_var):
    noise = np.random.normal(0, noise_var, state.shape[0])
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
    x = np.array(list(range(len(arr[0])))) * gap
    plt.plot(x, arr[0], marker='', color='steelblue', linewidth=0.8, alpha=0.9, label='Reward')
    plt.plot(x, arr[1], marker='', color='Green', linewidth=0.8, alpha=0.9, label='Lossx40')

    plt.legend(loc='lower right')
    plt.title(f"{env_id}", fontsize=14)
    plt.xlabel("episode", fontsize=12)
    plt.ylabel("score", fontsize=12)

    plt.savefig(os.path.abspath('../') + f'/output/[{time_now(now)}]{env_id}.png')
    plt.close(fig)


def plot_single(arr, label, msg):
    fig = plt.figure()
    x = np.array(list(range(len(arr))))
    plt.plot(x, arr, marker='', color='steelblue', linewidth=0.8, alpha=0.9, label=label)
    plt.legend(loc='upper right')
    plt.xlabel("episode", fontsize=12)

    plt.savefig(os.path.abspath('../') + f'/output/[{time_now(now)}]{msg}.png')
    plt.close(fig)


def save_model(model, env_id, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, f'[{time_now(now)}]{env_id}.ckpt'))


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
    def __init__(self, optimizer, warmup_epoch, T_max, eta_min=0, last_epoch=-1):
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


def make_env(env_id, rank=0, seed=0, log_dir=None, wrapper_class=None, env_kwargs=None):
    """
    Helper function to multiprocess training
    and log the progress.

    :param env_id: (str)
    :param rank: (int)
    :param seed: (int)
    :param log_dir: (str)
    :param wrapper: (type) a subclass of gym.Wrapper to wrap the original
                    env with
    :param env_kwargs: (Dict[str, Any]) Optional keyword argument to pass to the env constructor
    """
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    if env_kwargs is None:
        env_kwargs = {}

    def _init():
        set_global_seeds(seed + rank)
        env = gym.make(env_id, **env_kwargs)

        # Dict observation space is currently not supported.
        # https://github.com/hill-a/stable-baselines/issues/321
        # We allow a Gym env wrapper (a subclass of gym.Wrapper)
        if wrapper_class:
            env = wrapper_class(env)

        env.seed(seed + rank)
        log_file = os.path.join(log_dir, str(rank)) if log_dir is not None else None
        env = Monitor(env, log_file)
        return env

    return _init


def create_test_env(env_id, n_envs=1, is_atari=False,
                    stats_path=None, seed=0,
                    log_dir='', should_render=True, hyperparams=None, env_kwargs=None):
    """
    Create environment for testing a trained agent

    :param env_id: (str)
    :param n_envs: (int) number of processes
    :param is_atari: (bool)
    :param stats_path: (str) path to folder containing saved running averaged
    :param seed: (int) Seed for random number generator
    :param log_dir: (str) Where to log rewards
    :param should_render: (bool) For Pybullet env, display the GUI
    :param env_wrapper: (type) A subclass of gym.Wrapper to wrap the original
                        env with
    :param hyperparams: (dict) Additional hyperparams (ex: n_stack)
    :param env_kwargs: (Dict[str, Any]) Optional keyword argument to pass to the env constructor
    :return: (gym.Env)
    """

    if hyperparams is None:
        hyperparams = {}

    if env_kwargs is None:
        env_kwargs = {}

    if is_atari:
        print("Using Atari wrapper")
        env = make_atari_env(env_id, n_envs=n_envs, seed=seed)
        # Frame-stacking with 4 frames
        env = VecFrameStack(env, n_stack=4)
    elif n_envs > 1:
        # start_method = 'spawn' for thread safe
        env = SubprocVecEnv([make_env(env_id, i, seed, log_dir, wrapper_class=None, env_kwargs=env_kwargs) for i in range(n_envs)])
    # Pybullet envs does not follow gym.render() interface
    elif "Bullet" in env_id:
        # HACK: force SubprocVecEnv for Bullet env
        env = SubprocVecEnv([make_env(env_id, 0, seed, log_dir, wrapper_class=None, env_kwargs=env_kwargs)])
    else:
        env = DummyVecEnv([make_env(env_id, 0, seed, log_dir, wrapper_class=None, env_kwargs=env_kwargs)])

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if stats_path is not None:
        if hyperparams.get('normalize'):
            print("Loading running average")
            print("with params: {}".format(hyperparams['normalize_kwargs']))
            env = VecNormalize(env, training=False, **hyperparams['normalize_kwargs'])

            if os.path.exists(os.path.join(stats_path, 'vecnormalize.pkl')):
                env = VecNormalize.load(os.path.join(stats_path, 'vecnormalize.pkl'), env)
                # Deactivate training and reward normalization
                env.training = False
                env.norm_reward = False
            else:
                # Legacy:
                env.load_running_average(stats_path)

        n_stack = hyperparams.get('frame_stack', 0)
        if n_stack > 0:
            print("Stacking {} frames".format(n_stack))
            env = VecFrameStack(env, n_stack)
    return env