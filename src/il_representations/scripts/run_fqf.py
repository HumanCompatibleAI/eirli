#!/usr/bin/env python3
"""Demo script showing how to run FQF."""
from contextlib import ExitStack, closing
import faulthandler
import logging
import os
# readline import is black magic to stop PDB from segfaulting; do not remove it
import readline  # noqa: F401
import signal

import imitation.util.logger as im_log
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from stable_baselines3.common.utils import get_device
import torch
import torch as th

from il_representations.algos.utils import set_global_seeds
import il_representations.envs.auto as auto_env
from il_representations.envs.config import (env_cfg_ingredient,
                                            venv_opts_ingredient)
from il_representations.rl.fqf import FQFAgent

sacred.SETTINGS['CAPTURE_MODE'] = 'no'  # workaround for sacred issue#740
train_ex = Experiment('train',
                      ingredients=[
                          env_cfg_ingredient,
                          venv_opts_ingredient,
                      ])


@train_ex.config
def default_config():
    # identifier for use in viskit & other analysis scripts
    exp_ident = None

    # stop Torch taking up all cores needlessly
    torch_num_threads = 1

    # will default to GPU if available, otherwise CPU
    device = "auto"

    _ = locals()
    del _


@train_ex.main
def train(seed, torch_num_threads, device, exp_ident, _config):
    faulthandler.register(signal.SIGUSR1)
    set_global_seeds(seed)
    # python built-in logging
    logging.basicConfig(level=logging.INFO)
    # `imitation` logging
    log_dir = os.path.abspath(train_ex.observers[0].dir)
    im_log.configure(log_dir, ["stdout", "csv", "tensorboard"])
    if torch_num_threads is not None:
        th.set_num_threads(torch_num_threads)
    device = get_device(device)

    with ExitStack() as exit_stack:
        logging.info("Setting up vecenvs")
        venv = auto_env.load_vec_env(n_envs=1, venv_parallel=False)
        exit_stack.push(closing(venv))
        test_venv = auto_env.load_vec_env()
        exit_stack.push(closing(test_venv))

        logging.info("Constructing agent")
        agent = FQFAgent(venv, test_venv)

        logging.info("Running agent")
        agent.run(im_log.sb_logger)

        logging.info("Done!")


def add_fso():
    train_ex.observers.append(FileStorageObserver('runs/fqf_runs'))


if __name__ == '__main__':
    add_fso()
    train_ex.run_commandline()
