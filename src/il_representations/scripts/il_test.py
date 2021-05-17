#!/usr/bin/env python3
"""Run an IL algorithm in some selected domain."""
import logging

import imitation.util.logger as imitation_logger
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch as th

from il_representations.algos.utils import set_global_seeds
from il_representations.envs.config import (env_cfg_ingredient,
                                            venv_opts_ingredient)
from il_representations.pol_eval import do_final_eval

sacred.SETTINGS['CAPTURE_MODE'] = 'no'  # workaround for sacred issue#740
il_test_ex = Experiment('il_test',
                        ingredients=[
                            # We need env_cfg_ingredient to know which
                            # environment to test on, and venv_opts_ingredient
                            # to construct a vecenv (or vecenvs) for that
                            # environment.
                            env_cfg_ingredient, venv_opts_ingredient
                        ])


@il_test_ex.config
def default_config():
    # exp_ident is an arbitrary string. Set it to a meaningful value to help
    # you identify runs in viskit.
    exp_ident = None
    torch_num_threads = 1
    policy_path = None
    n_rollouts = 100
    device_name = 'auto'
    # use deterministic policy?
    deterministic_policy = False
    # run_id is written into the produced DataFrame to indicate what model is
    # being tested
    run_id = 'test'
    # if True, then we'll add a video named <video_file_name> as a Sacred
    # artifact
    write_video = False
    video_file_name = "rollouts.mp4"

    _ = locals()
    del _


@il_test_ex.main
def run(policy_path, env_cfg, venv_opts, seed, n_rollouts, device_name, run_id,
        write_video, video_file_name, torch_num_threads, deterministic_policy):
    set_global_seeds(seed)
    # FIXME(sam): this is not idiomatic way to do logging (as in il_train.py)
    logging.basicConfig(level=logging.INFO)
    log_dir = il_test_ex.observers[0].dir
    imitation_logger.configure(log_dir, ["stdout", "csv", "tensorboard"])
    if torch_num_threads is not None:
        th.set_num_threads(torch_num_threads)

    if policy_path is None:
        raise ValueError(
            "must pass a string-valued policy_path to this command")

    return do_final_eval(
        policy_path=policy_path,
        out_dir=log_dir,
        write_video=write_video,
        video_file_name=video_file_name,
        n_rollouts=n_rollouts,
        seed=seed,
        run_id=run_id,
        device=device_name,
        deterministic_policy=deterministic_policy)


if __name__ == '__main__':
    il_test_ex.observers.append(FileStorageObserver('runs/il_test_runs'))
    il_test_ex.run_commandline()
