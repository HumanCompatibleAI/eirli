#!/bin/bash

# Tuning BC on HalfCheetah

set -e

# note that this requires Ray to be running on port 42000
base_cfgs=("cfg_base_skopt_4cpu_pt3gpu_no_retry"
           "tune_run_kwargs.resources_per_trial.cpu=1"
           "tune_run_kwargs.resources_per_trial.gpu=0.2"
           "ray_init_kwargs.address=localhost:42000")

python -m il_representations.scripts.pretrain_n_adapt with \
    "${base_cfgs[@]}" env_cfg.benchmark_name=dm_control \
    env_cfg.task_name=cheetah-run cfg_repl_none \
    bc_tune exp_ident=tune_bc_hc_2021_03_01
