#!/bin/bash

exp_id="test"
gpu_number=0
benchmark_name=procgen
task_name=coinrun

CUDA_VISIBLE_DEVICES=${gpu_number} python src/il_representations/scripts/pretrain_n_adapt.py with \
    cfg_dqn_nofreeze \
    cfg_rl_only \
    exp_ident=${exp_id} \
    tune_run_kwargs.num_samples=5 \
    tune_run_kwargs.resources_per_trial.gpu=1 \
    env_cfg.benchmark_name=${benchmark_name} \
    env_cfg.task_name=${task_name}