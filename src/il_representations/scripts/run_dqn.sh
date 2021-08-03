#!/bin/bash

exp_id="test"
n_batches=10
n_traj=10
gpu_number=0
benchmark_name=procgen
task_name=coinrun

CUDA_VISIBLE_DEVICES=${gpu_number} python src/il_representations/scripts/pretrain_n_adapt.py with \
    cfg_rl_only \
    exp_ident=${exp_id} \
    env_cfg.benchmark_name=${benchmark_name} \
    env_cfg.task_name=${task_name} \
    dqn_train.n_batches=$n_batches \
    dqn_train.n_traj=$n_traj
