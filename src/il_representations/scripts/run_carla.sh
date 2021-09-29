#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=1 xvfb-run -a python src/il_representations/scripts/pretrain_n_adapt.py with \
#  cfg_repl_none \
#  cfg_il_bc_nofreeze \
#  cfg_use_mlp_bc_200 \
#  cfg_use_carla \
#  cfg_base_5seed_pt1gpu

CUDA_VISIBLE_DEVICES=2 xvfb-run -a python src/il_representations/scripts/pretrain_n_adapt.py with \
 cfg_repl_none \
 cfg_il_bc_nofreeze \
 cfg_base_5seed_pt1gpu \
 il_train.bc.n_batches=300000 \
 tune_run_kwargs.num_samples=5 \
 tune_run_kwargs.resources_per_trial.gpu=0.3 \
 tune_run_kwargs.max_failures=0 \
 env_cfg.benchmark_name=carla \
 env_cfg.task_name=carla-lane-v0

#  cfg_use_mlp_bc_200 \
#  cfg_repl_none \
#  repl.batches_per_epoch=1 \
#  cfg_use_carla \
#  cfg_force_use_repl \
#  cfg_repl_simclr \