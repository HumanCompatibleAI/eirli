#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python src/il_representations/scripts/pretrain_n_adapt.py with \
 cfg_base_3seed_1cpu_pt2gpu_2envs \
 cfg_repl_none \
 cfg_il_bc_nofreeze \
 exp_ident=procgen-augs \
 env_cfg.benchmark_name=procgen \
 env_cfg.task_name=ninja \
 il_train.bc.n_batches=2000000 \
 tune_run_kwargs.num_samples=5 \
 tune_run_kwargs.resources_per_trial.gpu=0.2 \
 # il_train.bc.augs=None 
