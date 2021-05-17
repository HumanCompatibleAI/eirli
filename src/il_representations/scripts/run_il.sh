#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 xvfb-run -a python src/il_representations/scripts/pretrain_n_adapt.py with \
 cfg_repl_none \
 cfg_il_bc_nofreeze \
 cfg_bench_micro_sweep_dm_control \
 cfg_run_few_trajs_long_dm_control \
 il_train.bc.n_trajs=10 \
 exp_ident=dmc_long_ntrajs_10 \
 tune_run_kwargs.num_samples=1 \
 tune_run_kwargs.resources_per_trial.gpu=0.3 \
 # il_train.bc.n_batches=400000 \
 # il_train.bc.nominal_length=10000 \
 # env_cfg.benchmark_name=dm_control \
 # env_cfg.task_name=finger-spin

