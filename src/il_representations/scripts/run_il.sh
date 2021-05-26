#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python src/il_representations/scripts/pretrain_n_adapt.py with \
 cfg_repl_none \
 cfg_il_bc_nofreeze \
 cfg_run_few_trajs_2m_updates \
 cfg_bench_micro_sweep_dm_control \
 exp_ident=dmc-few-trajs-noaugs \
 il_train.bc.augs=None \
 tune_run_kwargs.num_samples=5 \
 tune_run_kwargs.resources_per_trial.gpu=0.1
