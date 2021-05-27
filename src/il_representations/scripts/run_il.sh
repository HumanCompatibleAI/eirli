#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 xvfb-run -a python src/il_representations/scripts/pretrain_n_adapt.py with \
 cfg_base_3seed_1cpu_pt2gpu_2envs \
 cfg_repl_none \
 cfg_il_bc_nofreeze \
 cfg_run_few_trajs_2m_updates \
 cfg_bench_micro_sweep_dm_control \
 exp_ident=dmc-few-trajs-augs \
 tune_run_kwargs.num_samples=5 \
 tune_run_kwargs.resources_per_trial.gpu=0.3
