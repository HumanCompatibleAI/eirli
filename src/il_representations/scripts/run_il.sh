#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 xvfb-run -a python src/il_representations/scripts/pretrain_n_adapt.py with \
 cfg_force_use_repl \
 cfg_repl_jigsaw \
 cfg_il_bc_nofreeze \
 tune_run_kwargs.num_samples=1 \
 tune_run_kwargs.resources_per_trial.gpu=0.5 \
 tune_run_kwargs.max_failures=0 \
 exp_ident=magical-small \
 repl.batches_per_epoch=1 \
 il_train.bc.n_batches=4 \
 il_train.bc.batch_size=512 \
 env_cfg.benchmark_name=dm_control \
 env_cfg.task_name=finger-spin

 # cfg_repl_none \
