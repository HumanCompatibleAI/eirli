#!/bin/bash

exp_id="dmc"
n_batches=1000000

for repl in \
    "cfg_repl_none" \
    ; do
    CUDA_VISIBLE_DEVICES=2 python src/il_representations/scripts/pretrain_n_adapt.py with \
        ${repl} \
        cfg_il_bc_nofreeze \
        exp_ident=${exp_id}-${repl} \
        env_cfg.benchmark_name=dm_control \
        env_cfg.task_name=finger-spin \
        il_train.bc.augs=translate \
        il_train.bc.n_batches=$n_batches \
        tune_run_kwargs.num_samples=5 \
        tune_run_kwargs.resources_per_trial.gpu=0.1
        # il_train.bc.n_batches=$n_batches &
done
