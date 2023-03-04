#!/bin/bash

exp_id="vae"
n_batches=1000000
repl_configs=("cfg_repl_vae")
gpu_number=0

for repl in "${repl_configs[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpu_number} python src/il_representations/scripts/pretrain_n_adapt.py with \
        cfg_force_use_repl \
        ${repl} \
        cfg_il_bc_nofreeze \
        cfg_bench_micro_sweep_dm_control \
        exp_ident=${exp_id}-${repl} \
        tune_run_kwargs.num_samples=5 \
        tune_run_kwargs.max_failures=0 \
        tune_run_kwargs.resources_per_trial.gpu=0.25 \
        il_train.bc.n_batches=$n_batches &
    ((gpu_number++))
done
        # cfg_bench_full_sweep_procgen \
