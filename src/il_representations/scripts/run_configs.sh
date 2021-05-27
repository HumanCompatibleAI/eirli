#!/bin/bash

exp_id="procgen-repl"
n_batches=2000000

for repl in \
    "cfg_repl_vae" \
    ; do
    CUDA_VISIBLE_DEVICES=1,3 xvfb-run -a python src/il_representations/scripts/pretrain_n_adapt.py with \
        cfg_force_use_repl \
        ${repl} \
        cfg_il_bc_nofreeze \
        cfg_bench_micro_sweep_procgen \
        exp_ident=${exp_id}-${repl} \
        tune_run_kwargs.num_samples=5 \
        tune_run_kwargs.resources_per_trial.gpu=0.2 \
        il_train.bc.n_batches=$n_batches &
done
