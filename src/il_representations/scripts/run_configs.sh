#!/bin/bash

exp_id="dmc"
# n_batches=2000000

for repl in \
    "cfg_repl_inv_dyn" \
    ; do
    CUDA_VISIBLE_DEVICES=0 python src/il_representations/scripts/pretrain_n_adapt.py with \
        cfg_force_use_repl \
        ${repl} \
        cfg_il_bc_nofreeze \
        cfg_bench_micro_sweep_dm_control \
        cfg_run_few_trajs_2m_updates \
        exp_ident=${exp_id}-${repl} \
        tune_run_kwargs.num_samples=5 \
        tune_run_kwargs.resources_per_trial.gpu=0.1 &
done
