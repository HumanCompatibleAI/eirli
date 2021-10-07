#!/bin/bash

exp_id="gail"
gpu_number=3
# total_timesteps=10
benchmark_name="procgen"
task_name="coinrun"
repl_configs=("cfg_repl_inv_dyn" "cfg_repl_temporal_cpc" "cfg_repl_vae" "cfg_repl_simclr")

for repl in "${repl_configs[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpu_number} python src/il_representations/scripts/pretrain_n_adapt.py with \
	${repl} \
        cfg_force_use_repl \
	cfg_il_gail_dmc_500k_nofreeze \
	exp_ident=${exp_id} \
        env_cfg.benchmark_name=${benchmark_name} \
        env_cfg.task_name=${task_name} \
        tune_run_kwargs.num_samples=5 \
        tune_run_kwargs.max_failures=0 \
        tune_run_kwargs.resources_per_trial.gpu=0.9 &
    ((gpu_number++))
done
