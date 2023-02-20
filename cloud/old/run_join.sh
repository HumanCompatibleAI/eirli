#!/usr/bin/env bash

env_names=("miner")
bench_name="procgen"
repl_configs=("repl_id" "repl_fd" "repl_vae" "repl_tcpc8" "repl_simclr")

for env_name in "${env_names[@]}"; do
    for repl_config in "${repl_configs[@]}"; do
        CUDA_VISIBLE_DEVICES=3 python src/il_representations/scripts/joint_training.py with \
        ${repl_config} \
        env_cfg.benchmark_name=${bench_name}\
        env_cfg.task_name=${env_name} \
        model_save_interval=25000 \
        repl.batch_save_interval=25000 \
        n_batches=1000000 &
    done
done
