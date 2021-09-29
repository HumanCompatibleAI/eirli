#!/bin/bash

# Last minute GAIL experiments on Procgen

set -e

# WARNING: 5 demos here!
base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=5"
           "cfg_data_il_5demos" "env_cfg.benchmark_name=procgen")
# 8 runs per GPU by default
gpu_default=0.125
declare -A gpu_overrides=(
    # 5 runs per GPU for TCPC8/SimCLR
    ["cfg_repl_tcpc8"]="0.2"
    ["cfg_repl_simclr"]="0.2"
)
gpu_config() {
    # figures out GPU configuration string based on repL config, if any
    repl_config="$1"
    override="${gpu_overrides[$repl_config]}"
    if [ -z "$override" ]; then
        override="$gpu_default"
    fi
    echo "tune_run_kwargs.resources_per_trial.gpu=$override"
}
declare -a pg_dataset_configs=("cfg_data_repl_demos")

# we are evaluating on only a subset of cases for the rebuttal; we can do the
# rest later (which will require 4x as much compute)
# declare -a repl_configs=("icml_inv_dyn" "icml_dynamics" "cfg_repl_tcpc8" "cfg_repl_simclr" "icml_vae")
declare -a repl_configs=("icml_inv_dyn" "icml_dynamics" "cfg_repl_tcpc8" "cfg_repl_simclr" "icml_vae")
declare -a procgen_envs=("coinrun" "miner")

run_expt() {
    xvfb-run -a python -m il_representations.scripts.pretrain_adapt run with \
           -- "${base_cfgs[@]}" "$@" &
}


for procgen_env in "${procgen_envs[@]}"; do
    for repl_config in "${repl_configs[@]}"; do
        for pg_dataset_config in "${pg_dataset_configs[@]}"; do
            echo -e "\n *** TRAINING $repl_config ON $procgen_env WITH DATASET $pg_dataset_config *** \n "
            run_expt cfg_repl_5k_il cfg_repl_augs \
                "$repl_config" "$(gpu_config "$repl_config")" \
                "$pg_dataset_config" "cfg_il_gail_procgen_1m_nofreeze" \
                "env_cfg.task_name=$procgen_env" \
                exp_ident="neurips_repl_gail_${repl_config}_${pg_dataset_config}"
        done
    done
    for use_augs in augs noaugs; do
        run_expt cfg_repl_none \
            "$(gpu_config "no_repl")" "cfg_il_gail_procgen_1m_nofreeze" \
            "gail_disc_${use_augs}" "env_cfg.task_name=$procgen_env" \
            exp_ident="neurips_control_gail_${use_augs}"
    done
done
wait
