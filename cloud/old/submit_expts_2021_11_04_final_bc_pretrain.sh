#!/bin/bash

# What this file does:
# - repL/augs: Dynamics, InvDyn, SimCLR, TemporalCPC, VAE, BC+augs, BC-augs
# - Tasks: 3 MAGICAL tasks, 4 Proxgen tasks, 3 DMC tasks
# - #seeds: 5
# - repL batch size of 64 for all the non-contrastive methods, 192 for
#   contrastive ones (so that we finish on time).
# - RepL datasets: demos + random rollouts. MAGICAL is limited to 5 demos; all
#   other datasets (MAGICAL random rollouts, demos/rollouts for rest) are
#   unlimited.
# - # of steps: 20k for MAGICAL, 500,000 for the rest (should be doing more, but
#   # we have limited time)
# Note that this file does only BC, and only demos+random repL data

set -e

# WARNING: unlimited demos here! (for DMC/Procgen)
cluster_cfg_path="./gcp_cluster_sam_new_vis.yaml"
dmc_procgen_base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=5")
dmc_procgen_n_batches=500000  # in the interests of finishing on time
repl_configs=("icml_inv_dyn" "icml_dynamics" "cfg_repl_tcpc8_192"
              "cfg_repl_simclr_192" "icml_vae")
dataset_configs=("cfg_data_repl_demos_random")
# dmc_procgen_env_configs=("cfg_bench_micro_sweep_dm_control" "cfg_bench_procgen_cmfn")
dmc_procgen_env_configs=("cfg_bench_procgen_cmfn")
# WARNING: 5 demos here! (for MAGICAL)
magical_base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=5"
                   "cfg_data_il_5demos")
magical_envs=("MatchRegions-Demo-v0" "MoveToRegion-Demo-v0" "MoveToCorner-Demo-v0")
mg_dataset_configs=("cfg_data_repl_5demos_random")

gpu_default=0.11
declare -A gpu_overrides=(
    ["cfg_repl_tcpc8_192"]="0.16"
    ["cfg_repl_simclr_192"]="0.16"
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

submit_expt() {
    # submit experiment to cluster using given args
    ray submit --tmux "$cluster_cfg_path" \
        ./submit_pretrain_n_adapt.py -- "$@"
}

for env_config in "${dmc_procgen_env_configs[@]}"; do
    for repl_config in "${repl_configs[@]}"; do
        for dataset_config in "${dataset_configs[@]}"; do
            echo -e "\n *** TRAINING $repl_config ON $env_config WITH DATASET $dataset_config *** \n "
            submit_expt "${dmc_procgen_base_cfgs[@]}" cfg_repl_5k_il \
                cfg_repl_augs "$repl_config" "$(gpu_config "$repl_config")" \
                "$dataset_config" cfg_il_bc_500k_nofreeze \
                exp_ident="neurips_repl_bc_${repl_config}_${dataset_config}" \
                "$env_config" "il_train.bc.n_batches=$dmc_procgen_n_batches"
        done
    done
    for use_augs in augs noaugs; do
        submit_expt "${dmc_procgen_base_cfgs[@]}" cfg_repl_none \
            cfg_il_bc_500k_nofreeze "$(gpu_config "no_repl")" \
            "cfg_il_bc_${use_augs}" \
            exp_ident="neurips_control_bc_${use_augs}" "$env_config" \
            "il_train.bc.n_batches=$dmc_procgen_n_batches"
    done
    wait
done

# for magical_env in "${magical_envs[@]}"; do
#     for repl_config in "${repl_configs[@]}"; do
#         for mg_dataset_config in "${mg_dataset_configs[@]}"; do
#             echo -e "\n *** TRAINING $repl_config ON $magical_env WITH DATASET $mg_dataset_config *** \n "
#             submit_expt "${magical_base_cfgs[@]}" cfg_repl_5k_il cfg_repl_augs \
#                 cfg_use_magical "$repl_config" "$(gpu_config "$repl_config")" \
#                 "$mg_dataset_config" cfg_il_bc_20k_nofreeze \
#                 exp_ident="neurips_repl_bc_${repl_config}_${mg_dataset_config}" \
#                 "env_cfg.task_name=$magical_env"
#         done
#     done
#     for use_augs in augs noaugs; do
#         submit_expt "${magical_base_cfgs[@]}" cfg_repl_none cfg_use_magical \
#             cfg_il_bc_20k_nofreeze "$(gpu_config "no_repl")" \
#             "cfg_il_bc_${use_augs}" \
#             exp_ident="neurips_control_bc_${use_augs}" \
#             "env_cfg.task_name=$magical_env"
#     done
# done
# wait
