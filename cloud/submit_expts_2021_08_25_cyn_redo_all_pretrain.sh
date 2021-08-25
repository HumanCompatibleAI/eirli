#!/bin/bash

# What this file does:
# - repL/augs: Dynamics, InvDyn, SimCLR, TemporalCPC, VAE, BC+augs, BC-augs
# - Tasks: 4 procgen tasks, 3 DMC tasks (via namedconfigs)
# - #seeds: 5
# - repL batch size of 64 for all the non-contrastive methods, 192 for
#   contrastive ones (so that we finish on time).
# - RepL datasets: demos + random rollouts. Uses however much data is available.
# - # of steps: 500,000 (should be doing more, but we have limited time)
# Does only BC, and only demos+random repL data

# TODO:
# - **DONE** Set n_batches appropriately
# - **In progress** Generate random rollouts for procgen. Push to GCP.
# - **DONE** Create cfg_repl_{tcpc8,simclr}_192 configs
# - Shellcheck, final proof-read.
# - Figure out how to push all my git changes to gcp. May require manually
#   shelling into each machine and doing a pull.
# - Submit & twiddle thumbs.

set -e

# WARNING: unlimited demos here!
base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=5")
n_batches=500000  # in the interests of finishing on time
cluster_cfg_path="./gcp_cluster_sam_new_vis.yaml"
declare -a repl_configs=("icml_inv_dyn" "icml_dynamics"
                         "cfg_repl_tcpc8_192"
                         "cfg_repl_simclr_192"
                         "icml_vae")
declare -a dataset_configs=("cfg_data_repl_demos_random")
declare -a env_configs=("cfg_bench_micro_sweep_dm_control"
                        "cfg_bench_procgen_cmfn")
gpu_default=0.11
declare -A gpu_overrides=(
    ["cfg_repl_tcpc8"]="0.16"
    ["cfg_repl_simclr"]="0.16"
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
        ./submit_pretrain_n_adapt.py -- "${base_cfgs[@]}" "$@"
}

for env_config in "${env_configs[@]}"; do
    for repl_config in "${repl_configs[@]}"; do
        for dataset_config in "${dataset_configs[@]}"; do
            echo -e "\n *** TRAINING $repl_config ON $env_config WITH DATASET $dataset_config *** \n "
            submit_expt cfg_repl_5k_il cfg_repl_augs \
                "$repl_config" "$(gpu_config "$repl_config")" \
                "$dataset_config" cfg_il_bc_500k_nofreeze \
                exp_ident="neurips_repl_bc_${repl_config}_${dataset_config}" \
                "$env_config" "il_train.bc.n_batches=$n_batches"
        done
    done
    for use_augs in augs noaugs; do
        submit_expt cfg_repl_none cfg_il_bc_500k_nofreeze \
            "$(gpu_config "no_repl")" "cfg_il_bc_${use_augs}" \
            exp_ident="neurips_control_bc_${use_augs}" \
            "$env_config" "il_train.bc.n_batches=$n_batches"
    done
done
