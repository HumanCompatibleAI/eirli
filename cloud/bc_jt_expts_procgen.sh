#!/bin/bash

# Procgen joint train runs for NeurIPS benchmarks track:
# - Everything uses the same augmentations (augs_neurips_repl_bc), except the
#   no-augs control.
# - Algos: control (noid), no-augs control (noid_noaugs), ID, FD, VAE, SimCLR,
#   TCPC-8.
# - IL data: 5 demos for MAGICAL, however many demos are on disk for the rest.
# - repL data: same demos + all available random rollouts too.
# - Batch size of 64 for all repL algos except the contrastive ones, which are
#   192.
# - RepL weight is always 1.0, except in the controls where it is 0.0 (I can't
#   disable repL entirely, so I just zero the weight).

set -e

ray_ncpus=1.0
ray_address="localhost:42000"
nseeds=5
n_batches=500000  # hopefully finishes fast enough, may be too large even
dmc_procgen_base_cfgs=(
    "n_batches=$n_batches" "bc.short_eval_interval=50000" "augs_neurips_repl_bc"
    "repl_data_demos_random")
repl_configs=(
    "repl_simclr_192" "repl_tcpc8_192" "repl_id" "repl_vae" "repl_fd"
    "repl_noid" "repl_noid_noaugs")
gpu_default=0.11
declare -A gpu_overrides=(
    ["repl_tcpc8_192"]="0.16"
    ["repl_simclr_192"]="0.16"
)
gpu_config() {
    # figures out GPU configuration string based on repL config, if any
    override="$gpu_default"
    for cfg_string in "$@"; do
        this_override="${gpu_overrides[$cfg_string]}"
        if [ -n "$this_override" ]; then
            override="$this_override"
        fi
    done
    echo "$override"
}

set_dir() {
    # assert that we receive two args: data_dir and run_dir
    if [ "$#" -ne 2 ]; then
        echo "USAGE: $0 data_dir run_dir"
        exit 1
    fi
    data_dir="$1"
    run_dir="$2"

    # get dir containing this current script (bashism)
    this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

    # now make that data_link and runs_link are symlinks pointing to data_dir
    # and run_dir, respectively
    data_link="$(realpath "$this_dir/../data")"
    runs_link="$(realpath "$this_dir/../runs")"
    ln -snT "$data_dir" "$data_link"
    ln -snT "$run_dir" "$runs_link"
}
set_dir "$@"

submit_expt() {
    # submit joint training experiment to local Ray server
    n_gpus="$(gpu_config "$@")"
    python -m il_representations.scripts.joint_training_cluster \
        --ray-address="$ray_address" --ray-ncpus "$ray_ncpus" \
        --ray-ngpus "$n_gpus" --nseeds "$nseeds" "$@" &
}

dmc_procgen_launch_one() {
    for repl_config in "${repl_configs[@]}"; do
        # always demos + random rollouts for repL, hence the exp_ident
        # (see base_cfgs for actual dataset config)
        submit_expt "${dmc_procgen_base_cfgs[@]}" \
            exp_ident="${repl_config}_demos_random" "${repl_config[@]}" "$@"
    done
}

for procgen_task in jumper coinrun fruitbot miner; do
    dmc_procgen_launch_one "env_cfg.benchmark_name=procgen" "env_cfg.task_name=$procgen_task"
done
wait