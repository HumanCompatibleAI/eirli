#!/bin/bash

# I have a suspicion that the `venv_opts.parallel_workers` option is somehow
# messing up sampling. This env expt re-runs controls to test that hunch.

set -euo pipefail

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu"
           "gail_mr_config_2021_03_29"
           "tune_run_kwargs.num_samples=5"
           "tune_run_kwargs.resources_per_trial.gpu=0.25"
           "ray_init_kwargs.address=localhost:42000")
declare -a control_configs=("control_no_ortho_init")
declare -a magical_envs=("MatchRegions-Demo-v0" "MoveToCorner-Demo-v0" "MoveToRegion-Demo-v0")

submit_expt() {
    # submit experiments to local Ray server
    python -m il_representations.scripts.pretrain_n_adapt \
        with "${base_cfgs[@]}" "$@" &
}

for magical_env in "${magical_envs[@]}"; do
    for control_config in "${control_configs[@]}"; do
        submit_expt stages_to_run=IL_ONLY cfg_use_magical \
            "$control_config" exp_ident="magigail_${control_config}_p8" \
            "env_cfg.task_name=$magical_env" "venv_opts.parallel_workers=8"
    done
done
wait
