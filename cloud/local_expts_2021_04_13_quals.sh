#!/bin/bash

set -euo pipefail

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu"
           "tune_run_kwargs.num_samples=3"
           "tune_run_kwargs.resources_per_trial.gpu=0.3"
           "ray_init_kwargs.address=localhost:42000"
           "cfg_use_magical")
repl_configs=("icml_inv_dyn" "icml_identity_cpc" "icml_vae" "icml_dynamics")
declare -A env_names=(
    ["matchregions"]="MatchRegions-Demo-v0"
    ["movetocorner"]="MoveToCorner-Demo-v0"
)
declare -A il_flags=(
    ["gail"]="gail_mr_config_2021_03_29"
    ["bc"]="cfg_il_bc_20k_nofreeze"
)
common_control_flags=("stages_to_run=IL_ONLY" "control_no_ortho_init")
common_repl_flags=("stages_to_run=REPL_AND_IL")

submit_expt() {
    # submit experiments to local Ray server
    python -m il_representations.scripts.pretrain_n_adapt \
        with "${base_cfgs[@]}" "$@" &
}

submit_control() {
    submit_expt "${common_control_flags[@]}" "$@"
}

for env_nick in "${!env_names[@]}"; do
    full_env_name="${env_names[$env_nick]}"
    for il_alg in bc gail; do
        for repl_config in "${repl_configs[@]}"; do
            # alias for running repL experiments
            submit_repl() {
                submit_expt "${common_repl_flags[@]}" "${il_flags[$il_alg]}" \
                    "$repl_config" "$@"
            }

            # demos + random rollouts
            submit_repl "cfg_data_repl_demos_random" "env_cfg.task_name=$full_env_name" \
                exp_ident="quals_${il_alg}_${repl_config}_rand_demos"
            # multitask demos + random rollouts
            submit_repl "cfg_data_repl_rand_demos_magical_mt" "env_cfg.task_name=$full_env_name" \
                exp_ident="quals_${il_alg}_${repl_config}_rand_demos_magical_mt"
            # test variant demos + random rollouts
            submit_repl "env_cfg.task_name=${full_env_name}" "repl.is_multitask=True" \
                "cfg_data_repl_${env_nick}_rand_demos_magical_test" \
                "cfg_data_repl_rand_demos_magical_mt" "env_cfg.task_name=$full_env_name" \
                exp_ident="quals_${il_alg}_${repl_config}_rand_demos_test"
        done

        # controls
        submit_control \
            "${il_flags[$il_alg]}" "env_cfg.task_name=$full_env_name" \
            exp_ident="quals_${il_alg}_control"
    done
    # one env at a time (avoids "too many open files" error)
    wait
done
wait
