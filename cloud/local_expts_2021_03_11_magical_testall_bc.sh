#!/bin/bash

set -e

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=5" "ray_init_kwargs.address=localhost:42000")

# let's cast a wiiiide net
declare -a repl_configs=("icml_inv_dyn" "icml_dynamics" "icml_identity_cpc" "icml_vae")
declare -a control_configs=("control_no_ortho_init")
declare -a magical_envs=("MatchRegions-Demo-v0" "MoveToRegion-Demo-v0" "MoveToCorner-Demo-v0")
# configs to test:
# - plain single-variant demos vs. mix of single-variant demos and demos on the
#   test variant
# - MT demos+random (demo) vs. MT demos+random (test) (note that defining other
#   configs is super annoying with the full name env specification, so going to
#   leave it off for now)
declare -a mg_dataset_configs=(
    "cfg_data_repl_rand_demos_magical_mt"
    "cfg_data_repl_rand_demos_magical_mt_test"
)

submit_expt() {
    # submit experiments to local Ray server
    python -m il_representations.scripts.pretrain_n_adapt \
           with "${base_cfgs[@]}" "$@" &
}

for magical_env in "${magical_envs[@]}"; do
    for repl_config in "${repl_configs[@]}"; do
        for mg_dataset_config in "${mg_dataset_configs[@]}"; do
            echo -e "\n *** TRAINING $repl_config ON $magical_env WITH DATASET $mg_dataset_config *** \n "
            submit_expt stages_to_run=REPL_AND_IL cfg_use_magical \
                "$repl_config" "$mg_dataset_config" cfg_il_bc_20k_nofreeze  \
                exp_ident="mtest_${repl_config}_${mg_dataset_config}" \
                "env_cfg.task_name=$magical_env"
        done
    done
    for control_config in "${control_configs[@]}"; do
        submit_expt stages_to_run=IL_ONLY cfg_use_magical \
            "$control_config" cfg_il_bc_20k_nofreeze  \
            exp_ident="mtest_${control_config}" \
            "env_cfg.task_name=$magical_env"
    done
done
