#!/bin/bash

set -e

dest="submit_expts_2021_02_04_set3_newbc4pm_filtered.sh"

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu")

cluster_cfg_path="./gcp_cluster_sam_4pm.yaml"

# note that this does not contain identity CPC, but does contain temporal CPC
declare -a repl_configs=("icml_inv_dyn" "icml_dynamics" "icml_ac_tcpc"
                         "icml_tcpc" "icml_vae")
declare -a control_configs=("control_ortho_init" "control_no_ortho_init")
# Sam: skipping control_lsi_… configs because I don't expect them to make a difference.
# (this hyperparameter is also only used in DMC)
#                             "control_lsi_one" "control_lsi_zero")
declare -a magical_envs=("MatchRegions-Demo-v0" "MoveToRegion-Demo-v0" "MoveToCorner-Demo-v0")
declare -a mg_dataset_configs=("cfg_data_repl_demos_random" "cfg_data_repl_random"
                               "cfg_data_repl_demos_magical_mt" "cfg_data_repl_rand_demos_magical_mt")

submit_expt() {
    # submit experiment to cluster using given args
    ray submit --tmux "$cluster_cfg_path" \
        ./submit_pretrain_n_adapt.py -- "${base_cfgs[@]}" "$@"
}

# prelude to output
# (only print necessary configs, definitions, etc.)
echo "#!/bin/bash" > "$dest"
echo "set -e" >> "$dest"
declare -p cluster_cfg_path base_cfgs >> "$dest"
# echo 'cluster_cfg_path="'"$cluster_cfg_path"'"' >> "$dest"
# echo 'base_cfgs=('"${base_cfgs[@]}"')' >> "$dest"
type submit_expt | tail -n +2 >> "$dest"

# write actual configs
for magical_env in "${magical_envs[@]}"; do
    echo "# Environment: $magical_env" >> "$dest"
    for repl_config in "${repl_configs[@]}"; do
        echo "# RepL algo: $repl_config" >> "$dest"
        for mg_dataset_config in "${mg_dataset_configs[@]}"; do
            echo submit_expt icml_il_on_repl_sweep cfg_use_magical \
                "$repl_config" "$mg_dataset_config" cfg_il_bc_20k_nofreeze  \
                "tune_run_kwargs.num_samples=1" \
                exp_ident="newbcaugs_${repl_config}_${mg_dataset_config}" \
                "env_cfg.task_name=$magical_env" >> "$dest"
        done
    done
    echo "# Control" >> "$dest"
    for control_config in "${control_configs[@]}"; do
        echo submit_expt icml_control cfg_use_magical \
            "$control_config" cfg_il_bc_20k_nofreeze  \
            "tune_run_kwargs.num_samples=9" \
            exp_ident="newbcaugs_${control_config}" \
            "env_cfg.task_name=$magical_env" >> "$dest"
    done
    echo >> "$dest"
done
