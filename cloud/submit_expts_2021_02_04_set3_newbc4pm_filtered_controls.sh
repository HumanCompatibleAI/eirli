#!/bin/bash
set -e
declare -- cluster_cfg_path="./gcp_cluster_sam_4pm.yaml"
declare -a base_cfgs=([0]="cfg_base_5seed_1cpu_pt25gpu" [1]="tune_run_kwargs.num_samples=1")
submit_expt ()
{
    ray submit --tmux "$cluster_cfg_path" ./submit_pretrain_n_adapt.py -- "${base_cfgs[@]}" "$@"
}
# Control
submit_expt icml_control cfg_use_magical control_ortho_init cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_control_ortho_init env_cfg.task_name=MatchRegions tune_run_kwargs.num_samples=9
submit_expt icml_control cfg_use_magical control_no_ortho_init cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_control_no_ortho_init env_cfg.task_name=MatchRegions tune_run_kwargs.num_samples=9

# Environment: MoveToRegion
# RepL algo: icml_inv_dyn
# Control
submit_expt icml_control cfg_use_magical control_ortho_init cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_control_ortho_init env_cfg.task_name=MoveToRegion tune_run_kwargs.num_samples=9
submit_expt icml_control cfg_use_magical control_no_ortho_init cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_control_no_ortho_init env_cfg.task_name=MoveToRegion tune_run_kwargs.num_samples=9

# Environment: MoveToCorner
# Control
submit_expt icml_control cfg_use_magical control_ortho_init cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_control_ortho_init env_cfg.task_name=MoveToCorner tune_run_kwargs.num_samples=9
submit_expt icml_control cfg_use_magical control_no_ortho_init cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_control_no_ortho_init env_cfg.task_name=MoveToCorner tune_run_kwargs.num_samples=9
