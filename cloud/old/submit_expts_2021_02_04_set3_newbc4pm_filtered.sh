#!/bin/bash
set -e
declare -- cluster_cfg_path="./gcp_cluster_sam_4pm.yaml"
declare -a base_cfgs=([0]="cfg_base_5seed_1cpu_pt25gpu" [1]="tune_run_kwargs.num_samples=1")
submit_expt ()
{
    ray submit --tmux "$cluster_cfg_path" ./submit_pretrain_n_adapt.py -- "${base_cfgs[@]}" "$@"
}
# Environment: MatchRegions
# RepL algo: icml_ac_tcpc
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_ac_tcpc cfg_data_repl_demos_random cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_ac_tcpc_cfg_data_repl_demos_random env_cfg.task_name=MatchRegions
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_ac_tcpc cfg_data_repl_random cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_ac_tcpc_cfg_data_repl_random env_cfg.task_name=MatchRegions
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_ac_tcpc cfg_data_repl_rand_demos_magical_mt cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_ac_tcpc_cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions
# RepL algo: icml_tcpc
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_tcpc cfg_data_repl_demos_random cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_tcpc_cfg_data_repl_demos_random env_cfg.task_name=MatchRegions
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_tcpc cfg_data_repl_random cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_tcpc_cfg_data_repl_random env_cfg.task_name=MatchRegions
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_tcpc cfg_data_repl_demos_magical_mt cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_tcpc_cfg_data_repl_demos_magical_mt env_cfg.task_name=MatchRegions
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_tcpc cfg_data_repl_rand_demos_magical_mt cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_tcpc_cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions
# Control
submit_expt icml_control cfg_use_magical control_ortho_init cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_control_ortho_init env_cfg.task_name=MatchRegions
submit_expt icml_control cfg_use_magical control_no_ortho_init cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_control_no_ortho_init env_cfg.task_name=MatchRegions

# Environment: MoveToRegion
# RepL algo: icml_ac_tcpc
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_ac_tcpc cfg_data_repl_demos_random cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_ac_tcpc_cfg_data_repl_demos_random env_cfg.task_name=MoveToRegion
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_ac_tcpc cfg_data_repl_random cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_ac_tcpc_cfg_data_repl_random env_cfg.task_name=MoveToRegion
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_ac_tcpc cfg_data_repl_rand_demos_magical_mt cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_ac_tcpc_cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MoveToRegion
# RepL algo: icml_tcpc
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_tcpc cfg_data_repl_demos_random cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_tcpc_cfg_data_repl_demos_random env_cfg.task_name=MoveToRegion
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_tcpc cfg_data_repl_random cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_tcpc_cfg_data_repl_random env_cfg.task_name=MoveToRegion
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_tcpc cfg_data_repl_demos_magical_mt cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_tcpc_cfg_data_repl_demos_magical_mt env_cfg.task_name=MoveToRegion
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_tcpc cfg_data_repl_rand_demos_magical_mt cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_tcpc_cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MoveToRegion
# RepL algo: icml_vae
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_vae cfg_data_repl_rand_demos_magical_mt cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_vae_cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MoveToRegion
# Control
submit_expt icml_control cfg_use_magical control_ortho_init cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_control_ortho_init env_cfg.task_name=MoveToRegion tune_run_kwargs.num_samples=9
submit_expt icml_control cfg_use_magical control_no_ortho_init cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_control_no_ortho_init env_cfg.task_name=MoveToRegion tune_run_kwargs.num_samples=9

# Environment: MoveToCorner
# RepL algo: icml_ac_tcpc
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_ac_tcpc cfg_data_repl_demos_random cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_ac_tcpc_cfg_data_repl_demos_random env_cfg.task_name=MoveToCorner
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_ac_tcpc cfg_data_repl_random cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_ac_tcpc_cfg_data_repl_random env_cfg.task_name=MoveToCorner
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_ac_tcpc cfg_data_repl_rand_demos_magical_mt cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_ac_tcpc_cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MoveToCorner
# RepL algo: icml_tcpc
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_tcpc cfg_data_repl_demos_random cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_tcpc_cfg_data_repl_demos_random env_cfg.task_name=MoveToCorner
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_tcpc cfg_data_repl_random cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_tcpc_cfg_data_repl_random env_cfg.task_name=MoveToCorner
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_tcpc cfg_data_repl_demos_magical_mt cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_tcpc_cfg_data_repl_demos_magical_mt env_cfg.task_name=MoveToCorner
submit_expt icml_il_on_repl_sweep cfg_use_magical icml_tcpc cfg_data_repl_rand_demos_magical_mt cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_icml_tcpc_cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MoveToCorner
# Control
submit_expt icml_control cfg_use_magical control_ortho_init cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_control_ortho_init env_cfg.task_name=MoveToCorner tune_run_kwargs.num_samples=9
submit_expt icml_control cfg_use_magical control_no_ortho_init cfg_il_bc_20k_nofreeze exp_ident=newbcaugs_control_no_ortho_init env_cfg.task_name=MoveToCorner tune_run_kwargs.num_samples=9
