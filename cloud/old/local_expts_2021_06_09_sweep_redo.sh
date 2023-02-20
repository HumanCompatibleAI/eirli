#!/bin/bash

# Re-does sweep from June 9th with missing runs (all the MAGICAL
# MoveToCorner/MoveToRegion runs, plus a subset of MatchRegions runs).

set -euo pipefail

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu"
           "tune_run_kwargs.num_samples=5"
           "tune_run_kwargs.resources_per_trial.gpu=0.3"
           "ray_init_kwargs.address=localhost:42000")
repl_configs=("icml_inv_dyn" "icml_identity_cpc" "icml_tcpc_8step" "icml_vae" "icml_dynamics")
# I've removed MatchRegions
declare -A magical_env_names=(
    ["movetocorner"]="MoveToCorner-Demo-v0"
    ["movetoregion"]="MoveToRegion-Demo-v0"
)
declare -A magical_il_flags=(
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

# # MatchRegions partial runs
# 
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=3 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL cfg_il_bc_20k_nofreeze icml_inv_dyn cfg_data_repl_demos_random env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=bc_icml_inv_dyn_rand_demos &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=3 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL cfg_il_bc_20k_nofreeze icml_identity_cpc cfg_data_repl_demos_random env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=bc_icml_identity_cpc_rand_demos &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=2 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL cfg_il_bc_20k_nofreeze icml_identity_cpc cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=bc_icml_identity_cpc_rand_demos_magical_mt &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=2 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL cfg_il_bc_20k_nofreeze icml_identity_cpc env_cfg.task_name=MatchRegions-Demo-v0 repl.is_multitask=True cfg_data_repl_matchregions_rand_demos_magical_test cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=bc_icml_identity_cpc_rand_demos_test &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=1 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL cfg_il_bc_20k_nofreeze icml_tcpc_8step cfg_data_repl_demos_random env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=bc_icml_tcpc_8step_rand_demos &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=4 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL cfg_il_bc_20k_nofreeze icml_tcpc_8step cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=bc_icml_tcpc_8step_rand_demos_magical_mt &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=4 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL cfg_il_bc_20k_nofreeze icml_tcpc_8step env_cfg.task_name=MatchRegions-Demo-v0 repl.is_multitask=True cfg_data_repl_matchregions_rand_demos_magical_test cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=bc_icml_tcpc_8step_rand_demos_test &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=1 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL cfg_il_bc_20k_nofreeze icml_vae cfg_data_repl_demos_random env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=bc_icml_vae_rand_demos &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=3 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL cfg_il_bc_20k_nofreeze icml_vae env_cfg.task_name=MatchRegions-Demo-v0 repl.is_multitask=True cfg_data_repl_matchregions_rand_demos_magical_test cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=bc_icml_vae_rand_demos_test &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=2 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL cfg_il_bc_20k_nofreeze icml_dynamics cfg_data_repl_demos_random env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=bc_icml_dynamics_rand_demos &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=1 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL cfg_il_bc_20k_nofreeze icml_dynamics cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=bc_icml_dynamics_rand_demos_magical_mt &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=1 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL cfg_il_bc_20k_nofreeze icml_dynamics env_cfg.task_name=MatchRegions-Demo-v0 repl.is_multitask=True cfg_data_repl_matchregions_rand_demos_magical_test cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=bc_icml_dynamics_rand_demos_test &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=2 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL gail_mr_config_2021_03_29 icml_inv_dyn env_cfg.task_name=MatchRegions-Demo-v0 repl.is_multitask=True cfg_data_repl_matchregions_rand_demos_magical_test cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=gail_icml_inv_dyn_rand_demos_test &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=2 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL gail_mr_config_2021_03_29 icml_identity_cpc cfg_data_repl_demos_random env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=gail_icml_identity_cpc_rand_demos &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=2 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL gail_mr_config_2021_03_29 icml_identity_cpc cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=gail_icml_identity_cpc_rand_demos_magical_mt &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=4 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL gail_mr_config_2021_03_29 icml_identity_cpc env_cfg.task_name=MatchRegions-Demo-v0 repl.is_multitask=True cfg_data_repl_matchregions_rand_demos_magical_test cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=gail_icml_identity_cpc_rand_demos_test &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=1 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL gail_mr_config_2021_03_29 icml_tcpc_8step cfg_data_repl_demos_random env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=gail_icml_tcpc_8step_rand_demos &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=4 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL gail_mr_config_2021_03_29 icml_tcpc_8step cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=gail_icml_tcpc_8step_rand_demos_magical_mt &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=4 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL gail_mr_config_2021_03_29 icml_tcpc_8step env_cfg.task_name=MatchRegions-Demo-v0 repl.is_multitask=True cfg_data_repl_matchregions_rand_demos_magical_test cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=gail_icml_tcpc_8step_rand_demos_test &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=1 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL gail_mr_config_2021_03_29 icml_vae cfg_data_repl_demos_random env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=gail_icml_vae_rand_demos &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=1 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL gail_mr_config_2021_03_29 icml_vae cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=gail_icml_vae_rand_demos_magical_mt &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=1 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL gail_mr_config_2021_03_29 icml_vae env_cfg.task_name=MatchRegions-Demo-v0 repl.is_multitask=True cfg_data_repl_matchregions_rand_demos_magical_test cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=gail_icml_vae_rand_demos_test &
# python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=1 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 stages_to_run=IL_ONLY control_no_ortho_init cfg_use_magical gail_mr_config_2021_03_29 env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=gail_control &
# # python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=5 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL cfg_il_bc_20k_nofreeze icml_inv_dyn cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=bc_icml_inv_dyn_rand_demos_magical_mt &
# # python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=5 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL cfg_il_bc_20k_nofreeze icml_inv_dyn env_cfg.task_name=MatchRegions-Demo-v0 repl.is_multitask=True cfg_data_repl_matchregions_rand_demos_magical_test cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=bc_icml_inv_dyn_rand_demos_test &
# # python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=5 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL cfg_il_bc_20k_nofreeze icml_vae cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=bc_icml_vae_rand_demos_magical_mt &
# # python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=5 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL gail_mr_config_2021_03_29 icml_dynamics cfg_data_repl_demos_random env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=gail_icml_dynamics_rand_demos &
# # python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=5 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL gail_mr_config_2021_03_29 icml_dynamics cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=gail_icml_dynamics_rand_demos_magical_mt &
# # python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=5 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL gail_mr_config_2021_03_29 icml_dynamics env_cfg.task_name=MatchRegions-Demo-v0 repl.is_multitask=True cfg_data_repl_matchregions_rand_demos_magical_test cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=gail_icml_dynamics_rand_demos_test &
# # python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=5 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 stages_to_run=IL_ONLY control_no_ortho_init cfg_use_magical cfg_il_bc_20k_nofreeze env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=bc_control &
# # python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=5 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL gail_mr_config_2021_03_29 icml_inv_dyn cfg_data_repl_demos_random env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=gail_icml_inv_dyn_rand_demos &
# # python -m il_representations.scripts.pretrain_n_adapt with cfg_base_5seed_1cpu_pt25gpu tune_run_kwargs.num_samples=5 tune_run_kwargs.resources_per_trial.gpu=0.3 ray_init_kwargs.address=localhost:42000 cfg_use_magical stages_to_run=REPL_AND_IL gail_mr_config_2021_03_29 icml_inv_dyn cfg_data_repl_rand_demos_magical_mt env_cfg.task_name=MatchRegions-Demo-v0 exp_ident=gail_icml_inv_dyn_rand_demos_magical_mt &
# wait

# Remaining runs

for magical_env_nick in "${!magical_env_names[@]}"; do
    full_env_name="${magical_env_names[$magical_env_nick]}"
    for il_alg in bc gail; do
        for repl_config in "${repl_configs[@]}"; do
            # alias for running repL experiments
            submit_repl() {
                submit_expt cfg_use_magical "${common_repl_flags[@]}" "${magical_il_flags[$il_alg]}" \
                    "$repl_config" "$@"
            }

            # demos + random rollouts
            submit_repl "cfg_data_repl_demos_random" "env_cfg.task_name=$full_env_name" \
                exp_ident="${il_alg}_${repl_config}_rand_demos"
            # multitask demos + random rollouts
            submit_repl "cfg_data_repl_rand_demos_magical_mt" "env_cfg.task_name=$full_env_name" \
                exp_ident="${il_alg}_${repl_config}_rand_demos_magical_mt"
            # test variant demos + random rollouts
            submit_repl "env_cfg.task_name=${full_env_name}" "repl.is_multitask=True" \
                "cfg_data_repl_${magical_env_nick}_rand_demos_magical_test" \
                "cfg_data_repl_rand_demos_magical_mt" "env_cfg.task_name=$full_env_name" \
                exp_ident="${il_alg}_${repl_config}_rand_demos_test"
        done

        # controls
        submit_control cfg_use_magical \
            "${magical_il_flags[$il_alg]}" "env_cfg.task_name=$full_env_name" \
            exp_ident="${il_alg}_control"
    done
    # one env at a time (avoids "too many open files" error)
    wait
done
wait
