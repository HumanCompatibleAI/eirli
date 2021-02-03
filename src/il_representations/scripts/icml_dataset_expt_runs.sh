declare -a testing_configs=("icml_inv_dyn" "icml_dynamics" "icml_ac_tcpc"
                            "icml_identity_cpc" "icml_vae" "icml_four_tcpc"
                            "control_ortho_init" "control_no_ortho_init")
# Sam: skipping control_lsi_… configs because I don't expect them to make a difference.
# (this hyperparameter is also only used in DMC)
#                             "control_lsi_one" "control_lsi_zero")

declare -a dmc_envs=("finger-spin" "cheetah-run")

declare -a magical_envs=("MatchRegions" "MoveToRegion")

declare -a dmc_dataset_configs=("cfg_data_repl_demos_random" "cfg_data_repl_random")

declare -a mg_dataset_configs=("cfg_data_repl_demos_random" "cfg_data_repl_random"
                               "cfg_data_repl_demos_magical_mt", "cfg_data_repl_rand_demos_magical_mt")


for test_config in ${testing_configs[@]}; do
  for dmc_env in ${dmc_envs[@]}; do
    for dmc_dataset_config in ${dmc_dataset_configs[@]}; do
      printf "\n ***TRAINING $algo_config ON  $dmc_env *** \n "
      CUDA_VISIBLE_DEVICES="1" xvfb-run -a python -m il_representations.scripts.pretrain_n_adapt\
       with icml_il_on_repl_sweep cfg_use_dm_control\
        $test_config $dmc_dataset_config cfg_il_bc_200k_nofreeze\
         exp_ident=$test_config"_"$dmc_dataset_config env_cfg.task_name=$dmc_env
    done
  done
  for magical_env in ${magical_envs[@]}; do
    for mg_dataset_config in ${mg_dataset_configs[@]}; do
      printf "\n ***TRAINING $algo_config ON $magical_env *** \n "
      CUDA_VISIBLE_DEVICES="1" xvfb-run -a python -m il_representations.scripts.pretrain_n_adapt\
      with icml_il_on_repl_sweep cfg_use_magical\
      $test_config $mg_dataset_config cfg_il_bc_20k_nofreeze  \
       exp_ident=$test_config"_"$mg_dataset_config env_cfg.task_name=$magical_env
    done
done
