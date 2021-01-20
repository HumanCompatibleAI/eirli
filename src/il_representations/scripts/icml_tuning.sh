declare -a tuning_configs=("tune_ceb" "tune_momentum"
                           "main_contrastive_tuning" "tune_vae",
                           "tune_projection_heads")

declare -a dmc_envs=("finger-spin" "cheetah-run")

declare -a magical_envs=("MatchRegions" "MoveToRegion")

for algo_config in ${tuning_configs[@]}; do
  for dmc_env in ${dmc_envs[@]}; do
    printf "\n ***TRAINING $algo_config ON  $dmc_env *** \n "
    CUDA_VISIBLE_DEVICES="1" xvfb-run -a python -m il_representations.scripts.pretrain_n_adapt\
     with icml_tuning cfg_use_dm_control $algo_config exp_ident=$algo_config env_cfg.task_name=$dmc_env
  done
  for magical_env in ${magical_envs[@]}; do
    printf "\n ***TRAINING $algo_config ON $magical_env *** \n "
    CUDA_VISIBLE_DEVICES="1" xvfb-run -a python -m il_representations.scripts.pretrain_n_adapt\
    with icml_tuning cfg_use_magical $algo_config exp_ident=$algo_config env_cfg.task_name=$magical_env
  done
done