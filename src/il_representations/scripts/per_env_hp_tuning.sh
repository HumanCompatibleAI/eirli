declare -a tuning_configs=("temporal_cpc_tune" "temporal_cpc_aug_tune"
                           "ac_temporal_cpc_tune", "identity_cpc_aug_tune"
                           "temporal_ceb_tune" "temporal_ceb_fixed_variance_tune"
                           "vae_tune" "temporal_cpc_momentum_tune")

declare -a dmc_envs=("finger-spin" "cheetah-run" "walker-walk"
                     "cartpole-swingup" "reacher-easy" "ball-in-cup-catch")

declare -a magical_envs=("MatchRegions" "MoveToRegion" "MoveToCorner"
                         "MakeLine" "FindDupe" "ClusterShape")

for algo_config in ${tuning_configs[@]}; do
  for dmc_env in ${dmc_envs[@]}; do
    printf "\n ***TRAINING $algo_config ON  $dmc_env *** \n "
    CUDA_VISIBLE_DEVICES="1" xvfb-run -a python -m il_representations.scripts.pretrain_n_adapt\
     with tuning cfg_use_dm_control $algo_config benchmark.dm_control_env=$dmc_env
  done
  for magical_env in ${magical_envs[@]}; do
    printf "\n ***TRAINING $algo_config ON $magical_env *** \n "
    CUDA_VISIBLE_DEVICES="1" xvfb-run -a python -m il_representations.scripts.pretrain_n_adapt\
    with tuning cfg_use_magical $algo_config benchmark.magical_env_prefix=$magical_env
  done
done