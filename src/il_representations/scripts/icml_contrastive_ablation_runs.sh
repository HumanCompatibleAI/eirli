declare -a testing_configs=("icml_temporal_cpc", "icml_ac_tcpc",
                            "icml_identity_cpc", "icml_temporal_cpc_asym_proj",
                            "icml_tcpc_no_augs" "icml_tceb"
                            "icml_tcpc_momentum")

declare -a dmc_envs=("finger-spin" "cheetah-run")

declare -a magical_envs=("MatchRegions" "MoveToRegion")

declare -a dataset_configs=("cfg_data_repl_demos_random")


for dmc_env in ${dmc_envs[@]}; do
    for dataset_config in ${dataset_configs[@]}; do
        for test_config in ${testing_configs[@]}; do
        printf "\n ***TRAINING $algo_config ON  $dmc_env *** \n "
         CUDA_VISIBLE_DEVICES="1" xvfb-run -a python -m il_representations.scripts.pretrain_n_adapt\
         with icml_il_on_repl_sweep cfg_use_dm_control\
         $test_config $dataset_config cfg_il_bc_200k_nofreeze\
         exp_ident=$test_config"_"$dataset_config env_cfg.task_name=$dmc_env
        done
        # control run
        CUDA_VISIBLE_DEVICES="1" xvfb-run -a python -m il_representations.scripts.pretrain_n_adapt\
         with cfg_use_dm_control icml_control\
          $dataset_config cfg_il_bc_200k_nofreeze\
          exp_ident="control_"$dataset_config env_cfg.task_name=$dmc_env
    done
done



for magical_env in ${magical_envs[@]}; do
    for dataset_config in ${dataset_configs[@]}; do
        for test_config in ${testing_configs[@]}; do
            printf "\n ***TRAINING $algo_config ON $magical_env *** \n "
              CUDA_VISIBLE_DEVICES="1" xvfb-run -a python -m il_representations.scripts.pretrain_n_adapt\
              with icml_il_on_repl_sweep cfg_use_magical\
              $test_config $dataset_config cfg_il_bc_20k_nofreeze  \
              exp_ident=$test_config"_"$dataset_config env_cfg.task_name=$magical_env
        done

        # control run
        CUDA_VISIBLE_DEVICES="1" xvfb-run -a python -m il_representations.scripts.pretrain_n_adapt\
              with cfg_use_magical icml_control\
              $dataset_config cfg_il_bc_20k_nofreeze  \
              exp_ident="control_"$dataset_config env_cfg.task_name=$magical_env
    done
done
