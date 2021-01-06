#!/bin/bash

# Some DMC runs with a huge complement of algorithms and data sources.

set -e

# override three-seed config to do five seeds instead. Also make sure that
# HalfCheetah always uses the extended dataset.
base_cfgs="cfg_base_3seed_1cpu_pt5gpu_2envs tune_run_kwargs.num_samples=5 cfg_data_il_hc_extended"
cluster_cfg_path="./gcp_cluster_sam.yaml"

for il in cfg_il_bc_nofreeze cfg_il_bc_500k_nofreeze; do
    froco="$([[ "$il" == *_nofreeze  ]] || echo '_froco')"

    # we have no multitask runs because DMC does not support multitask
    for bench in \
        "env_cfg.benchmark_name=dm_control env_cfg.task_name=cheetah-run" \
        "env_cfg.benchmark_name=dm_control env_cfg.task_name=finger-spin" \
        ; do
        python -m il_representations.scripts.pretrain_n_adapt \
            -- $base_cfgs $bench cfg_repl_none cfg_data_il_hc_extended $il exp_ident=control_il$froco

        for repl in \
            "exp_ident=inv_dyn_rand_demos_il$froco cfg_data_repl_demos_random cfg_repl_inv_dyn" \
            "exp_ident=inv_dyn_rand_only_il$froco cfg_data_repl_random cfg_repl_inv_dyn" \
            "exp_ident=vae_rand_demos_il$froco cfg_data_repl_demos_random cfg_repl_vae" \
            "exp_ident=vae_rand_only_il$froco cfg_data_repl_random cfg_repl_vae" \
            "exp_ident=tcpc_aug_rand_demos_il$froco cfg_data_repl_demos_random repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_augment_both_magical repl.algo_params.batch_size=192" \
            "exp_ident=cpc_identity_aug_rand_demos_il$froco cfg_data_repl_demos_random repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_aug repl.algo_params.batch_size=192" \
            "exp_ident=tcpc_aug_rand_only_il$froco cfg_data_repl_random repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_augment_both_magical repl.algo_params.batch_size=192" \
            "exp_ident=cpc_identity_aug_rand_only_il$froco cfg_data_repl_random repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_aug repl.algo_params.batch_size=192" \
            "exp_ident=ac_tcpc_aug_rand_demos_il$froco cfg_data_repl_demos_random repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_action_conditioned_augment_both repl.algo_params.batch_size=192" \
            "exp_ident=ac_tcpc_aug_rand_only_il$froco cfg_data_repl_random repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_action_conditioned_augment_both repl.algo_params.batch_size=192" \
            "exp_ident=ac_tceb_aug_rand_demos_il$froco cfg_data_repl_demos_random repl.stooke_contrastive_hyperparams_dmc repl.temporal_ceb_action_conditioned_augment_both repl.algo_params.batch_size=192" \
            "exp_ident=ac_tceb_aug_rand_only_il$froco cfg_data_repl_random repl.stooke_contrastive_hyperparams_dmc repl.temporal_ceb_action_conditioned_augment_both repl.algo_params.batch_size=192" \
            "exp_ident=ac_cpc_aug_rand_demos_il$froco cfg_data_repl_demos_random repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_action_conditioned_augment_both repl.algo_params.batch_size=192" \
            "exp_ident=ac_cpc_aug_rand_only_il$froco cfg_data_repl_random repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_action_conditioned_augment_both repl.algo_params.batch_size=192" \
            ; do
            python -m il_representations.scripts.pretrain_n_adapt \
                -- $base_cfgs $bench cfg_base_repl_5000_batches cfg_force_use_repl $il $repl
        done
    done
done
