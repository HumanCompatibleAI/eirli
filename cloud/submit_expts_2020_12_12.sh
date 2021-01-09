#!/bin/bash

set -e

base_cfgs="cfg_base_3seed_1cpu_pt5gpu_2envs"
cluster_cfg_path="./gcp_cluster_sam.yaml"

for il in cfg_il_bc_nofreeze; do
    froco="$([[ "$il" == *_nofreeze  ]] || echo '_froco')"

    # control runs for all tasks
    # (cfg_data_il_hc_extended ensures that HalfCheetah run uses extended
    # dataset)
    for bench in cfg_bench_magical_mr cfg_bench_magical_mtc cfg_bench_one_task_dm_control; do
        ray submit --tmux "$cluster_cfg_path" \
            ./submit_pretrain_n_adapt.py \
            -- $base_cfgs $bench cfg_repl_none cfg_data_il_hc_extended $il exp_ident=control$froco
    done

    # MAGICAL multi-task runs (data sources are MT vs. random only vs. demos +
    # random; all IL runs use 5 trajectories)
    for bench in cfg_bench_magical_mr cfg_bench_magical_mtc; do
        for repl in \
            "exp_ident=inv_dyn_rand_demos$froco cfg_data_il_5traj cfg_data_repl_demos_random cfg_repl_inv_dyn" \
            "exp_ident=inv_dyn_rand_only$froco cfg_data_il_5traj cfg_data_repl_random cfg_repl_inv_dyn" \
            "exp_ident=inv_dyn_mt$froco cfg_data_il_5traj cfg_data_repl_demos_magical_mt cfg_repl_inv_dyn" \
            "exp_ident=vae_rand_demos$froco cfg_data_il_5traj cfg_data_repl_demos_random cfg_repl_vae" \
            "exp_ident=vae_rand_only$froco cfg_data_il_5traj cfg_data_repl_random cfg_repl_vae" \
            "exp_ident=vae_mt$froco cfg_data_il_5traj cfg_data_repl_demos_magical_mt cfg_repl_vae" \
            "exp_ident=tcpc_aug_rand_demos$froco cfg_data_il_5traj cfg_data_repl_demos_random repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_augment_both_magical repl.algo_params.batch_size=192" \
            "exp_ident=cpc_identity_aug_rand_demos$froco cfg_data_il_5traj cfg_data_repl_demos_random repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_aug repl.algo_params.batch_size=192" \
            "exp_ident=tcpc_aug_rand_only$froco cfg_data_il_5traj cfg_data_repl_random repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_augment_both_magical repl.algo_params.batch_size=192" \
            "exp_ident=cpc_identity_aug_rand_only$froco cfg_data_il_5traj cfg_data_repl_random repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_aug repl.algo_params.batch_size=192" \
            "exp_ident=tcpc_aug_mt$froco cfg_data_il_5traj cfg_data_repl_demos_magical_mt repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_augment_both_magical repl.algo_params.batch_size=192" \
            "exp_ident=cpc_identity_aug_mt$froco cfg_data_il_5traj cfg_data_repl_demos_magical_mt repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_aug repl.algo_params.batch_size=192" \
            ; do
            ray submit --tmux "$cluster_cfg_path" \
                ./submit_pretrain_n_adapt.py \
                -- $base_cfgs $bench cfg_base_repl_5000_batches cfg_force_use_repl $il $repl
        done
    done

    # DMC runs (data sources are only random)
    for bench in cfg_bench_one_task_dm_control; do
        for repl in \
            "exp_ident=inv_dyn_rand_only$froco cfg_data_il_hc_extended cfg_data_repl_random cfg_repl_inv_dyn" \
            "exp_ident=vae_rand_only$froco cfg_data_il_hc_extended cfg_data_repl_random cfg_repl_vae" \
            "exp_ident=tcpc_aug_rand_only$froco cfg_data_il_hc_extended cfg_data_repl_random repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_augment_both_magical repl.algo_params.batch_size=192" \
            "exp_ident=cpc_identity_aug_rand_only$froco cfg_data_il_hc_extended cfg_data_repl_random repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_aug repl.algo_params.batch_size=192" \
            ; do
            # all other options must come BEFORE repl.$repl_nc, because Sacred
            # has a bug in how it handles namedconfigs for ingredients (ask
            # Sam/Cody about this)
            ray submit --tmux "$cluster_cfg_path" \
                ./submit_pretrain_n_adapt.py \
                -- $base_cfgs $bench cfg_base_repl_5000_batches cfg_force_use_repl $il $repl
        done
    done
done
