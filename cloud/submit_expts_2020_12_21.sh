#!/bin/bash

set -e

base_cfgs="cfg_base_3seed_1cpu_pt5gpu_2envs"
cluster_cfg_path="./gcp_cluster_sam.yaml"

for il in cfg_il_bc_nofreeze; do
    froco="$([[ "$il" == *_nofreeze  ]] || echo '_froco')"

    # control runs for all tasks
    for bench in cfg_bench_magical_mr cfg_bench_magical_mtc; do
        ray submit --tmux "$cluster_cfg_path" \
            ./submit_pretrain_n_adapt.py \
            -- $base_cfgs $bench cfg_repl_none cfg_data_il_hc_extended $il exp_ident=control_il_25traj$froco
    done

    # MAGICAL multi-task runs (data sources are MT vs. random only vs. demos +
    # random; all IL runs use full 25 trajectories)
    for bench in cfg_bench_magical_mr cfg_bench_magical_mtc; do
        for repl in \
            "exp_ident=inv_dyn_rand_demos_il_25traj$froco cfg_data_repl_demos_random cfg_repl_inv_dyn" \
            "exp_ident=inv_dyn_rand_only_il_25traj$froco cfg_data_repl_random cfg_repl_inv_dyn" \
            "exp_ident=inv_dyn_mt_il_25traj$froco cfg_data_repl_demos_magical_mt cfg_repl_inv_dyn" \
            "exp_ident=vae_rand_demos_il_25traj$froco cfg_data_repl_demos_random cfg_repl_vae" \
            "exp_ident=vae_rand_only_il_25traj$froco cfg_data_repl_random cfg_repl_vae" \
            "exp_ident=vae_mt_il_25traj$froco cfg_data_repl_demos_magical_mt cfg_repl_vae" \
            "exp_ident=tcpc_aug_rand_demos_il_25traj$froco cfg_data_repl_demos_random repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_augment_both_magical repl.algo_params.batch_size=192" \
            "exp_ident=cpc_identity_aug_rand_demos_il_25traj$froco cfg_data_repl_demos_random repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_aug repl.algo_params.batch_size=192" \
            "exp_ident=tcpc_aug_rand_only_il_25traj$froco cfg_data_repl_random repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_augment_both_magical repl.algo_params.batch_size=192" \
            "exp_ident=cpc_identity_aug_rand_only_il_25traj$froco cfg_data_repl_random repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_aug repl.algo_params.batch_size=192" \
            "exp_ident=tcpc_aug_mt_il_25traj$froco cfg_data_repl_demos_magical_mt repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_augment_both_magical repl.algo_params.batch_size=192" \
            "exp_ident=cpc_identity_aug_mt_il_25traj$froco cfg_data_repl_demos_magical_mt repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_aug repl.algo_params.batch_size=192" \
            ; do
            ray submit --tmux "$cluster_cfg_path" \
                ./submit_pretrain_n_adapt.py \
                -- $base_cfgs $bench cfg_base_repl_5000_batches cfg_force_use_repl $il $repl
        done
    done
done
