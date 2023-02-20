#!/bin/bash

set -e

base_cfgs="cfg_base_3seed_1cpu_1gpu_2envs"
cluster_cfg_path="./gcp_cluster_sam.yaml"

for il in cfg_il_bc_nofreeze; do
    froco="$([[ "$il" == *_nofreeze  ]] || echo '_froco')"

    # additional multi-task runs (MAGICAL only; dm_control has different action spaces for different environments)
    for bench in cfg_bench_magical_mr cfg_bench_magical_mtc; do
        for repl in \
            "exp_ident=tcpc_aug_mt$froco cfg_data_repl_demos_magical_mt repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_augment_both_magical repl.algo_params.batch_size=192" \
            "exp_ident=tcpc_act_cond_mt$froco cfg_data_repl_demos_magical_mt repl.algo=ActionConditionedTemporalCPC repl.stooke_contrastive_hyperparams_dmc repl.algo_params.batch_size=192" \
            "exp_ident=cpc_identity_aug_mt$froco cfg_data_repl_demos_magical_mt repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_aug repl.algo_params.batch_size=192" \
            ; do
            ray submit --tmux "$cluster_cfg_path" \
                ./submit_pretrain_n_adapt.py \
                -- $base_cfgs $bench cfg_base_repl_5000_batches cfg_force_use_repl $il $repl
        done
    done


    for bench in cfg_bench_magical_mr cfg_bench_magical_mtc cfg_bench_one_task_dm_control; do
        # "control" config without repL
        ray submit --tmux "$cluster_cfg_path" \
            ./submit_pretrain_n_adapt.py \
            -- $base_cfgs $bench cfg_repl_none $il exp_ident=control$froco

        # if [[ "$bench" == *"magical"* ]]; then
        #     # multitask runs should train on all MAGICAL tasks
        #     mt_cfg=cfg_data_repl_demos_mt_magical
        # else
        #     # multitask runs should train on all DMC tasks
        #     mt_cfg=cfg_data_repl_demos_mt_dmc
        # fi

        for repl in \
            "exp_ident=tcpc_aug_rand$froco cfg_data_repl_demos_random repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_augment_both_magical repl.algo_params.batch_size=192" \
            "exp_ident=tcpc_act_cond_rand$froco cfg_data_repl_demos_random repl.algo=ActionConditionedTemporalCPC repl.stooke_contrastive_hyperparams_dmc repl.algo_params.batch_size=192" \
            "exp_ident=cpc_identity_aug_rand$froco cfg_data_repl_demos_random repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_aug repl.algo_params.batch_size=192" \
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
