#!/bin/bash

set -e

base_cfgs="cfg_base_3seed_1cpu_pt5gpu_2envs"
cluster_cfg_path="./gcp_cluster_sam.yaml"

for il in cfg_il_bc_nofreeze; do
    froco="$([[ "$il" == *_nofreeze  ]] || echo '_froco')"
    for bench in cfg_bench_one_task_magical cfg_bench_one_task_dm_control; do
        # "control" config without repL
        ray submit --tmux "$cluster_cfg_path" \
            ./submit_pretrain_n_adapt.py \
            -- $base_cfgs $bench cfg_repl_none $il exp_ident=control$froco
        for repl in \
            "exp_ident=tcpc_aug$froco repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_augment_both_magical repl.algo_params.batch_size=192" \
            "exp_ident=tcpc_act_cond$froco repl.algo=ActionConditionedTemporalCPC repl.stooke_contrastive_hyperparams_dmc repl.algo_params.batch_size=192" \
            "exp_ident=cpc_identity_aug$froco repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_aug repl.algo_params.batch_size=192" \
            "exp_ident=tcpc_aug_rand$froco cfg_data_repl_demos_random repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_augment_both_magical repl.algo_params.batch_size=192" \
            "exp_ident=tcpc_act_cond_rand$froco cfg_data_repl_demos_random repl.algo=ActionConditionedTemporalCPC repl.stooke_contrastive_hyperparams_dmc repl.algo_params.batch_size=192" \
            "exp_ident=cpc_identity_aug_rand$froco cfg_data_repl_demos_random repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_aug repl.algo_params.batch_size=192" \
            ; do
            # all other options must come BEFORE repl.$repl_nc, because Sacred
            # has a bug in how it handles namedconfigs for ingredients (ask
            # Sam/Cody about this)
            ray submit --tmux "$cluster_cfg_path" \
                ./submit_pretrain_n_adapt.py \
                -- $base_cfgs $bench cfg_base_repl_10000_batches cfg_force_use_repl $il $repl
        done
    done
done
