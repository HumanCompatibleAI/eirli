#!/bin/bash

set -e

base_cfgs="cfg_base_3seed_1cpu_pt2gpu_2envs"
cluster_cfg_path="./gcp_cluster_sam.yaml"

# running ALL THE REPL ALGORITHMS so that we know what works/what doesn't
for il in cfg_il_bc_nofreeze; do
    for bench in cfg_bench_micro_sweep_magical cfg_bench_micro_sweep_dm_control; do
        # "control" config without repL
        ray submit --tmux "$cluster_cfg_path" \
            ./submit_pretrain_n_adapt.py \
            -- $base_cfgs $bench cfg_repl_none $il exp_ident=control
        for repl in \
            "exp_ident=tcpc_momentum_new repl.condition_two_temporal_cpc_momentum repl.stooke_contrastive_hyperparams_dmc repl.cody_momentum_bn" \
            "exp_ident=tcpc_aug repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_augment_both_magical" \
            "exp_ident=tcpc_act_cond repl.algo=ActionConditionedTemporalCPC repl.stooke_contrastive_hyperparams_dmc" \
            "exp_ident=cpc_identity repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc" \
            "exp_ident=ceb repl.stooke_contrastive_hyperparams_dmc repl.temporal_ceb_no_projection" \
            ; do
            # all other options must come BEFORE repl.$repl_nc, because Sacred
            # has a bug in how it handles namedconfigs for ingredients (ask
            # Sam/Cody about this)
            ray submit --tmux "$cluster_cfg_path" \
                ./submit_pretrain_n_adapt.py \
                -- $base_cfgs $bench cfg_base_repl_5000 cfg_force_use_repl $il $repl
        done
    done
done
