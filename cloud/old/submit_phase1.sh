#!/bin/bash

set -e

base_cfgs="cfg_base_3seed_1cpu_pt2gpu_2envs"
cluster_cfg_path="./gcp_cluster_sam.yaml"

# standard configs, no skopt or anything
for il in cfg_il_bc_nofreeze; do
    for bench in cfg_bench_micro_sweep_magical cfg_bench_micro_sweep_dm_control; do
        # "control" config without repL
        ray submit --tmux "$cluster_cfg_path" \
            ./submit_pretrain_n_adapt.py \
            -- $base_cfgs $bench cfg_repl_none $il
        for repl in \
            "repl.condition_one_temporal_cpc repl.stooke_contrastive_hyperparams_dmc" \
            "repl.condition_two_temporal_cpc_momentum repl.stooke_contrastive_hyperparams_dmc repl.stooke_momentum_hyperparams_dmc" \
            "repl.condition_three_temporal_cpc_sym_proj repl.stooke_contrastive_hyperparams_dmc" \
            "repl.condition_four_temporal_cpc_asym_proj repl.stooke_contrastive_hyperparams_dmc" \
            ; do
            # all other options must come BEFORE repl.$repl_nc, because Sacred
            # has a bug in how it handles namedconfigs for ingredients (ask
            # Sam/Cody about this)
            ray submit --tmux "$cluster_cfg_path" \
                ./submit_pretrain_n_adapt.py \
                -- $base_cfgs $bench cfg_base_repl_10000 cfg_force_use_repl $il $repl
        done
    done
done


echo
echo
echo "Done basic configs, starting expensive configs in 30s"
echo
echo
sleep 30

# expensive HP tuning configs with skopt
for repl in \
    "cfg_tune_augmentations repl.condition_five_temporal_cpc_augment_both repl.stooke_contrastive_hyperparams_dmc" \
    "cfg_tune_vae_learning_rate repl.condition_ten_vae repl.stooke_vae_hyperparams_dmc" \
    ; do
    for il in cfg_il_bc_nofreeze; do
        for bench in cfg_bench_one_task_magical cfg_bench_one_task_dm_control; do
            ray submit --tmux "$cluster_cfg_path" \
                ./submit_pretrain_n_adapt.py \
                -- $base_cfgs $bench cfg_force_use_repl $il $repl
        done
    done
done
