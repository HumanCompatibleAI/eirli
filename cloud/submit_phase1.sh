#!/bin/bash

set -e

base_cfgs="cfg_base_3seed_1cpu_pt2gpu_2envs cfg_base_repl_1500"
cluster_cfg_path="./gcp_cluster_sam.yaml"

for il in cfg_il_bc_nofreeze; do
    for bench in cfg_bench_micro_sweep_magical cfg_bench_micro_sweep_dm_control; do
        # "control" config without repL
        ray submit --tmux "$cluster_cfg_path" \
            ./submit_pretrain_n_adapt.py \
            -- $base_cfgs $bench repl_none $il
        for repl in condition_one_temporal_cpc condition_two_temporal_cpc_momentum \
            condition_three_temporal_cpc_sym_proj condition_four_temporal_cpc_asym_proj \
            condition_five_temporal_cpc_augment_both; do
            ray submit --tmux "$cluster_cfg_path" \
                ./submit_pretrain_n_adapt.py \
                -- $base_cfgs $bench $repl $il
        done
    done
done
