#!/bin/bash

set -e

base_cfgs="cfg_base_3seed_1cpu_pt2gpu_2envs"
cluster_cfg_path="./gcp_cluster_sam.yaml"

for repl in cfg_repl_{none,simclr,moco,ceb,temporal_cpc}; do
    for il in cfg_il_bc_nofreeze; do
        for bench in cfg_bench_short_sweep_magical cfg_bench_short_sweep_dm_control; do
            ray submit --tmux "$cluster_cfg_path" \
                ./submit_pretrain_n_adapt.py \
                -- $base_cfgs $bench $repl $il
        done
    done
done
