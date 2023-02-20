#!/bin/bash

set -e

base_cfgs="cfg_base_3seed_1cpu_pt5gpu_2envs"
cluster_cfg_path="./gcp_cluster_sam.yaml"

for il in cfg_il_bc_nofreeze; do
    froco="$([[ "$il" == *_nofreeze  ]] || echo '_froco')"

    # control runs for all tasks
    # (cfg_data_il_hc_extended ensures that HalfCheetah run uses extended
    # dataset; cfg_data_il_5traj enoforces use of 5 trajectorieso nly)
    for bench in cfg_bench_magical_mr cfg_bench_magical_mtc ; do
        ray submit --tmux "$cluster_cfg_path" \
            ./submit_pretrain_n_adapt.py \
            -- $base_cfgs $bench cfg_repl_none cfg_data_il_5traj \
            $il exp_ident=control_5traj$froco
    done
    ray submit --tmux "$cluster_cfg_path" \
        ./submit_pretrain_n_adapt.py \
        -- $base_cfgs cfg_bench_one_task_dm_control cfg_repl_none \
            cfg_data_il_hc_extended $il exp_ident=control_500traj$froco
done
