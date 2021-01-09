#!/bin/bash

set -e

base_cfgs="cfg_base_3seed_1cpu_pt5gpu_2envs tune_run_kwargs.resources_per_trial.gpu=1.0"
cluster_cfg_path="./gcp_cluster_sam.yaml"

for il in cfg_il_bc_nofreeze; do
    froco="$([[ "$il" == *_nofreeze  ]] || echo '_froco')"
    # MAGICAL multi-task runs (data sources are MT vs. random only vs. demos +
    # random; all IL runs use full 25 trajectories)
    for bench in cfg_bench_magical_mtc; do
        for bs in 32 64 128 192 256 320 384 448 512; do
            for repl in \
                "exp_ident=bsweep_cpc_identity_aug_mt_il_b${bs}$froco cfg_data_repl_demos_magical_mt repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_aug repl.algo_params.batch_size=${bs}"; do
                ray submit --tmux "$cluster_cfg_path" \
                    ./submit_pretrain_n_adapt.py \
                    -- $base_cfgs $bench cfg_base_repl_5000_batches cfg_force_use_repl $il $repl
            done
        done
    done
done
