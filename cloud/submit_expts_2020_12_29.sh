#!/bin/bash

set -e

base_cfgs="cfg_base_3seed_1cpu_pt5gpu_2envs tune_run_kwargs.num_samples=15"
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
            "exp_ident=ac_tceb_aug_rand_only_il_25traj$froco cfg_data_repl_random repl.stooke_contrastive_hyperparams_dmc repl.temporal_ceb_action_conditioned_augment_both repl.algo_params.batch_size=192" \
            "exp_ident=ac_tcpc_aug_rand_only_il_25traj$froco cfg_data_repl_random repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_action_conditioned_augment_both repl.algo_params.batch_size=192" \
            "exp_ident=inv_dyn_mt_il_25traj$froco cfg_data_repl_demos_magical_mt cfg_repl_inv_dyn" \
            ; do
            ray submit --tmux "$cluster_cfg_path" \
                ./submit_pretrain_n_adapt.py \
                -- $base_cfgs $bench cfg_base_repl_5000_batches cfg_force_use_repl $il $repl
        done
    done
done
