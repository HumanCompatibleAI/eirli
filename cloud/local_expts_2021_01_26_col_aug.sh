#!/bin/bash

# Investigating impact of colour augmentation (which randomly reassigns hues) on
# repL and IL in MAGICAL

set -e

# probably we need 5 seeds to get something meaningful
# (note that this requires Ray to be running on port 42000)
base_cfgs="cfg_base_5seed_1cpu_pt3gpu ray_init_kwargs.address=localhost:42000 venv_opts.n_envs=5 "

for il in cfg_il_bc_20k_nofreeze; do
    froco="$([[ "$il" == *_nofreeze  ]] || echo '_froco')"

    # we have no multitask runs because DMC does not support multitask
    for bench in \
        "env_cfg.benchmark_name=magical env_cfg.task_name=MoveToCorner" \
        "env_cfg.benchmark_name=magical env_cfg.task_name=MatchRegions" \
        ; do
        # the _col_aug_il run has color augmentations applied to repL, but not IL;
        # the _il_col_aug run has color augmentations applied to IL, but not repL.
        for repl in \
            "exp_ident=cpc_identity_aug_rand_demos_il$froco cfg_data_repl_demos_random repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_aug repl.algo_params.batch_size=192" \
            "exp_ident=cpc_identity_aug_rand_demos_col_aug_il$froco cfg_data_repl_demos_random repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_aug repl.algo_params.batch_size=192 repl.algo_params.augmenter_kwargs.augmenter_spec='translate,rotate,gaussian_blur,color_jitter_ex'" \
            "exp_ident=cpc_identity_aug_rand_demos_il_col_aug$froco cfg_data_repl_demos_random repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc_aug repl.algo_params.batch_size=192 il_train.bc.augs='rotate,translate,noise,color_jitter_ex'" \
            "exp_ident=tcpc_aug_mt_il_25traj$froco cfg_data_repl_demos_magical_mt repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_augment_both_magical" \
            "exp_ident=tcpc_aug_mt_col_aug_il_25traj$froco cfg_data_repl_demos_magical_mt repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_augment_both_magical repl.algo_params.augmenter_kwargs.augmenter_spec='translate,rotate,gaussian_blur,color_jitter_ex'" \
            "exp_ident=tcpc_aug_mt_il_col_aug_25traj$froco cfg_data_repl_demos_magical_mt repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_augment_both_magical il_train.bc.augs='rotate,translate,noise,color_jitter_ex'" \
            ; do
            python -m il_representations.scripts.pretrain_n_adapt with \
                $base_cfgs $bench cfg_base_repl_5000_batches cfg_force_use_repl $il $repl &
            sleep 1
        done

        python -m il_representations.scripts.pretrain_n_adapt with \
            $base_cfgs $bench cfg_repl_none $il exp_ident=control_il$froco &
        sleep 1
    done
done

wait
