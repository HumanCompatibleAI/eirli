#!/bin/bash
# Big architecture search for cheetah-run w/ 3 demos and 1,000,000 batches of
# optimisation. 3 seeds per run (I'm only looking for BIG differences, so few
# seeds is fine). I'm aiming for <100 trials in total here.

set -euo pipefail

submit_job() {
    # note that this has token L2 regularisation and entropy regularisation
    # coefficients of 1e-7; I'm hoping this is going to prevent accidental
    # blowups due to architecture weirdness
    DISPLAY=:420 python src/il_representations/scripts/joint_training_cluster.py \
        --ray-address=localhost:42000 --nseeds=3 --ray-ngpus=0.1 n_batches=1000000 \
        env_use_dm_control env_cfg.task_name=cheetah-run repl_noid arch_use_flexcnn \
        bc.ent_weight=1e-7 bc.l2_weight=1e-7 bc_data_3demos "$@"
}

# baseline run
submit_job exp_ident="baseline_run"

# "super" run with wide/deep network, dropout, skip connections, batch norm,
# spectral norm, GELU activations, batch size 256
submit_job exp_ident="super_run" bc.batch_size=256 \
    obs_encoder_kwargs.{"depth_factor=4","width_factor=4","activation=GELU"} \
    obs_encoder_kwargs.{"coord_conv=True","skip_conns=True"} \
    obs_encoder_kwargs.{"spectral_norm=True","dropout=0.1","act_norm=BATCH_NORM"}

# modifying width (default is 2)
for width_factor in 1 3 4; do
    submit_job exp_ident="width_${width_factor}" \
        "obs_encoder_kwargs.width_factor=${width_factor}"
done
unset width_factor

# modifying depth (default is 2)
for depth_factor in 1 3 4; do
    submit_job exp_ident="depth_${depth_factor}" \
        "obs_encoder_kwargs.depth_factor=${depth_factor}"
done
unset depth_factor

# modifying dropout rate (default is 0)
for dropout_rate in 0.1 0.5; do
    submit_job exp_ident="dropout_${dropout_rate}" \
        "obs_encoder_kwargs.dropout_rate=${dropout_rate}"
done
unset dropout_rate

# modifying action norm (default is BATCH_NORM)
for act_norm in NO_NORM LAYER_NORM; do
    submit_job exp_ident="act_norm_${act_norm}" \
       "obs_encoder_kwargs.act_norm=${act_norm}"
done
unset act_norm

# turning on spectral norm (default is False)
# shellcheck disable=SC2043
for spectral_norm in True; do
    submit_job exp_ident="spectral_norm_${spectral_norm}" \
        "obs_encoder_kwargs.spectral_norm=${spectral_norm}"
done
unset spectral_norm

# turning on skip connections (default is False)
# shellcheck disable=SC2043
for skip_conns in True; do
    submit_job exp_ident="skip_conns_${skip_conns}" \
        "obs_encoder_kwargs.skip_conns=${skip_conns}"
done
unset skip_conns

# turning on coordconv (default is False)
# shellcheck disable=SC2043
for coord_conv in False; do
    submit_job exp_ident="coord_conv_${coord_conv}" \
        "obs_encoder_kwargs.coord_conv=${coord_conv}"
done
unset skip_conns

# modifying activation type (default is ReLU)
for activation in ELU GELU Tanh; do
    submit_job exp_ident="activation_${activation}" \
        obs_encoder_kwargs.activation="$activation"
done
unset activation

# modifying batch size (default is 64)
for batch_size in 16 256 1024; do
    submit_job exp_ident="batch_size_${batch_size}" \
        "bc.batch_size=$batch_size"
done
unset batch_size
