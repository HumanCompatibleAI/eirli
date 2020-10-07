# TODO I don't know how to add the right flag for run over all environments for pretrain_n_adapt
# TODO I don't know how best to construct a `pretrain_n_adapt` config that searches over a given number of seeds per environment


# Run set 1 -> Augmentation hyperparameter search over with TemporalCPC, over all environments
python -m il_representations.scripts.pretrain_n_adapt with cfg_tune_augmentations\
        repl.condition_five_temporal_cpc_augment_both repl.stooke_contrastive_hyperparams_dmc


# Run set 2 -> Final run over different seeds of baseline TemporalCPC, over all environments
python -m il_representations.scripts.pretrain_n_adapt with repl.condition_one_temporal_cpc\
  repl.stooke_contrastive_hyperparams_dmc


# Run set 3 -> Final run over different seeds of TemporalCPC with momentum, over all environments
python -m il_representations.scripts.pretrain_n_adapt with repl.condition_two_temporal_cpc_momentum\
        repl.stooke_contrastive_hyperparams_dmc repl.stooke_momentum_hyperparams_dmc


# Run set 4 -> Final run over different seeds of TemporalCPC with a symmetric projection head, over all environments
python -m il_representations.scripts.pretrain_n_adapt with repl.condition_three_temporal_cpc_sym_proj\
        repl.stooke_contrastive_hyperparams_dmc


# Run set 5 -> Final run over different seeds of TemporalCPC with an asymmetric projection head, over all environments
python -m il_representations.scripts.pretrain_n_adapt with repl.condition_four_temporal_cpc_asym_proj\
        repl.stooke_contrastive_hyperparams_dmc


# Run set 6 -> Final run over different seeds of TemporalCPC with augmentations, over all environments
# TODO - DO NOT RUN UNTIL hp_tuned_augmentation_set is filled in with the results of tuning Run Set #1
python -m il_representations.scripts.pretrain_n_adapt with repl.condition_five_temporal_cpc_augment_both\
        repl.stooke_contrastive_hyperparams_dmc repl.hp_tuned_augmentation_set


# Run set 7 -> Hyperparameter tuning over learning rate for VAE
python -m il_representations.scripts.pretrain_n_adapt with cfg_tune_vae_learning_rate repl.condition_ten_vae\
        repl.stooke_vae_hyperparams_dmc




###
# TODO DO NOT RUN Run Sets 8-13 until hp_tuned_vae_lr is filled in with the results of tuning Run Set #7
###

# Run set 8 -> Final run over different seeds of temporal VAE/AE with beta=0
python -m il_representations.scripts.pretrain_n_adapt with repl.condition_eight_temporal_autoencoder\
 repl.stooke_vae_hyperparams_dmc repl.hp_tuned_vae_lr


# Run set 9 -> Final run over different seeds of VAE/AE with beta=0
python -m il_representations.scripts.pretrain_n_adapt with repl.condition_nine_autoencoder\
 repl.stooke_vae_hyperparams_dmc repl.hp_tuned_vae_lr

# Run set 10 -> Final run over different seeds of VAE with beta=0.1
python -m il_representations.scripts.pretrain_n_adapt with repl.condition_ten_vae\
 repl.stooke_vae_hyperparams_dmc repl.hp_tuned_vae_lr

# Run set 11 -> Final run over different seeds of Temporal VAE with beta=0.1
python -m il_representations.scripts.pretrain_n_adapt with repl.condition_thirteen_temporal_vae_lowbeta\
 repl.stooke_vae_hyperparams_dmc repl.hp_tuned_vae_lr


# Run set 12 -> Final run over different seeds of Temporal VAE with beta=1.0
python -m il_representations.scripts.pretrain_n_adapt with repl.condition_fourteen_temporal_vae_highbeta\
 repl.stooke_vae_hyperparams_dmc repl.hp_tuned_vae_lr


# Run set 13 -> Final run over different seeds of AC Temporal VAE with beta=0.1
python -m il_representations.scripts.pretrain_n_adapt with repl.condition_eighteen_ac_temporal_vae_lowbeta\
 repl.stooke_vae_hyperparams_dmc repl.hp_tuned_vae_lr
