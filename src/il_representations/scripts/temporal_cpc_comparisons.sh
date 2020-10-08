# TemporalCPC
python -m il_representations.scripts.pretrain_n_adapt with cfg_use_magical\
repl.stooke_contrastive_hyperparams_dmc repl.condition_one_temporal_cpc


# TemporalCPC with Action Conditioning
python -m il_representations.scripts.pretrain_n_adapt with cfg_use_magical\
repl.stooke_contrastive_hyperparams_dmc repl.algo=ActionConditionedTemporalCPC\
 repl.use_random_rollouts=False



 # Regular/Identity CPC
python -m il_representations.scripts.pretrain_n_adapt with cfg_use_magical\
repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc


 # CEB
python -m il_representations.scripts.pretrain_n_adapt with cfg_use_magical\
repl.stooke_contrastive_hyperparams_dmc repl.temporal_ceb_no_projection


# TemporalVAE
python -m il_representations.scripts.pretrain_n_adapt with cfg_use_magical\
repl.stooke_contrastive_hyperparams_dmc repl.condition_thirteen_temporal_vae_lowbeta


# VAE
python -m il_representations.scripts.pretrain_n_adapt with cfg_use_magical\
repl.stooke_contrastive_hyperparams_dmc repl.condition_ten_vae