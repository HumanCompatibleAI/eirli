## TemporalCPC - commented out if we don't want to re-run
#python -m il_representations.scripts.pretrain_n_adapt with cfg_use_magical\
#repl.stooke_contrastive_hyperparams_dmc repl.condition_one_temporal_cpc

for seed_var in {1..3}:
do
# TemporalCPC with augmentations
  python -m il_representations.scripts.pretrain_n_adapt with cfg_use_magical\
  repl.stooke_contrastive_hyperparams_dmc repl.temporal_cpc_augment_both_magical seed=$seed_var

  # TemporalCPC with Action Conditioning
  python -m il_representations.scripts.pretrain_n_adapt with cfg_use_magical\
  repl.stooke_contrastive_hyperparams_dmc repl.algo=ActionConditionedTemporalCPC\
   repl.use_random_rollouts=False seed=$seed_var


   # Regular/Identity CPC
  python -m il_representations.scripts.pretrain_n_adapt with cfg_use_magical\
  repl.stooke_contrastive_hyperparams_dmc repl.identity_cpc seed=$seed_var


   # CEB
  python -m il_representations.scripts.pretrain_n_adapt with cfg_use_magical\
  repl.stooke_contrastive_hyperparams_dmc repl.temporal_ceb_no_projection seed=$seed_var


  # TemporalVAE
  python -m il_representations.scripts.pretrain_n_adapt with cfg_use_magical\
  repl.condition_thirteen_temporal_vae_lowbeta seed=$seed_var


  # VAE
  python -m il_representations.scripts.pretrain_n_adapt with cfg_use_magical\
  repl.condition_ten_vae seed=$seed_var

done