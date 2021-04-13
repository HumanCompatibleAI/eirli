from il_representations.algos import batch_extenders, encoders, losses, decoders, augmenters, pair_constructors
from il_representations.scripts.run_rep_learner import represent_ex
from torch.optim.lr_scheduler import CosineAnnealingLR

@represent_ex.named_config
def condition_one_temporal_cpc():
    # Baseline Temporal CPC with expert demonstrations
    algo = 'TemporalCPC'
    _ = locals()
    del _


@represent_ex.named_config
def condition_two_temporal_cpc_momentum():
    # Baseline Temporal CPC with momentum added
    algo = 'TemporalCPC'
    algo_params = {
        'batch_extender': batch_extenders.QueueBatchExtender,
        'encoder': encoders.MomentumEncoder,
        'loss_calculator': losses.QueueAsymmetricContrastiveLoss
    }
    _ = locals()
    del _


@represent_ex.named_config
def condition_three_temporal_cpc_sym_proj():
    # Baseline Temporal CPC with a symmetric projection head
    algo = 'TemporalCPC'
    algo_params = {'decoder': decoders.SymmetricProjectionHead}
    _ = locals()
    del _


@represent_ex.named_config
def condition_four_temporal_cpc_asym_proj():
    # Baseline Temporal CPC with an asymmetric projection head
    algo = 'TemporalCPC'
    algo_params = {'decoder': decoders.AsymmetricProjectionHead}
    _ = locals()
    del _


@represent_ex.named_config
def condition_five_temporal_cpc_augment_both():
    # Baseline Temporal CPC with augmentation of both context and target
    algo = 'TemporalCPC'
    algo_params = {'augmenter': augmenters.AugmentContextAndTarget}
    _ = locals()
    del _


@represent_ex.named_config
def condition_eight_temporal_autoencoder():
    # A variational autoencoder with weight on KLD loss set to 0, and temporal offset
    # between encoded image and target image
    algo = 'VariationalAutoencoder'
    algo_params = {
            'target_pair_constructor': pair_constructors.TemporalOffsetPairConstructor,
            'loss_calculator_kwargs': {'beta': 0}}
    _ = locals()
    del _



@represent_ex.named_config
def condition_nine_autoencoder():
    # A variational autoencoder with weight on KLD loss set to 0
    algo = 'VariationalAutoencoder'
    algo_params = {
            'loss_calculator_kwargs': {'beta': 0},
                    }
    _ = locals()
    del _


@represent_ex.named_config
def condition_ten_vae():
    # A variational autoencoder with weight on KLD loss set to 1.0
    algo = 'VariationalAutoencoder'
    algo_params = {
        'loss_calculator_kwargs': {'beta': 0.1}} # TODO What is a good default beta here?
    _ = locals()
    del _


@represent_ex.named_config
def condition_thirteen_temporal_vae_lowbeta():
    # A variational autoencoder with weight on KLD loss set to 0.01, and temporal offset
    # between encoded image and target image
    algo = 'VariationalAutoencoder'
    algo_params = {'loss_calculator_kwargs': {'beta': 0.1},
                   'target_pair_constructor': pair_constructors.TemporalOffsetPairConstructor,
                   }
    _ = locals()
    del _

@represent_ex.named_config
def condition_fourteen_temporal_vae_highbeta():
    # A variational autoencoder with weight on KLD loss set to 1.0, and temporal offset
    # between encoded image and target image
    algo = 'VariationalAutoencoder'
    algo_params = {'loss_calculator_kwargs': {'beta': 1.0},
                   'target_pair_constructor': pair_constructors.TemporalOffsetPairConstructor,
                   }
    _ = locals()
    del _


@represent_ex.named_config
def condition_eighteen_ac_temporal_vae_lowbeta():
    # An action-conditioned variational autoencoder with weight on KLD loss set to 0.1, and temporal offset
    # between encoded image and target image
    algo = 'ActionConditionedTemporalVAE'
    algo_params = {'loss_calculator_kwargs': {'beta': 0.1}}
    _ = locals()
    del _


@represent_ex.named_config
def stooke_contrastive_hyperparams_dmc():
    # To be used with contrastive algorithms that do not use momentum
    # Currently doesn't contain CosineAnnealing, despite being in Stooke,
    # because it's too hard to debug last minute
    # TODO add in augmentations once determined via hyperparameter search
    algo_params = {'representation_dim': 128,
                   'batch_size': 256,
                   'optimizer_kwargs': {'lr': 0.001}}
    _ = locals()
    del _


@represent_ex.named_config
# TODO CURRENTLY NONSENSE, REPLACE AFTER HP TUNING
def hp_tuned_augmentation_set():
    algo_params = {'augmenter_kwargs': {'augmenter_spec': 'REPLACE-ME'}}
    _ = locals()
    del _


@represent_ex.named_config
# TODO CURRENTLY NONSENSE, REPLACE AFTER HP TUNING
def hp_tuned_vae_lr():
    algo_params = {'optimizer_kwargs': {'lr': 'REPLACE-ME'}}
    _ = locals()
    del _

@represent_ex.named_config
def stooke_momentum_hyperparams_dmc():
    # To be added to stooke_contrastive_hyperparams_dmc for algorithms that use momentum
    algo_params = {'encoder_kwargs': {'momentum_weight': 0.95}}
    _ = locals()
    del _


@represent_ex.named_config
def cody_momentum_bn():
    algo_params = {
        'encoder_kwargs': {
            'momentum_weight': 0.995,
        },
        'loss_calculator_kwargs': {
            'use_batch_neg': True,
        }
    }
    _ = locals()
    del _

@represent_ex.named_config
# TODO add learning rate once determined by hyperparameter search
def stooke_vae_hyperparams_dmc():
    algo_params = {'batch_size': 128}
    _ = locals()
    del _



@represent_ex.named_config
def identity_cpc():
    algo = 'TemporalCPC'
    algo_params = {'target_pair_constructor': pair_constructors.IdentityPairConstructor}
    _ = locals()
    del _


@represent_ex.named_config
def identity_cpc_aug():
    algo = 'TemporalCPC'
    algo_params = {'target_pair_constructor': pair_constructors.IdentityPairConstructor,
                   'augmenter': augmenters.AugmentContextAndTarget,
                   'augmenter_kwargs': {'augmenter_spec': "translate,rotate"}}
    _ = locals()
    del _


@represent_ex.named_config
def temporal_ceb_no_projection():
    algo = 'TemporalCPC'
    algo_params = {'loss_calculator': losses.CEBLoss}
    _ = locals()
    del _


@represent_ex.named_config
def temporal_ceb_no_projection_beta0():
    algo = 'TemporalCPC'
    algo_params = {'loss_calculator': losses.CEBLoss,
                   'loss_calculator_kwargs': {'beta': 0.0}}
    _ = locals()
    del _


@represent_ex.named_config
def temporal_cpc_augment_both_magical():
    # Baseline Temporal CPC with augmentation of both context and target
    algo = 'TemporalCPC'
    algo_params = {'augmenter': augmenters.AugmentContextAndTarget,
                   'augmenter_kwargs': {'augmenter_spec': "translate,rotate"}}
    _ = locals()
    del _


@represent_ex.named_config
def temporal_cpc_action_conditioned_augment_both():
    # Action-conditioned Temporal CPC with augmentation of both context and
    # target
    algo = 'ActionConditionedTemporalCPC'
    algo_params = {'augmenter': augmenters.AugmentContextAndTarget,
                   'augmenter_kwargs': {'augmenter_spec': "translate,rotate"}}
    _ = locals()
    del _


@represent_ex.named_config
def temporal_ceb_action_conditioned_augment_both():
    # Action-conditioned Temporal CPC with augmentation of both context and
    # target
    algo = 'ActionConditionedTemporalCPC'
    algo_params = {'augmenter': augmenters.AugmentContextAndTarget,
                   'augmenter_kwargs': {'augmenter_spec': "translate,rotate"},
                   'loss_calculator': losses.CEBLoss}
    _ = locals()
    del _


@represent_ex.named_config
def identity_cpc_action_conditioned_augment_both():
    # Action-conditioned CPC with augmentation of both context and target
    algo = 'ActionConditionedTemporalCPC'
    # We use TemporalOffsetPairConstructor instead of IdentityPairConstructor
    # because the former lets us set mode='dynamics', which includes actions in
    # the extra_context.
    algo_params = {'target_pair_constructor': pair_constructors.TemporalOffsetPairConstructor,
                   'target_pair_constructor_kwargs': dict(mode='dynamics', temporal_offset=0),
                   'augmenter': augmenters.AugmentContextAndTarget,
                   'augmenter_kwargs': {'augmenter_spec': "translate,rotate"}}
    _ = locals()
    del _
