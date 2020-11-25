from il_representations.algos import augmenters, pair_constructors, encoders, losses, batch_extenders
import collections
from copy import deepcopy


def get_space_and_ref_configs(contrastive=True):
    if contrastive:
        space = collections.OrderedDict([
                ('repl:algo_params:batch_size', (64, 512)),
                ('repl:algo_params:optimizer_kwargs:lr', (1e-6, 1e-2, 'log-uniform')),
                ('repl:algo_params:representation_dim', (8, 256)),
                ('repl:algo_params:encoder_kwargs:obs_encoder_cls', ['BasicCNN',
                                                                     'MAGICALCNN']),
                ('il_train:freeze_encoder', [True, False]),
                ('il_test:n_rollouts', [20])
                                                ])
        base_refs = [
                {'repl:algo_params:batch_size': 256,
                 'repl:algo_params:optimizer_kwargs:lr': 0.0003,
                 'repl:algo_params:representation_dim': 128,
                 'repl:algo_params:encoder_kwargs:obs_encoder_cls': 'BasicCNN',
                 'il_train:freeze_encoder':  True,
                 'il_test:n_rollouts': 20
                    }]
    else:
        # Do not include batch size in search space
        space = collections.OrderedDict([
                ('repl:algo_params:representation_dim', (8, 256)),
                ('repl:algo_params:optimizer_kwargs:lr', (1e-6, 1e-2, 'log-uniform')),
                ('repl:algo_params:encoder_kwargs:obs_encoder_cls', ['BasicCNN',
                                                                     'MAGICALCNN']),
                ('il_train:freeze_encoder', [True, False]),
                ('il_test:n_rollouts', [20])
                ])
        base_refs = [
                {'repl:algo_params:representation_dim': 128,
                 'repl:algo_params:optimizer_kwargs:lr': 0.0003,
                 'repl:algo_params:encoder_kwargs:obs_encoder_cls': 'BasicCNN',
                 'il_train:freeze_encoder':  True,
                 'il_test:n_rollouts': 20
               }]

    return space, base_refs


def make_hp_tuning_configs(experiment_obj):
    @experiment_obj.named_config
    def tuning():
        use_skopt = True
        skopt_search_mode = 'max'
        metric = 'return_mean'
        stages_to_run = "REPL_AND_IL"
        repl = {
            # this isn't a lot of training, but should be enough to tell whether
            # loss goes down quickly
            'batches_per_epoch': 1000,
            'n_epochs': 10,
        }
        tune_run_kwargs = dict(num_samples=50)
        _ = locals()
        del _

    @experiment_obj.named_config
    def temporal_cpc_tune():
        repl = {'algo': 'TemporalCPC'}
        skopt_space, skopt_ref_configs = get_space_and_ref_configs()
        _ = locals()
        del _

    @experiment_obj.named_config
    def temporal_cpc_aug_tune():
        repl = {'algo': 'TemporalCPC',
                'algo_params': {'augmenter': augmenters.AugmentContextAndTarget}
                }

        skopt_space, skopt_ref_configs = get_space_and_ref_configs()
        skopt_space[
            'repl:algo_params:augmenter_kwargs:augmenter_spec'] = ["translate,rotate,gaussian_blur",
                                                                   "translate,rotate",
                                                                   "translate,rotate,flip_ud,flip_lr"]
        skopt_ref_configs[0][
            'repl:algo_params:augmenter_kwargs:augmenter_spec'] = "translate,rotate,gaussian_blur"
        _ = locals()
        del _


    @experiment_obj.named_config
    def ac_temporal_cpc_tune():
        repl = {'algo': 'ActionConditionedTemporalCPC'}
        skopt_space, skopt_ref_configs = get_space_and_ref_configs()

        skopt_space['repl:algo_params:decoder_kwargs:action_encoding_dim'] = (64, 512)
        skopt_space['repl:algo_params:decoder_kwargs:action_embedding_dim'] = (5, 30)

        skopt_ref_configs[0]['repl:algo_params:decoder_kwargs:action_encoding_dim'] = 64
        skopt_ref_configs[0]['repl:algo_params:decoder_kwargs:action_embedding_dim'] = 5
        _ = locals()
        del _

    @experiment_obj.named_config
    def identity_cpc_aug_tune():
        repl = {'algo': 'TemporalCPC',
                'algo_params': {'target_pair_constructor': pair_constructors.IdentityPairConstructor,
                                'augmenter': augmenters.AugmentContextAndTarget}
                }
        skopt_space, skopt_ref_configs = get_space_and_ref_configs()
        skopt_space[
            'repl:algo_params:augmenter_kwargs:augmenter_spec'] = ["translate,rotate,gaussian_blur",
                                                                   "translate,rotate",
                                                                   "translate,rotate,flip_ud,flip_lr"]
        skopt_ref_configs[0][
            'repl:algo_params:augmenter_kwargs:augmenter_spec'] = "translate,rotate,gaussian_blur"

        _ = locals()
        del _

    @experiment_obj.named_config
    def temporal_ceb_tune():
        repl = {'algo': 'CEB'}
        skopt_space, skopt_ref_configs = get_space_and_ref_configs()
        skopt_space['repl:algo_params:loss_constructor_kwargs:beta'] = (0, 0.1)
        skopt_ref_configs[0]['repl:algo_params:loss_constructor_kwargs:beta'] = 0.01
        _ = locals()
        del _


    @experiment_obj.named_config
    def temporal_ceb_fixed_variance_tune():
        repl = {'algo': 'FixedVarianceCEB'}
        skopt_space, skopt_ref_configs = get_space_and_ref_configs()
        skopt_space['repl:algo_params:loss_constructor_kwargs:beta'] = (0, 0.1)
        skopt_ref_configs[0]['repl:algo_params:loss_constructor_kwargs:beta'] = 0.01
        _ = locals()
        del _

    @experiment_obj.named_config
    def identity_ceb_aug_tune():
        repl = {'algo': 'CEB',
                'algo_params': {'target_pair_constructor': pair_constructors.IdentityPairConstructor,
                                'augmenter': augmenters.AugmentContextAndTarget}
                }
        skopt_space, skopt_ref_configs = get_space_and_ref_configs()
        skopt_space['repl:algo_params:loss_constructor_kwargs:beta'] = (0, 0.1)
        skopt_space[
            'repl:algo_params:augmenter_kwargs:augmenter_spec'] = ["translate,rotate,gaussian_blur",
                                                                   "translate,rotate",
                                                                   "translate,rotate,flip_ud,flip_lr"]

        skopt_ref_configs[0]['repl:algo_params:loss_constructor_kwargs:beta'] = 0.01
        skopt_ref_configs[0][
            'repl:algo_params:augmenter_kwargs:augmenter_spec'] = "translate,rotate,gaussian_blur"
        _ = locals()
        del _

    @experiment_obj.named_config
    def temporal_cpc_momentum_tune():
        repl = {'algo': 'TemporalCPC',
                'algo_params': {'batch_extender': batch_extenders.QueueBatchExtender,
                                'encoder': encoders.MomentumEncoder,
                                'loss_calculator': losses.QueueAsymmetricContrastiveLoss}}
        skopt_space, skopt_ref_configs = get_space_and_ref_configs()
        skopt_space[
            'repl:algo_params:augmenter_kwargs:augmenter_spec'] = ["translate,rotate,gaussian_blur",
                                                                   "translate,rotate",
                                                                   "translate,rotate,flip_ud,flip_lr"]
        skopt_space['repl:algo_params:encoder_kwargs:momentum_weight'] = (0.99, 1.0)
        skopt_ref_configs[0][
            'repl:algo_params:augmenter_kwargs:augmenter_spec'] = "translate,rotate,gaussian_blur"
        skopt_ref_configs[0]['repl:algo_params:encoder_kwargs:momentum_weight'] = 0.995
        _ = locals()
        del _


    ### Non-Contrastive ###

    @experiment_obj.named_config
    def vae_tune():
        repl = {'algo': 'VariationalAutoencoder', 'algo_params': {'batch_size': 64}}
        skopt_space, skopt_ref_configs = get_space_and_ref_configs(contrastive=False)
        skopt_space['repl:algo_params:loss_constructor_kwargs:beta'] = (0, 1)
        skopt_ref_configs[0]['repl:algo_params:loss_constructor_kwargs:beta'] = 0.01
        _ = locals()
        del _

    @experiment_obj.named_config
    def dynamics_tune():
        repl = {'algo': 'DynamicsPrediction', 'algo_params': {'batch_size': 64}}
        skopt_space, skopt_ref_configs = get_space_and_ref_configs(contrastive=False)
        skopt_space['repl:algo_params:encoder_kwargs:action_encoding_dim'] = (8, 128)
        skopt_space['repl:algo_params:encoder_kwargs:action_embedding_dim'] = (5, 30)
        skopt_ref_configs[0]['repl:algo_params:decoder_kwargs:action_encoding_dim'] = 64
        skopt_ref_configs[0]['repl:algo_params:decoder_kwargs:action_embedding_dim'] = 5
        _ = locals()
        del _

    @experiment_obj.named_config
    def inverse_dynamics_tune():
        repl = {'algo': 'InverseDynamicsPrediction', 'algo_params': {'batch_size': 64}}
        skopt_space, skopt_ref_configs = get_space_and_ref_configs(contrastive=False)
        _ = locals()
        del _


