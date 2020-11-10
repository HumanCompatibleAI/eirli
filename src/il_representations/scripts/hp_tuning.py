from il_representations.algos import augmenters, pair_constructors, encoders, losses, batch_extenders
import collections


BASE_SKOPT_SPACE = collections.OrderedDict([
        ('repl:algo_params:representation_dim', (8, 512)),
        ('repl:algo_params:batch_size', (64, 1024)),
        ('repl:algo_params:optimizer_kwargs:lr', (1e-6, 1e-2, 'log-uniform')),
        ('repl:algo_params:encoder_kwargs:obs_encoder_cls', ['BasicCNN',
                                                             'MAGICALCNN']),
        ('il_train:freeze_encoder', [True, False]),
        ('il_test:n_rollouts', [10])
])

BASE_REF_CONFIGS = [
        {'repl:algo_params:batch_size': 256,
         'repl:algo_params:optimizer_kwargs:lr': 0.0003,
         'repl:algo_params:representation_dim': 128,
         'repl:algo_params:encoder_kwargs:obs_encoder_cls': 'BasicCNN',
         'repl:algo_params:augmenter_kwargs:augmenter_spec': "translate,rotate,gaussian_blur",
         'il_train:freeze_encoder':  True,
         'il_test:n_rollouts':  50
       }]


def make_hp_tuning_configs(experiment_obj):

    @experiment_obj.named_config
    def tuning():
        use_skopt = True
        skopt_search_mode = 'max'
        metric = 'return_mean'
        stages_to_run = "REPL_AND_IL"
        repl = {
            'use_random_rollouts': False,
            'ppo_finetune': False,
            # this isn't a lot of training, but should be enough to tell whether
            # loss goes down quickly
            'pretrain_batches': 5,  # TODO testing value, change
            'pretrain_epochs': None # TODO testing value, change
        }
        tune_run_kwargs = dict(num_samples=2) # TODO testing value, change
        _ = locals()
        del _


    @experiment_obj.named_config
    def temporal_cpc_tune():
        repl = {'algo': 'TemporalCPC'}
        skopt_space = BASE_SKOPT_SPACE
        skopt_ref_configs = BASE_REF_CONFIGS
        _ = locals()
        del _


    @experiment_obj.named_config
    def temporal_cpc_aug_tune():
        repl = {'algo': 'TemporalCPC',
                'algo_params': {'augmenter': augmenters.AugmentContextAndTarget}
                }

        skopt_space = BASE_SKOPT_SPACE
        skopt_space[
            'repl:algo_params:augmenter_kwargs:augmenter_spec'] = ["translate,rotate,gaussian_blur",
                                                                   "translate,rotate",
                                                                   "translate,rotate,flip_ud,flip_lr"]
        skopt_ref_configs = BASE_REF_CONFIGS
        skopt_ref_configs[0][
            'repl:algo_params:augmenter_kwargs:augmenter_spec'] = "translate,rotate,gaussian_blur"
        _ = locals()
        del _


    @experiment_obj.named_config
    def ac_temporal_cpc_tune():
        repl = {'algo': 'ActionConditionedTemporalCPC'}
        skopt_space = BASE_SKOPT_SPACE
        skopt_ref_configs = BASE_REF_CONFIGS
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
        skopt_space = BASE_SKOPT_SPACE
        skopt_ref_configs = BASE_REF_CONFIGS
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
        skopt_space = BASE_SKOPT_SPACE
        skopt_ref_configs = BASE_REF_CONFIGS
        skopt_space['repl:algo_params:loss_constructor_kwargs:beta'] = (0, 0.1)
        skopt_ref_configs[0]['repl:algo_params:loss_constructor_kwargs:beta'] = 0.01
        _ = locals()
        del _


    @experiment_obj.named_config
    def temporal_ceb_fixed_variance_tune():
        repl = {'algo': 'FixedVarianceCEB'}
        skopt_space = BASE_SKOPT_SPACE
        skopt_ref_configs = BASE_REF_CONFIGS
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
        skopt_space = BASE_SKOPT_SPACE
        skopt_ref_configs = BASE_REF_CONFIGS
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
    def vae_tune():
        repl = {'algo': 'VAE'}
        skopt_space = BASE_SKOPT_SPACE
        skopt_ref_configs = BASE_REF_CONFIGS
        skopt_space['repl:algo_params:loss_constructor_kwargs:beta'] = (0, 1)
        skopt_ref_configs[0]['repl:algo_params:loss_constructor_kwargs:beta'] = 0.01
        _ = locals()
        del _

    @experiment_obj.named_config
    def temporal_cpc_momentum_tune():
        repl = {'algo': 'TemporalCPC',
                'algo_params': {'batch_extender': batch_extenders.QueueBatchExtender,
                                'encoder': encoders.MomentumEncoder,
                                'loss_calculator': losses.QueueAsymmetricContrastiveLoss}}
        skopt_space = BASE_SKOPT_SPACE
        skopt_ref_configs = BASE_REF_CONFIGS
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

    @experiment_obj.named_config
    def dynamics_tune():
        repl = {'algo': 'DynamicsPrediction'}
        skopt_space = BASE_SKOPT_SPACE
        skopt_ref_configs = BASE_REF_CONFIGS
        skopt_space['repl:algo_params:encoder_kwargs:action_encoding_dim'] = (8, 128)
        skopt_space['repl:algo_params:encoder_kwargs:action_embedding_dim'] = (5, 30)
        skopt_ref_configs[0]['repl:algo_params:decoder_kwargs:action_encoding_dim'] = 64
        skopt_ref_configs[0]['repl:algo_params:decoder_kwargs:action_embedding_dim'] = 5
        _ = locals()
        del _

    @experiment_obj.named_config
    def inverse_dynamics_tune():
        repl = {'algo': 'InverseDynamicsPrediction'}
        skopt_space = BASE_SKOPT_SPACE
        skopt_ref_configs = BASE_REF_CONFIGS
        _ = locals()
        del _


