from collections import OrderedDict

from skopt.space import Categorical

from il_representations.algos import decoders, losses, pair_constructors
from il_representations.scripts.utils import StagesToRun


def make_icml_tuning_configs(experiment_obj):


    # Tuning Config #1: (Main Tuning)
    # Base: AC Temporal CPC
    # Axes to test: Batch size, learning rate, representation_dim,
    #               augmentation (Y/N), ActionConditioned (Y/N),
    #               projection head (None, Symmetric, Asymmetric),
    #               loss CEB/CPC

    # Note: This doesn't directly have momentum as an axis
    # because momentum requires that encoder, decoder, loss,
    # and batch_extender are all modified in unison, which
    # isn't ACAICT possible.

    # Similarly, this tests the presence or absence of a CEB loss
    # with default beta, but doesn't test over different betas,
    # since beta would be an invalid loss parameter for the normal
    # CPC loss
    @experiment_obj.named_config
    def icml_tuning():
        use_skopt = True
        skopt_search_mode = 'max'
        metric = 'return_mean'
        stages_to_run = StagesToRun.REPL_AND_IL
        repl = {
            'batches_per_epoch': 500,
            'n_epochs': 10,
        }
        _ = locals()
        del _

    @experiment_obj.named_config
    def main_contrastive_tuning():
        tune_run_kwargs = dict(num_samples=50)
        skopt_space = OrderedDict([
                        ('repl:algo_params:batch_size', (256, 512)),
                        ('repl:optimizer_kwargs:lr', (1e-6, 1e-2, 'log-uniform')),
                        ('repl:algo_params:representation_dim', (64, 256)),
                        ('repl:algo', Categorical(['TemporalCPC',
                                                   'ActionConditionedTemporalCPC'],
                                                  prior=[0.2, 0.8])),

                        ('repl:algo_params:loss_calculator', Categorical([losses.CEBLoss,
                                                                          losses.BatchAsymmetricContrastiveLoss],
                                                                         prior=[0.2, 0.8]),),
                        ('repl:algo_params:augmenter_kwargs:augmenter_spec', Categorical(["translate,rotate,gaussian_blur",
                                                                                          "translate,rotate",
                                                                                          "translate,rotate,flip_ud,flip_lr"]))
                        ])
        skopt_ref_configs = [
                {'repl:algo_params:batch_size': 256,
                 'repl:optimizer_kwargs:lr': 0.0003,
                 'repl:algo_params:representation_dim': 128,
                 'repl:algo': 'ActionConditionedTemporalCPC',
                 'repl:algo_params:loss_calculator': losses.BatchAsymmetricContrastiveLoss,
                 'repl:algo_params:augmenter_kwargs:augmenter_spec': "translate,rotate",
                    }]

        _ = locals()
        del _

    # Tuning Config #2
    # This config tests features of momentum: queue size, momentum weight, and batch size
    # (Testing batch size separately because it has a different impact in a momentum context)
    @experiment_obj.named_config
    def tune_momentum():
        repl = {'algo': 'MoCo'}
        tune_run_kwargs = dict(num_samples=25)
        skopt_space = OrderedDict([
            ('repl:algo_params:batch_size', (128, 512)),
            ('repl:algo_params:batch_extender_kwargs:queue_size', (1024, 8192)),
            ('repl:algo_params:encoder_kwargs:momentum_weight', (0.985, 0.999)) #,
            #('il_test:n_rollouts', [20])
        ])
        skopt_ref_configs = [
            {'repl:algo_params:batch_size': 256,
             'repl:algo_params:batch_extender_kwargs:queue_size': 8192,
             'repl:algo_params:encoder_kwargs:momentum_weight': 0.999,
             #'il_test:n_rollouts': 20
             }]

        _ = locals()
        del _

    # Tuning Config #3
    # This config tests different beta parameters of Temporal CEB
    @experiment_obj.named_config
    def tune_ceb():
        repl = {'algo': 'TemporalCPC',
                'algo_params': {'loss_calculator': losses.CEBLoss}}
        tune_run_kwargs = dict(num_samples=15)
        skopt_space = OrderedDict([
            ('repl:algo_params:loss_calculator_kwargs:beta', (1e-10, 0.25, 'log-uniform')),
            # ('il_test:n_rollouts', [20])
        ])
        skopt_ref_configs = [
            {'repl:algo_params:loss_calculator_kwargs:beta': 0.1,
             # 'il_test:n_rollouts': 20
             }]

        _ = locals()
        del _

    # Tuning Config #4
    # This config tests different beta parameters of VAE
    @experiment_obj.named_config
    def tune_vae():
        repl = {'algo': 'VariationalAutoencoder',
                'algo_params': {'batch_size': 64}}
        tune_run_kwargs = dict(num_samples=15)
        skopt_space = OrderedDict([
            ('repl:algo_params:loss_calculator_kwargs:beta', (1e-10, 0.1, 'log-uniform')),
            # ('il_test:n_rollouts', [20])
        ])
        skopt_ref_configs = [
            {'repl:algo_params:loss_calculator_kwargs:beta': 1e-5,
             # 'il_test:n_rollouts': 20
             }]

        _ = locals()
        del _

    # Tuning Config #5
    # This config tests different projection decoders. It can't be done
    # with ActionConditionedTemporalCPC as a base without writing action-conditioned
    # projection heads for each of the projection head types, which
    # probably doesn't make sense unless they actually seem valuable
    @experiment_obj.named_config
    def tune_projection_heads():
        repl = {'algo': 'TemporalCPC',
                'algo_params': {'target_pair_constructor': pair_constructors.IdentityPairConstructor}
                }
        tune_run_kwargs = dict(num_samples=15)
        skopt_space = OrderedDict([
            ('repl:algo_params:decoder', Categorical([decoders.NoOp,
                                                      decoders.SymmetricProjectionHead,
                                                      decoders.AsymmetricProjectionHead],
                                                     prior=[0.5, 0.25, 0.25])),
            # ('il_test:n_rollouts', [20])
        ])
        skopt_ref_configs = [
            {'repl:algo_params:decoder': decoders.NoOp,
             # 'il_test:n_rollouts': 20
             }]

        _ = locals()
        del _



