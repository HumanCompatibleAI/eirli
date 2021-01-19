from il_representations.algos import augmenters, pair_constructors, encoders, losses, batch_extenders, decoders
from collections import OrderedDict
from copy import deepcopy
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
            'batches_per_epoch': 5000,
        }
        il_test = {'num_rollouts': 20}
        _ = locals()
        del _

    @experiment_obj.named_config
    def main_contrastive_tuning():
        tune_run_kwargs = dict(num_samples=50)
        skopt_space = OrderedDict([
                        ('repl:algo_params:batch_size', (256, 512)),
                        ('repl:algo_params:optimizer_kwargs:lr', (1e-6, 1e-2, 'log-uniform')),
                        ('repl:algo_params:representation_dim', (64, 256)),
                        ('repl:algo_params:augmenter', [augmenters.AugmentContextAndTarget,
                                                        augmenters.NoAugmentation]),
                        ('repl:algo', ['TemporalCPC', 'ActionConditionedTemporalCPC']),
                        ('repl:algo_params:decoder', [decoders.NoOp,
                                                      decoders.SymmetricProjectionHead,
                                                      decoders.AsymmetricProjectionHead]),
                        ('repl:algo_params:loss_calculator', [losses.CEBLoss,
                                                              losses.BatchAsymmetricContrastiveLoss])
                        ])
        skopt_ref_configs = [
                {'repl:algo_params:batch_size': 256,
                 'repl:algo_params:optimizer_kwargs:lr': 0.0003,
                 'repl:algo_params:representation_dim': 128,
                 'repl:algo_params:augmenter': augmenters.AugmentContextAndTarget,
                 'repl:algo': 'ActionConditionedTemporalCPC',
                 'repl:algo_params:decoder': decoders.NoOp,
                 'repl:algo_params:loss_calculator': losses.BatchAsymmetricContrastiveLoss
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
            ('repl:algo_params:encoder_kwargs:momentum_weight', (0.985, 0.999))
        ])
        skopt_ref_configs = [
            {'repl:algo_params:batch_size': 256,
             'repl:algo_params:batch_extender_kwargs:queue_size': 8192,
             'repl:algo_params:encoder_kwargs:momentum_weight': 0.999
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
        ])
        skopt_ref_configs = [
            {'repl:algo_params:loss_calculator_kwargs:': 0.1,
             }]

        _ = locals()
        del _

    # Tuning Config #4
    # This config tests different beta parameters of Temporal CEB
    @experiment_obj.named_config
    def tune_vae():
        repl = {'algo': 'VariationalAutoencoder'}
        tune_run_kwargs = dict(num_samples=15)
        skopt_space = OrderedDict([
            ('repl:algo_params:loss_calculator_kwargs:beta', (0.0, 0.2, 'log-uniform')),
        ])
        skopt_ref_configs = [
            {'repl:algo_params:loss_calculator_kwargs:': 0.01,
             }]

        _ = locals()
        del _
