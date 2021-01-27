from ray import tune
from il_representations.scripts.utils import StagesToRun, ReuseRepl, partial_repl_class
from il_representations.algos import (ActionConditionedTemporalCPC, TemporalCPC,
                                      VariationalAutoencoder, DynamicsPrediction,
                                      InverseDynamicsPrediction, GaussianPrior)
from il_representations.algos import augmenters, pair_constructors, encoders, losses, batch_extenders, decoders

# This set of configs is meant to correspond to Hypothesis #3 in Test Conditions spreadsheet
# It implements:
# For each benchmark:
#   For each environment:
#       For each dataset structure:
#           if MAGICAL:
                # train k seeds of RepL for each algo, then subsequently k*j seeds of IL on those pretrained RepL
#           if DMC:
#               # train k*j seeds of RepL+IL without reuse
#

contrastive_kwargs_standin = {}
ICMLActionConditionedTemporalCPC = partial_repl_class(ActionConditionedTemporalCPC,
                                                      new_class_name="ICMLActionConditionedTemporalCPC",
                                                      **contrastive_kwargs_standin)
ICMLIdentityCPC = partial_repl_class(TemporalCPC,
                                     new_class_name="ICMLIdentityCPC",
                                     target_pair_constructor=pair_constructors.IdentityPairConstructor,
                                     **contrastive_kwargs_standin)
best_hp_vae_beta = 0 # fill in
ICMLVariationalAutoencoder = partial_repl_class(VariationalAutoencoder,
                                                new_class_name="ICMLVariationalAutoencoder",
                                                loss_calculator_kwargs={'beta': best_hp_vae_beta})

MAGICAL_MULTITASK_DEMOS_CONFIG = [
                {
                    'type': 'demos',
                    'env_cfg': {
                        'benchmark_name': 'magical',
                        'task_name': magical_task_name,
                    }
                } for magical_task_name in [
                    'MoveToCorner',
                    'MoveToRegion',
                    'MatchRegions',
                    'MakeLine',
                    'FixColour',
                    'FindDupe',
                    'ClusterColour',
                    'ClusterShape',
                ]
            ]

MAGICAL_MULTITASK_RAND_DEMOS_CONFIG = [
            [
                {
                    'type': dataset_type,
                    'env_cfg': {
                        'benchmark_name': 'magical',
                        'task_name': magical_task_name,
                    }
                } for magical_task_name in [
                    'MoveToCorner',
                    'MoveToRegion',
                    'MatchRegions',
                    'MakeLine',
                    'FixColour',
                    'FindDupe',
                    'ClusterColour',
                    'ClusterShape',
                ]
            ]
            for dataset_type in ["demos", "random"]
    ]


RAND_DEMOS_CONFIG = [{'type': 'demos'}, {'type': 'random'}]
RAND_ONLY_CONFIG = [{'type': 'random'}]
NUM_SEEDS = 9



def make_dataset_experiment_configs(experiment_obj):
    @experiment_obj.named_config
    def dataset_sweep_with_multitask():
        # TODO
        spec = {'repl.dataset_configs': tune.grid_search([RAND_DEMOS_CONFIG,
                                                          MAGICAL_MULTITASK_RAND_DEMOS_CONFIG,
                                                          MAGICAL_MULTITASK_DEMOS_CONFIG,
                                                          RAND_ONLY_CONFIG])}
        _ = locals()
        del _

    @experiment_obj.named_config
    def dataset_sweep_no_multitask():
        spec = {'repl.dataset_configs': tune.grid_search([RAND_DEMOS_CONFIG,
                                                          RAND_ONLY_CONFIG])}
        _ = locals()
        del _

    @experiment_obj.named_config
    def algo_sweep():
        spec = dict(repl=tune.grid_search([{'algo': algo} for algo in [InverseDynamicsPrediction, DynamicsPrediction,
                              ICMLIdentityCPC, ICMLActionConditionedTemporalCPC,
                              GaussianPrior, ICMLVariationalAutoencoder]]))
        _ = locals()
        del _

    # To be used for DMC, where Sam said we might as well retrain RepL for each new IL seed
    @experiment_obj.named_config
    def il_on_repl_sweep():
        repl = {'batches_per_epoch': 500,
                'n_epochs': 10}
        stages_to_run = StagesToRun.REPL_AND_IL
        reuse_repl = ReuseRepl.NO
        tune_run_kwargs = dict(num_samples=NUM_SEEDS)
        _ = locals()
        del _



# Notional shell invocations:
# MAGICAL:
# Currently, this will do a new Multitask Repl run for each
#         # MAGICAL environment, because the REPL_ONLY part of the code doesn't have logic
#         # for loading in RepL. I could either change that logic, or else we could have two RepL invocations
#         # for Magical: One that uses the smaller set of dataset configs x all envs, and also one that uses
#         # the multitask set of configs x 1 env (which doesn't actually matter)

# Also, we may not want to use these exact bench configs, but may prefer to explicitly specify environment

# python pretrain_n_adapt.py with repl_seed_sweep cfg_bench_short_sweep_magical dataset_sweep_with_multitask
#           algo_sweep

# python pretrain_n_adapt.py with il_on_pretrained_repl_seed_sweep cfg_bench_short_sweep_magical
#       dataset_sweep_with_multitask algo_sweep
# DMC:

# python pretrain_n_adapt.py with il_on_repl_sweep cfg_bench_short_sweep_dm_control dataset_sweep_no_multitask
#       algo_sweep



## Control sweep over initializations

    @experiment_obj.named_config
    def control_sweep():
        stages_to_run = StagesToRun.IL_ONLY
        spec = {'il_train.ortho_init': tune.grid_search([True, False]),
                'il_train.log_std_init': tune.grid_search([])} # TODO reasonable values for log std init?
        tune_run_kwargs = dict(num_samples=NUM_SEEDS)
        _ = locals()
        del _




