from ray import tune
from il_representations.scripts.utils import StagesToRun, ReuseRepl
from il_representations.algos import (ActionConditionedTemporalCPC, TemporalCPC,
                                      VariationalAutoencoder, DynamicsPrediction,
                                      InverseDynamicsPrediction, GaussianPriorControl)
from il_representations.algos import pair_constructors
from functools import partialmethod
from copy import deepcopy


# metaclass logic pulled from here: https://stackoverflow.com/a/55053439
class MetaclassACTCPC(type):
    def __repr__(self):
        return "ICMLActionConditionedTemporalCPC"


class MetaclassTCPC(type):
    def __repr__(self):
        return "ICMLTemporalCPC"


class MetaclassVAE(type):
    def __repr__(self):
        return "ICMLVariationalAutoencoder"


contrastive_kwargs_standin = {}


class ICMLActionConditionedTemporalCPC(ActionConditionedTemporalCPC, metaclass=MetaclassACTCPC):
    __init__ = partialmethod(ActionConditionedTemporalCPC.__init__, **contrastive_kwargs_standin)


tcpc_args = deepcopy(contrastive_kwargs_standin)
tcpc_args["target_pair_constructor"] = pair_constructors.IdentityPairConstructor


class ICMLIdentityCPC(TemporalCPC, metaclass=MetaclassTCPC):
    __init__ = partialmethod(TemporalCPC.__init__, **contrastive_kwargs_standin)


best_hp_vae_beta = 0.0
vae_args = {'loss_calculator_kwargs': {'beta': best_hp_vae_beta}}


class ICMLVariationalAutoencoder(VariationalAutoencoder, metaclass=MetaclassVAE):
    __init__ = partialmethod(VariationalAutoencoder.__init__, **vae_args)


NUM_SEEDS = 9


def make_dataset_experiment_configs(experiment_obj):
    # To be used for DMC, where Sam said we might as well retrain RepL for each new IL seed
    @experiment_obj.named_config
    def icml_il_on_repl_sweep():
        repl = {'batches_per_epoch': 500,
                'n_epochs': 10}
        stages_to_run = StagesToRun.REPL_AND_IL
        reuse_repl = ReuseRepl.NO
        tune_run_kwargs = dict(num_samples=NUM_SEEDS)
        _ = locals()
        del _

    @experiment_obj.named_config
    def algo_sweep():
        spec = dict(repl=tune.grid_search([{'algo': algo}
                                           for algo in [InverseDynamicsPrediction, DynamicsPrediction,
                                                        ICMLActionConditionedTemporalCPC, ICMLIdentityCPC,
                                                        ICMLVariationalAutoencoder,
                                                        GaussianPriorControl]]))
        _ = locals()
        del _

    @experiment_obj.named_config
    def control_ortho_init_sweep():
        stages_to_run = StagesToRun.IL_ONLY
        spec = {'il_train.ortho_init': tune.grid_search([True, False])}
        tune_run_kwargs = dict(num_samples=NUM_SEEDS)
        _ = locals()
        del _

    @experiment_obj.named_config
    def control_log_std_init_sweep():
        stages_to_run = StagesToRun.IL_ONLY
        spec = {'il_train.log_std_init': tune.grid_search([])} # TODO reasonable values for log std init?
        tune_run_kwargs = dict(num_samples=NUM_SEEDS)
        _ = locals()
        del _
