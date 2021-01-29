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


best_hp_vae_beta = 1e-5
vae_args = {'loss_calculator_kwargs': {'beta': best_hp_vae_beta},
            'batch_size': 64}


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
    def icml_inv_dyn():
        repl = {'algo': InverseDynamicsPrediction}
        _ = locals()
        del _

    @experiment_obj.named_config
    def icml_dynamics():
        repl = {'algo': DynamicsPrediction}
        _ = locals()
        del _

    @experiment_obj.named_config
    def icml_ac_tcpc():
        repl = {'algo': ICMLActionConditionedTemporalCPC}
        _ = locals()
        del _

    @experiment_obj.named_config
    def icml_identity_cpc():
        repl = {'algo': ICMLIdentityCPC}
        _ = locals()
        del _

    @experiment_obj.named_config
    def icml_vae():
        repl = {'algo': ICMLVariationalAutoencoder}
        _ = locals()
        del _

    @experiment_obj.named_config
    def icml_gaussian_prior():
        repl = {'algo': GaussianPriorControl}
        _ = locals()
        del _

    @experiment_obj.named_config
    def control_ortho_init():
        il_train = {'ortho_init': True}
        _ = locals()
        del _

    @experiment_obj.named_config
    def control_no_ortho_init():
        il_train = {'ortho_init': False}
        _ = locals()
        del _

    @experiment_obj.named_config
    def control_lsi_one():
        il_train = {'log_std_init': 1.0}
        _ = locals()
        del _

    @experiment_obj.named_config
    def control_lsi_zero():
        # I think this is the SB default value, but having it here
        # to make testing clearer
        il_train = {'log_std_init': 0.0}
        _ = locals()
        del _
