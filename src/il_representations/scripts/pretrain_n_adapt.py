"""
This file should support:
1. Running representation pretrain for X seeds, then adapt on an IL algo for Y seeds.
2. Grid search over different configurations
"""
import os.path as osp
from sacred import Experiment
from sacred.observers import FileStorageObserver
import ray
from ray import tune
from il_representations.scripts.run_rep_learner import represent_ex
from il_representations.scripts.il_train import il_train_ex
from utils import sacred_copy, update, detect_ec2

chain_ex = Experiment('chain', ingredients=[represent_ex, il_train_ex])


def run_experiment(ex, ex_config, config, observer_name):
    ex_dict = dict(ex_config)
    merged_config = update(ex_dict, config)

    observer = FileStorageObserver(osp.join(observer_name))
    ex.observers.append(observer)
    ret_val = ex.run(config_updates=merged_config)
    ray.tune.track.log(accuracy=ret_val.result)


@chain_ex.config
def base_config(representation_learning, il_train):
    exp_name = "grid_search"
    run_rep = True
    run_il = True
    spec = {
        'rep': {
            'algo': tune.grid_search(['MoCo', 'SimCLR'])
        },
        'il': {
            'lr': tune.grid_search([0.01, 0.02])
        }
    }
    modified_rep_ex = dict(representation_learning)
    modified_il_ex = dict(il_train)


@chain_ex.main
def run(exp_name, run_rep, run_il, spec, modified_rep_ex, modified_il_ex):
    def runnable_function(inner_exp, modified_inner_ex, exp_name_, spec_):
        inner_ex_config = sacred_copy(modified_inner_ex)
        spec_copy = sacred_copy(spec_)
        print('spec copy', spec_copy)

        def run_exp(config, exp_name_):
            run_experiment(inner_exp, inner_ex_config, config, exp_name_)

        analysis = tune.run(
            run_exp,
            name=exp_name_,
            config=spec_copy
        )
        best_config = analysis.get_best_config(metric="accuracy")
        print(f"Best config is: {best_config}")
        print("Results available at: ")
        print(analysis._get_trial_paths())

    if run_rep:
        runnable_function(represent_ex, modified_rep_ex, f"{exp_name}_rep", spec['rep'])
    if run_il:
        runnable_function(il_train_ex, modified_il_ex, f"{exp_name}_il", spec['il'])


def main():
    observer = FileStorageObserver('chain_runs')
    chain_ex.observers.append(observer)
    chain_ex.run_commandline()


if __name__ == '__main__':
    main()
