import logging

import numpy as np
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch

from il_representations import algos
from il_representations.algos.representation_learner import \
    RepresentationLearner
from il_representations.algos.utils import LinearWarmupCosine
from il_representations.envs import auto
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient)

sacred.SETTINGS['CAPTURE_MODE'] = 'sys'  # workaround for sacred issue#740
represent_ex = Experiment(
    'repl', ingredients=[env_cfg_ingredient, env_data_ingredient])


@represent_ex.config
def default_config():
    # exp_ident is an arbitrary string. Set it to a meaningful value to help
    # you identify runs in viskit.
    exp_ident = None

    # see docs for `load_new_style_ilr_dataset()` for more information on
    # syntax for dataset_configs.
    dataset_configs = [{'type': 'demos'}]
    algo = "ActionConditionedTemporalCPC"
    torch_num_threads = 1
    n_envs = 1
    pretrain_epochs = None
    pretrain_batches = 10000
    algo_params = {
        'representation_dim': 128,
        'optimizer': torch.optim.Adam,
        'optimizer_kwargs': {'lr': 1e-4},
        'augmenter_kwargs': {
            # augmenter_spec is a comma-separated list of enabled augmentations.
            # See `help(imitation.augment.StandardAugmentations)` for available
            # augmentations.
            "augmenter_spec": "translate,rotate,gaussian_blur",
        },
    }
    device = "auto"
    # this is useful for constructing tests where we want to truncate the
    # dataset to be small

    _ = locals()
    del _


@represent_ex.named_config
def cosine_warmup_scheduler():
    algo_params = {"scheduler": LinearWarmupCosine, "scheduler_kwargs": {'warmup_epoch': 2, 'T_max': 10}}
    _ = locals()
    del _

@represent_ex.named_config
def ceb_breakout():
    env_id = 'BreakoutNoFrameskip-v4'
    train_from_expert = True
    algo = algos.FixedVarianceCEB
    pretrain_batches = None
    pretrain_batches = 5
    _ = locals()
    del _


@represent_ex.named_config
def target_projection():
    algo = algos.FixedVarianceTargetProjectedCEB
    _ = locals()
    del _


def initialize_non_features_extractor(sb3_model):
    # This is a hack to get around the fact that you can't initialize only some of the components of a SB3 policy
    # upon creation, and we in fact want to keep the loaded representation frozen, but orthogonally initalize other
    # components.
    sb3_model.policy.init_weights(sb3_model.policy.mlp_extractor, np.sqrt(2))
    sb3_model.policy.init_weights(sb3_model.policy.action_net, 0.01)
    sb3_model.policy.init_weights(sb3_model.policy.value_net, 1)
    return sb3_model


@represent_ex.main
def run(dataset_configs, algo, algo_params, seed, pretrain_epochs,
        pretrain_batches, torch_num_threads, _config):
    # TODO fix to not assume FileStorageObserver always present
    log_dir = represent_ex.observers[0].dir
    if torch_num_threads is not None:
        torch.set_num_threads(torch_num_threads)

    if isinstance(algo, str):
        algo = getattr(algos, algo)

    # setup environment & dataset
    webdataset = auto.load_new_style_ilr_dataset(configs=dataset_configs)
    color_space = webdataset.meta['color_space']
    observation_space = webdataset.meta['observation_space']
    action_space = webdataset.meta['action_space']

    assert issubclass(algo, RepresentationLearner)
    algo_params = dict(algo_params)
    algo_params['augmenter_kwargs'] = {
        'color_space': color_space,
        **algo_params['augmenter_kwargs'],
    }
    logging.info(f"Running {algo} with parameters: {algo_params}")
    model = algo(
        observation_space=observation_space,
        action_space=action_space,
        log_dir=log_dir,
        **algo_params)

    # setup model
    loss_record, most_recent_encoder_path = model.learn(
        webdataset, pretrain_epochs, pretrain_batches)

    return {
        'encoder_path': most_recent_encoder_path,
        # return average loss from final epoch for HP tuning
        'repl_loss': loss_record[-1],
    }


if __name__ == '__main__':
    represent_ex.observers.append(FileStorageObserver('runs/rep_learning_runs'))
    represent_ex.run_commandline()
