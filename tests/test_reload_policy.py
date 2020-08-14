import glob
import os

from sacred.observers import FileStorageObserver

from il_representations.algos import MoCo
from il_representations.scripts.il_train import il_train_ex
from il_representations.scripts.run_rep_learner import represent_ex
from il_representations.test_support.configuration import BENCHMARK_CONFIGS

# FIXME(sam): figure out a better way of doing this. Probably make each
# experiment a "set up once" fixture that adds FileStorageObservers.
for ex in (represent_ex, il_train_ex):
    if not any(isinstance(o, FileStorageObserver) for o in ex.observers):
        ex.observers.append(FileStorageObserver('test_observer'))


def test_reload_policy():
    """Test saving a policy with one specific representation learner, then loading
    it with the IL code.

    (I only test one IL algorithm, one dataset, and one representation learner
    because the process is roughly the same in all cases)"""
    rep_result = represent_ex.run(
        config_updates={
            'pretrain_epochs': 1,
            'batch_size': 7,
            'unit_test_max_train_steps': 2,
            'representation_dim': 3,
            'algo': MoCo,
            'use_random_rollouts': False,
            'benchmark': BENCHMARK_CONFIGS[0],
            'ppo_finetune': False,
        })

    # train BC using learnt representation
    encoder_list = glob.glob(
        os.path.join(
            rep_result.observers[0].dir,
            'training_logs/checkpoints/representation_encoder/*.ckpt'))
    policy_path = encoder_list[0]
    il_train_ex.run(
        config_updates={
            'bc_n_epochs': 1,
            'device_name': 'cpu',
            'benchmark': BENCHMARK_CONFIGS[0],
            'encoder_path': policy_path,
        })
