import os
import sacred
import pandas as pd
import json

from sacred import Experiment
from sacred.observers import FileStorageObserver

from il_representations.scripts.il_test import il_test_ex

test_multiple_ex = Experiment('test_multiple')


@test_multiple_ex.config
def base_config():
    exp_path = '/scratch/cynthiachen/ilr-results/run-long-epochs-2021-04-06/98/il_train/1'
    ckpt_dir = os.path.join(exp_path, 'snapshots')

    # How many ckpts to test?
    num_test_ckpts = 20


@test_multiple_ex.main
def run(exp_path, ckpt_dir, num_test_ckpts):
    progress_df = pd.read_csv(os.path.join(exp_path, 'progress.csv'))
    total_updates = list(progress_df['n_updates'])[-1]

    module_ckpt_names = sorted(os.listdir(ckpt_dir))
    num_ckpts = len(module_ckpt_names)
    test_ckpt_interval = int(num_ckpts / num_test_ckpts)

    updates_per_ckpt = int(total_updates / len(module_ckpt_names))

    with open(os.path.join(exp_path, 'config.json')) as f:
        exp_config = json.load(f)

    print(f"Total saved ckpts: {len(module_ckpt_names)} for {total_updates} recorded updates. \n"
          f"Updates per ckpt: {updates_per_ckpt}")

    for ckpt_idx in range(0, num_ckpts, test_ckpt_interval):
        policy_path = os.path.join(ckpt_dir, module_ckpt_names[ckpt_idx])
        il_test_ex.observers.append(FileStorageObserver('runs/il_test_runs'))
        result = il_test_ex.run(config_updates={'policy_path': policy_path,
                                                'env_cfg': {
                                                    'benchmark_name': exp_config['env_cfg']['benchmark_name'],
                                                    'task_name': exp_config['env_cfg']['task_name']
                                                }})
        print(result)


if __name__ == '__main__':
    sacred.SETTINGS['CAPTURE_MODE'] = 'sys'
    test_multiple_ex.observers.append(FileStorageObserver('runs/test_multiple_runs'))
    test_multiple_ex.run_commandline()
