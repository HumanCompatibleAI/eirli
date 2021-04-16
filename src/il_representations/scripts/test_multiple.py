import os
import sacred
import pandas as pd
import json
import time
import datetime

from sacred import Experiment
from sacred.observers import FileStorageObserver

test_multiple_ex = Experiment('test_multiple')


@test_multiple_ex.config
def base_config():
    exp_path = '/scratch/cynthiachen/ilr-results/run-long-epochs-2021-04-06/98/il_train/1'
    ckpt_dir = os.path.join(exp_path, 'snapshots')
    exp_ident = None

    # How many ckpts to test?
    num_test_ckpts = 20
    cuda_device = 1
    test_script_path = 'src/il_representations/scripts/il_test.py'


@test_multiple_ex.main
def run(exp_path, ckpt_dir, exp_ident, num_test_ckpts, cuda_device, test_script_path):
    progress_df = pd.read_csv(os.path.join(exp_path, 'progress.csv'))
    total_updates = list(progress_df['n_updates'])[-1]

    module_ckpt_names = sorted(os.listdir(ckpt_dir))
    num_ckpts = len(module_ckpt_names)
    test_ckpt_interval = int(num_ckpts / num_test_ckpts)

    updates_per_ckpt = int(total_updates / len(module_ckpt_names))

    with open(os.path.join(exp_path, 'config.json')) as f:
        exp_config = json.load(f)

    exp_ident = exp_config['exp_ident'] if exp_ident is None else exp_ident

    print(f"Total saved ckpts: {len(module_ckpt_names)} for {total_updates} recorded updates. \n"
          f"Updates per ckpt: {updates_per_ckpt}")

    count = 0
    start_time = time.time()
    for ckpt_idx in range(0, num_ckpts, test_ckpt_interval):
        count += 1
        print(f"Start testing [{count}/{num_test_ckpts}] model...")
        policy_path = os.path.join(ckpt_dir, module_ckpt_names[ckpt_idx])

        os.system(f"CUDA_VISIBLE_DEVICES={cuda_device} xvfb-run -a python {test_script_path} with "
                  f"policy_path={policy_path} "
                  f"exp_ident={exp_ident} "
                  f"env_cfg.benchmark_name={exp_config['env_cfg']['benchmark_name']} "
                  f"env_cfg.task_name={exp_config['env_cfg']['task_name']}")

        print(f"Finished testing [{count}/{num_test_ckpts}] models. "
              f"Time elapsed: {str(datetime.timedelta(seconds=time.time() - start_time))}")


if __name__ == '__main__':
    sacred.SETTINGS['CAPTURE_MODE'] = 'sys'
    test_multiple_ex.observers.append(FileStorageObserver('runs/test_multiple_runs'))
    test_multiple_ex.run_commandline()
