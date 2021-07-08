"""A fixing script for continue running failed experiments."""
import os
import json
import sacred
from sacred import Experiment, Ingredient
from sacred.observers import FileStorageObserver
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)
from il_representations.scripts.il_test import il_test_ex
from il_representations.scripts.il_train import il_train_ex
from il_representations.scripts.run_rep_learner import represent_ex
from il_representations.scripts.pretrain_n_adapt import run_single_exp

sacred.SETTINGS['CAPTURE_MODE'] = 'sys'  # workaround for sacred issue#740
fix_ex = Experiment(
    'fix',
    ingredients=[
        # explicitly list every ingredient we want to configure
        represent_ex,
        il_train_ex,
        il_test_ex,
        env_cfg_ingredient,
        env_data_ingredient,
        venv_opts_ingredient,
    ])
cwd = os.getcwd()


@fix_ex.config
def base_config():
    exp_name = "fix"
    exp_ident = None
    stage_to_run = 'il_test'

    # Assuming exp_path is the folder that's at the top of an experiment. i.e.
    # The one contains 'repl', 'il_train', and 'il_test'.
    exp_path = '/userhome/cs/cyn0531/il-representations/runs/chain_runs/26'
    cuda_devices = '3'
    num_test_ckpts = 20
    write_video = True
    start_test_threshold = 1900000
    test_range = [16, 24]

@fix_ex.main
def run(stage_to_run, exp_path, cuda_devices, num_test_ckpts, write_video,
        start_test_threshold, test_range):
    # Identify all the snapshot/ dir to find the latest saved models.
    il_train_dir = os.path.join(exp_path, 'il_train')
    il_train_exps = [name for name in os.listdir(il_train_dir)
                     if os.path.isdir(os.path.join(il_train_dir, name))]
    il_train_exps = sorted(il_train_exps)
    if test_range:
        il_train_exps = il_train_exps[test_range[0]:test_range[1]]
    il_train_exps = [os.path.join(il_train_dir, exp_dir) for exp_dir in
                     il_train_exps]

    tested_exps = []
    # identify which exp has been tested and skip that later.
    il_test_dir = os.path.join(exp_path, 'il_test')
    if os.path.isdir(il_test_dir):
        il_test_exps = [name for name in os.listdir(il_test_dir)
                        if os.path.isdir(os.path.join(il_test_dir, name))]
        il_test_exps = sorted(il_test_exps)
        il_test_exps = [os.path.join(il_test_dir, exp_dir) for exp_dir
                        in il_test_exps]
        for exp_dir in il_test_exps:
            test_config_file = os.path.join(exp_dir, 'config.json')

            if not os.path.isfile(test_config_file):
                continue

            with open(test_config_file, 'r') as json_file:
                test_config = json.load(json_file)

            test_files = [name for name in os.listdir(exp_dir)]

            # The below line can be pretty flexible; you can modify
            # the condition according to your testing config. Directories
            # added to tested_exps will not be tested again.
            if 'eval_19.json' in test_files:
                tested_exps.append(test_config['policy_dir'])

    print('Detected il_train_exps:', il_train_exps)
    for exp_dir in il_train_exps:
        config_file = os.path.join(exp_dir, 'config.json')

        # Skip this dir if it's not an experiment dir.
        if not os.path.isfile(config_file):
            continue

        with open(config_file, 'r') as json_file:
            config = json.load(json_file)
        assert config['algo'] == 'bc', 'Currently only support continue ' \
            'running BC exps.'

        model_save_dir = os.path.join(exp_dir, 'snapshots')
        benchmark_name = config['env_cfg']['benchmark_name']
        task_name = config['env_cfg']['task_name']
        exp_ident = config['exp_ident']

        # Skip this dir if it doesn't have any saved models.
        if not os.path.isdir(model_save_dir):
            continue

        saved_models = os.listdir(model_save_dir)
        # Saved models usually have name "policy_{nupdate}_batches.pt".
        # Here we are sorting models according to nupdate with ascending
        # order.
        saved_models = sorted(saved_models, key=lambda p:
                              int(p.split('_')[-2]))
        last_model = os.path.join(model_save_dir, saved_models[-1])
        last_saved_batch = int(saved_models[-1].split('_')[-2])

        if stage_to_run == 'il_train':

            config = {
                'log_start_batch': last_saved_batch,
                'encoder_path': last_model,
                'bc': dict(n_batches=config['bc']['n_batches'] -
                           last_saved_batch),
                'env_cfg': dict(benchmark_name=benchmark_name,
                                task_name=task_name)
            }

            exp_result = run_single_exp(config, log_dir=exp_dir,
                                        exp_name='il_train')
        if stage_to_run == 'il_test':
            if model_save_dir in tested_exps:
                continue
            if last_saved_batch > start_test_threshold:
                print(f'Evaluate {model_save_dir}')
                os.system(f'CUDA_VISIBLE_DEVICES={cuda_devices} '
                          f'python ./src/il_representations/scripts/il_test.py with \\'
                          f'policy_path={last_model} \\'
                          f'exp_ident={exp_ident} \\'
                          f'num_test_ckpts={num_test_ckpts} \\'
                          f'write_video={write_video} \\'
                          f'env_cfg.benchmark_name={benchmark_name} \\'
                          f'env_cfg.task_name={task_name} &')



def main(argv=None):
    fix_ex.run_commandline(argv)


if __name__ == '__main__':
    main()
