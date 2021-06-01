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

def run_single_exp(merged_config, log_dir, exp_name):
    """
    Run a specified experiment. We could not pass each Sacred experiment in
    because they are not pickle serializable, which is not supported by Ray
    (when running this as a remote function).

    params:
        merged_config: The config that should be used in the sub-experiment;
                       formed by calling merge_configs()
        log_dir: The log directory of current chain experiment.
        exp_name: Specify the experiment type in ['repl', 'il_train',
            'il_test']
    """
    # we need to run the workaround in each raylet, so we do it at the start of
    # run_single_exp
    sacred.SETTINGS['CAPTURE_MODE'] = 'sys'  # workaround for sacred issue#740
    from il_representations.scripts.il_test import il_test_ex
    from il_representations.scripts.il_train import il_train_ex
    from il_representations.scripts.run_rep_learner import represent_ex

    if exp_name == 'repl':
        inner_ex = represent_ex
    elif exp_name == 'il_train':
        inner_ex = il_train_ex
    elif exp_name == 'il_test':
        inner_ex = il_test_ex
    else:
        raise NotImplementedError(
            f"exp_name must be one of repl, il_train, or il_test. Value passed in: {exp_name}")

    observer = FileStorageObserver(os.path.join(log_dir, exp_name))
    inner_ex.observers.append(observer)
    ret_val = inner_ex.run(config_updates=merged_config)
    return {
        "type": exp_name,
        "result": ret_val.result,
        # FIXME(sam): this is dependent on us having exactly one
        # observer, and it being a FileStorageObserver
        "dir": ret_val.observers[0].dir,
    }

@fix_ex.config
def base_config():
    exp_name = "fix"
    exp_ident = None
    stage_to_run = 'il_test'

    # Assuming exp_path is the folder that's at the top of an experiment. i.e.
    # The one contains 'repl', 'il_train', and 'il_test'.
    exp_path = None
    cuda_devices = '0'
    num_test_ckpts = 20
    write_video = True
    start_test_threshold = 1500000

@fix_ex.main
def run(stage_to_run, exp_path, cuda_devices, num_test_ckpts, write_video,
        start_test_threshold):
    # Identify all the snapshot/ dir to find the latest saved models.
    il_train_dir = os.path.join(exp_path, 'il_train')
    il_train_exps = [name for name in os.listdir(il_train_dir) \
                     if os.path.isdir(os.path.join(il_train_dir, name))]
    il_train_exps = sorted(il_train_exps)
    breakpoint()
    il_train_exps = [os.path.join(il_train_dir, exp_dir) for exp_dir in
                     il_train_exps]
    for exp_dir in il_train_exps:
        config_file = os.path.join(exp_dir, 'config.json')

        # Skip this dir if it's not an experiment dir.
        if not os.path.isfile(config_file):
            continue

        with open(config_file, 'r') as json_file:
            config = json.load(json_file)
        assert config['algo'] == 'bc', 'Currently only support continue \
            running BC exps.'

        model_save_dir = os.path.join(exp_dir, 'snapshots')
        benchmark_name = config['env_cfg']['benchmark_name']
        task_name = config['env_cfg']['task_name']

        # Skip this dir if it doesn't have any saved models.
        if not os.path.isdir(model_save_dir):
            continue

        saved_models = [m for m in os.listdir(model_save_dir)]
        sorted_idx = sorted(range(len(saved_models)), key=lambda
                            k: int(saved_models[k].split('_')[-2]))
        saved_models = [saved_models[k] for k in sorted_idx]
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
            if last_saved_batch > start_test_threshold:
                os.system(f'CUDA_VISIBLE_DEVICES={cuda_devices} '
                          f'python ./src/il_representations/scripts/il_test.py with \\'
                          f'policy_dir={model_save_dir} \\'
                          f'num_test_ckpts={num_test_ckpts} \\'
                          f'write_video={write_video} \\'
                          f'env_cfg.benchmark_name={benchmark_name} \\'
                          f'env_cfg.task_name={task_name} &')



def main(argv=None):
    fix_ex.run_commandline(argv)


if __name__ == '__main__':
    main()

