#!/usr/bin/env python3
"""Run an IL algorithm in some selected domain."""
import logging
import tempfile

import torch as th
from sacred import Experiment
from sacred.observers import FileStorageObserver
from stable_baselines3.common.utils import get_device

import imitation.util.logger as imitation_logger

from il_representations.envs import magical_envs
from il_representations.envs.config import benchmark_ingredient

il_test_ex = Experiment('il_test', ingredients=[benchmark_ingredient])


@il_test_ex.config
def default_config():
    policy_path = None
    seed = 42
    n_rollouts = 100
    eval_batch_size = 32
    dev_name = 'auto'
    # run_id is written into the produced DataFrame to indicate what model is
    # being tested
    run_id = 'test'


@il_test_ex.main
def test(policy_path, benchmark, seed, n_rollouts, eval_batch_size, dev_name,
         run_id):
    # FIXME(sam): this is not idiomatic way to do logging (as in il_train.py)
    logging.basicConfig(level=logging.INFO)
    log_dir = il_test_ex.observers[0].dir
    imitation_logger.configure(log_dir, ["stdout", "tensorboard"])

    if policy_path is None:
        raise ValueError(
            "must pass a string-valued policy_path to this command")
    policy = th.load(policy_path)

    device = get_device(dev_name)
    policy = policy.to(device)

    if benchmark['benchmark_name'] == 'magical':
        env_prefix = benchmark['magical_env_prefix']
        env_preproc = benchmark['magical_preproc']
        demo_env_name = f'{env_prefix}-Demo-{env_preproc}-v0'
        eval_protocol = magical_envs.SB3EvaluationProtocol(
            demo_env_name=demo_env_name,
            policy=policy,
            n_rollouts=n_rollouts,
            seed=seed,
            batch_size=eval_batch_size,
            run_id=run_id,
        )
        eval_data_frame = eval_protocol.do_eval(verbose=False)
        # display to stdout
        logging.info("Evaluation finished, results:\n" +
                     eval_data_frame.to_string())
        with tempfile.NamedTemporaryFile('w') as fp:
            eval_data_frame.to_csv(fp)
            fp.flush()
            il_test_ex.add_artifact(fp.name, 'eval.csv')
    else:
        raise NotImplementedError("policy evaluation on benchmark_name="
                                  f"{benchmark['benchmark_name']!r} is not "
                                  "yet supported")


if __name__ == '__main__':
    il_test_ex.observers.append(FileStorageObserver('il_test_runs'))
    il_test_ex.run_commandline()
