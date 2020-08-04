#!/usr/bin/env python3
"""Run an IL algorithm in some selected domain."""
import collections
import json
import logging
import tempfile

import imitation.util.logger as imitation_logger
import imitation.data.rollout as il_rollout
from imitation.util.util import make_vec_env
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from stable_baselines3.common.utils import get_device
import torch as th

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
        from il_representations.envs import magical_envs
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

    elif benchmark['benchmark_name'] == 'dm_control':
        # must import this to register envs
        from il_representations.envs import dm_control_envs  # noqa: F401

        # get env name
        short_env_name = benchmark['dm_control_env']
        all_full_names = benchmark['dm_control_full_env_names']
        full_env_name = all_full_names[short_env_name]

        # sample some trajectories
        rng = np.random.RandomState(seed)
        vec_env = make_vec_env(full_env_name,
                               n_envs=eval_batch_size,
                               parallel=False,
                               seed=seed)
        trajectories = il_rollout.generate_trajectories(
            policy, vec_env, il_rollout.min_episodes(n_rollouts), rng=rng)

        # the "stats" dict has keys {return,len}_{min,max,mean,std}
        stats = il_rollout.rollout_stats(trajectories)
        stats = collections.OrderedDict([(key, stats[key])
                                         for key in sorted(stats)])

        # print it out
        kv_message = '\n'.join(f"  {key}={value}"
                               for key, value in stats.items())
        logging.info(f"Evaluation stats on '{full_env_name}': {kv_message}")

        # save to a .json file
        with tempfile.NamedTemporaryFile('w') as fp:
            full_stats_dict = collections.OrderedDict([
                ('short_env_name', short_env_name),
                ('full_env_name', full_env_name),
                ('policy_path', policy_path),
                ('seed', seed),
                *stats.items(),
            ])
            json.dump(full_stats_dict, fp, indent=2, sort_keys=False)
            fp.flush()
            il_test_ex.add_artifact(fp.name, 'eval.json')

    else:
        raise NotImplementedError("policy evaluation on benchmark_name="
                                  f"{benchmark['benchmark_name']!r} is not "
                                  "yet supported")


if __name__ == '__main__':
    il_test_ex.observers.append(FileStorageObserver('il_test_runs'))
    il_test_ex.run_commandline()
