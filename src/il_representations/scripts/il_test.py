#!/usr/bin/env python3
"""Run an IL algorithm in some selected domain."""
import collections
import json
import logging
import tempfile

import imitation.data.rollout as il_rollout
import imitation.util.logger as imitation_logger
import numpy as np
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from stable_baselines3.common.utils import get_device
import torch as th

from il_representations.algos.utils import set_global_seeds
from il_representations.envs import auto
from il_representations.envs.config import benchmark_ingredient
from il_representations.utils import TensorFrameWriter

sacred.SETTINGS['CAPTURE_MODE'] = 'sys'  # workaround for sacred issue#740
il_test_ex = Experiment('il_test', ingredients=[benchmark_ingredient])


@il_test_ex.config
def default_config():
    torch_num_threads = 1
    policy_path = None
    n_rollouts = 100
    device_name = 'auto'
    # run_id is written into the produced DataFrame to indicate what model is
    # being tested
    run_id = 'test'
    # if True, then we'll add a video named <video_file_name> as a Sacred
    # artifact
    write_video = False
    video_file_name = "rollouts.mp4"


@il_test_ex.main
def run(policy_path, benchmark, seed, n_rollouts, device_name, run_id,
        write_video, video_file_name, torch_num_threads):
    set_global_seeds(seed)
    # FIXME(sam): this is not idiomatic way to do logging (as in il_train.py)
    logging.basicConfig(level=logging.INFO)
    log_dir = il_test_ex.observers[0].dir
    imitation_logger.configure(log_dir, ["stdout", "csv", "tensorboard"])
    if torch_num_threads is not None:
        th.set_num_threads(torch_num_threads)

    if policy_path is None:
        raise ValueError(
            "must pass a string-valued policy_path to this command")
    policy = th.load(policy_path)

    device = get_device(device_name)
    policy = policy.to(device)
    policy.eval()

    if write_video:
        video_fp = tempfile.NamedTemporaryFile("wb", suffix=video_file_name)
        video_writer = TensorFrameWriter(video_fp.name,
                                         color_space=auto.load_color_space())

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
            run_id=run_id,
            video_writer=video_writer if write_video else None,
        )
        eval_data_frame = eval_protocol.do_eval(verbose=False)
        # display to stdout
        logging.info("Evaluation finished, results:\n" +
                     eval_data_frame.to_string())
        final_stats_dict = {
            'demo_env_name': demo_env_name,
            'policy_path': policy_path,
            'seed': seed,
            'ntraj': n_rollouts,
            'full_data': json.loads(eval_data_frame.to_json(orient='records')),
            # return_mean is included for hyperparameter tuning; we also get
            # the same value for other environments (dm_control, Atari). (in
            # MAGICAL, it averages across all test environments)
            'return_mean': eval_data_frame['mean_score'].mean(),
        }

    elif (benchmark['benchmark_name'] == 'dm_control'
          or benchmark['benchmark_name'] == 'atari'):
        # must import this to register envs
        from il_representations.envs import dm_control_envs  # noqa: F401

        full_env_name = auto.get_gym_env_name()
        vec_env = auto.load_vec_env()

        # sample some trajectories
        rng = np.random.RandomState(seed)
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

        final_stats_dict = collections.OrderedDict([
            ('full_env_name', full_env_name),
            ('policy_path', policy_path),
            ('seed', seed),
            *stats.items(),
        ])
        vec_env.close()

        if write_video:
            assert len(trajectories) > 0
            # write the trajectories in sequence
            for traj in trajectories:
                for step_tensor in traj.obs:
                    video_writer.add_tensor(th.FloatTensor(step_tensor) / 255.)

    else:
        raise NotImplementedError("policy evaluation on benchmark_name="
                                  f"{benchmark['benchmark_name']!r} is not "
                                  "yet supported")

    # save to a .json file
    with tempfile.NamedTemporaryFile('w') as fp:
        json.dump(final_stats_dict, fp, indent=2, sort_keys=False)
        fp.flush()
        il_test_ex.add_artifact(fp.name, 'eval.json')

    # also save video
    if write_video:
        video_writer.close()
        il_test_ex.add_artifact(video_fp.name, video_file_name)
        video_fp.close()

    return final_stats_dict


if __name__ == '__main__':
    il_test_ex.observers.append(FileStorageObserver('runs/il_test_runs'))
    il_test_ex.run_commandline()
