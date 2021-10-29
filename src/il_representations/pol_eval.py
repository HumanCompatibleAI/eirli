"""Utilities for evaluating policies."""
import collections
import json
import logging
import os

import imitation.data.rollout as il_rollout
import numpy as np
from stable_baselines3.common.utils import get_device
import torch as th

from il_representations.envs import auto
from il_representations.envs.config import env_cfg_ingredient
from il_representations.utils import TensorFrameWriter


@env_cfg_ingredient.capture
def _get_global_env_cfg(_config):
    return _config


def do_final_eval(*,
                  policy_path,
                  n_rollouts,
                  out_dir,
                  seed,
                  run_id,
                  deterministic_policy,
                  device,
                  write_video=False,
                  eval_file_name='eval.json',
                  video_file_name=None):
    """Do final evaluation of a policy & write eval.json file.

    Args:
        policy_path (str): path to SB3 `BasePolicy` that will be evaluated.
        n_rollouts (int): number of rollouts to perform.
        out_dir (str): output directory to write files to.
        seed (int): random seed for environment.
        run_id (Optional[str]): optional extra string to be written to JSON
            file as run identifier.
        deterministic_policy (bool): should actions be chosen
            deterministically?
        device (Union[str, th.Device]): Torch device or device name to use for
            evaluation.
        write_video (bool): should trajectories be rendered as video?
        eval_file_name (Optional[str]): filename for writing evaluation
            results.
        video_file_name (Optional[str]): filename for when write_video=True
    """
    policy = th.load(policy_path)

    device = get_device(device)
    policy = policy.to(device).eval()
    policy.eval()
    env_cfg = _get_global_env_cfg()

    if write_video:
        video_writer = TensorFrameWriter(
            os.path.join(out_dir, video_file_name),
            color_space=auto.load_color_space())

    if env_cfg['benchmark_name'] == 'magical':
        from magical.benchmarks import update_magical_env_name

        from il_representations.envs import magical_envs  # noqa: I003
        task_name = env_cfg['task_name']
        env_preproc = env_cfg['magical_preproc']
        demo_env_name = update_magical_env_name(task_name,
                                                preproc=env_preproc,
                                                variant='Demo')
        eval_protocol = magical_envs.SB3EvaluationProtocol(
            demo_env_name=demo_env_name,
            policy=policy,
            n_rollouts=n_rollouts,
            seed=seed,
            run_id=run_id,
            video_writer=video_writer if write_video else None,
            deterministic_policy=deterministic_policy,
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

    elif env_cfg['benchmark_name'] in ('dm_control', 'atari', 'minecraft'):
        # must import this to register envs
        from il_representations.envs import dm_control_envs  # noqa: F401

        full_env_name = auto.get_gym_env_name()
        vec_env = auto.load_vec_env()

        # sample some trajectories
        rng = np.random.RandomState(seed)
        trajectories = il_rollout.generate_trajectories(
            policy,
            vec_env,
            il_rollout.make_min_episodes(n_rollouts),
            rng=rng,
            deterministic_policy=deterministic_policy)
        # make sure all the actions are finite
        for traj in trajectories:
            assert np.all(np.isfinite(traj.acts)), traj.acts

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

    elif env_cfg['benchmark_name'] == ('procgen'):
        full_env_name = auto.get_gym_env_name()
        final_stats_dict = collections.OrderedDict([
            ('full_env_name', full_env_name),
            ('policy_path', policy_path),
            ('seed', seed),
        ])

        # In Procgen, we use start_level=0 to test on train level, and 1000 for
        # test level.
        for start_level in [0, 1000]:
            vec_env = auto.load_vec_env(procgen_start_level=start_level)

            # sample some trajectories
            rng = np.random.RandomState(seed)
            trajectories = il_rollout.generate_trajectories(
                policy, vec_env, il_rollout.make_min_episodes(n_rollouts),
                rng=rng)
            # make sure all the actions are finite
            for traj in trajectories:
                assert np.all(np.isfinite(traj.acts)), traj.acts

            # the "stats" dict has keys {return,len}_{min,max,mean,std}
            stats = il_rollout.rollout_stats(trajectories)
            stats = collections.OrderedDict([(key, stats[key])
                                             for key in sorted(stats)])

            game_level = 'train_level' if start_level == 0 else 'test_level'
            final_stats_dict.update({game_level: stats})

            # print it out
            kv_message = '\n'.join(f"  {key}={value}"
                                   for key, value in stats.items())
            logging.info(
                f"Evaluation stats on '{full_env_name}': {kv_message}")

            vec_env.close()

            if write_video:
                assert len(trajectories) > 0

                policy_filename = os.path.basename(policy_path)
                # Remove file extension
                policy_filename = os.path.splitext(policy_filename)[0]
                video_file_name = f"rollout_{policy_filename}_{game_level}.mp4"
                video_writer = TensorFrameWriter(
                    os.path.join(out_dir, video_file_name),
                    color_space=auto.load_color_space())

                # write the trajectories in sequence
                for traj in trajectories:
                    for step_tensor in traj.obs:
                        video_writer.add_tensor(th.FloatTensor(step_tensor) / 255.)

                video_writer.close()

    else:
        raise NotImplementedError("policy evaluation on benchmark_name="
                                  f"{env_cfg['benchmark_name']!r} is not "
                                  "yet supported")

    # save to a .json file
    with open(os.path.join(out_dir, eval_file_name), 'w') as fp:
        json.dump(final_stats_dict, fp, indent=2, sort_keys=False)

    # also save video
    if write_video:
        video_writer.close()

    return final_stats_dict
