from il_representations.script_utils import StagesToRun
from ray import tune


def make_chain_configs(experiment_obj):

    @experiment_obj.named_config
    def cfg_use_magical():
        # see il_representations/envs/config for examples of what should go here
        env_cfg = {
            'benchmark_name': 'magical',
            # MatchRegions-Demo-v0 is of intermediate difficulty
            # (TODO(sam): allow MAGICAL to load data from _all_ tasks at once, so
            # we can try multi-task repL)
            'task_name': 'MatchRegions-Demo-v0',
            # we really need magical_remove_null_actions=True for BC; for RepL it
            # shouldn't matter so much (for action-based RepL methods)
            'magical_remove_null_actions': False,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_use_dm_control():
        env_cfg = {
            'benchmark_name': 'dm_control',
            # walker-walk is difficult relative to other dm-control tasks that we
            # use, but RL solves it quickly. Plateaus around 850-900 reward (see
            # https://docs.google.com/document/d/1YrXFCmCjdK2HK-WFrKNUjx03pwNUfNA6wwkO1QexfwY/edit#).
            'task_name': 'reacher-easy',
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_base_3seed_4cpu_pt3gpu():
        """Basic config that does three samples per config, using 5 CPU cores and
        0.3 of a GPU. Reasonable idea for, e.g., GAIL on svm/perceptron."""
        use_skopt = False
        tune_run_kwargs = dict(num_samples=3,
                               # retry on (node) failure
                               max_failures=2,
                               fail_fast=False,
                               resources_per_trial=dict(
                                   cpu=5,
                                   gpu=0.32,
                               ))

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_base_5seed_1cpu_pt3gpu():
        use_skopt = False
        tune_run_kwargs = dict(num_samples=5,
                               # retry on (node) failure
                               max_failures=2,
                               fail_fast=False,
                               resources_per_trial=dict(
                                   cpu=1,
                                   gpu=0.32,
                               ))

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_base_5seed_1cpu_pt25gpu():
        use_skopt = False
        tune_run_kwargs = dict(num_samples=5,
                               # retry on (node) failure
                               max_failures=5,
                               fail_fast=False,
                               resources_per_trial=dict(
                                   cpu=1,
                                   gpu=0.25,
                               ))

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_base_skopt_4cpu_pt3gpu_no_retry():
        # config that is used for skopt tuning runs in lead-up to icml
        use_skopt = True
        tune_run_kwargs = dict(num_samples=1,
                               # never retry, since these are just HP tuning
                               # runs
                               max_failures=0,
                               fail_fast=False,
                               resources_per_trial=dict(
                                   cpu=4,
                                   gpu=0.32,
                               ))

        _ = locals()
        del _


    @experiment_obj.named_config
    def cfg_base_skopt_1cpu_pt25gpu_no_retry():
        # another config that is used for skopt tuning runs in lead-up to icml
        use_skopt = True
        tune_run_kwargs = dict(num_samples=1,
                               # never retry, since these are just HP tuning
                               # runs
                               max_failures=0,
                               fail_fast=False,
                               resources_per_trial=dict(
                                   cpu=1,
                                   gpu=0.25,
                               ))

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_base_3seed_1cpu_pt2gpu_2envs():
        """Another config that uses only one CPU per run, and .2 of a GPU. Good for
        running GPU-intensive algorithms (repL, BC) on GCP."""
        use_skopt = False
        tune_run_kwargs = dict(num_samples=3,
                               # retry on node failure
                               max_failures=3,
                               fail_fast=False,
                               resources_per_trial=dict(
                                   cpu=1,
                                   gpu=0.2,
                               ))
        venv_opts = {
            'n_envs': 2,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_base_3seed_1cpu_pt5gpu_2envs():
        """As above, but one GPU per run."""
        use_skopt = False
        tune_run_kwargs = dict(num_samples=3,
                               # retry on node failure
                               max_failures=3,
                               fail_fast=False,
                               resources_per_trial=dict(
                                   cpu=1,
                                   gpu=0.5,
                               ))
        venv_opts = {
            'n_envs': 2,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_base_3seed_1cpu_1gpu_2envs():
        """As above, but one GPU per run."""
        use_skopt = False
        tune_run_kwargs = dict(num_samples=3,
                               # retry on node failure
                               max_failures=3,
                               fail_fast=False,
                               resources_per_trial=dict(
                                   cpu=1,
                                   gpu=1,
                               ))
        venv_opts = {
            'n_envs': 2,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_no_log_to_driver():
        # disables sending stdout of Ray workers back to head node
        # (only useful for huge clusters)
        ray_init_kwargs = {
            'log_to_driver': False,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_bench_short_sweep_magical():
        """Sweeps over four easiest MAGICAL instances."""
        spec = dict(env_cfg=tune.grid_search(
            # MAGICAL configs
            [
                {
                    'benchmark_name': 'magical',
                    'task_name': magical_env_name,
                    'magical_remove_null_actions': True,
                } for magical_env_name in [
                    'MoveToCorner-Demo-v0',
                    'MoveToRegion-Demo-v0',
                    'FixColour-Demo-v0',
                    'MatchRegions-Demo-v0',
                    # 'FindDupe-Demo-v0',
                    # 'MakeLine-Demo-v0',
                    # 'ClusterColour-Demo-v0',
                    # 'ClusterShape-Demo-v0',
                ]
            ]))

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_bench_short_sweep_dm_control():
        """Sweeps over four easiest dm_control instances."""
        spec = dict(env_cfg=tune.grid_search(
            # dm_control configs
            [
                {
                    'benchmark_name': 'dm_control',
                    'task_name': dm_control_env_name
                } for dm_control_env_name in [
                # to gauge how hard these are, see
                # https://docs.google.com/document/d/1YrXFCmCjdK2HK-WFrKNUjx03pwNUfNA6wwkO1QexfwY/edit#heading=h.akt76l1pl1l5
                'reacher-easy',
                'finger-spin',
                'ball-in-cup-catch',
                'cartpole-swingup',
                # 'cheetah-run',
                # 'walker-walk',
                # 'reacher-easy',
            ]
            ]))

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_bench_micro_sweep_magical():
        """Tiny sweep over MAGICAL configs, both of which are "not too hard",
        but still provide interesting generalisation challenges."""
        spec = dict(env_cfg=tune.grid_search(
            [
                {
                    'benchmark_name': 'magical',
                    'task_name': magical_env_name,
                    'magical_remove_null_actions': True,
                } for magical_env_name in [
                    'MoveToRegion-Demo-v0',
                    'MatchRegions-Demo-v0',
                    'MoveToCorner-Demo-v0',
                ]
            ]))

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_bench_micro_sweep_dm_control():
        """Tiny sweep over two dm_control configs (finger-spin is really easy for
        RL, and cheetah-run is really hard for RL)."""
        spec = dict(env_cfg=tune.grid_search(
            [
                {
                    'benchmark_name': 'dm_control',
                    'task_name': dm_control_env_name
                } for dm_control_env_name in [
                'finger-spin', 'cheetah-run', 'reacher-easy'
            ]
            ]))

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_bench_micro_sweep_procgen():
        """Tiny sweep over three procgen configs."""
        spec = dict(env_cfg=tune.grid_search(
            [
                {
                    'benchmark_name': 'procgen',
                    'task_name': procgen_env_name
                } for procgen_env_name in [
                'coinrun', 'miner', 'fruitbot'
            ]
            ]))

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_bench_full_sweep_procgen():
        """Sweep over all five procgen configs."""
        spec = dict(env_cfg=tune.grid_search(
            [
                {
                    'benchmark_name': 'procgen',
                    'task_name': procgen_env_name
                } for procgen_env_name in [
                'coinrun', 'miner', 'fruitbot', 'jumper', 'ninja'
            ]
            ]))

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_bench_one_task_magical():
        """Just one simple MAGICAL config."""
        env_cfg = {
            'benchmark_name': 'magical',
            'task_name': 'MatchRegions-Demo-v0',
            'magical_remove_null_actions': True,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_bench_magical_mr():
        """Bench on MAGICAL MatchRegions-Demo-v0."""
        env_cfg = {
            'benchmark_name': 'magical',
            'task_name': 'MatchRegions-Demo-v0',
            'magical_remove_null_actions': True,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_bench_magical_mtc():
        """Bench on MAGICAL MoveToCorner-Demo-v0."""
        env_cfg = {
            'benchmark_name': 'magical',
            'task_name': 'MoveToCorner-Demo-v0',
            'magical_remove_null_actions': True,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_bench_one_task_dm_control():
        """Just one simple dm_control config."""
        env_cfg = {
            'benchmark_name': 'dm_control',
            'task_name': 'cheetah-run',
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_base_repl_5000_batches():
        repl = {
            'batches_per_epoch': 500,
            'n_epochs': 10,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_base_repl_10000_batches():
        repl = {
            'batches_per_epoch': 1000,
            'n_epochs': 10,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_force_use_repl():
        stages_to_run = StagesToRun.REPL_AND_IL

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_repl_none():
        stages_to_run = StagesToRun.IL_ONLY

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_rl_only():
        stages_to_run = StagesToRun.RL_ONLY

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_repl_rl():
        stages_to_run = StagesToRun.REPL_AND_RL

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_repl_moco():
        stages_to_run = StagesToRun.REPL_AND_IL
        repl = {
            'algo': 'MoCoWithProjection',
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_repl_simclr():
        stages_to_run = StagesToRun.REPL_AND_IL
        repl = {
            'algo': 'SimCLR',
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_repl_dynamics():
        stages_to_run = StagesToRun.REPL_AND_IL
        repl = {
            'algo': 'DynamicsPrediction',
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_repl_temporal_cpc():
        stages_to_run = StagesToRun.REPL_AND_IL
        repl = {
            'algo': 'TemporalCPC',
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_repl_tcpc8():
        stages_to_run = StagesToRun.REPL_AND_IL
        repl = {
            'algo': 'TemporalCPC',
            'algo_params': {
                'target_pair_constructor_kwargs':{
                    'temporal_offset': 8
                }
            }
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_repl_tcpc4():
        stages_to_run = StagesToRun.REPL_AND_IL
        repl = {
            'algo': 'TemporalCPC',
            'algo_params': {
                'target_pair_constructor_kwargs':{
                    'temporal_offset': 4
                }
            }
        }

        _ = locals()
        del _


    @experiment_obj.named_config
    def cfg_data_repl_demos():
        """Training on both demos and random rollouts for the current
        environment."""
        repl = {
            'dataset_configs': [{'type': 'demos'}],
        }
        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_data_repl_demos_random():
        """Training on both demos and random rollouts for the current
        environment."""
        repl = {
            'dataset_configs': [{'type': 'demos'}, {'type': 'random'}],
        }
        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_data_repl_random():
        """Training on both demos and random rollouts for the current
        environment."""
        repl = {
            'dataset_configs': [{'type': 'random'}],
        }
        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_data_use_icml_on_chai_machines():
        # use data for ICML when running on perceptron/svm
        env_data = {
            'data_root': '/scratch/sam/ilr-data-icml/',
        }

        _ = locals()
        del _


    @experiment_obj.named_config
    def cfg_data_repl_demos_magical_mt():
        """Multi-task training on all MAGICAL tasks."""
        repl = {
            'dataset_configs': [
                {
                    'type': 'demos',
                    'env_cfg': {
                        'benchmark_name': 'magical',
                        'task_name': magical_task_name,
                    }
                } for magical_task_name in [
                    'MoveToCorner-Demo-v0',
                    'MoveToRegion-Demo-v0',
                    'MatchRegions-Demo-v0',
                    'MakeLine-Demo-v0',
                    'FixColour-Demo-v0',
                    'FindDupe-Demo-v0',
                    'ClusterColour-Demo-v0',
                    'ClusterShape-Demo-v0',
                ]
            ],
            'is_multitask': True,
        }
        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_data_repl_rand_demos_magical_mt():
        """Multi-task training on all MAGICAL tasks."""
        repl = {
            'dataset_configs': [
                {
                    'type': dataset_type,
                    'env_cfg': {
                        'benchmark_name': 'magical',
                        'task_name': magical_task_name,
                    }
                } for magical_task_name in [
                    'MoveToCorner-Demo-v0',
                    'MoveToRegion-Demo-v0',
                    'MatchRegions-Demo-v0',
                    'MakeLine-Demo-v0',
                    'FixColour-Demo-v0',
                    'FindDupe-Demo-v0',
                    'ClusterColour-Demo-v0',
                    'ClusterShape-Demo-v0',
                ]
                for dataset_type in ["demos", "random"]
            ],
            'is_multitask': True,
        }
        _ = locals()
        del _

    # Adding configs programatically like this seems like a terrible idea, but
    # I don't care
    def _():
        for task_name in [
                'MoveToCorner-Demo-v0', 'MoveToRegion-Demo-v0',
                'MatchRegions-Demo-v0', 'MakeLine-Demo-v0',
                'FixColour-Demo-v0', 'FindDupe-Demo-v0',
                'ClusterColour-Demo-v0', 'ClusterShape-Demo-v0'
        ]:
            prefix = task_name.split('-')[0]
            prefix_lower = prefix.lower()

            # this namedconfig is to train on demos from both demo variant and
            # test variant
            experiment_obj.add_named_config(
                f'cfg_data_repl_{prefix_lower}_demos_magical_test', {
                    'repl': {
                        'dataset_configs':
                        [{
                            'type': 'demos',
                            'env_cfg': {
                                'benchmark_name': 'magical',
                                'task_name': tn,
                            },
                        } for tn in
                         [f'{prefix}-Demo-v0', f'{prefix}-TestAll-v0']]
                    }
                })

            # this named config is to train on both demos and random rollouts,
            # each taken from both demo and test variant (i.e. we train on
            # everything related to this task)
            experiment_obj.add_named_config(
                f'cfg_data_repl_{prefix_lower}_rand_demos_magical_test', {
                    'repl': {
                        'dataset_configs':
                        [{
                            'type': data_type,
                            'env_cfg': {
                                'benchmark_name': 'magical',
                                'task_name': tn,
                            }
                        } for tn in
                         [f'{prefix}-Demo-v0', f'{prefix}-TestAll-v0']
                         for data_type in ['demos', 'random']]
                    }
                })

            # here we train on demos from the demo variant, along with random
            # rollouts from both the demo variant and from test variants
            experiment_obj.add_named_config(
                (f'cfg_data_repl_{prefix_lower}_test_demos_and_all_random'
                 '_rollouts'), {
                    'repl': {
                        'dataset_configs': [{
                            'type': 'demos',
                            'env_cfg': {
                                'benchmark_name': 'magical',
                                'task_name': task_name,
                            }
                        }] + [{
                            'type': 'random',
                            'env_cfg': {
                                'benchmark_name': 'magical',
                                'task_name': tn,
                            }
                        } for tn in
                              [f'{prefix}-Demo-v0', f'{prefix}-TestAll-v0']]
                    }
                })

            _ = locals()
            del _

    # call fn above & then delete it (avoids polluting namespace of this outer
    # fn)
    _()
    del _

    @experiment_obj.named_config
    def cfg_data_repl_rand_demos_magical_mt_test():
        """Multi-task training on all MAGICAL tasks."""
        repl = {
            'dataset_configs': [
                {
                    'type': dataset_type,
                    'env_cfg': {
                        'benchmark_name': 'magical',
                        'task_name': magical_task_name,
                    }
                }
                for variant in ['Demo', 'TestAll']
                for magical_task_name in [
                    f'MoveToCorner-{variant}-v0',
                    f'MoveToRegion-{variant}-v0',
                    f'MatchRegions-{variant}-v0',
                    f'MakeLine-{variant}-v0',
                    f'FixColour-{variant}-v0',
                    f'FindDupe-{variant}-v0',
                    f'ClusterColour-{variant}-v0',
                    f'ClusterShape-{variant}-v0',
                ]
                for dataset_type in ["demos", "random"]
            ],
            'is_multitask': True,
        }
        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_data_il_5traj():
        """Use only 5 trajectories for IL training."""
        il_train = {
            'n_traj': 5,
        }
        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_data_il_hc_extended():
        """Use extended HalfCheetah dataset for IL training."""
        env_data = {
            'dm_control_demo_patterns': {
                'cheetah-run':
                    'data/dm_control/extended-cheetah-run-*500traj.pkl.gz',
            }
        }
        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_repl_ceb():
        stages_to_run = StagesToRun.REPL_AND_IL
        repl = {
            'algo': 'CEB',
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_repl_vae():
        repl = {
            'algo': 'VariationalAutoencoder',
            'algo_params': {'batch_size': 32},
        }
        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_repl_inv_dyn():
        repl = {
            'algo': 'InverseDynamicsPrediction',
            'algo_params': {'batch_size': 32},
        }
        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_il_bc_nofreeze():
        il_train = {
            'algo': 'bc',
            'bc': {
                'n_batches': 15000,
            },
            'freeze_encoder': False,
        }

        _ = locals()
        del _


    @experiment_obj.named_config
    def cfg_dqn_nofreeze():
        dqn_train = {
            'freeze_encoder': False,
        }

        _ = locals()
        del _


    @experiment_obj.named_config
    def cfg_il_bc_500k_nofreeze():
        il_train = {
            'algo': 'bc',
            'bc': {
                'n_batches': 500000,
            },
            'freeze_encoder': False,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_il_bc_200k_nofreeze():
        il_train = {
            'algo': 'bc',
            'bc': {
                'n_batches': 200000,
            },
            'freeze_encoder': False,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_il_bc_20k_nofreeze():
        il_train = {
            'algo': 'bc',
            'bc': {
                'n_batches': 20000,
            },
            'freeze_encoder': False,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_il_bc_15k_freeze():
        il_train = {
            'algo': 'bc',
            'bc': {
                'n_batches': 15000,
            },
            'freeze_encoder': True,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_il_gail_200k_nofreeze():
        il_train = {
            'algo': 'gail',
            'gail': {
                # TODO(sam): make a new config with a larger value once you
                # know what works for most envs.
                'total_timesteps': 200000,
            },
            'freeze_encoder': False,
        }
        venv_opts = {
            'n_envs': 32,
            'venv_parallel': True,
            'parallel_workers': 8,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_il_gail_magical_250k_nofreeze():
        """GAIL config tailored to MAGICAL tasks."""
        il_train = {
            'algo': 'gail',
            'gail': {
                # These HP values based on defaults in il_train.py as of
                # 2021-02-01 (which I believe I had already tuned for MAGICAL)
                # Update 2021-03-17: looks like those didn't work; now trying
                # hyperparams that I used for tuning (which were surprisingly
                # good).
                # Update 2021-03-19: updating these AGAIN to match best run
                # during MatchRegions tuning. Still NFI why this is so hard to
                # train :(
                # (current hypothesis: the HPs don't really matter, but the
                # fact that I'm only using 5 trajectories is killing
                # performance)
                'total_timesteps': 250000,
                'ppo_n_steps': 5,
                'ppo_n_epochs': 10,
                'ppo_batch_size': 48,
                'ppo_init_learning_rate': 8.87e-5,
                'ppo_final_learning_rate': 0.0,
                'ppo_gamma': 0.99,
                'ppo_gae_lambda': 0.70,
                'ppo_ent': 1.5e-7,
                'ppo_adv_clip': 0.037,
                'disc_n_updates_per_round': 3,
                'disc_batch_size': 48,
                'disc_lr': 0.001,
                'disc_augs': {
                    'color_jitter_mid': False,
                    'erase': True,
                    # this looks a bit sus (but still leaving it in)
                    'flip_lr': True,
                    'gaussian_blur': True,
                    'noise': True,
                    'rotate': False,
                    'translate': True,
                }
            },
            'freeze_encoder': False,
        }
        venv_opts = {
            'n_envs': 32,
            'venv_parallel': True,
            'parallel_workers': 8,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_il_gail_dmc_250k_nofreeze():
        """GAIL config tailored to dm_control tasks. This was specifically
        tuned for HalfCheetah at 500k steps, but should work for other tasks
        too (HalfCheetah is just the hardest one)."""
        il_train = {
            'algo': 'gail',
            'gail': {
                # Tuning guide for HalfCheetah is at
                # https://docs.google.com/document/d/1k6cEgszHWEmYZG7X8R3m6XySHqr_inoFTuNLbRnSZ8M/edit#bookmark=id.n3ge30xuplwx
                # (in Jan/Feb shared notebook)
                'total_timesteps': 250000,
                'ppo_n_steps': 8,
                'ppo_n_epochs': 12,
                'ppo_batch_size': 64,
                'ppo_init_learning_rate': 1e-4,
                'ppo_gamma': 0.99,
                'ppo_gae_lambda': 0.8,
                'ppo_ent': 1e-8,
                'ppo_adv_clip': 0.02,
                'disc_n_updates_per_round': 6,
                'disc_batch_size': 48,
                'disc_lr': 1e-3,
                'disc_augs': {
                    'rotate': True,
                    'noise': True,
                    'erase': True,
                    'gaussian_blur': True,
                    # note lack of color_jitter_mid, flip_lr, translate_ex; I
                    # put those into HP optimiser, but didn't find that they
                    # worked well
                }
            },
            'freeze_encoder': False,
        }
        il_test = {
            'deterministic_policy': True,
        }
        venv_opts = {
            'n_envs': 32,
            'venv_parallel': True,
            'parallel_workers': 8,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def gail_mr_config_2021_03_29():
        """500k-step GAIL config tuned (and tested) for MatchRegions. Should get
        between ~0.42 and ~0.49 mean score (averaged across all variants)."""
        il_train = {
            "algo": "gail",
            "dataset_configs": [{"type": "demos"}],
            "freeze_encoder": False,
            "gail": {
                "disc_augs": {
                    "color_jitter_mid": True,
                    "erase": True,
                    "flip_lr": True,
                    "gaussian_blur": True,
                    "noise": True,
                    "rotate": True,
                    "translate_ex": False
                },
                "disc_batch_size": 48,
                "disc_lr": 0.0005718485231390044,
                "disc_n_updates_per_round": 2,
                "log_interval_steps": 10000,
                "ppo_adv_clip": 0.006035475207613115,
                "ppo_batch_size": 48,
                "ppo_clip_reward": float('inf'),
                "ppo_ent": 4.5401344442176644e-08,
                "ppo_final_learning_rate": 0.0,
                "ppo_gae_lambda": 0.7648620566937083,
                "ppo_gamma": 0.9853500136087187,
                "ppo_init_learning_rate": 0.00025070798829149955,
                "ppo_max_grad_norm": 1.0,
                "ppo_n_epochs": 7,
                "ppo_n_steps": 7,
                "ppo_norm_reward": True,
                "ppo_reward_std": 0.01,
                "save_every_n_steps": 10000000000,
                "total_timesteps": 500000
            },
            "log_std_init": 0.0,
            "ortho_init": False,
            "postproc_arch": [],
            "print_policy_summary": True,
            "shuffle_buffer_size": 1024,
            "torch_num_threads": 1
        }
        env_cfg = {
            "benchmark_name": "magical",
            "task_name": "MatchRegions-Demo-v0"
        }
        venv_opts = {
            # FIXME(sam): figure out whether having only 8 procs (instead of
            # 32) crashes perf---might be a bug in my env stuff
            "n_envs": 32,
            "venv_parallel": True
        }

        _ = locals()
        del _
