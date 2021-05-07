from il_representations.scripts.utils import StagesToRun
from ray import tune
# TODO(sam): GAIL configs


def make_chain_configs(experiment_obj):

    @experiment_obj.named_config
    def cfg_use_magical():
        # see il_representations/envs/config for examples of what should go here
        env_cfg = {
            'benchmark_name': 'magical',
            # MatchRegions is of intermediate difficulty
            # (TODO(sam): allow MAGICAL to load data from _all_ tasks at once, so
            # we can try multi-task repL)
            'task_name': 'MatchRegions',
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
    def cfg_base_3seed_1cpu_pt2gpu():
        """Basic config that does three samples per config, using 1 CPU cores and
        0.2 of a GPU."""
        use_skopt = False
        tune_run_kwargs = dict(num_samples=3,
                               # retry on (node) failure
                               max_failures=2,
                               fail_fast=False,
                               resources_per_trial=dict(
                                   cpu=1,
                                   gpu=0.2,
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
                'MoveToCorner',
                'MoveToRegion',
                'FixColour',
                'MatchRegions',
                # 'FindDupe',
                # 'MakeLine',
                # 'ClusterColour',
                # 'ClusterShape',
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
                'MoveToRegion', 'MatchRegions', 'MoveToCorner'
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
    def cfg_run_few_trajs_long_dm_control():
        """For experiments running very few BC trajs"""
        spec = dict(il_train={
            'bc': {
                'n_batches': 10000000,
                # 'n_trajs': tune.grid_search([1, 10, 30]),
                'save_every_n_batches': 5e4
            }
        })

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_bench_one_task_magical():
        """Just one simple MAGICAL config."""
        env_cfg = {
            'benchmark_name': 'magical',
            'task_name': 'MatchRegions',
            'magical_remove_null_actions': True,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_bench_magical_mr():
        """Bench on MAGICAL MatchRegions."""
        env_cfg = {
            'benchmark_name': 'magical',
            'task_name': 'MatchRegions',
            'magical_remove_null_actions': True,
        }

        _ = locals()
        del _

    @experiment_obj.named_config
    def cfg_bench_magical_mtc():
        """Bench on MAGICAL MoveToCorner."""
        env_cfg = {
            'benchmark_name': 'magical',
            'task_name': 'MoveToCorner',
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
    def cfg_repl_temporal_cpc():
        stages_to_run = StagesToRun.REPL_AND_IL
        repl = {
            'algo': 'TemporalCPC',
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
                    'MoveToCorner',
                    'MoveToRegion',
                    'MatchRegions',
                    'MakeLine',
                    'FixColour',
                    'FindDupe',
                    'ClusterColour',
                    'ClusterShape',
                ]
            ],
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
    def cfg_il_bc_freeze():
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
