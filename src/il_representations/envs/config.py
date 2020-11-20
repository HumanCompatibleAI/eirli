"""Common config utilities for all benchmarks."""
import os

from sacred import Ingredient

env_cfg_ingredient = Ingredient('env_cfg')
ALL_BENCHMARK_NAMES = {"atari", "magical", "dm_control"}


@env_cfg_ingredient.config
def env_cfg_defaults():
    """Minimal set of variables that are necessary to construct an environment.
    Include environment names and benchmark-specific settings (e.g. observation
    preprocessing)."""

    # set this to one of the options in ALL_BENCHMARK_NAMES
    benchmark_name = 'dm_control'
    # format for env_name depends in benchmark:
    # - MAGICAL: use env name prefixes like MoveToRegion, ClusterShape, etc.
    # - dm_control: in dm_control parlance, use [domain name]-[task name], like
    #  'finger-spin', 'cheetah-run', etc.
    # - Atari: use fully qualified Gym names (e.g. PongNoFrameskip-v4)
    task_name = 'finger-spin'

    # #################################
    # MAGICAL-specific config variables
    # #################################

    magical_preproc = 'LoResCHW4E'
    # null actions should be ignored when computing losses
    # (this is useful for BC, to stop the agent from getting 'stuck' in some
    # position)
    magical_remove_null_actions = False

    # ####################################
    # dm_control-specific config variables
    # ####################################

    dm_control_frame_stack = 3
    # mapping from short dm_control env names to complete Gym env names
    # registered by dm_control_envs.py
    # FIXME(sam): maybe get rid of this eventually, it doesn't save much time
    dm_control_full_env_names = {
        'finger-spin': 'DMC-Finger-Spin-v0',
        'cheetah-run': 'DMC-Cheetah-Run-v0',
        'walker-walk': 'DMC-Walker-Walk-v0',
        'cartpole-swingup': 'DMC-Cartpole-Swingup-v0',
        'reacher-easy': 'DMC-Reacher-Easy-v0',
        'ball-in-cup-catch': 'DMC-Ball-In-Cup-Catch-v0',
    }

    # ###############################
    # Atari-specific config variables
    # (none currently present)
    # ###############################

    _ = locals()
    del _


venv_opts_ingredient = Ingredient('venv_opts')


@venv_opts_ingredient.config
def venv_opts_defaults():
    """Configuration variables for tasks that use vec envs."""
    # should venvs be parallel?
    venv_parallel = True
    # how many envs constitute a batch step (regardless of parallelisation)
    n_envs = 2

    _ = locals()
    del _


env_data_ingredient = Ingredient('env_data')


@env_data_ingredient.config
def env_data_defaults():
    """Config variables for scripts that need to load data. Mostly these are
    default paths for data; they should not have to be changed unless you want
    use different default paths."""

    # Root directory for data; useful when script is being run under Ray Tune,
    # which changes the working directory. The default tries to point at the
    # root of the repo, which should contain a symlink to the data directory.
    _this_file_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(os.path.join(_this_file_dir, '../../../'))
    del _this_file_dir

    processed_data_dirs = {
        'magical': {
            'MoveToCorner': {
                'demos': 'data/processed/magical/demos/move-to-corner/',
                'random': 'data/processed/magical/random/move-to-corner/',
            },
            'MoveToRegion': {
                'demos': 'data/processed/magical/demos/move-to-region/',
                'random': 'data/processed/magical/random/move-to-region/',
            },
            'MatchRegions': {
                'demos': 'data/processed/magical/demos/match-regions/',
                'random': 'data/processed/magical/random/match-regions/',
            },
            'MakeLine': {
                'demos': 'data/processed/magical/demos/make-line/',
                'random': 'data/processed/magical/random/make-line/',
            },
            'FixColour': {
                'demos': 'data/processed/magical/demos/fix-colour/',
                'random': 'data/processed/magical/random/fix-colour/',
            },
            'FindDupe': {
                'demos': 'data/processed/magical/demos/find-dupe/',
                'random': 'data/processed/magical/random/find-dupe/',
            },
            'ClusterColour': {
                'demos': 'data/processed/magical/demos/cluster-colour/',
                'random': 'data/processed/magical/random/cluster-colour/',
            },
            'ClusterShape': {
                'demos': 'data/processed/magical/demos/cluster-shape/',
                'random': 'data/processed/magical/random/cluster-shape/',
            },
        },
        'dm_control': {
            'finger-spin': {
                'demos': 'data/processed/dm_control/demos/finger-spin/',
                'random': 'data/processed/dm_control/random/finger-spin/',
            },
            'cheetah-run': {
                'demos': 'data/processed/dm_control/demos/cheetah-run/',
                'random': 'data/processed/dm_control/random/cheetah-run/',
            },
            'walker-walk': {
                'demos': 'data/processed/dm_control/demos/walker-walk/',
                'random': 'data/processed/dm_control/random/walker-walk/',
            },
            'cartpole-swingup': {
                'demos': 'data/processed/dm_control/demos/cartpole-swingup/',
                'random': 'data/processed/dm_control/random/cartpole-swingup/',
            },
            'reacher-easy': {
                'demos': 'data/processed/dm_control/demos/reacher-easy/',
                'random': 'data/processed/dm_control/random/reacher-easy/',
            },
            'ball-in-cup-catch': {
                'demos': 'data/processed/dm_control/demos/ball-in-cup-catch/',
                'random': 'data/processed/dm_control/random/ball-in-cup-catch/',
            },
        },
        'atari': {
            'BreakoutNoFrameskip-v4': {
                'demos': "data/processed/atari/demos/breakout/",
                'random': "data/processed/atari/random/breakout/",
            },
            'PongNoFrameskip-v4': {
                'demos': "data/processed/atari/demos/pong/",
                'random': "data/processed/atari/random/pong/",
            },
        },
    }

    # ########################
    # MAGICAL config variables
    # ########################

    magical_demo_dirs = {
        # These defaults work well if you set up symlinks.
        # For example, on perceptron/svm, you can set up the appropriate
        # symlink structure with a script like this:
        #
        # mkdir -p data/magical;
        # for dname in move-to-corner match-regions make-line fix-colour \
        #     find-dupe cluster-colour cluster-shape move-to-region;
        # do
        #     ln -s /scratch/sam/il-demos/magical/${dname}-2020-*/ \
        #         ./data/magical/${dname};
        # done
        'MoveToCorner': 'data/magical/move-to-corner/',
        'MoveToRegion': 'data/magical/move-to-region/',
        'MatchRegions': 'data/magical/match-regions/',
        'MakeLine': 'data/magical/make-line/',
        'FixColour': 'data/magical/fix-colour/',
        'FindDupe': 'data/magical/find-dupe/',
        'ClusterColour': 'data/magical/cluster-colour/',
        'ClusterShape': 'data/magical/cluster-shape/',
    }

    # ###########################
    # dm_control config variables
    # ###########################

    dm_control_demo_patterns = {
        'finger-spin': 'data/dm_control/finger-spin-*.pkl.gz',
        'cheetah-run': 'data/dm_control/cheetah-run-*.pkl.gz',
        'walker-walk': 'data/dm_control/walker-walk-*.pkl.gz',
        'cartpole-swingup': 'data/dm_control/cartpole-swingup-*.pkl.gz',
        'reacher-easy': 'data/dm_control/reacher-easy-*.pkl.gz',
        'ball-in-cup-catch': 'data/dm_control/ball-in-cup-catch-*.pkl.gz',
    }

    # ###########################
    # Atari config variables
    # ###########################

    atari_demo_paths = {
        'BreakoutNoFrameskip-v4':
        'data/atari/BreakoutNoFrameskip-v4_rollouts_500_ts_100_traj.npz',
        'PongNoFrameskip-v4':
        'data/atari/PongNoFrameskip-v4_rollouts_500_ts_100_traj.npz',
    }

    _ = locals()
    del _
