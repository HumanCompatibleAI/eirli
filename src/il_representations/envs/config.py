"""Common config utilities for all benchmarks, including configuration
variables for creating environments (`env_cfg_ingredient`), for creating
vecenvs (`venv_opts_ingredient`), and for loading data
(`env_data_ingredient`)."""
import os

from sacred import Ingredient

ALL_BENCHMARK_NAMES = {"atari", "magical", "dm_control", "minecraft",
                       "procgen"}

# see env_cfg_defaults docstring for description of this ingredient
env_cfg_ingredient = Ingredient('env_cfg')


@env_cfg_ingredient.config
def env_cfg_defaults():
    """The `env_cfg` ingredient contains all (and only) the config variables
    that would be necessary to construct a Gym environment (*not* a vec env,
    which would require the additional settings from the venv_opts
    ingredient). `env_cfg` is thus used by every script, including the repL
    script, the IL training script, and the IL testing script."""

    # set this to one of the options in ALL_BENCHMARK_NAMES
    benchmark_name = 'dm_control'
    # format for env_name depends in benchmark:
    # - MAGICAL: use env names without preprocessors, like
    #   MoveToRegion-Demo-v0, ClusterShape-TestAll-v0, etc.
    # - dm_control: in dm_control parlance, use [domain name]-[task name], like
    #  'finger-spin', 'cheetah-run', etc.
    # - Atari: use fully qualified Gym names (e.g. PongNoFrameskip-v4)
    task_name = 'finger-spin'

    # #################################
    # MAGICAL-specific config variables
    # #################################

    # Name of preprocessor to use with MAGICAL. See
    # https://github.com/qxcv/magical/blob/pyglet1.5/README.md#preprocessors
    # for list of preprocessors.
    magical_preproc = 'LoResCHW4E'
    # null actions should be ignored when computing losses
    # (this is useful for BC, to stop the agent from getting 'stuck' in some
    # position)
    magical_remove_null_actions = False

    # ####################################
    # dm_control-specific config variables
    # ####################################

    # number of successive frames to stack for dm_control
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

    # ###############################
    # Minecraft-specific config variables
    # ###############################
    minecraft_max_env_steps = None

    # ###############################
    # Procgen-specific config variables
    # ###############################
    procgen_frame_stack = 4

    locals()


# see venv_opts_defaults docstring for description of this ingredient
venv_opts_ingredient = Ingredient('venv_opts')


@venv_opts_ingredient.config
def venv_opts_defaults():
    """Configuration ingredient for Sacred experiments that need to construct
    vectorised environments (aka vecenvs, or just venvs). Usually this
    ingredient will be used in conjunction with env_cfg_defaults, so the code
    knows the name of the environment that it needs to construct, etc. However,
    the config variables in this ingredient have been split out from
    env_cfg_defaults because there are some situations in which we only need
    metadata about the environment itself, and do not need to a vecenv or
    generate samples (e.g. the repL script is in this category)."""

    # should venvs be parallel?
    venv_parallel = True
    # how many envs constitute a batch step (regardless of parallelisation)
    n_envs = 2

    locals()


# see env_data_ingredient docstring for description of this ingredient
env_data_ingredient = Ingredient('env_data')


@env_data_ingredient.config
def env_data_defaults():
    """Additional config variables for Sacred experiments that need to load
    saved data. That includes the repL script and IL training script, but not
    the IL testing script, for instance."""

    # Root directory for data; useful when script is being run under Ray Tune,
    # which changes the working directory. The default tries to point at the
    # root of the repo, which should contain a symlink to the data directory.
    _this_file_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(os.path.join(_this_file_dir, '../../../'))
    del _this_file_dir

    # Maximum number of trajectories to load from webdatasets when using
    # `auto.load_wds_datasets`. Set to None to have no limit. Code will error
    # if a numerical limit is supplied but there are fewer trajectories in the
    # dataset.
    wds_n_trajs = None
    # Similar, but limits maximum number of transitions. User should not supply
    # both wds_n_trajs and wds_n_trans.
    wds_n_trans = None

    # ########################
    # MAGICAL config variables
    # ########################

    # demonstration directories for each MAGICAL task
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
        'MoveToCorner-Demo-v0': 'data/magical/move-to-corner/',
        'MoveToRegion-Demo-v0': 'data/magical/move-to-region/',
        'MatchRegions-Demo-v0': 'data/magical/match-regions/',
        'MakeLine-Demo-v0': 'data/magical/make-line/',
        'FixColour-Demo-v0': 'data/magical/fix-colour/',
        'FindDupe-Demo-v0': 'data/magical/find-dupe/',
        'ClusterColour-Demo-v0': 'data/magical/cluster-colour/',
        'ClusterShape-Demo-v0': 'data/magical/cluster-shape/',
    }

    # ###########################
    # dm_control config variables
    # ###########################

    # glob patterns used to locate demonstrations for each dm_control task
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

    # paths to saved demonstrations for some Atari environments
    atari_demo_paths = {
        'BreakoutNoFrameskip-v4':
        'data/atari/BreakoutNoFrameskip-v4_rollouts_500_ts_100_traj.npz',
        'PongNoFrameskip-v4':
        'data/atari/PongNoFrameskip-v4_rollouts_500_ts_100_traj.npz',
    }

    # ###########################
    # ProcGen config variables
    # ###########################
    procgen_demo_paths = {
        'coinrun': 'procgen/demo_coinrun.pickle',
        'fruitbot': 'procgen/demo_fruitbot.pickle',
        'ninja': 'procgen/demo_ninja.pickle',
        'jumper': 'procgen/demo_jumper.pickle',
        'miner': 'procgen/demo_miner.pickle',
        'fruitbot': 'procgen/demo_fruitbot.pickle',
    }

    _ = locals()
    del _
