"""Common config utilities for all benchmarks."""

from sacred import Ingredient

benchmark_ingredient = Ingredient('benchmark')


@benchmark_ingredient.config
def bench_defaults():
    # set this to "atari", "magical", "dm_control"
    benchmark_name = 'dm_control'
    # should venvs be parallel?
    venv_parallel = True
    # how many envs constitute a batch step (regardless of parallelisation)
    n_envs = 8
    # this should be a number of trajectories to return, or None if returning
    # all available trajectories is okay
    n_traj = None
    # root directory for data; useful when script is being run under Ray Tune,
    # which changes the working directory
    data_root = '.'

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
    magical_env_prefix = 'MoveToRegion'
    magical_preproc = 'LoResCHW4E'
    # this should probably be True for all BC runs, False for GAIL runs
    # (although TBH it doesn't really matter for GAIL)
    magical_remove_null_actions = False

    # ###########################
    # dm_control config variables
    # ###########################

    # mapping from short dm_control env names to complete Gym env names
    # registered by dm_control_envs.py
    dm_control_full_env_names = {
        'finger-spin': 'DMC-Finger-Spin-v0',
        'cheetah-run': 'DMC-Cheetah-Run-v0',
        'walker-walk': 'DMC-Walker-Walk-v0',
        'cartpole-swingup': 'DMC-Cartpole-Swingup-v0',
        'reacher-easy': 'DMC-Reacher-Easy-v0',
        'ball-in-cup-catch': 'DMC-Ball-In-Cup-Catch-v0',
    }
    dm_control_demo_patterns = {
        'finger-spin': 'data/dm_control/finger-spin-*.pkl.gz',
        'cheetah-run': 'data/dm_control/cheetah-run-*.pkl.gz',
        'walker-walk': 'data/dm_control/walker-walk-*.pkl.gz',
        'cartpole-swingup': 'data/dm_control/cartpole-swingup-*.pkl.gz',
        'reacher-easy': 'data/dm_control/reacher-easy-*.pkl.gz',
        'ball-in-cup-catch': 'data/dm_control/ball-in-cup-catch-*.pkl.gz',
    }
    dm_control_env = 'finger-spin'

    # ###########################
    # Atari config variables
    # ###########################

    atari_env_id = 'PongNoFrameskip-v4'
    atari_demo_paths = {
        'BreakoutNoFrameskip-v4':
        "data/atari/BreakoutNoFrameskip-v4_rollouts_500_ts_100_traj.npz",
        'PongNoFrameskip-v4':
        "data/atari/pong.npz",
    }
