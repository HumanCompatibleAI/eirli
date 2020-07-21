"""Common config utilities for all benchmarks."""

from sacred import Ingredient

benchmark_ingredient = Ingredient('benchmark')


@benchmark_ingredient.config
def bench_default():
    benchmark_name = 'Benchmark not configured'


@benchmark_ingredient.named_config
def bench_magical():
    benchmark_name = 'magical'
    magical_demo_dirs = {
        # these defaults work well if you set up symlinks
        'MoveToCorner': 'data/magical/move-to-corner/',
        'MoveToRegion': 'data/magical/move-to-region/',
        'MatchRegions': 'data/magical/match-regions/',
        'MakeLine': 'data/magical/make-line/',
        'FixColour': 'data/magical/fix-colour/',
        'FindDupe': 'data/magical/find-dupe/',
        'ClusterColour': 'data/magical/cluster-colour/',
        'ClusterShape': 'data/magical/cluster-shape/',

    }
    magical_env_name = 'MoveToCorner'


@benchmark_ingredient.named_config
def bench_dm_control():
    # FIXME(sam): implement all of this
    benchmark_name = 'dm_control'
