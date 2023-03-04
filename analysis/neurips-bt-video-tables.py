#!/usr/bin/env python3
"""Tool to make some last-minute plots for the NeurIPS BT video.

These tables are similar to the ones in the paper, but show _only_ significance
of results, and not the actual results themselves. This makes it easier to tell
at a glance what works and what doesn't. It is designed to be run on data
extracted straight from our preprint using Tabula; the resulting input CSV
looked like this (note that the columns "BC aug" and "BC no aug" are treated
specially by this code; they are hardcoded as `BASELINE_COLUMNS` for use in
t-tests):

    Task,Dynamics,InvDyn,SimCLR,TemporalCPC,VAE,BC aug,BC no aug
    cheetah-run,723±14,716±23,717±11,716±16,724±12,690±17,617±34
    finger-spin,755±6,755±12,732±15,725±12,755±3,730±9,940±4
    reacher-easy,898±19,903±10,889±14,912±18,903±8,874±21,452±34
    …
"""
import argparse

import pandas as pd
from scipy.stats import ttest_ind_from_stats
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('file',
                    type=argparse.FileType(mode='r'),
                    help='file to read from when generating new table')
BASELINE_COLUMNS = ['BC aug', 'BC no aug']


def main(args):
    # Data is 2D grid with repL methods as columns and environments as rows.
    # Each cell is "mean±std" (very approximately).
    data = pd.read_csv(args.file, header=0, index_col=None)
    melted = pd.melt(data,
                     id_vars=['Task'],
                     value_name='Return',
                     var_name='Method')
    returns = melted['Return']
    melted['Mean'] = returns.map(lambda c: float(c.split('±', 1)[0]))
    melted['Std'] = returns.map(lambda c: float(c.split('±', 1)[1]))
    for baseline in BASELINE_COLUMNS:
        baseline_entries = melted[melted['Method'] == baseline]
        baseline_dict = {}

        def dict_builder(r):
            baseline_dict[r['Task']] = [r['Mean'], r['Std']]

        baseline_entries.apply(dict_builder, axis=1)
        baseline_mean_std = melted.apply(
            lambda r: baseline_dict.get(r['Task']),
            axis=1,
            result_type='expand')
        data_w_baseline = melted.copy()
        data_w_baseline['Baseline mean'] = baseline_mean_std[0]
        data_w_baseline['Baseline std'] = baseline_mean_std[1]
        _, p_values = ttest_ind_from_stats(data_w_baseline['Mean'],
                                           data_w_baseline['Std'],
                                           5,
                                           data_w_baseline['Baseline mean'],
                                           data_w_baseline['Baseline std'],
                                           5,
                                           equal_var=False,
                                           alternative='greater')
        # add marks for p<0.05
        sig_indices = (p_values < 0.01) + 0
        data_w_baseline['Significant'] = np.array(['', '✱'])[sig_indices]

        def single_aggfunc(s):
            assert len(s) == 1, s
            return next(iter(s))

        pivoted = pd.pivot_table(data_w_baseline,
                                 columns='Method',
                                 index='Task',
                                 values='Significant',
                                 aggfunc=single_aggfunc)

        print(f'Pivot table for baseline {baseline}:')
        print(pivoted)
        print()
        pivoted.to_csv(f'Pivot table with baseline {baseline}.csv')


if __name__ == '__main__':
    main(parser.parse_args())
