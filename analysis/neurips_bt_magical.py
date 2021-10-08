"""Script MAGICAL results as LaTeX for the NeurIPS benchmark track.

Ingests data from this spreadsheet:

    https://docs.google.com/spreadsheets/u/1/d/1bIRKNlLHOeZSsQTkP1oluL1KfJACb-WeTID7Ve-3R_U/edit#gid=1421822029

It then does some t-tests and applies some formatting, before writing out
results as LaTeX. Designed to generate table rows for the existing tables in
our paper; I don't know what I'm going to do for the appendix tables."""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats

HIGHLIGHT_COLOR = 'fff7df'
# hard-coded, bleh
CONTROL_NAMES = {'repl_noid_5demos_random', 'neurips_control_bc_augs',
                 'neurips_control_gail_augs'}
# CONTROL_NAMES = {'neurips_repl_bc_cfg_repl_simclr_cfg_data_repl_5demos_random'}
# Ordering for column names (first 7 are pretrain, second 7 are joint).
# Order in Overleaf:
# Dynamics, InvDyn, SimCLR, TemporalCPC, VAE, Augs, No Augs
COL_ORDER = [
    # for pretraining + BC
    'neurips_repl_bc_icml_dynamics_cfg_data_repl_5demos_random',
    'neurips_repl_bc_icml_inv_dyn_cfg_data_repl_5demos_random',
    'neurips_repl_bc_cfg_repl_simclr_asymm_proj_cfg_data_repl_5demos_random',
    'neurips_repl_bc_cfg_repl_simclr_no_proj_cfg_data_repl_5demos_random',
    'neurips_repl_bc_cfg_repl_simclr_ceb_loss_cfg_data_repl_5demos_random',
    'neurips_repl_bc_cfg_repl_simclr_momentum_cfg_data_repl_5demos_random',
    'neurips_repl_bc_cfg_repl_simclr_cfg_data_repl_5demos_random',
    'neurips_repl_bc_cfg_repl_tcpc8_cfg_data_repl_5demos_random',
    'neurips_repl_bc_icml_vae_cfg_data_repl_5demos_random',
    'neurips_control_bc_augs',
    'neurips_control_bc_noaugs',
    # for joint training + BC
    'repl_fd_5demos_random',
    'repl_id_5demos_random',
    'repl_simclr_5demos_random',
    'repl_tcpc8_5demos_random',
    'repl_vae_5demos_random',
    'repl_noid_5demos_random',
    'repl_noid_noaugs_5demos_random',
    # for pretraining + GAIL
    'neurips_repl_gail_icml_dynamics_cfg_data_repl_5demos_random',
    'neurips_repl_gail_icml_inv_dyn_cfg_data_repl_5demos_random',
    'neurips_repl_gail_cfg_repl_simclr_cfg_data_repl_5demos_random',
    'neurips_repl_gail_cfg_repl_tcpc8_cfg_data_repl_5demos_random',
    'neurips_repl_gail_icml_vae_cfg_data_repl_5demos_random',
    # for pretraining + GAIL (procgen)
    'neurips_repl_gail_icml_dynamics_cfg_data_repl_demos',
    'neurips_repl_gail_icml_inv_dyn_cfg_data_repl_demos',
    'neurips_repl_gail_cfg_repl_simclr_cfg_data_repl_demos',
    'neurips_repl_gail_cfg_repl_tcpc8_cfg_data_repl_demos',
    'neurips_repl_gail_icml_vae_cfg_data_repl_demos',
    # controls for pretraining + GAIL
    'neurips_control_gail_augs',
    'neurips_control_gail_noaugs',
]
# Ordering of MAGICAL variants as we go down the rows.
MAGICAL_VARIANT_ORDER = [
  '-Demo',
  '-TestDynamics',
  '-TestColour',
  '-TestShape',
  '-TestJitter',
  '-TestLayout',
  '-TestCountPlus',
  '-TestAll',
  'Average',
]
EXP_IDENT_TO_NUM = {exp_ident: idx for idx, exp_ident in enumerate(COL_ORDER)}

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    '--full', action='store_true', default=False,
    help='create a table with scores for all variants, not just the average')
parser.add_argument('csv_files', nargs='*', help='paths to CSV files')


def get_p_val_df(mean_df, std_df, n_df, control_idx, alternative="greater"):
    """Get p-values under one-sided t-test from dataframes of mean and
    stddev."""
    test_means = mean_df[mean_df.index != control_idx]
    control_means = mean_df.loc[control_idx]

    test_std = std_df[std_df.index != control_idx]
    control_std = std_df.loc[control_idx]

    test_ns = n_df[n_df.index != control_idx]
    control_ns = n_df.loc[control_idx]

    t_stats, p_vals = ttest_ind_from_stats(test_means,
                                           test_std,
                                           test_ns,
                                           control_means,
                                           control_std,
                                           control_ns,
                                           equal_var=False,
                                           alternative=alternative)

    return p_vals


def process_df(df, *, full=False):
    # DF columns:
    # train_env, exp_ident, test_env, return_mean, return_std, count

    # first we find controls indices for each (train_env, test_env) pair
    controls = {}

    def compute_key(row):
        return (row['train_env'], row['test_env'])

    def find_control(row):
        if row['exp_ident'] not in CONTROL_NAMES:
            return False
        key = compute_key(row)
        if key in controls:
            old_row = df.loc[controls[key]]
            raise ValueError(
                f"Found duplicate controls:\n{old_row}\n  and\n{row}")
        controls[key] = row.name
        return True

    df.apply(find_control, axis=1)

    def compute_p_value(row):
        """Row mapper that does one-sided t-test on return of this method vs.
        return of corresponding control."""
        key = compute_key(row)
        control_index = controls[key]
        if control_index == row.name:
            return 1.0
        control_row = df.loc[control_index]
        t_stat, p_val = ttest_ind_from_stats(row["return_mean"],
                                             row["return_std"],
                                             row["count"],
                                             control_row["return_mean"],
                                             control_row["return_std"],
                                             control_row["count"],
                                             equal_var=False,
                                             alternative="greater")
        if not np.isfinite(p_val):
            # this happens when stddev of both things is 0.0
            assert control_row["count"] > 0 \
                and np.allclose(control_row["return_std"], 0.0), control_row
            assert row["count"] > 0 \
                and np.allclose(row["return_std"], 0.0), row
            return 1.0
        return p_val
    df['p_value'] = df.apply(compute_p_value, axis=1)

    def compute_mean_diff(row):
        """Row mapper that computes whether mean return for current row is
        greater than control return mean."""
        key = compute_key(row)
        control_index = controls[key]
        if control_index == row.name:
            return False
        control_row = df.loc[control_index]
        return row["return_mean"] > control_row["return_mean"]
    df['return_greater'] = df.apply(compute_mean_diff, axis=1)

    # make cell contents
    def compute_cell_contents(row):
        """Row mapper that computes LaTeX table cell contents from the extra
        columns added above."""
        mean = row["return_mean"]
        std = row["return_std"]
        # FIXME(sam): figure out how to format this in a generic way
        final_str = f'{mean:.2f}$\\pm${std:.2f}'
        p_value = row['p_value']
        if p_value < 0.05:
            final_str += '**'
        greater = row['return_greater']
        if greater:
            final_str = f'\\cellcolor[HTML]{{{HIGHLIGHT_COLOR}}}{final_str}'
        return final_str
    df['contents'] = df.apply(compute_cell_contents, axis=1)

    if not full:
        # get just the average test env scores
        df = df[df['test_env'] == 'Average']

    # want a new table like this:
    #      | exp_ident_1 | exp_ident_2 | … | exp_ident_N |
    # env1 |
    # env2 |
    #   …  |
    # envN |
    # …and that means pivoting!
    def unique_agg(values):
        """Aggregation function that simply asserts that only one thing is
        being aggregated over."""
        assert len(values) == 1, values
        val, = values
        return val

    def index_sort_key(index):
        """Sorts nested index where each index item is a (train env, variant)
        pair. Aim is to sort train environments alphabetically, then sort
        variants according to MAGICAL_VARIANT_ORDER."""
        def mapper(index_item):
            if '-v0' in index_item:
                # this is a train env name (first level of the index)
                return index_item
            # otherwise assume we get a test variant label (second level of the
            # index)
            test_variant = index_item
            try:
                test_variant_idx = MAGICAL_VARIANT_ORDER.index(test_variant)
            except ValueError:
                test_variant_idx = -1
            return test_variant_idx
            # return (train_env, test_variant_idx)
        return index.map(mapper)

    if full:
        index = ['train_env', 'test_env']
    else:
        index = 'train_env'
    pivoted = pd.pivot_table(
        df, columns='exp_ident', index=index,
        values='contents', aggfunc=unique_agg) \
        .sort_index(key=index_sort_key if full else None)

    def ei_index(exp_ident):
        """Sort key that sorts by index in `EXP_IDENT_TO_NUM`."""
        return EXP_IDENT_TO_NUM.get(exp_ident, -1)
    sorted_pivot = pivoted.reindex(sorted(pivoted.columns, key=ei_index),
                                   axis=1)

    def format_row(row):
        """Row mapper that joins columns together with ampersand, followed by
        newline at EOL."""
        if full:
            train_env, test_variant = row.name
            if test_variant == '-Demo':
                row_label = train_env.split('-')[0] + '-Demo-v0'
            else:
                row_label = r'\ \ \ \ ' + test_variant
                if test_variant.startswith('-'):
                    row_label = row_label + '-v0'
            return row_label + ' & ' + ' & '.join(row) + r' \\'
        return row.name + ' & ' + ' & '.join(row) + r' \\'
    rows = sorted_pivot.apply(format_row, axis=1)
    return '% ' + ', '.join(sorted_pivot.columns) + '\n' + '\n'.join(rows)


def main(args):
    first = True
    for csv_path in args.csv_files:
        if not first:
            print()
        first = False
        print(f'Result for {csv_path}')
        df = pd.read_csv(csv_path)
        print(process_df(df, full=args.full))


if __name__ == '__main__':
    main(parser.parse_args())
