import pandas as pd
import re

csv_filepath = '/Users/cynthiachen/Google Drive/CHAI/Paper/neurips/tables/joint.csv'
highlight_color = 'fff2cc'

def main():
    df = pd.read_csv(csv_filepath)
    first_line = True
    for index, row in df.iterrows():
        # We didn't do ninja exps on joint training.
        if 'ninja' in row["exp_ident"]:
            continue
        # Procgen exp_ident has format similar to 'coinrun-train_level'
        if '_' in row["exp_ident"]:
            exp_ident = row["exp_ident"].split('_')[0]
            row_print = f'{exp_ident} & '
        else:
            row_print = f'{row["exp_ident"]} & '
        bc_mean = row['BC with augs'].split('±')[0]
        for key, value in row.items():
            if key == 'exp_ident':
                continue
            # Print table headings
            if first_line:
                key_list = row.keys()
                print(' & '.join(key_list))
                first_line = False
                continue
            mean = value.split('±')[0]
            std = value.split('±')[1].split(' ')[0]
            asterisk = '**' if '*' in value else ''
            # This expression is too long for our paper
            expression = f'{mean} $\\pm$ {std}{asterisk}'
            # expression = f'{mean}{asterisk}'

            if key != 'BC with augs':
                if float(mean) > float(bc_mean):
                    row_print += '\cellcolor[HTML]{fff2cc} '
            row_print += f'{expression} & '

        # Remove trailing &
        row_print = row_print[:-1] + ' \\\\'
        if 'reacher' in row["exp_ident"] or 'miner' in row["exp_ident"]:
            row_print += ' \\hline'
        print(row_print)

if __name__ == '__main__':
    main()
