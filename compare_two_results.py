import argparse
from os import path as osp

import pandas as pd
import yaml


DEFAULT_EXCLUDE_COLUMNS = {
    'image_id',
    'instance_id',
    '#vertices',
    'area',
    'size',
    'touch_border',
    'orientation',
    'Unnamed: 0',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare two exported result workbooks and summarize metric differences.'
    )
    parser.add_argument('file1', nargs='?', type=str, default=None,
                        help='Path to the first result file.')
    parser.add_argument('file2', nargs='?', type=str, default=None,
                        help='Path to the second result file.')
    parser.add_argument('--label1', type=str, default=None,
                        help='Label for the first file in the output tables.')
    parser.add_argument('--label2', type=str, default=None,
                        help='Label for the second file in the output tables.')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional output xlsx path for detailed comparison tables.')
    return parser.parse_args()


def load_results(file_path):
    df = pd.read_excel(file_path)
    unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def infer_key_columns(df1, df2):
    if {'image_id', 'instance_id'}.issubset(df1.columns) and \
       {'image_id', 'instance_id'}.issubset(df2.columns):
        return ['image_id', 'instance_id']
    if 'image_id' in df1.columns and 'image_id' in df2.columns:
        return ['image_id']
    raise ValueError('Could not infer key columns. Expected image_id or image_id + instance_id.')


def infer_metric_columns(df1, df2, key_columns):
    common_columns = set(df1.columns).intersection(df2.columns)
    metric_columns = []
    for column in sorted(common_columns):
        if column in key_columns or column in DEFAULT_EXCLUDE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df1[column]) and pd.api.types.is_numeric_dtype(df2[column]):
            metric_columns.append(column)
    return metric_columns


def build_comparison_table(df1, df2, key_columns, metric_columns, label1, label2):
    metadata_columns = [
        column for column in ['#vertices', 'area', 'size', 'touch_border', 'orientation']
        if column in df1.columns and column not in key_columns
    ]

    left_columns = key_columns + metadata_columns + metric_columns
    right_columns = key_columns + [column for column in metric_columns if column in df2.columns]

    merged = df1[left_columns].merge(
        df2[right_columns],
        on=key_columns,
        how='outer',
        suffixes=(f'_{label1}', f'_{label2}'),
        indicator=True,
    )

    for metric in metric_columns:
        col1 = f'{metric}_{label1}'
        col2 = f'{metric}_{label2}'
        merged[f'delta_{metric}'] = merged[col1] - merged[col2]

    return merged


def build_summary_table(comparison_df, metric_columns, label1, label2):
    matched = comparison_df[comparison_df['_merge'] == 'both']
    summary_rows = []
    for metric in metric_columns:
        delta_col = f'delta_{metric}'
        col1 = f'{metric}_{label1}'
        col2 = f'{metric}_{label2}'
        delta = matched[delta_col].dropna()
        if len(delta) == 0:
            continue
        summary_rows.append({
            'metric': metric,
            'n_compared': int(delta.count()),
            f'mean_{label1}': matched[col1].mean(),
            f'mean_{label2}': matched[col2].mean(),
            'mean_delta': delta.mean(),
            'mean_abs_delta': delta.abs().mean(),
            'median_delta': delta.median(),
            'max_delta': delta.max(),
            'min_delta': delta.min(),
        })
    return pd.DataFrame(summary_rows)


def print_alignment_summary(comparison_df, label1, label2):
    only_file1 = int((comparison_df['_merge'] == 'left_only').sum())
    only_file2 = int((comparison_df['_merge'] == 'right_only').sum())
    both = int((comparison_df['_merge'] == 'both').sum())

    print('Row alignment summary:')
    print(f'- matched rows: {both}')
    print(f'- only in {label1}: {only_file1}')
    print(f'- only in {label2}: {only_file2}')


def default_output_path(file1, file2):
    name1 = osp.splitext(osp.basename(file1))[0]
    name2 = osp.splitext(osp.basename(file2))[0]
    return f'compare_{name1}_vs_{name2}.xlsx'


def resolve_existing_workbook(preferred_path):
    if preferred_path is None:
        return None
    if osp.exists(preferred_path):
        return preferred_path

    basename = osp.basename(preferred_path)
    if osp.exists(basename):
        print(
            f'Configured workbook not found at {preferred_path}. '
            f'Using {basename} instead.'
        )
        return basename
    return preferred_path


def resolve_inputs(args):
    try:
        with open('config.yaml') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        cfg = {}

    file1 = args.file1 or resolve_existing_workbook(cfg.get('instance_results_file'))
    file2 = args.file2 or resolve_existing_workbook(cfg.get('instance_results_file2'))

    if file1 is None or file2 is None:
        raise FileNotFoundError(
            'Need two result workbooks to compare. Pass file1/file2 explicitly '
            'or set instance_results_file and instance_results_file2 in config.yaml.'
        )
    if not osp.exists(file1):
        raise FileNotFoundError(f'Could not find first workbook: {file1}')
    if not osp.exists(file2):
        raise FileNotFoundError(f'Could not find second workbook: {file2}')

    label1 = args.label1 or cfg.get('prediction_label') or 'file1'
    label2 = args.label2 or cfg.get('prediction_label2') or 'file2'
    return file1, file2, label1, label2


if __name__ == "__main__":
    args = parse_args()
    file1, file2, label1, label2 = resolve_inputs(args)

    df1 = load_results(file1)
    df2 = load_results(file2)
    key_columns = infer_key_columns(df1, df2)
    metric_columns = infer_metric_columns(df1, df2, key_columns)

    if not metric_columns:
        raise ValueError('No shared numeric metric columns found to compare.')

    comparison_df = build_comparison_table(df1, df2, key_columns,
                                           metric_columns,
                                           label1, label2)
    summary_df = build_summary_table(comparison_df, metric_columns,
                                     label1, label2)

    print(f'Using key columns: {key_columns}')
    print(f'Comparing metric columns: {metric_columns}')
    print_alignment_summary(comparison_df, label1, label2)
    print('\nMetric summary:')
    print(summary_df.to_string(index=False))

    output_path = args.output or default_output_path(file1, file2)
    with pd.ExcelWriter(output_path) as writer:
        summary_df.to_excel(writer, index=False, sheet_name='summary')
        comparison_df.to_excel(writer, index=False, sheet_name='comparison')

    print(f'\nComparison workbook saved to: {output_path}')
