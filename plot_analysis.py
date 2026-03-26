import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


SIZE_ORDER = ['small', 'medium', 'large']
AREA_BIN_EDGES = [0, 32 ** 2, 96 ** 2, 256 ** 2, 512 ** 2, np.inf]
AREA_BIN_LABELS = ['0-32^2', '32^2-96^2', '96^2-256^2', '256^2-512^2', '512^2+']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create analysis plots from exported instance-based xlsx files.'
    )
    parser.add_argument('--file', type=str, default=None,
                        help='Path to the first instance_based xlsx file')
    parser.add_argument('--file2', type=str, default=None,
                        help='Optional second instance_based xlsx file for comparison')
    parser.add_argument('--label1', type=str, default=None,
                        help='Label for the first workbook')
    parser.add_argument('--label2', type=str, default=None,
                        help='Label for the second workbook')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory where plots will be saved')
    parser.add_argument('--max_scatter_points', type=int, default=15000,
                        help='Maximum number of points used in scatter plots')
    return parser.parse_args()


def sanitize_name(name):
    return ''.join(char if char.isalnum() or char in {'-', '_'} else '_'
                   for char in str(name))


def resolve_label(file_path, cli_label, default_label):
    if cli_label:
        return cli_label
    if default_label:
        return default_label
    if file_path:
        return osp.splitext(osp.basename(file_path))[0]
    return 'results'


def resolve_existing_workbook(preferred_path):
    if osp.exists(preferred_path):
        return preferred_path

    basename = osp.basename(preferred_path)
    fallback_path = basename
    if osp.exists(fallback_path):
        print(
            f'Configured workbook not found at {preferred_path}. '
            f'Using {fallback_path} instead.'
        )
        return fallback_path

    return preferred_path


def load_results(file_path):
    df = pd.read_excel(file_path)
    unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def add_analysis_columns(df):
    enriched = df.copy()
    enriched['abs_N_diff'] = pd.to_numeric(enriched['N_diff'], errors='coerce').abs()
    enriched['area_bin'] = pd.cut(pd.to_numeric(enriched['area'], errors='coerce'),
                                  bins=AREA_BIN_EDGES,
                                  labels=AREA_BIN_LABELS,
                                  include_lowest=True,
                                  right=True)
    enriched['size'] = pd.Categorical(enriched['size'],
                                      categories=SIZE_ORDER,
                                      ordered=True)
    return enriched


def valid_metric_series(df, metric):
    series = pd.to_numeric(df[metric], errors='coerce')
    return series[series >= 0]


def valid_rows(df, columns):
    mask = pd.Series(True, index=df.index)
    for column in columns:
        if column in {'size', 'area_bin'}:
            mask &= df[column].notna()
            continue

        values = pd.to_numeric(df[column], errors='coerce')
        mask &= values.notna()
        if column in {'polis', 'mta', 'iou', 'ciou', 'box_iou'}:
            mask &= values >= 0
    return df.loc[mask].copy()


def sample_for_scatter(df, max_points):
    if len(df) <= max_points:
        return df
    return df.sample(max_points, random_state=0)


def plot_ecdf(ax, values, label):
    sorted_values = np.sort(np.asarray(values))
    if len(sorted_values) == 0:
        return
    y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax.plot(sorted_values, y, linewidth=2, label=label)
    ax.set_ylabel('ECDF')
    ax.grid(alpha=0.3)


def save_figure(fig, output_path):
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f'Saved: {output_path}')


def create_distribution_overview(df, output_dir, label):
    metrics = [
        ('ciou', 'C-IoU Distribution'),
        ('polis', 'PoLiS Distribution'),
        ('mta', 'MTA Distribution'),
        ('abs_N_diff', '|N_diff| Distribution'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, metrics):
        values = valid_metric_series(df, metric)
        ax.hist(values, bins=40, alpha=0.85, color='#3A78B4')
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.set_ylabel('Count')
        ax.grid(alpha=0.3)
    save_figure(fig, osp.join(output_dir, f'{sanitize_name(label)}_distribution_overview.png'))


def create_vertices_scatter(df, output_dir, label, max_points):
    scatter_df = valid_rows(df, ['#vertices', '#vertices_pred'])
    scatter_df = sample_for_scatter(scatter_df, max_points)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(scatter_df['#vertices'], scatter_df['#vertices_pred'],
               s=10, alpha=0.25, color='#0F766E')
    max_value = max(scatter_df['#vertices'].max(), scatter_df['#vertices_pred'].max())
    ax.plot([0, max_value], [0, max_value], linestyle='--', color='black', linewidth=1)
    ax.set_title(f'Simplicity: GT vs Pred Vertices ({label})')
    ax.set_xlabel('GT #vertices')
    ax.set_ylabel('Pred #vertices')
    ax.grid(alpha=0.3)
    save_figure(fig, osp.join(output_dir, f'{sanitize_name(label)}_vertices_scatter.png'))


def create_metric_scatter(df, x_col, y_col, output_dir, label, title, filename, max_points):
    scatter_df = valid_rows(df, [x_col, y_col])
    scatter_df = sample_for_scatter(scatter_df, max_points)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(scatter_df[x_col], scatter_df[y_col], s=10, alpha=0.25, color='#A61E4D')
    ax.set_title(f'{title} ({label})')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(alpha=0.3)
    save_figure(fig, osp.join(output_dir, f'{sanitize_name(label)}_{filename}.png'))


def create_grouped_boxplots(df, output_dir, label):
    plot_specs = [
        ('ciou', 'C-IoU by Size'),
        ('polis', 'PoLiS by Size'),
        ('mta', 'MTA by Size'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (metric, title) in zip(axes, plot_specs):
        metric_df = valid_rows(df, ['size', metric])
        data = [metric_df.loc[metric_df['size'] == size, metric].values for size in SIZE_ORDER]
        filtered = [(size, values) for size, values in zip(SIZE_ORDER, data) if len(values) > 0]
        if not filtered:
            continue
        labels, values = zip(*filtered)
        ax.boxplot(values, labels=labels, patch_artist=True,
                   boxprops=dict(facecolor='#A7C7E7'))
        ax.set_title(title)
        ax.set_xlabel('size')
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    save_figure(fig, osp.join(output_dir, f'{sanitize_name(label)}_grouped_boxplots_size.png'))


def create_heatmap(df, value_col, output_dir, label):
    heatmap_df = valid_rows(df, ['area_bin', '#vertices', value_col])
    pivot = heatmap_df.pivot_table(index='area_bin',
                                   columns='#vertices',
                                   values=value_col,
                                   aggfunc='mean',
                                   observed=True)
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
    ax.set_title(f'{value_col} mean over area_bin x #vertices ({label})')
    ax.set_xlabel('#vertices')
    ax.set_ylabel('area_bin')
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(col)) for col in pivot.columns], rotation=90)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(idx) for idx in pivot.index])
    fig.colorbar(im, ax=ax, label=f'mean {value_col}')
    save_figure(fig, osp.join(output_dir, f'{sanitize_name(label)}_heatmap_{value_col}.png'))


def create_compare_ecdfs(df1, df2, label1, label2, output_dir):
    metrics = [
        ('ciou', 'C-IoU'),
        ('polis', 'PoLiS'),
        ('mta', 'MTA'),
        ('abs_N_diff', '|N_diff|'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, metrics):
        values1 = valid_metric_series(df1, metric)
        values2 = valid_metric_series(df2, metric)
        plot_ecdf(ax, values1, label1)
        plot_ecdf(ax, values2, label2)
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.legend()
    save_figure(fig, osp.join(output_dir,
                              f'compare_{sanitize_name(label1)}_vs_{sanitize_name(label2)}_ecdf.png'))


def create_compare_size_means(df1, df2, label1, label2, output_dir):
    metrics = ['ciou', 'polis', 'mta']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, metrics):
        metric_df1 = valid_rows(df1, ['size', metric])
        metric_df2 = valid_rows(df2, ['size', metric])
        means1 = metric_df1.groupby('size', observed=True)[metric].mean().reindex(SIZE_ORDER)
        means2 = metric_df2.groupby('size', observed=True)[metric].mean().reindex(SIZE_ORDER)
        x = np.arange(len(SIZE_ORDER))
        ax.plot(x, means1.values, marker='o', linewidth=2, label=label1)
        ax.plot(x, means2.values, marker='o', linewidth=2, label=label2)
        ax.set_title(f'{metric} mean by size')
        ax.set_xticks(x)
        ax.set_xticklabels(SIZE_ORDER)
        ax.set_xlabel('size')
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend()
    save_figure(fig, osp.join(output_dir,
                              f'compare_{sanitize_name(label1)}_vs_{sanitize_name(label2)}_size_means.png'))


def generate_single_report(df, label, output_dir, max_points):
    os.makedirs(output_dir, exist_ok=True)
    create_distribution_overview(df, output_dir, label)
    create_vertices_scatter(df, output_dir, label, max_points)
    create_metric_scatter(df, 'iou', 'polis', output_dir, label,
                          'Fidelity: IoU vs PoLiS', 'iou_vs_polis', max_points)
    create_metric_scatter(df, 'mta', 'ciou', output_dir, label,
                          'Regularity vs Fidelity: MTA vs C-IoU',
                          'mta_vs_ciou', max_points)
    create_grouped_boxplots(df, output_dir, label)
    create_heatmap(df, 'mta', output_dir, label)
    create_heatmap(df, 'ciou', output_dir, label)


def main():
    try:
        with open('config.yaml') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print('You need a "config.yaml" file with your configs set in')
        raise

    args = parse_args()
    default_file = cfg.get('instance_results_file') or osp.join(
        cfg['output_dir'], f"{cfg['name']}_instance_based.xlsx"
    )
    file1 = args.file or resolve_existing_workbook(default_file)
    if not osp.exists(file1):
        raise FileNotFoundError(
            f'Could not find instance workbook: {file1}. '
            'Run create_tables.py first or pass --file explicitly.'
        )
    label1 = resolve_label(file1, args.label1, cfg.get('prediction_label', cfg['name']))

    df1 = add_analysis_columns(load_results(file1))

    config_file2 = cfg.get('instance_results_file2')
    file2 = args.file2 or (resolve_existing_workbook(config_file2) if config_file2 else None)

    if file2:
        file2 = resolve_existing_workbook(file2)
        if not osp.exists(file2):
            raise FileNotFoundError(f'Could not find comparison workbook: {file2}')
        label2 = resolve_label(file2, args.label2, cfg.get('prediction_label2', 'comparison'))
        compare_root = args.output_dir or osp.join(
            cfg['output_dir'],
            f"compare_{sanitize_name(label1)}_vs_{sanitize_name(label2)}_plots",
        )
        df2 = add_analysis_columns(load_results(file2))

        generate_single_report(df1, label1, osp.join(compare_root, sanitize_name(label1)),
                               args.max_scatter_points)
        generate_single_report(df2, label2, osp.join(compare_root, sanitize_name(label2)),
                               args.max_scatter_points)
        os.makedirs(compare_root, exist_ok=True)
        create_compare_ecdfs(df1, df2, label1, label2, compare_root)
        create_compare_size_means(df1, df2, label1, label2, compare_root)
        print(f'Analysis plots saved to: {compare_root}')
    else:
        output_dir = args.output_dir or osp.join(cfg['output_dir'],
                                                 f"{sanitize_name(label1)}_analysis_plots")
        generate_single_report(df1, label1, output_dir, args.max_scatter_points)
        print(f'Analysis plots saved to: {output_dir}')


if __name__ == '__main__':
    main()
