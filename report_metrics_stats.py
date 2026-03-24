from os import path as osp

import numpy as np
import pandas as pd
import yaml
from scipy import stats


METRIC_CONFIGS = [
    {"name": "ciou", "thresholds": [0.1, 0.5, 0.75], "larger_better": True},
    {"name": "iou", "thresholds": [0.1, 0.5, 0.75], "larger_better": True},
    {"name": "polis", "thresholds": [3, 5, 10], "larger_better": False},
    {"name": "mta", "thresholds": [40, 60], "larger_better": False},
]
MIN_HIGH_SIGNAL_COUNT = 30

AREA_BIN_EDGES = [0, 32 ** 2, 96 ** 2, 256 ** 2, 512 ** 2, np.inf]
AREA_BIN_LABELS = [
    '0-32^2',
    '32^2-96^2',
    '96^2-256^2',
    '256^2-512^2',
    '512^2+',
]
N_RATIO_BIN_EDGES = [0, 0.5, 0.8, 1.2, 1.5, 2.0, np.inf]
N_RATIO_BIN_LABELS = [
    '<=0.5',
    '0.5-0.8',
    '0.8-1.2',
    '1.2-1.5',
    '1.5-2.0',
    '>2.0',
]
ORIENTATION_BIN_EDGES = [-0.1, 15, 30, 45, 60, 75, 90]
ORIENTATION_BIN_LABELS = [
    '0-15',
    '15-30',
    '30-45',
    '45-60',
    '60-75',
    '75-90',
]


def load_results(config):
    ins_based = osp.join(config['output_dir'],
                         f"{config['name']}_instance_based.xlsx")
    return pd.read_excel(ins_based)


def add_grouping_columns(results: pd.DataFrame) -> pd.DataFrame:
    enriched = results.copy()
    enriched['touch_border'] = enriched['touch_border'].map(
        lambda value: 'touches_border' if bool(value) else 'no_border_touch'
    )
    enriched['vertices_group'] = pd.to_numeric(enriched['#vertices'],
                                               errors='coerce').astype('Int64')
    enriched['area_bin'] = pd.cut(pd.to_numeric(enriched['area'],
                                                errors='coerce'),
                                  bins=AREA_BIN_EDGES,
                                  labels=AREA_BIN_LABELS,
                                  include_lowest=True,
                                  right=True)
    orientation = pd.to_numeric(enriched['orientation'], errors='coerce')
    orientation = np.mod(np.abs(orientation), 90.0)
    enriched['orientation_bin'] = pd.cut(orientation,
                                         bins=ORIENTATION_BIN_EDGES,
                                         labels=ORIENTATION_BIN_LABELS,
                                         include_lowest=True,
                                         right=True)
    enriched['n_ratio_bin'] = pd.cut(pd.to_numeric(enriched['N_ratio'],
                                                   errors='coerce'),
                                     bins=N_RATIO_BIN_EDGES,
                                     labels=N_RATIO_BIN_LABELS,
                                     include_lowest=True,
                                     right=True)
    n_diff = pd.to_numeric(enriched['N_diff'], errors='coerce')
    enriched['n_diff_group'] = pd.Series(np.select(
        [
            n_diff <= -5,
            (n_diff >= -4) & (n_diff <= -2),
            n_diff == -1,
            n_diff == 0,
            n_diff == 1,
            (n_diff >= 2) & (n_diff <= 4),
            n_diff >= 5,
        ],
        [
            '<=-5',
            '-4:-2',
            '-1',
            '0',
            '+1',
            '+2:+4',
            '>=+5',
        ],
        default=np.nan,
    ), index=enriched.index)
    return enriched


def valid_metric_values(metric: pd.Series) -> pd.Series:
    metric = pd.to_numeric(metric, errors='coerce')
    return metric[metric >= 0.0]


def threshold_label(threshold: float, larger_better: bool) -> str:
    prefix = "rate_ge" if larger_better else "rate_le"
    return f"{prefix}_{threshold:g}"


def threshold_pass_rates(metric: pd.Series,
                         thresholds: list[float],
                         larger_better: bool) -> dict:
    rates = {}
    for threshold in thresholds:
        if larger_better:
            passed = metric >= threshold
        else:
            passed = metric <= threshold
        rates[threshold_label(threshold, larger_better)] = passed.mean()
    return rates


def compute_outlier_masks(metric: pd.Series) -> tuple[pd.Series, pd.Series]:
    if len(metric) == 0:
        empty = pd.Series([], dtype=bool, index=metric.index)
        return empty, empty

    if len(metric) < 4:
        out_iqr = pd.Series(False, index=metric.index)
    else:
        q1 = metric.quantile(0.25)
        q3 = metric.quantile(0.75)
        iqr = q3 - q1
        if np.isclose(iqr, 0.0):
            out_iqr = pd.Series(False, index=metric.index)
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            out_iqr = (metric < lower) | (metric > upper)

    if len(metric) < 2:
        out_zscore = pd.Series(False, index=metric.index)
    else:
        z_scores = pd.Series(stats.zscore(metric, nan_policy='omit'),
                             index=metric.index)
        if z_scores.isna().all():
            out_zscore = pd.Series(False, index=metric.index)
        else:
            out_zscore = z_scores.abs() > 3

    return out_iqr, out_zscore


def metric_stats(metric: pd.Series,
                 thresholds: list[float],
                 larger_better: bool = True) -> tuple[dict, pd.Series, pd.Series]:
    valid_metric = valid_metric_values(metric)
    stats_dict = {
        'valid_count': int(valid_metric.count()),
        'invalid_count': int(metric.shape[0] - valid_metric.count()),
        'average': valid_metric.mean(),
        'std': valid_metric.std(),
        'min': valid_metric.min(),
        'max': valid_metric.max(),
        'median': valid_metric.median(),
    }

    if len(valid_metric) == 0:
        stats_dict.update({
            'mean_IQR': np.nan,
            'mean_zscore': np.nan,
            'iqr_outlier_count': 0,
            'zscore_outlier_count': 0,
        })
        stats_dict.update(threshold_pass_rates(valid_metric,
                                               thresholds,
                                               larger_better))
        empty = pd.Series([], dtype=bool, index=valid_metric.index)
        return stats_dict, empty, empty

    out_iqr, out_zscore = compute_outlier_masks(valid_metric)
    stats_dict.update({
        'mean_IQR': valid_metric.loc[~out_iqr].mean(),
        'mean_zscore': valid_metric.loc[~out_zscore].mean(),
        'iqr_outlier_count': int(out_iqr.sum()),
        'zscore_outlier_count': int(out_zscore.sum()),
    })
    stats_dict.update(threshold_pass_rates(valid_metric,
                                           thresholds,
                                           larger_better))
    return stats_dict, out_iqr, out_zscore


def build_outlier_rows(results: pd.DataFrame,
                       metric_name: str,
                       metric_values: pd.Series,
                       outlier_mask: pd.Series,
                       method: str) -> list[dict]:
    rows = []
    if len(metric_values) == 0:
        return rows

    outlier_rows = results.loc[metric_values.index[outlier_mask]]
    for _, row in outlier_rows.iterrows():
        rows.append({
            'metric': metric_name,
            'method': method,
            'image_id': row['image_id'],
            'instance_id': row['instance_id'],
            'value': row[metric_name],
        })
    return rows


def grouped_metric_stats(results: pd.DataFrame,
                         metric_cfg: dict,
                         group_by: str) -> list[dict]:
    rows = []
    grouped = results.groupby(group_by, dropna=False, observed=True)

    for group_value, group_df in grouped:
        if pd.isna(group_value):
            continue

        stats_dict, _, _ = metric_stats(group_df[metric_cfg['name']],
                                        thresholds=metric_cfg['thresholds'],
                                        larger_better=metric_cfg['larger_better'])
        stats_dict.update({
            'metric': metric_cfg['name'],
            'group_by': group_by,
            'group': str(group_value),
            'total_count': int(len(group_df)),
        })
        rows.append(stats_dict)

    return rows


def save_outputs(config: dict,
                 stats_df: pd.DataFrame,
                 grouped_stats_df: pd.DataFrame,
                 high_signal_df: pd.DataFrame,
                 outliers_df: pd.DataFrame) -> None:
    stats_path = osp.join(config['output_dir'],
                          f"{config['name']}_metric_stats.xlsx")
    outliers_path = osp.join(config['output_dir'],
                             f"{config['name']}_outliers.xlsx")

    with pd.ExcelWriter(stats_path) as writer:
        stats_df.to_excel(writer, index=False, sheet_name='metric_stats')
        grouped_stats_df.to_excel(writer, index=False, sheet_name='grouped_stats')
        high_signal_df.to_excel(writer, index=False, sheet_name='high_signal_groups')

    outliers_df.to_excel(outliers_path, index=False, sheet_name='outliers')


def reorder_columns(df: pd.DataFrame, first_columns: list[str]) -> pd.DataFrame:
    ordered = [col for col in first_columns if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered]
    return df[ordered + remaining]


def build_high_signal_groups(grouped_stats_df: pd.DataFrame) -> pd.DataFrame:
    if grouped_stats_df.empty:
        return grouped_stats_df

    metric_direction = {
        metric_cfg['name']: metric_cfg['larger_better']
        for metric_cfg in METRIC_CONFIGS
    }

    filtered = grouped_stats_df[grouped_stats_df['total_count'] >= MIN_HIGH_SIGNAL_COUNT].copy()
    if filtered.empty:
        return filtered

    severity_score = []
    for _, row in filtered.iterrows():
        larger_better = metric_direction[row['metric']]
        avg = row['average']
        severity_score.append(-avg if larger_better else avg)

    filtered['severity_score'] = severity_score
    filtered = filtered.sort_values(
        by=['metric', 'severity_score', 'total_count'],
        ascending=[True, False, False],
    )
    return filtered


if __name__ == '__main__':

    try:
        with open("config.yaml") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print('You need a "config.yaml" file with your configs set in')
        raise

    results = load_results(cfg)
    results = add_grouping_columns(results)
    stats_rows = []
    grouped_rows = []
    outlier_rows = []

    for metric_cfg in METRIC_CONFIGS:
        metric_name = metric_cfg['name']
        valid_metric = valid_metric_values(results[metric_name])
        stats_dict, out_iqr, out_zscore = metric_stats(
            results[metric_name],
            thresholds=metric_cfg['thresholds'],
            larger_better=metric_cfg['larger_better'],
        )
        stats_dict['metric'] = metric_name
        stats_rows.append(stats_dict)

        outlier_rows.extend(build_outlier_rows(results,
                                               metric_name,
                                               valid_metric,
                                               out_iqr,
                                               'iqr'))
        outlier_rows.extend(build_outlier_rows(results,
                                               metric_name,
                                               valid_metric,
                                               out_zscore,
                                               'zscore'))
        grouped_rows.extend(grouped_metric_stats(results, metric_cfg, 'size'))
        grouped_rows.extend(grouped_metric_stats(results,
                                                metric_cfg,
                                                'touch_border'))
        grouped_rows.extend(grouped_metric_stats(results,
                                                metric_cfg,
                                                'vertices_group'))
        grouped_rows.extend(grouped_metric_stats(results,
                                                metric_cfg,
                                                'area_bin'))
        grouped_rows.extend(grouped_metric_stats(results,
                                                metric_cfg,
                                                'orientation_bin'))
        grouped_rows.extend(grouped_metric_stats(results,
                                                metric_cfg,
                                                'n_ratio_bin'))
        grouped_rows.extend(grouped_metric_stats(results,
                                                metric_cfg,
                                                'n_diff_group'))

    stats_df = pd.DataFrame(stats_rows)
    grouped_stats_df = pd.DataFrame(grouped_rows)
    high_signal_df = build_high_signal_groups(grouped_stats_df)
    outliers_df = pd.DataFrame(outlier_rows)
    stats_df = reorder_columns(stats_df, ['metric'])
    if len(grouped_stats_df):
        grouped_stats_df = reorder_columns(grouped_stats_df,
                                           ['metric', 'group_by', 'group',
                                            'total_count'])
    if len(high_signal_df):
        high_signal_df = reorder_columns(high_signal_df,
                                         ['metric', 'group_by', 'group',
                                          'total_count', 'severity_score'])
    if len(outliers_df):
        outliers_df = reorder_columns(outliers_df,
                                      ['metric', 'method', 'image_id',
                                       'instance_id', 'value'])

    print(stats_df)
    if len(grouped_stats_df):
        print(grouped_stats_df)
    if len(high_signal_df):
        print(high_signal_df)
    if len(outliers_df):
        print(outliers_df)
    else:
        print('No outliers detected.')

    save_outputs(cfg, stats_df, grouped_stats_df, high_signal_df, outliers_df)
