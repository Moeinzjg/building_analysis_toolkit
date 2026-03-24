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


def load_results(config):
    ins_based = osp.join(config['output_dir'],
                         f"{config['name']}_instance_based.xlsx")
    return pd.read_excel(ins_based)


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


def save_outputs(config: dict,
                 stats_df: pd.DataFrame,
                 outliers_df: pd.DataFrame) -> None:
    stats_path = osp.join(config['output_dir'],
                          f"{config['name']}_metric_stats.xlsx")
    outliers_path = osp.join(config['output_dir'],
                             f"{config['name']}_outliers.xlsx")

    stats_df.to_excel(stats_path, index=False, sheet_name='metric_stats')
    outliers_df.to_excel(outliers_path, index=False, sheet_name='outliers')


def reorder_columns(df: pd.DataFrame, first_columns: list[str]) -> pd.DataFrame:
    ordered = [col for col in first_columns if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered]
    return df[ordered + remaining]


if __name__ == '__main__':

    try:
        with open("config.yaml") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print('You need a "config.yaml" file with your configs set in')
        raise

    results = load_results(cfg)
    stats_rows = []
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

    stats_df = pd.DataFrame(stats_rows)
    outliers_df = pd.DataFrame(outlier_rows)
    stats_df = reorder_columns(stats_df, ['metric'])
    if len(outliers_df):
        outliers_df = reorder_columns(outliers_df,
                                      ['metric', 'method', 'image_id',
                                       'instance_id', 'value'])

    print(stats_df)
    if len(outliers_df):
        print(outliers_df)
    else:
        print('No outliers detected.')

    save_outputs(cfg, stats_df, outliers_df)
