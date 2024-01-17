import yaml
from os import path as osp
import pandas as pd
import numpy as np


def load_results(config):
    ins_based = config['name'] + '_instance_based.xlsx'
    results = pd.read_excel(ins_based)
    return results


def metric_stats(metric: pd.DataFrame, thrs: list[float], larger_better=True):
    # check if the metris are positive (valid)
    metric = metric[metric >= 0.0]

    # stats
    avg = metric.mean()
    std = metric.std()
    minim = metric.min()
    maxim = metric.max()
    med = metric.median()

    # average after removing outliers by 3-sigma
    from scipy import stats
    z_score = np.abs(stats.zscore(metric))
    out_zscore = z_score > 3
    metric_z = metric[np.logical_not(out_zscore)]
    mean_zscore = metric_z.mean()

    # average after removing outliers by IQR
    q1 = metric.quantile(0.25)
    q3 = metric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr
    # removing the outliers
    out_iqr = np.logical_or(metric >= upper, metric <= lower)
    metric_iqr = metric[np.logical_not(out_iqr)]
    mean_iqr = metric_iqr.mean()

    # threshold-based metrics
    metric_thrs = {}
    for thr in thrs:
        if larger_better:
            cond = metric >= thr
        else:
            cond = metric <= thr
        metric_thr = metric[cond]
        metric_thrs.update({str(thr): metric_thr.mean()})

    metric_stats = [avg, std, minim, maxim, med, mean_iqr, mean_zscore]
    outliers_idx = [out_iqr, out_zscore]
    return metric_stats, metric_thrs, outliers_idx


if __name__ == '__main__':

    try:
        with open("config.yaml") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print('You need a "config.yaml" file with your configs set in')

    results = load_results(cfg)
    row_stats = []
    col_outlier = {}

    # C-IoU
    stats, metr_thrs, outliers = metric_stats(results['ciou'],
                                              thrs=[0.1, 0.5, 0.75],
                                              larger_better=True)
    stats_dict = {'metric': 'ciou', 'average': stats[0],
                  'std': stats[1], 'min': stats[2],
                  'max': stats[3], 'median': stats[4],
                  'mean_IQR': stats[5], 'mean_zscore': stats[6]
                  }
    stats_dict.update(metr_thrs)
    row_stats.append(stats_dict)
    res = results[results['ciou'] >= 0.0]
    col_outlier.update({'ciou_iqr': res[outliers[0]]['instance_id'],
                        'ciou_z-score': res[outliers[1]]['instance_id']})

    # IoU
    stats, metr_thrs, outliers = metric_stats(results['iou'],
                                              thrs=[0.1, 0.5, 0.75],
                                              larger_better=True)
    stats_dict = {'metric': 'iou', 'average': stats[0],
                  'std': stats[1], 'min': stats[2],
                  'max': stats[3], 'median': stats[4],
                  'mean_IQR': stats[5], 'mean_zscore': stats[6]
                  }
    stats_dict.update(metr_thrs)
    row_stats.append(stats_dict)

    res = results[results['iou'] >= 0.0]
    col_outlier.update({'iou_iqr': res[outliers[0]]['instance_id'],
                        'iou_z-score': res[outliers[1]]['instance_id']})

    # PoLiS
    stats, metr_thrs, outliers = metric_stats(results['polis'],
                                              thrs=[3, 5, 10],
                                              larger_better=False)
    stats_dict = {'metric': 'polis', 'average': stats[0],
                  'std': stats[1], 'min': stats[2],
                  'max': stats[3], 'median': stats[4],
                  'mean_IQR': stats[5], 'mean_zscore': stats[6]
                  }
    stats_dict.update(metr_thrs)
    row_stats.append(stats_dict)

    res = results[results['polis'] >= 0.0]
    col_outlier.update({'polis_iqr': res[outliers[0]]['instance_id'],
                        'polis_z-score': res[outliers[1]]['instance_id']})

    # MTA
    stats, metr_thrs, outliers = metric_stats(results['mta'],
                                              thrs=[60, 40],
                                              larger_better=False)
    stats_dict = {'metric': 'mta', 'average': stats[0],
                  'std': stats[1], 'min': stats[2],
                  'max': stats[3], 'median': stats[4],
                  'mean_IQR': stats[5], 'mean_zscore': stats[6]
                  }
    stats_dict.update(metr_thrs)
    row_stats.append(stats_dict)
    res = results[results['mta'] >= 0.0]
    col_outlier.update({'mta_iqr': res[outliers[0]]['instance_id'],
                        'mta_z-score': res[outliers[1]]['instance_id']})

    # Create the metric stats table
    df = pd.DataFrame(row_stats)
    print(df)  # Disply the DataFrame
    # Write to xlsx
    df.to_excel(osp.join(
                cfg['output_dir'],
                f"./{cfg['name']}_metric_stats.xlsx"),
                sheet_name='metric stats')

    # Create the metric outlier table
    df2 = pd.DataFrame({})
    df2 = df2.assign(**col_outlier)
    # Write to xlsx
    df2.to_excel(osp.join(
                 cfg['output_dir'],
                 f"./{cfg['name']}_outleirs.xlsx"),
                 sheet_name='ins_ids of outleirs')
