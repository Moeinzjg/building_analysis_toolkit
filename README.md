# Building Analysis Toolkit

A toolkit for quantitative and qualitative analysis of building polygon / instance segmentation results in MS COCO JSON format.

## What It Does

Given:

- a COCO ground-truth annotation file
- a COCO prediction file
- the corresponding image directory

the toolkit can:

- compute per-instance metrics
- compute image-level IoU / C-IoU summaries
- export Excel tables for analysis
- generate grouped metric reports and top-k failure groups
- visualize predictions and annotations per image or per instance

## Environment Setup

Create and use the provided conda environment:

```bash
conda env create -f env.yml
conda activate atk
```

All commands below assume you are running inside `atk`.

## Config

Copy the sample config and edit it:

```bash
cp sample_config.yaml config.yaml
```

Required fields in `config.yaml`:

```yaml
name: "your_run_name"
prediction_file: "/absolute/path/to/predictions.json"
prediction_label: "Model A"
prediction_file2: "/absolute/path/to/second_predictions.json"
prediction_label2: "Model B"
annotation_file: "/absolute/path/to/instances_test.json"
image_dir: "/absolute/path/to/image_directory"
output_dir: "."
```

Field meanings:

- `name`: prefix used for exported files
- `prediction_file`: model predictions in COCO result format
- `prediction_label`: optional display title for the first prediction in visualization panels
- `prediction_file2`: optional second prediction file for side-by-side visualization / comparison plots
- `prediction_label2`: optional display title for the second prediction in visualization panels
- `annotation_file`: COCO ground-truth annotations
- `image_dir`: directory containing the corresponding images
- `output_dir`: where generated Excel files are written

## Typical Workflow

### 1. Generate the base metric tables

```bash
python create_tables.py
```

This creates:

- `<name>_instance_based.xlsx`
- `<name>_image_based.xlsx`

`create_tables.py` now shows a progress bar while per-instance metrics are being computed.

### 2. Generate analysis-oriented metric reports

```bash
python report_metrics_stats.py
```

This creates:

- `<name>_metric_stats.xlsx`
- `<name>_outliers.xlsx`

`<name>_metric_stats.xlsx` contains these sheets:

- `metric_stats`: overall summary per metric
- `grouped_stats`: grouped analysis by `size`, `touch_border`, `#vertices`, `area_bin`, `orientation_bin`, `n_ratio_bin`, and `n_diff_group`
- `high_signal_groups`: grouped rows filtered to groups with enough samples and sorted by worst-performing groups
- `top_k_groups`: the worst-ranked groups per metric for quick triage

`<name>_outliers.xlsx` contains:

- `outliers`: long-form list of detected outlier instances with `metric`, `method`, `image_id`, `instance_id`, and metric `value`

### 3. Visualize results

Image-level visualization:

```bash
python vis.py --img_id <img_id>
```

Instance-level visualization:

```bash
python vis.py --instance --img_id <img_id> --ins_id <ins_id>
```

The required `img_id` and `ins_id` values are available in the exported instance table.

If `prediction_file2` is set in `config.yaml`, `vis.py` will automatically show:
- `GT`
- `prediction_file`
- `prediction_file2`

You can also override either prediction path from the CLI with `--pred_file1` and `--pred_file2`, and override subplot titles with `--label1` and `--label2`.

If you want saved plots instead of interactive display:

```bash
python vis.py --img_id <img_id> --save
python vis.py --instance --img_id <img_id> --ins_id <ins_id> --save
```

Two-result comparison example:

```bash
python vis.py --img_id <img_id> --pred_file2 /path/to/second_results.json
```

`--save` writes plots into `output_dir` from `config.yaml`.

### 4. Generate analysis plots from the exported instance workbook

```bash
python plot_analysis.py
```

This reads `<output_dir>/<name>_instance_based.xlsx` and saves a compact set of analysis plots for:
- simplicity
- regularity
- fidelity

Typical outputs include:
- metric distribution overviews
- `#vertices` vs `#vertices_pred`
- `iou` vs `polis`
- `mta` vs `ciou`
- grouped boxplots by `size`
- heatmaps over `area_bin x #vertices`

To compare two exported workbooks:

```bash
python plot_analysis.py --file path/to/model_a_instance_based.xlsx --file2 path/to/model_b_instance_based.xlsx --label1 ModelA --label2 ModelB
```

## Main Files

- [create_tables.py](create_tables.py): main entry point for metric extraction and Excel export
- [report_metrics_stats.py](report_metrics_stats.py): summary statistics, grouped analysis, and top-k failure groups
- [vis.py](vis.py): interactive or saved visualization
- [plot_analysis.py](plot_analysis.py): ready-made plots for polygon simplicity, regularity, and fidelity
- [compare_two_results.py](compare_two_results.py): compare two exported result files
- [metrics/polis.py](metrics/polis.py): POLIS metric
- [metrics/maxtan.py](metrics/maxtan.py): max tangent angle / contour metric
- [metrics/ciou.py](metrics/ciou.py): IoU and C-IoU logic

## Exported Instance Table

The per-instance table includes columns such as:

- `image_id`
- `instance_id`
- `#vertices`
- `area`
- `size`
- `orientation`
- `touch_border`
- `polis`
- `box_iou`
- `mta`
- `iou`
- `ciou`
- `#vertices_pred`
- `N_diff`
- `N_ratio`

These are the main inputs used by `report_metrics_stats.py` for grouped analysis.

## Notes

- Predictions and annotations should be polygon-based COCO segmentation data.
- Some metrics use `-1` as an invalid / unmatched sentinel in the instance table; the reporting script filters these out before computing summary statistics.
- Empty prediction files are handled in the metric pipeline.

## ToDo

- [x] Instance-based visualization
- [x] Image-based table
- [x] Add the option to save the plots
- [x] Add plots and analysis of results/annotations
- [x] Add interactive graph interface
- [x] Add tables of metrics useful for analysis like mean / median / min / max
- [x] List `img_id` and `ins_id` for high-error cases
- [ ] TP / FP / FN mask visualization
- [ ] Optional web-based interactive interface
