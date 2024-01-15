# building_analysis_toolkit
A toolkit for both quantitative and qualitative analysis of building instance segmentation results in <br>
MS COCO json format.

## Requirements
To install all the packages needed, you can simply use conda:
```
conda env create -f env.yml\
conda activate atk
```

## Usage
You need your annotation and prediction results files to be in MS COCO format.<br>
Then, copy "sample_config.yaml", rename it to "config.yaml", and enter the file names and configs of yours.<br>
After that, you can export the tables including instance-wise characteristics and metrics by:
```
python create_tables.py
```
To visualize the results and annotation you have two options:

1) image-based
```
python vis.py --img_id <img_id>
```
img_id is available in the output excel files.

2) instance-based
```
python vis.py --instance --img_id <img_id> --ins_id <ins_id>
```
img_id and ins_id are available in the output excel files.

## ToDo
- [x] Instance-based Visualization
- [x] Image-based table
- [x] Add the option to save the plots
- [x] Add plots and analysis of results/annotations
- [x] Add interactive graph interface
- [ ] Add tables of metrics useful for analysis like their med, average, and min/max
- [ ] List of img_id, and ins_id of highest errors
- [ ] TP/FP/FN Mask Visualization
- [ ] Add web-based interactive interface using maube Dash (Optional)

