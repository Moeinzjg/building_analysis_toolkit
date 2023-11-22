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

## ToDo
- [ ] Instance-based Visualization 
- [ ] Image-based table 
- [ ] TP/FP/FN Mask Visualization 
