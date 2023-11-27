from os import path as osp
import json
import math

import yaml
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from pycocotools.coco import COCO

from metrics.polis import PolisEval
from metrics.maxtan import ContourEval


def det_size(area: float) -> str:
    small_rng = [0 ** 2, 32 ** 2]
    med_rng = [32 ** 2, 96 ** 2]
    large_rng = [96 ** 2, 1e5 ** 2]

    if area <= small_rng[1] and area >= small_rng[0]:
        return 'small'
    if area <= med_rng[1] and area >= med_rng[0]:
        return 'medium'
    if area >= large_rng[0] and area <= large_rng[1]:
        return 'large'

    return 'not in range'


def det_orient(poly: list) -> float:
    poly = np.array(poly).reshape(-1, 2)
    polygon = Polygon(poly)

    # Calculate the minimum area bounding rectangle
    rect = polygon.minimum_rotated_rectangle

    # Extract the coordinates of the rectangle
    try:
        x, y = rect.exterior.coords.xy
    except:
        import pdb; pdb.set_trace()

    # Calculate edge lengths and determine the longer edge
    edge_lengths = [math.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2) for
                    i in range(1, len(x))]
    longer_edge_index = np.argmax(edge_lengths)

    # Calculate the orientation of the longer edge
    dx = x[longer_edge_index] - x[longer_edge_index - 1]
    dy = y[longer_edge_index] - y[longer_edge_index - 1]
    angle = math.degrees(math.atan2(dy, dx))

    return angle


def touch_border(poly: list, width: int, height: int) -> bool:
    touch_right = max(poly[0][::2]) >= width
    touch_left = min(poly[0][::2]) <= 0
    touch_bottom = max(poly[0][1::2]) >= height
    touch_top = min(poly[0][1::2]) <= 0
    if touch_right or touch_left or touch_top or touch_bottom:
        return True
    else:
        return False


def create_image_table():
    # TODO: columns: img_id, iou, coco, ciou, polis
    pass


def load_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def polis_eval(ann_file, pred_file):
    print('\nCalculating POLIS ...\n')
    gt_coco = COCO(ann_file)
    pred_coco = gt_coco.loadRes(pred_file)
    polis_evaluator = PolisEval(gt_coco, pred_coco)
    polis_evaluator.evaluate()
    return polis_evaluator


def max_angle_error_eval(ann_file, pred_file):
    print('\nCalculating Max Angle Error ...\n')
    gt_coco = COCO(ann_file)
    pred_coco = gt_coco.loadRes(pred_file)
    mta_evaluator = ContourEval(gt_coco, pred_coco)
    return mta_evaluator


def create_instance_table(config, annotations: dict,
                          evaluators: tuple) -> None:
    polis_evaluator = evaluators[0]
    mta_evaluator = evaluators[1]

    # Get the dataset info
    name = cfg['name']
    height = annotations['images'][0]['height']
    width = annotations['images'][0]['width']

    # Process each annotation
    row_list = []
    for ann in annotations['annotations']:

        instance_id = ann['id']
        img_id = ann['image_id']
        seg = ann['segmentation']
        polygon = [[min(max(point, 0), min(height, width)) for
                    point in seg[0]]]

        area = Polygon(np.array(polygon).reshape(-1, 2)).area
        if area <= 0:
            continue
        size = det_size(area)
        orient = det_orient(polygon)
        border = touch_border(polygon, width, height)

        # Calculate #vertices for polygons
        vertices = sum([len(poly) // 2 for poly in polygon])

        # Get the corresponding prediction
        polis, box_iou = polis_evaluator.evaluateIns(img_id, instance_id)
        mta = mta_evaluator.evaluate_ins(img_id, instance_id, pool=None)

        # Create the row for instance table
        row_list.append({'image_id': img_id, 'instance_id': instance_id,
                         '#vertices': vertices, 'area': area, 'size': size,
                         'orientation': orient, 'touch_border': border,
                         'polis': polis, 'box_iou': box_iou, 'mta': mta})
        # NOTE: Box IoU is the IoU between the pred and gt boxes derived
        # from the extrema of the polygons
        # -1 for polis means no polis is calculated because box IoU <= 0.5 and
        # it was excluded from the metric calculation
        # -1 for MTA also means there was no matching polygon with gt and
        # it was excluded from the metric calculation

    # Create the instance table
    df = pd.DataFrame(row_list)
    print(df)  # Disply the DataFrame
    # Write to xlsx
    df.to_excel(osp.join(
        config['output_dir'], './instance_based analysis.xlsx'),
        sheet_name=name)


if __name__ == '__main__':

    try:
        with open("config.yaml") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print('You need a "config.yaml" file with your configs set in')

    ann_file = cfg['annotation_file']
    pred_file = cfg['prediction_file']

    # Load coco format files
    annotations = load_json(ann_file)
    results = load_json(pred_file)
    polis_evaluator = polis_eval(ann_file, pred_file)
    mta_evaluator = max_angle_error_eval(ann_file, pred_file)

    # Create tables
    create_instance_table(cfg, annotations, (polis_evaluator, mta_evaluator))
