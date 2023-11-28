import json
import argparse
import os.path as osp
from collections import defaultdict

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from metrics.polis import bounding_box


plt.rcParams["savefig.bbox"] = 'tight'

colormap = (
    (0.10987878, 0.8754545, 0.0980392156862745),
    (0.8901960784313725, 0.10196078431372549, 0.10980392156862745),
    (0.6509803921568628, 0.807843137254902, 0.8901960784313725),
    (0.12156862745098039, 0.47058823529411764, 0.7058823529411765),
    (0.984313725490196, 0.6039215686274509, 0.6),
    (0.9921568627450981, 0.7490196078431373, 0.43529411764705883),
    (1.0, 0.4980392156862745, 0.0),
    (0.792156862745098, 0.6980392156862745, 0.8392156862745098),
    (0.41568627450980394, 0.23921568627450981, 0.6039215686274509),
    (1.0, 1.0, 0.6),
    (0.6941176470588235, 0.34901960784313724, 0.1568627450980392))

num_color = len(colormap)


def load_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def show_polygons(image, polys, title, color_id=None):
    plt.axis('off')
    plt.imshow(image)
    if not isinstance(polys, list):
        polys = [polys]
    for i, polygon in enumerate(polys):
        color = colormap[
            i % num_color] if color_id is None else colormap[color_id]
        node_color = color if color_id is None else (0.72, 0.02, 0.04)
        marker = '.' if color_id is None else 'o'
        plt.plot(polygon[:, 0], polygon[:, 1], color=node_color,
                 marker=marker, linestyle=None)
        plt.gca().add_patch(Patches.Polygon(polygon, fill=False,
                                            ec=color, linewidth=1.5))
        plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.3)

    plt.title(title)
    plt.show()


def visualize(ANN_FILE: str, PRED_FILE: str, IMAGE_DIR: str,
              INSTANCE: bool, IMG_ID: int, INS_ID: int) -> None:
    gt_coco = COCO(ANN_FILE)
    pred_coco = gt_coco.loadRes(PRED_FILE)
    img_ids = list(sorted(gt_coco.imgs.keys())) if IMG_ID is None else IMG_ID
    img_ids = [img_ids] if not isinstance(img_ids, list) else img_ids

    gts = gt_coco.loadAnns(gt_coco.getAnnIds(imgIds=img_ids))
    dts = pred_coco.loadAnns(pred_coco.getAnnIds(imgIds=img_ids))

    gt_polys = defaultdict(dict)
    dt_polys = defaultdict(dict)

    for gt in gts:
        gt_poly = {gt['id']: np.array(gt['segmentation']).reshape(-1, 2)}  # TODO: add img boundary limit
        gt_polys[gt['image_id']].update(gt_poly)
    for dt in dts:
        dt_poly = {dt['id']: np.array(dt['segmentation']).reshape(-1, 2)}
        dt_polys[dt['image_id']].update(dt_poly)

    for img_id in img_ids:
        img_info = gt_coco.loadImgs(img_id)[0]
        img_path = osp.join(IMAGE_DIR, img_info['file_name'])
        img = plt.imread(img_path)

        width = img_info['width']
        height = img_info['height']

        if INSTANCE:
            assert INS_ID is not None, 'You need to enter instance id!\n'
            assert IMG_ID is not None, 'You need to enter image id!\n'
            # TODO: as instance ids are unique, then only instance id
            #  must be enough; however, currently we ask for image id too.

            ins = gt_polys[img_id][INS_ID]
            dt_instances = [el for el in dt_polys[img_id].values()]

            gt_box = [bounding_box(ins)]
            dt_boxes = [bounding_box(dt_ins) for dt_ins in dt_instances]

            # IoU match; find the corresponding pred for our gt instance
            iscrowd = [0] * len(gt_box)
            ious = maskUtils.iou(dt_boxes, gt_box, iscrowd)
            matched_idx = np.argmax(ious[:, 0])
            dt_ins = dt_instances[matched_idx]
            # Repeat the first node to close the polygon
            dt_ins = np.concatenate([dt_ins, np.expand_dims(dt_ins[0], 0)], axis=0)

            ins_box = [ins[:, 0].min(), ins[:, 1].min(),
                       ins[:, 0].max(), ins[:, 1].max()]
            # 10 pixels box expansion to have some context
            left = int(max(ins_box[0] - 10, 0))
            top = int(max(ins_box[1] - 10, 0))
            right = int(min(ins_box[2] + 10, width))
            bottom = int(min(ins_box[3] + 10, height))

            ins_img = img[top:bottom, left:right, :]
            ins[:, 0] = ins[:, 0] - ins_box[0] + (int(ins_box[0]) - left)
            ins[:, 1] = ins[:, 1] - ins_box[1] + (int(ins_box[1]) - top)

            dt_ins[:, 0] = dt_ins[:, 0] - ins_box[0] + (int(ins_box[0]) - left)
            dt_ins[:, 1] = dt_ins[:, 1] - ins_box[1] + (int(ins_box[1]) - top)

            show_polygons(ins_img, ins, title='GT', color_id=0)
            show_polygons(ins_img, dt_ins, title='Pred', color_id=1)

        else:
            dt_polygons = [el for el in dt_polys[img_id].values()]
            gt_polygons = [el for el in gt_polys[img_id].values()]
            show_polygons(img, dt_polygons, title=f'Pred: img {img_id}')
            show_polygons(img, gt_polygons, title=f'GT: img {img_id}')


if __name__ == '__main__':

    try:
        with open("config.yaml") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print('You need a "config.yaml" file with your configs set in')

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', action='store_true', default=False,
                        help='Enables single-instance visualization')
    parser.add_argument('--img_id', type=int, default=None,
                        help='The image id you want to visualize')
    parser.add_argument('--ins_id', type=int, default=None,
                        help='The image id you want to visualize')
    args = parser.parse_args()

    visualize(cfg['annotation_file'], cfg['prediction_file'],
              cfg['image_dir'], args.instance, args.img_id,
              args.ins_id)
