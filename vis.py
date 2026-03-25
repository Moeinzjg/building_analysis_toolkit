import argparse
import os
import os.path as osp
from collections import defaultdict

import matplotlib.patches as Patches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

from metrics.coco_utils import load_res_or_empty
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
    (0.6941176470588235, 0.34901960784313724, 0.1568627450980392),
)

num_color = len(colormap)


def resolve_label(path, cli_label, config_label, fallback):
    if cli_label:
        return cli_label
    if config_label:
        return config_label
    if path:
        return osp.splitext(osp.basename(path))[0]
    return fallback


def ann_to_polygon(ann):
    segmentation = ann.get('segmentation', [])
    if not segmentation:
        return None
    return np.array(segmentation[0]).reshape(-1, 2)


def build_polygons_by_image(annotations):
    polygons = defaultdict(dict)
    for ann in annotations:
        polygon = ann_to_polygon(ann)
        if polygon is not None:
            polygons[ann['image_id']][ann['id']] = polygon
    return polygons


def load_prediction_polygons(gt_coco, pred_file, img_ids):
    pred_coco, _ = load_res_or_empty(gt_coco, pred_file)
    anns = pred_coco.loadAnns(pred_coco.getAnnIds(imgIds=img_ids))
    return build_polygons_by_image(anns)


def draw_polygons(ax, image, polys, title, color_id=None):
    ax.axis('off')
    ax.imshow(image)
    if not isinstance(polys, list):
        polys = [polys]
    for i, polygon in enumerate(polys):
        if polygon is None or len(polygon) == 0:
            continue
        color = colormap[i % num_color] if color_id is None else colormap[color_id % num_color]
        node_color = color if color_id is None else (0.72, 0.02, 0.04)
        marker = '.' if color_id is None else 'o'
        ax.plot(polygon[:, 0], polygon[:, 1], color=node_color,
                marker=marker, linestyle=None)
        ax.add_patch(Patches.Polygon(polygon, fill=False,
                                     ec=color, linewidth=1.5))
        ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.3)
    ax.set_title(title)


def render_panels(panels, save_path=None):
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6))
    if len(panels) == 1:
        axes = [axes]
    for ax, panel in zip(axes, panels):
        draw_polygons(ax, panel['image'], panel['polys'], panel['title'],
                      color_id=panel.get('color_id'))
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def match_prediction(gt_polygon, dt_polygons):
    dt_instances = [polygon for polygon in dt_polygons.values()]
    if not dt_instances:
        return None

    gt_box = [bounding_box(gt_polygon)]
    dt_boxes = [bounding_box(dt_polygon) for dt_polygon in dt_instances]
    ious = maskUtils.iou(dt_boxes, gt_box, [0])
    matched_idx = np.argmax(ious[:, 0])
    dt_polygon = dt_instances[matched_idx].copy()
    return np.concatenate([dt_polygon, np.expand_dims(dt_polygon[0], 0)], axis=0)


def crop_instance_view(image, gt_polygon, pred_polygons, width, height):
    gt_box = [gt_polygon[:, 0].min(), gt_polygon[:, 1].min(),
              gt_polygon[:, 0].max(), gt_polygon[:, 1].max()]
    left = int(max(gt_box[0] - 10, 0))
    top = int(max(gt_box[1] - 10, 0))
    right = int(min(gt_box[2] + 10, width))
    bottom = int(min(gt_box[3] + 10, height))

    cropped_image = image[top:bottom, left:right, :]

    gt_crop = gt_polygon.copy()
    gt_crop[:, 0] -= left
    gt_crop[:, 1] -= top

    pred_crops = []
    for pred_polygon in pred_polygons:
        if pred_polygon is None:
            pred_crops.append(None)
            continue
        pred_crop = pred_polygon.copy()
        pred_crop[:, 0] -= left
        pred_crop[:, 1] -= top
        pred_crops.append(pred_crop)

    return cropped_image, gt_crop, pred_crops


def visualize(ann_file, pred_file1, pred_file2, image_dir,
              instance, img_id, ins_id, label1, label2, save_dir=None):
    gt_coco = COCO(ann_file)
    img_ids = list(sorted(gt_coco.imgs.keys())) if img_id is None else [img_id]

    gts = gt_coco.loadAnns(gt_coco.getAnnIds(imgIds=img_ids))
    gt_polys = build_polygons_by_image(gts)

    pred_sets = []
    pred_polys1 = load_prediction_polygons(gt_coco, pred_file1, img_ids)
    pred_sets.append((label1, pred_polys1, 1))
    if pred_file2 is not None:
        pred_polys2 = load_prediction_polygons(gt_coco, pred_file2, img_ids)
        pred_sets.append((label2, pred_polys2, 2))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for current_img_id in img_ids:
        img_info = gt_coco.loadImgs(current_img_id)[0]
        img_path = osp.join(image_dir, img_info['file_name'])
        image = plt.imread(img_path)
        width = img_info['width']
        height = img_info['height']

        if instance:
            assert ins_id is not None, 'You need to enter instance id!\n'
            gt_instance = gt_polys[current_img_id].get(ins_id)
            if gt_instance is None:
                raise KeyError(
                    f'Instance {ins_id} not found in image {current_img_id}. '
                    f'Available ids: {list(gt_polys[current_img_id].keys())}'
                )

            matched_preds = [
                match_prediction(gt_instance, pred_polys.get(current_img_id, {}))
                for _, pred_polys, _ in pred_sets
            ]
            cropped_image, gt_crop, pred_crops = crop_instance_view(
                image, gt_instance.copy(), matched_preds, width, height
            )

            panels = [{'image': cropped_image, 'polys': gt_crop, 'title': 'GT', 'color_id': 0}]
            for (pred_label, _, color_id), pred_crop in zip(pred_sets, pred_crops):
                title = pred_label if pred_crop is not None else f'{pred_label} (no match)'
                panels.append({
                    'image': cropped_image,
                    'polys': [] if pred_crop is None else pred_crop,
                    'title': title,
                    'color_id': color_id,
                })
            save_path = None if save_dir is None else osp.join(
                save_dir, f'compare_instance_{current_img_id}_{ins_id}.png'
            )
            render_panels(panels, save_path=save_path)
        else:
            panels = [{
                'image': image,
                'polys': list(gt_polys[current_img_id].values()),
                'title': f'GT: img {current_img_id}',
            }]
            for pred_label, pred_polys, _ in pred_sets:
                panels.append({
                    'image': image,
                    'polys': list(pred_polys[current_img_id].values()),
                    'title': f'{pred_label}: img {current_img_id}',
                })
            save_path = None if save_dir is None else osp.join(
                save_dir, f'compare_image_{current_img_id}.png'
            )
            render_panels(panels, save_path=save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', action='store_true', default=False,
                        help='Enables single-instance visualization')
    parser.add_argument('--img_id', type=int, default=None,
                        help='The image id you want to visualize')
    parser.add_argument('--ins_id', type=int, default=None,
                        help='The instance id you want to visualize')
    parser.add_argument('--pred_file1', type=str, default=None,
                        help='Optional override for the first prediction json')
    parser.add_argument('--pred_file2', type=str, default=None,
                        help='Optional second prediction json for comparison')
    parser.add_argument('--label1', type=str, default=None,
                        help='Display label for the first prediction file')
    parser.add_argument('--label2', type=str, default=None,
                        help='Display label for the second prediction file')
    parser.add_argument('--save', action='store_true', default=False,
                        help='Save plots to output_dir instead of showing them')
    return parser.parse_args()


def main(default_save_dir=None):
    try:
        with open("config.yaml") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print('You need a "config.yaml" file with your configs set in')
        raise

    args = parse_args()
    pred_file1 = args.pred_file1 or cfg['prediction_file']
    pred_file2 = args.pred_file2 or cfg.get('prediction_file2')
    label1 = resolve_label(pred_file1,
                           args.label1,
                           cfg.get('prediction_label'),
                           'Pred 1')
    label2 = resolve_label(pred_file2,
                           args.label2,
                           cfg.get('prediction_label2'),
                           'Pred 2')
    save_dir = cfg['output_dir'] if args.save else default_save_dir

    visualize(cfg['annotation_file'], pred_file1, pred_file2,
              cfg['image_dir'], args.instance, args.img_id,
              args.ins_id, label1, label2, save_dir=save_dir)


if __name__ == '__main__':
    main()
