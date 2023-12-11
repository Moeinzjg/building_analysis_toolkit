"""
This is the code from https://github.com/zorzi-s/PolyWorldPretrainedNetwork.
@article{zorzi2021polyworld,
  title={PolyWorld: Polygonal Building Extraction with Graph Neural Networks
         in Satellite Images},
  author={Zorzi, Stefano and Bazrafkan, Shabab and Habenschuss, Stefan and
          Fraundorfer, Friedrich},
  journal={arXiv preprint arXiv:2111.15491},
  year={2021}
}
"""
import json
import numpy as np
from tqdm import tqdm
from pycocotools import mask as cocomask
from pycocotools.coco import COCO


def calc_IoU(a, b):
    i = np.logical_and(a, b)
    u = np.logical_or(a, b)
    sum_i = np.sum(i)
    sum_u = np.sum(u)

    iou = sum_i/(sum_u + 1e-9)

    is_void = sum_u == 0
    if is_void:
        return 1.0
    else:
        return iou


def compute_iou_ciou(input_json, gti_annotations):
    # Ground truth annotations
    coco_gti = COCO(gti_annotations)

    # Predictions annotations
    submission_file = json.loads(open(input_json).read())
    coco = COCO(gti_annotations)
    coco = coco.loadRes(submission_file)

    image_ids = coco.getImgIds(catIds=coco.getCatIds())
    pbar = tqdm(image_ids)

    list_iou = []
    list_ciou = []
    list_N = []
    list_N_GT = []
    list_N_ratio = []
    pss = []
    for image_id in pbar:

        img = coco.loadImgs(image_id)[0]

        annotation_ids = coco.getAnnIds(imgIds=img['id'])
        annotations = coco.loadAnns(annotation_ids)
        N = 0
        for _idx, annotation in enumerate(annotations):
            try:
                rle = cocomask.frPyObjects(annotation['segmentation'],
                                           img['height'], img['width'])
            except Exception:
                import pdb
                pdb.set_trace()
            m = cocomask.decode(rle)
            if _idx == 0:
                mask = m.reshape((img['height'], img['width']))
                N = len(annotation['segmentation'][0]) // 2
            else:
                mask = mask + m.reshape((img['height'], img['width']))
                N = N + len(annotation['segmentation'][0]) // 2

        mask = mask != 0

        annotation_ids = coco_gti.getAnnIds(imgIds=img['id'])
        annotations = coco_gti.loadAnns(annotation_ids)
        N_GT = 0
        for _idx, annotation in enumerate(annotations):
            rle = cocomask.frPyObjects(annotation['segmentation'],
                                       img['height'], img['width'])
            m = cocomask.decode(rle)
            if _idx == 0:
                mask_gti = m.reshape((img['height'], img['width']))
                N_GT = len(annotation['segmentation'][0]) // 2
            else:
                mask_gti = mask_gti + m.reshape((img['height'], img['width']))
                N_GT = N_GT + len(annotation['segmentation'][0]) // 2

        mask_gti = mask_gti != 0

        ps = 1 - np.abs(N - N_GT) / (N + N_GT + 1e-9)
        iou = calc_IoU(mask, mask_gti)
        list_iou.append(iou)
        list_ciou.append(iou * ps)
        list_N.append(N)
        list_N_GT.append(N_GT)
        list_N_ratio.append(N/N_GT)
        pss.append(ps)
        text = "iou: %2.4f, c-iou: %2.4f, ps:%2.4f"
        pbar.set_description(text % (np.mean(list_iou),
                                     np.mean(list_ciou),
                                     np.mean(pss)))
        pbar.refresh()

    print("Done!")
    print("Mean IoU: ", np.mean(list_iou))
    print("Mean C-IoU: ", np.mean(list_ciou))

    return image_ids, list_iou, list_ciou, list_N, list_N_GT, list_N_ratio
