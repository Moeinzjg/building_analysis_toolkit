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
The code is changed to contain instance-based evaluation too.
"""
import json
import numpy as np
from tqdm import tqdm
from pycocotools import mask as cocomask
from pycocotools.coco import COCO
from collections import defaultdict
from pycocotools import mask as maskUtils


def bounding_box(points):
    """returns a list containing the bottom left and the top right
    points in the sequence
    Here, we traverse the collection of points only once,
    to find the min and max for x and y
    """
    bot_left_x, bot_left_y = float('inf'), float('inf')
    top_right_x, top_right_y = float('-inf'), float('-inf')
    for x, y in points:
        bot_left_x = min(bot_left_x, x)
        bot_left_y = min(bot_left_y, y)
        top_right_x = max(top_right_x, x)
        top_right_y = max(top_right_y, y)

    return [bot_left_x, bot_left_y,
            top_right_x - bot_left_x, top_right_y - bot_left_y]


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
        list_N_ratio.append(N/(N_GT+ 1e-9))
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


class CiouEval():
    def __init__(self, cocoGt=None, cocoDt=None) -> None:
        self.cocoGt = cocoGt
        self.cocoDt = cocoDt

        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        self.imgIds = list(sorted(self.cocoGt.imgs.keys()))
        self._prepare()

    def _prepare(self):
        gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=self.imgIds))
        dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=self.imgIds))
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id']].append(dt)

    def eval_inst(self, img_id, ins_id):  # (input_json, gti_annotations):
        gts = self._gts[img_id]
        dts = self._dts[img_id]

        if len(gts) == 0:
            dt_polygons = [dt['segmentation'][0]
                           for dt in dts]
            N = len(dt_polygons[0]) // 2
            return -1, -1, N, 0, -1
        if len(dts) == 0:
            gt_polygons = [gt['segmentation'][0]
                           for gt in gts if gt['id'] == ins_id]
            N_GT = len(gt_polygons[0]) // 2
            return -1, -1, 0, N_GT, -1

        gt_bboxs = [bounding_box(np.array(gt['segmentation'][0]
                                          ).reshape(-1, 2)
                                 ) for gt in gts if gt['id'] == ins_id]
        dt_bboxs = [bounding_box(np.array(dt['segmentation'][0]
                                          ).reshape(-1, 2)) for dt in dts]
        gt_polygons = [gt['segmentation'][0]
                       for gt in gts if gt['id'] == ins_id]
        dt_polygons = [dt['segmentation'][0]
                       for dt in dts]

        # IoU match
        iscrowd = [0] * len(gt_bboxs)
        box_ious = maskUtils.iou(dt_bboxs, gt_bboxs, iscrowd)
        matched_idx = np.argmax(box_ious[:, 0])

        # Calculate C-IoU
        img = self.cocoGt.loadImgs(img_id)[0]

        # making gt mask
        rle = cocomask.frPyObjects(gt_polygons,
                                   img['height'], img['width'])
        gt_m = cocomask.decode(rle)
        gt_mask = gt_m.reshape((img['height'], img['width']))
        N_GT = len(gt_polygons[0]) // 2
        gt_mask = gt_mask != 0

        # making pred mask
        rle = cocomask.frPyObjects([dt_polygons[matched_idx]],
                                   img['height'], img['width'])
        m = cocomask.decode(rle)
        mask = m.reshape((img['height'], img['width']))
        N = len(dt_polygons[matched_idx]) // 2
        mask = mask != 0

        # Calculate IoU, C-IoU, and N Ratio
        ps = 1 - np.abs(N - N_GT) / (N + N_GT + 1e-9)
        iou = calc_IoU(mask, gt_mask)
        ciou = iou * ps
        N_ratio = N / N_GT

        return iou, ciou, N, N_GT, N_ratio
