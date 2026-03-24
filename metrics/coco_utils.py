import json


class EmptyCocoResults:
    def __init__(self, coco_gt):
        self.imgs = coco_gt.imgs
        self.cats = coco_gt.cats

    def getAnnIds(self, imgIds=None, catIds=None, areaRng=None, iscrowd=None):
        return []

    def loadAnns(self, ids=None):
        return []

    def getCatIds(self, catNms=None, supNms=None, catIds=None):
        if catIds:
            return [cat_id for cat_id in catIds if cat_id in self.cats]
        return list(self.cats.keys())


def load_res_or_empty(coco_gt, pred_source):
    if isinstance(pred_source, str):
        with open(pred_source) as f:
            predictions = json.load(f)
    else:
        predictions = pred_source

    if len(predictions) == 0:
        return EmptyCocoResults(coco_gt), predictions

    return coco_gt.loadRes(predictions), predictions
