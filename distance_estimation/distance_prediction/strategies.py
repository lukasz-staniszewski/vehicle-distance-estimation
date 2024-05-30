import numpy as np


def _depth_bbox_region(depth_mask, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return depth_mask[y1:y2, x1:x2]


def _depth_bbox_center_region(depth_mask, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    cx1, cy1 = x1 + int((x2 - x1) * 0.25), y1 + int((y2 - y1) * 0.25)
    cx2, cy2 = x1 + int((x2 - x1) * 0.75), y1 + int((y2 - y1) * 0.75)
    return depth_mask[cy1:cy2, cx1:cx2]


def bbox_depth(depth_mask, bbox, strategy):
    region_type, method = strategy.split("_")
    region = _depth_bbox_region(depth_mask, bbox) if region_type == "bbox" else _depth_bbox_center_region(depth_mask, bbox)

    if method == "mean":
        return np.mean(region)
    elif method == "median":
        return np.median(region)
    elif method == "min":
        return np.min(region)
    elif method == "percentile":
        return np.percentile(region, 25)
    else:
        raise NotImplementedError("Unknown method after '_'")
