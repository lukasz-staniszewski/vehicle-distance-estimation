import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import torch
from sklearn.metrics import mean_absolute_error

from distance_estimation.detection.constants import KITTI_DEFAULT_SIZE, KITTI_DETECTION_TRAIN_PATH, USED_KITTI_CLASS_NAMES
from distance_estimation.dummy_distance_prediction.ddp_predict import DistanceDetection
from distance_estimation.dummy_distance_prediction.ddp_prepare import get_focal_length

KITTI_DETECTION_TRAINDATA_DIR = Path(KITTI_DETECTION_TRAIN_PATH).resolve()

KITTI_TESTLABELS_DIR = KITTI_DETECTION_TRAINDATA_DIR / "processed_kitti" / "labels_test"
testlabels_files = sorted([f for f in KITTI_TESTLABELS_DIR.glob("*")])

KITTI_IMAGES_DIR = KITTI_DETECTION_TRAINDATA_DIR / "test" / "image_2"


def get_iou(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
    """Calculates IOU"""
    xmin_1, ymin_1, xmax_1, ymax_1 = bbox1
    xmin_2, ymin_2, xmax_2, ymax_2 = bbox2

    xmin_i = max(xmin_1, xmin_2)
    ymin_i = max(ymin_1, ymin_2)
    xmax_i = min(xmax_1, xmax_2)
    ymax_i = min(ymax_1, ymax_2)
    inter_area = max(0, xmax_i - xmin_i) * max(0, ymax_i - ymin_i)

    box1_area = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)
    box2_area = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area

    return iou


def classify_bbox_size(bbox: Tuple[float, float, float, float]) -> str:
    """Classifies bbox based on its size to one of categories: small, medium or large"""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height

    if area < 32 * 32:
        return "small"
    elif area < 96 * 96:
        return "medium"
    else:
        return "large"


def get_maes(imgs_preds: List[List[DistanceDetection]], imgs_targets: List[List[DistanceDetection]], iou_threshold: float = 0.75):
    """Calculates metric distance error as Mean Absolute Error. Returns MAE which calculated is divided on:
    - classes (MAE per class)
    - size (MAE for small, medium and large objects)
    - over all the objects (with micro- and macro- averaging)
    """
    per_class_mae = defaultdict(list)
    per_size_mae = {"small": [], "medium": [], "large": []}
    all_abs_errors = []

    for img_pred_list, img_gt_list in zip(imgs_preds, imgs_targets):
        for detection_pred in img_pred_list:
            best_iou = 0
            best_gt = None
            for detection_gt in img_gt_list:
                iou = get_iou(detection_pred.xyxy.tolist(), detection_gt.xyxy.tolist())
                if iou > best_iou:
                    best_iou = iou
                    best_gt = detection_gt

            if best_gt and best_iou >= iou_threshold:
                abs_error = torch.abs(detection_pred.distance - best_gt.distance).item()
                per_class_mae[detection_pred.class_name].append(abs_error)
                size_category = classify_bbox_size(detection_pred.xyxy.tolist())
                per_size_mae[size_category].append(abs_error)
                all_abs_errors.append(abs_error)

    per_class_mae = {cls: mean_absolute_error([0] * len(errors), errors) for cls, errors in per_class_mae.items()}
    per_size_mae = {size: mean_absolute_error([0] * len(errors), errors) for size, errors in per_size_mae.items()}
    macro_mae = sum(per_class_mae.values()) / len(per_class_mae) if per_class_mae else 0
    micro_mae = mean_absolute_error([0] * len(all_abs_errors), all_abs_errors) if all_abs_errors else 0

    return {"class_mae": per_class_mae, "size_mae": per_size_mae, "macro_mae": macro_mae, "micro_mae": micro_mae}


def get_test_distance_dataset() -> Dict[str, list]:
    """Returns KITTI distance test dataset"""
    images: List[Image.Image] = []
    focal_lens: List[float] = []
    targets: List[List[DistanceDetection]] = []

    testlabels_files = sorted([f for f in KITTI_TESTLABELS_DIR.glob("*")])

    for label_file_path in testlabels_files:
        focal_length = get_focal_length(img_path=label_file_path)
        img_path = KITTI_IMAGES_DIR / re.sub(r"\.txt", ".png", (str(label_file_path).split("/")[-1]))
        focal_lens.append(focal_length)
        images.append(Image.open(img_path))

        objects = []
        with open(label_file_path, "r") as file:
            for line in file:
                l = line.strip()
                l_spl = l.split(" ")
                cls = int(l_spl[0])
                dist = float(l_spl[5])
                x = float(l_spl[1]) * KITTI_DEFAULT_SIZE[0]
                y = float(l_spl[2]) * KITTI_DEFAULT_SIZE[1]
                w = float(l_spl[3]) * KITTI_DEFAULT_SIZE[0]
                h = float(l_spl[4]) * KITTI_DEFAULT_SIZE[1]
                xyxy = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
                objects.append(
                    DistanceDetection(
                        xyxy=torch.tensor(xyxy),
                        class_idx=torch.tensor(cls),
                        class_name=USED_KITTI_CLASS_NAMES[cls],
                        distance=torch.tensor(dist),
                    )
                )
        targets.append(objects)

    return {
        "images": images,
        "focal_lens": focal_lens,
        "targets": targets
    }
