import json
import os
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.metrics import mean_absolute_error

from distance_estimation.detection.constants import KITTI_DEFAULT_SIZE, KITTI_DETECTION_TRAIN_PATH, USED_KITTI_CLASS_NAMES

KITTI_DETECTION_TRAINDATA_DIR = Path(KITTI_DETECTION_TRAIN_PATH).resolve()

KITTI_TRAINLABELS_DIR = KITTI_DETECTION_TRAINDATA_DIR / "processed_kitti" / "labels_train"
KITTI_TESTLABELS_DIR = KITTI_DETECTION_TRAINDATA_DIR / "processed_kitti" / "labels_test"


def get_focal_length(img_path: Path) -> float:
    calib_dir_path = KITTI_DETECTION_TRAINDATA_DIR / "training" / "calib"
    filename = str(img_path).split("/")[-1]
    filename = re.sub(r"\.png", ".txt", filename)
    calib_txt_path = calib_dir_path / filename
    with open(calib_txt_path, "r") as file:
        for line in file:
            if line.startswith("P2:"):
                values = line.split()
                focal_length = float(values[1])
                break
    return focal_length


def prepare_kitti_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    trainlabels_files = sorted([f for f in KITTI_TRAINLABELS_DIR.glob("*")])
    testlabels_files = sorted([f for f in KITTI_TESTLABELS_DIR.glob("*")])

    results_train = {"cls": [], "h": [], "focal_length": [], "dist": []}
    results_test = {"cls": [], "h": [], "focal_length": [], "dist": []}

    for label_file_path in trainlabels_files:
        focal_length = get_focal_length(img_path=label_file_path)
        with open(label_file_path, "r") as file:
            for line in file:
                l = line.strip()
                l_spl = l.split(" ")
                results_train["cls"].append(int(l_spl[0]))
                results_train["h"].append(float(l_spl[4]) * KITTI_DEFAULT_SIZE[1])
                results_train["focal_length"].append(focal_length)
                results_train["dist"].append(float(l_spl[5]))

    for label_file_path in testlabels_files:
        focal_length = get_focal_length(img_path=label_file_path)
        with open(label_file_path, "r") as file:
            for line in file:
                l = line.strip()
                l_spl = l.split(" ")
                results_test["cls"].append(int(l_spl[0]))
                results_test["h"].append(float(l_spl[4]) * KITTI_DEFAULT_SIZE[1])
                results_test["focal_length"].append(focal_length)
                results_test["dist"].append(float(l_spl[5]))

    return pd.DataFrame(results_train), pd.DataFrame(results_test)


def prepare_class_datasets(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    num_classes = len(USED_KITTI_CLASS_NAMES)
    per_class_dataframes = {cls: df[df["cls"] == cls] for cls in range(num_classes)}

    splits = {}
    for cls, class_df in per_class_dataframes.items():
        if not class_df.empty:
            splits[cls] = class_df
    return splits


def calculate_mean_real_height(train_df):
    mean_real_height = (train_df["dist"] * train_df["h"]) / train_df["focal_length"]
    return mean_real_height.mean()


def save_model(mean_heights: Dict[int, float]) -> Path:
    out_model_path = os.path.join(os.path.dirname(__file__), "model.json")
    with open(out_model_path, "w") as f:
        json.dump(mean_heights, f)
    print(f"Model succesfully saved to {out_model_path}")


def validate_model(cls_datasets, mean_real_heights):
    mean_absolute_errors = {}
    sizes = {}
    for cls, test_df in cls_datasets.items():
        mean_real_height = mean_real_heights[cls]
        test_df = test_df.copy()
        test_df.loc[:, "predicted_dist"] = (mean_real_height * test_df["focal_length"]) / test_df["h"]
        mae = mean_absolute_error(test_df["dist"], test_df["predicted_dist"])
        mean_absolute_errors[cls] = mae
        sizes[cls] = test_df.shape[0]
    print("Mean Absolute Errors for each class:")
    for cls, error in mean_absolute_errors.items():
        print(f"{USED_KITTI_CLASS_NAMES[cls]}: {error:.04f} m")

    macro_sum = 0
    micro_sum = 0
    n_cls = 0
    sum_sizes = 0
    for cls in mean_absolute_errors.keys():
        macro_sum += mean_absolute_errors[cls]
        micro_sum += mean_absolute_errors[cls] * sizes[cls]
        n_cls += 1
        sum_sizes += sizes[cls]

    print(f"Macro Mean Absolute Error: {(macro_sum / n_cls):.04f} m")
    print(f"Micro Mean Absolute Error: {(micro_sum / sum_sizes):.04f} m")


def find_params():
    df_train, df_test = prepare_kitti_data()
    cls_train_datasets = prepare_class_datasets(df=df_train)
    cls_test_datasets = prepare_class_datasets(df=df_test)

    print("Creating model...")
    mean_real_heights = {}
    for cls, train_df in cls_train_datasets.items():
        mean_real_height = calculate_mean_real_height(train_df=train_df)
        mean_real_heights[cls] = mean_real_height
    print("Mean Real Heights for each class:", mean_real_heights)

    print(f"On training validation...")
    validate_model(cls_train_datasets, mean_real_heights)
    print(f"On testing validation...")
    validate_model(cls_test_datasets, mean_real_heights)

    save_model(mean_heights=mean_real_heights)


def main():
    find_params()


if __name__ == "__main__":
    main()
