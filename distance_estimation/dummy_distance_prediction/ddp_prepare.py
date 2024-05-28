import json
import os
import re
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from distance_estimation.detection.constants import KITTI_DEFAULT_SIZE, KITTI_DETECTION_TRAIN_PATH, USED_KITTI_CLASS_NAMES

KITTI_DETECTION_TRAINDATA_DIR = Path(KITTI_DETECTION_TRAIN_PATH).resolve()


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


def prepare_kitti_data(kitti_labels_dir: Path) -> pd.DataFrame:
    labels_files = sorted([f for f in kitti_labels_dir.glob("*")])
    results = {"cls": [], "h": [], "focal_length": [], "dist": []}

    for label_file_path in labels_files:
        focal_length = get_focal_length(img_path=label_file_path)
        with open(label_file_path, "r") as file:
            for line in file:
                l = line.strip()
                l_spl = l.split(" ")
                results["cls"].append(int(l_spl[0]))
                results["h"].append(float(l_spl[4]) * KITTI_DEFAULT_SIZE[1])
                results["focal_length"].append(focal_length)
                results["dist"].append(float(l_spl[5]))

    return pd.DataFrame(results)


def prepare_datasets(df: pd.DataFrame) -> Dict[int, Dict[str, pd.DataFrame]]:
    num_classes = len(USED_KITTI_CLASS_NAMES)
    per_class_dataframes = {cls: df[df["cls"] == cls] for cls in range(num_classes)}

    train_test_splits = {}
    for cls, class_df in per_class_dataframes.items():
        if not class_df.empty:
            train, test = train_test_split(class_df, test_size=0.1, random_state=42)
            train_test_splits[cls] = {"train": train, "test": test}
    return train_test_splits


def calculate_mean_real_height(train_df):
    """From training dataset, for each object, we take bbox height, camera focal length and metric distance to that object, to obtain avarage real object height (in meters)."""
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
    for cls, data in cls_datasets.items():
        mean_real_height = mean_real_heights[cls]
        test_df = data["test"]

        test_df["predicted_dist"] = (mean_real_height * test_df["focal_length"]) / test_df["h"]
        mae = mean_absolute_error(test_df["dist"], test_df["predicted_dist"])
        mean_absolute_errors[cls] = mae
        sizes[cls] = test_df.shape[0]
    print("Mean Absolute Errors for each class on valid:")
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

    print(f"Macro Mean Absolute Error on valid: {(macro_sum / n_cls):.04f} m")
    print(f"Micro Mean Absolute Error on valid: {(micro_sum / sum_sizes):.04f} m")


def find_params():
    kitti_labels_dir = KITTI_DETECTION_TRAINDATA_DIR / "processed_kitti" / "labels"
    df_all = prepare_kitti_data(kitti_labels_dir=kitti_labels_dir)
    cls_datasets = prepare_datasets(df=df_all)

    print("Creating model...")
    mean_real_heights = {}
    for cls, data in cls_datasets.items():
        train_df = data["train"]

        mean_real_height = calculate_mean_real_height(train_df=train_df)
        mean_real_heights[cls] = mean_real_height
    print("Mean Real Heights for each class:", mean_real_heights)

    validate_model(cls_datasets, mean_real_heights)

    save_model(mean_heights=mean_real_heights)


def main():
    find_params()


if __name__ == "__main__":
    main()
