import json
import os
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from distance_estimation.detection.constants import KITTI_DEFAULT_SIZE, KITTI_DETECTION_TRAIN_PATH, USED_KITTI_CLASS_NAMES


def get_focal_length(path: Path, calibration_dir_path: Path) -> float:
    filename = str(path).split("/")[-1]
    calib_txt_path = calibration_dir_path / filename
    with open(calib_txt_path, "r") as file:
        for line in file:
            if line.startswith("P2:"):
                values = line.split()
                focal_length = float(values[1])
                break
    return focal_length


def prepare_kitti_data(kitti_labels_dir: Path, calibration_dir_path: Path) -> pd.DataFrame:
    labels_files = sorted([f for f in kitti_labels_dir.glob("*")])
    results = {"cls": [], "h": [], "focal_length": [], "dist": []}

    for label_file_path in labels_files:
        focal_length = get_focal_length(path=label_file_path, calibration_dir_path=calibration_dir_path)
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
    mean_real_height = (train_df["dist"] * train_df["h"]) / train_df["focal_length"]
    return mean_real_height.mean()


def save_model(mean_heights: Dict[int, float]) -> Path:
    out_model_path = os.path.join(os.path.dirname(__file__), "model.json")
    with open(out_model_path, "w") as f:
        json.dump(mean_heights, f)
    print(f"Model succesfully saved to {out_model_path}")


def find_params():
    data_dir = Path(KITTI_DETECTION_TRAIN_PATH).resolve()
    kitti_labels_dir = data_dir / "processed_kitti" / "labels"
    calib_dir = data_dir / "training" / "calib"
    df_all = prepare_kitti_data(kitti_labels_dir=kitti_labels_dir, calibration_dir_path=calib_dir)
    cls_datasets = prepare_datasets(df=df_all)

    mean_real_heights = {}
    mean_absolute_errors = {}
    for cls, data in cls_datasets.items():
        train_df = data["train"]
        test_df = data["test"]

        mean_real_height = calculate_mean_real_height(train_df=train_df)
        mean_real_heights[cls] = mean_real_height

        test_df["predicted_dist"] = (mean_real_height * test_df["focal_length"]) / test_df["h"]
        mae = mean_absolute_error(test_df["dist"], test_df["predicted_dist"])
        mean_absolute_errors[cls] = mae

    print("Mean Real Heights for each class:", mean_real_heights)
    print("Mean Absolute Errors for each class on valid:", mean_absolute_errors)

    save_model(mean_heights=mean_real_heights)


def main():
    find_params()


if __name__ == "__main__":
    main()
