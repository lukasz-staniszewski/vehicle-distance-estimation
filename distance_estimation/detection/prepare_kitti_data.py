import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from distance_estimation.detection.constants import KEY_DONT_CARE, KITTI_CLASSNAME_TO_NUMBER, KITTI_COLUMNS, KITTI_DETECTION_TRAIN_PATH
from distance_estimation.detection.utils import UserKittiYoloConfig, read_user_config


@dataclass
class KittiTrainConfig:
    dir_path: Path
    processed_kitti_dir_path: Path
    processed_yolo_dir_path: Path
    raw_imgs_path: Path
    raw_labels_path: Path
    raw_calib_path: Path
    processed_kitti_labels_train_path: Path
    processed_kitti_labels_test_path: Path
    processed_kitti_classes_file_path: Path
    processed_yolo_train_path: Path
    processed_yolo_test_path: Path
    processed_yolo_valid_path: Path
    processed_yolo_config_file_path: Path
    images: list
    labels: list
    columns: list
    train_df: pd.DataFrame


def get_kitti_train_config(
    path: os.PathLike = KITTI_DETECTION_TRAIN_PATH,
) -> KittiTrainConfig:
    dir_path = Path(path)
    processed_kitti_dir_path = dir_path / "processed_kitti"
    processed_yolo_dir_path = dir_path / "processed_yolo"

    config = {
        "dir_path": dir_path,
        "processed_kitti_dir_path": processed_kitti_dir_path,
        "processed_yolo_dir_path": processed_yolo_dir_path,
        "raw_imgs_path": dir_path / "training" / "image_2",
        "raw_labels_path": dir_path / "training" / "label_2",
        "raw_calib_path": dir_path / "training" / "calib",
        "processed_kitti_labels_train_path": processed_kitti_dir_path / "labels_train",
        "processed_kitti_labels_test_path": processed_kitti_dir_path / "labels_test",
        "processed_kitti_classes_file_path": processed_kitti_dir_path / "classes.json",
        "processed_yolo_train_path": processed_yolo_dir_path / "train",
        "processed_yolo_test_path": processed_yolo_dir_path / "test",
        "processed_yolo_valid_path": processed_yolo_dir_path / "valid",
        "processed_yolo_config_file_path": processed_yolo_dir_path / "kitti.yaml",
    }

    config.update(
        {
            "images": sorted(list(config["raw_imgs_path"].glob("*"))),
            "labels": sorted(list(config["raw_labels_path"].glob("*"))),
            "columns": KITTI_COLUMNS,
        }
    )

    config.update({"train_df": pd.DataFrame({"image": config["images"], "label": config["labels"]})})

    return KittiTrainConfig(**config)


def get_labels(config: KittiTrainConfig, p: pd.Series) -> pd.DataFrame:
    data = pd.read_csv(p, sep=" ", names=config.columns, usecols=config.columns)
    return data


def open_image(path: Path) -> cv.typing.MatLike:
    im = cv.imread(filename=str(path))
    im = cv.cvtColor(src=im, code=cv.COLOR_BGR2RGB)
    return im


def draw_with_bbox(config: KittiTrainConfig, idx: int):
    sample = config.df_train.iloc[idx, :]
    image = open_image(path=sample["image"])
    labels = get_labels(config=config, p=sample["label"])

    for _, row in labels.iterrows():
        if row.label == KEY_DONT_CARE:
            continue

        left_corner = [int(row.bbox_xmin), int(row.bbox_ymin)]
        right_corner = [int(row.bbox_xmax), int(row.bbox_ymax)]
        label_color = config.label_colors.get(row.label, (0, 255, 0))

        image = cv.rectangle(img=image, pt1=left_corner, pt2=right_corner, color=label_color, thickness=2)
        image = cv.putText(
            img=image,
            text=row.label,
            org=(left_corner[0] + 10, left_corner[1] - 4),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=label_color,
            thickness=3,
        )

    plt.imshow(image)


def convert_bbox_to_yolo_format(bbox: Tuple[int, int, int, int], size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x = (bbox[0] + bbox[1]) / 2.0
    y = (bbox[2] + bbox[3]) / 2.0
    w = bbox[1] - bbox[0]
    h = bbox[3] - bbox[2]
    x, y = x * dw, y * dh
    w, h = w * dw, h * dh
    return (x, y, w, h)


def preprocess_kitti_data(kitti_config: KittiTrainConfig, user_config: UserKittiYoloConfig):
    def get_sample_id(path: Path) -> str:
        basename = os.path.basename(path)
        return os.path.splitext(basename)[0]

    def get_image_size(path: Path) -> Tuple[int, int]:
        return Image.open(path).size

    def get_sample_class_number(sample_class: str, use_dont_care: bool) -> int | None:
        if use_dont_care and sample_class == KEY_DONT_CARE:
            return KITTI_CLASSNAME_TO_NUMBER[sample_class]
        elif sample_class != KEY_DONT_CARE:
            return KITTI_CLASSNAME_TO_NUMBER[sample_class]
        return None

    def parse_sample(
        sample_labels_path: Path,
        sample_image_path: Path,
        use_dont_care: bool = False,
    ) -> Tuple[List[Tuple[int, float, float, float, float]], str]:
        yolo_labels = []

        with open(sample_labels_path) as csv_file:
            reader = csv.DictReader(
                csv_file,
                fieldnames=KITTI_COLUMNS,
                delimiter=" ",
            )
            for row in reader:
                class_number = get_sample_class_number(sample_class=row["label"], use_dont_care=use_dont_care)
                if class_number is not None:
                    size = get_image_size(path=sample_image_path)
                    bbox = (
                        float(row["bbox_xmin"]),
                        float(row["bbox_xmax"]),
                        float(row["bbox_ymin"]),
                        float(row["bbox_ymax"]),
                    )
                    dist = float(row["distance"])
                    yolo_bbox = convert_bbox_to_yolo_format(bbox=bbox, size=size)
                    yolo_labels.append((class_number,) + yolo_bbox + (dist,))

        return yolo_labels

    if not os.path.exists(kitti_config.processed_kitti_labels_train_path):
        os.makedirs(kitti_config.processed_kitti_labels_train_path, exist_ok=True)
    if not os.path.exists(kitti_config.processed_kitti_labels_test_path):
        os.makedirs(kitti_config.processed_kitti_labels_test_path, exist_ok=True)

    train_images = sorted(list((kitti_config.dir_path / "train" / "image_2").glob("*")))
    test_images = sorted(list((kitti_config.dir_path / "test" / "image_2").glob("*")))
    train_labels = sorted(list((kitti_config.dir_path / "train" / "label_2").glob("*")))
    test_labels = sorted(list((kitti_config.dir_path / "test" / "label_2").glob("*")))

    for img_list, lbl_list, lbl_output_dir in [
        (train_images, train_labels, kitti_config.processed_kitti_labels_train_path),
        (test_images, test_labels, kitti_config.processed_kitti_labels_test_path),
    ]:
        for img_path, lbl_path in tqdm(zip(img_list, lbl_list), desc="Generating labels...", total=len(img_list)):
            sample_id = get_sample_id(path=lbl_path)
            yolo_labels = parse_sample(
                sample_labels_path=lbl_path,
                sample_image_path=img_path,
                use_dont_care=user_config.use_dont_care_label,
            )

            with open(
                os.path.join(
                    lbl_output_dir,
                    "{}.txt".format(sample_id),
                ),
                "w",
            ) as kitti_label_file:
                for lbl in yolo_labels:
                    kitti_label_file.write("{} {} {} {} {} {}\n".format(*lbl))

    with open(kitti_config.processed_kitti_classes_file_path, "w") as f:
        json.dump(obj=KITTI_CLASSNAME_TO_NUMBER, fp=f)
    print("[2/6] Kitti data preprocessed")


def prepare_yolo_data(kitti_config: KittiTrainConfig, user_config: UserKittiYoloConfig):
    def remove_distance_label(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
        with open(label_path, "w") as f:
            for line in lines:
                f.write(" ".join(line.split()[:-1]) + "\n")

    with open(kitti_config.processed_kitti_classes_file_path, "r") as f:
        classes = json.load(f)

    train_images = sorted(list((kitti_config.dir_path / "train" / "image_2").glob("*")))
    test_images = sorted(list((kitti_config.dir_path / "test" / "image_2").glob("*")))
    train_labels = sorted(list((kitti_config.processed_kitti_labels_train_path).glob("*")))
    test_labels = sorted(list((kitti_config.processed_kitti_labels_test_path).glob("*")))

    train_pairs = list(zip(train_images, train_labels))
    test_pairs = list(zip(test_images, test_labels))

    train_set, valid_set = train_test_split(train_pairs, test_size=user_config.val_size, shuffle=True)

    for dir_path in [
        kitti_config.processed_yolo_train_path,
        kitti_config.processed_yolo_valid_path,
        kitti_config.processed_yolo_test_path,
    ]:
        os.makedirs(dir_path, exist_ok=True)

    for image_path, label_path in tqdm(train_set, desc="Preparing YOLO training data"):
        target_image_path = kitti_config.processed_yolo_train_path / image_path.name
        target_label_path = kitti_config.processed_yolo_train_path / label_path.name
        shutil.copy(image_path, target_image_path)
        shutil.copy(label_path, target_label_path)
        remove_distance_label(target_label_path)

    print("[3/6] YOLO training data processed")

    for image_path, label_path in tqdm(valid_set, desc="Preparing YOLO validation data"):
        target_image_path = kitti_config.processed_yolo_valid_path / image_path.name
        target_label_path = kitti_config.processed_yolo_valid_path / label_path.name
        shutil.copy(image_path, target_image_path)
        shutil.copy(label_path, target_label_path)
        remove_distance_label(target_label_path)

    print("[4/6] YOLO validation data processed")

    for image_path, label_path in tqdm(test_pairs, desc="Preparing YOLO test data"):
        target_image_path = kitti_config.processed_yolo_test_path / image_path.name
        target_label_path = kitti_config.processed_yolo_test_path / label_path.name
        shutil.copy(image_path, target_image_path)
        shutil.copy(label_path, target_label_path)
        remove_distance_label(target_label_path)

    print("[5/6] YOLO test data processed")

    create_yolo_yaml_file(classes=classes, kitti_config=kitti_config)
    print("[6/6] YOLO config file processed")


def create_yolo_yaml_file(classes: List[str], kitti_config: KittiTrainConfig):
    yaml_file = "names:\n"
    yaml_file += "\n".join(f"- {c}" for c in classes)
    yaml_file += f"\nnc: {len(classes)}"
    yaml_file += f"\ntrain: train\nval: valid\ntest: test"
    with open(kitti_config.processed_yolo_config_file_path, "w") as f:
        f.write(yaml_file)


def split_dataset(kitti_config: KittiTrainConfig, user_config: UserKittiYoloConfig):
    images = sorted(list(kitti_config.raw_imgs_path.glob("*.png")))
    labels = sorted(list(kitti_config.raw_labels_path.glob("*.txt")))
    calib = sorted(list(kitti_config.raw_calib_path.glob("*.txt")))

    data = list(zip(images, labels, calib))
    train_data, test_data = train_test_split(data, test_size=user_config.test_size, shuffle=True)

    train_dir = kitti_config.dir_path / "train"
    test_dir = kitti_config.dir_path / "test"

    for subset, subset_name in [(train_data, train_dir), (test_data, test_dir)]:
        image_dir = subset_name / "image_2"
        label_dir = subset_name / "label_2"
        calib_dir = subset_name / "calib"

        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(calib_dir, exist_ok=True)

        for img, lbl, cal in tqdm(subset, desc=f"Moving {subset_name}", total=len(subset)):
            shutil.copy(img, image_dir / img.name)
            shutil.copy(lbl, label_dir / lbl.name)
            shutil.copy(cal, calib_dir / cal.name)

    print("[1/6] Original KITTI data split into train and test sets")


def main():
    kitti_config = get_kitti_train_config(path=KITTI_DETECTION_TRAIN_PATH)
    user_config = read_user_config()

    split_dataset(kitti_config=kitti_config, user_config=user_config)
    preprocess_kitti_data(kitti_config=kitti_config, user_config=user_config)
    prepare_yolo_data(kitti_config=kitti_config, user_config=user_config)


if __name__ == "__main__":
    main()
