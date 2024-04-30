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
    processed_kitti_labels_path: Path
    processed_kitti_classes_file_path: Path
    processed_yolo_train_path: Path
    processed_yolo_test_path: Path
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
        "processed_kitti_labels_path": processed_kitti_dir_path / "labels",
        "processed_kitti_classes_file_path": processed_kitti_dir_path / "classes.json",
        "processed_yolo_train_path": processed_yolo_dir_path / "train",
        "processed_yolo_test_path": processed_yolo_dir_path / "test",
        "processed_yolo_config_file_path": processed_yolo_dir_path / "kitti.yaml",
    }

    config.update(
        {
            "images": sorted(list(config["raw_imgs_path"].glob("*"))),
            "labels": sorted(list(config["raw_labels_path"].glob("*"))),
            "columns": KITTI_COLUMNS,
        }
    )

    config.update({"train_df": pd.DataFrame({"image": config["images"], "label": config["labels"]}).iloc[:, :8]})

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
        labels_names = []
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
                        float(row["bbox_ymin"]),
                        float(row["bbox_xmax"]),
                        float(row["bbox_ymax"]),
                    )
                    yolo_bbox = convert_bbox_to_yolo_format(bbox=bbox, size=size)
                    yolo_labels.append((class_number,) + yolo_bbox)
                    labels_names.append(row["label"])

        return yolo_labels, labels_names

    if not os.path.exists(kitti_config.processed_kitti_labels_path):
        os.makedirs(kitti_config.processed_kitti_labels_path, exist_ok=True)

    samples_classes: List[str] = []
    samples_img_path: List[str] = []

    for dir_path, _, files in os.walk(kitti_config.raw_labels_path):
        for file_name in tqdm(files, desc="Generating labels..."):
            if file_name.endswith(".txt"):
                sample_label_path = os.path.join(dir_path, file_name)
                sample_id = get_sample_id(path=sample_label_path)
                sample_img_path = os.path.join(kitti_config.raw_imgs_path, "{}.png".format(sample_id))

                yolo_labels, classes_names = parse_sample(
                    sample_labels_path=sample_label_path,
                    sample_image_path=sample_img_path,
                    use_dont_care=user_config.use_dont_care_label,
                )

                samples_classes.extend(classes_names)
                samples_img_path.append(sample_img_path)

                with open(
                    os.path.join(
                        kitti_config.processed_kitti_labels_path,
                        "{}.txt".format(sample_id),
                    ),
                    "w",
                ) as yolo_label_file:
                    for lbl in yolo_labels:
                        yolo_label_file.write("{} {} {} {} {}\n".format(*lbl))

    with open(kitti_config.processed_kitti_classes_file_path, "w") as f:
        json.dump(obj=KITTI_CLASSNAME_TO_NUMBER, fp=f)
    print("[1/4] Kitti data preprocessed")


def create_yolo_yaml_file(classes: List[str], kitti_config: KittiTrainConfig):
    yaml_file = "names:\n"
    yaml_file += "\n".join(f"- {c}" for c in classes)
    yaml_file += f"\nnc: {len(classes)}"
    yaml_file += f"\ntrain: train\nval: test"
    with open(kitti_config.processed_yolo_config_file_path, "w") as f:
        f.write(yaml_file)


def prepare_yolo_data(kitti_config: KittiTrainConfig, user_config: UserKittiYoloConfig):
    with open(kitti_config.processed_kitti_classes_file_path, "r") as f:
        classes = json.load(f)

    images = sorted(list(kitti_config.raw_imgs_path.glob("*")))
    labels = sorted(list(kitti_config.processed_kitti_labels_path.glob("*")))
    pairs = list(zip(images, labels))

    train_dir_path = (kitti_config.processed_yolo_train_path).resolve()
    os.makedirs(train_dir_path, exist_ok=True)
    test_dir_path = (kitti_config.processed_yolo_test_path).resolve()
    os.makedirs(test_dir_path, exist_ok=True)

    train_set, test_set = train_test_split(pairs, test_size=user_config.test_size, shuffle=True)

    for image_path, label_path in tqdm(train_set, desc="Preparing YOLO training data"):
        target_image_path = train_dir_path / image_path.name
        target_label_path = train_dir_path / label_path.name
        shutil.copy(image_path, target_image_path)
        shutil.copy(label_path, target_label_path)

    print("[2/4] YOLO training data processed")

    for image_path, label_path in tqdm(test_set, desc="Preparing YOLO test data"):
        target_image_path = test_dir_path / image_path.name
        target_label_path = test_dir_path / label_path.name
        shutil.copy(image_path, target_image_path)
        shutil.copy(label_path, target_label_path)

    print("[3/4] YOLO test data processed")

    create_yolo_yaml_file(classes=classes, kitti_config=kitti_config)
    print("[4/4] YOLO config file processed")


def main():
    kitti_config = get_kitti_train_config(path=KITTI_DETECTION_TRAIN_PATH)
    user_config = read_user_config()

    preprocess_kitti_data(kitti_config=kitti_config, user_config=user_config)
    prepare_yolo_data(kitti_config=kitti_config, user_config=user_config)


if __name__ == "__main__":
    main()
