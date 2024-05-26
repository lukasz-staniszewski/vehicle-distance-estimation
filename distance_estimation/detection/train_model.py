from pathlib import Path

import onnx
from ultralytics import YOLO

from distance_estimation.detection.utils import UserKittiYoloConfig, read_user_config


def get_model(user_config: UserKittiYoloConfig) -> YOLO:
    return YOLO((user_config.experiment_path / "model" / "yolov8n.pt").resolve())


def export_model(yolo_model: YOLO):
    out_model_path = yolo_model.export(format="onnx", dynamic=True)
    onnx_model = onnx.load(out_model_path)
    onnx.checker.check_model(onnx_model)
    print(f"Model saved to path: {out_model_path}")


def train_model(user_config: UserKittiYoloConfig):
    model = get_model(user_config=user_config)
    print("Starting training...")
    model.train(
        data=Path("data/detection/processed_yolo/kitti.yaml").resolve(),
        epochs=user_config.n_epochs,
        patience=user_config.patience,
        mixup=0.1,
        project=(user_config.experiment_path / "yolov8-kitti-detection").resolve(),
        device=user_config.device,
    )
    print("Starting validation...")
    model.val()

    print("Exporting...")
    export_model(yolo_model=model)


def main():
    user_config = read_user_config()
    train_model(user_config=user_config)


if __name__ == "__main__":
    main()
