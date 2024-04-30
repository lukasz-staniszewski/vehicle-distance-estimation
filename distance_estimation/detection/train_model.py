from ultralytics import YOLO
from pathlib import Path
from distance_estimation.detection.utils import UserKittiYoloConfig, read_user_config


def get_model(user_config: UserKittiYoloConfig) -> YOLO:
    return YOLO((user_config.experiment_path / "model" / "yolov8n.pt").resolve())


def train_model(user_config: UserKittiYoloConfig):
    model = get_model(user_config=user_config)
    print("~~TRAINING~~")
    model.train(
        data=Path("data/detection/processed_yolo/kitti.yaml").resolve(),
        epochs=user_config.n_epochs,
        patience=user_config.patience,
        mixup=0.1,
        project=(user_config.experiment_path / "yolov8-kitti-detection"),
        device=user_config.device,
    )
    print("~~VALIDATION~~")
    model.val()
    out_model_path = model.export(format="onnx")
    print(f"Model saved to path: {out_model_path}")


def main():
    user_config = read_user_config()
    train_model(user_config=user_config)


if __name__ == "__main__":
    main()
