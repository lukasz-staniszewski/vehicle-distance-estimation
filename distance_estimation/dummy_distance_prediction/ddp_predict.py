import json
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from PIL import Image
from ultralytics import YOLO

from distance_estimation.detection.predict import Detection, load_yolo_model, predict_detection
from distance_estimation.distance_prediction.helpers import DistanceDetection, draw_dist_detection_bbox
from distance_estimation.dummy_distance_prediction.ddp_prepare import get_focal_length


class DummyDistancePredictor:

    def __init__(self, detection_model: YOLO, height_model: Dict[str, float]):
        self.detection_model = detection_model
        self.height_model = height_model

    def _process_detection(self, detection: Detection, focal_length: float) -> DistanceDetection:
        class_idx = detection.class_idx.item()
        real_height = self.height_model[str(class_idx)]
        pixel_height = (detection.xyxy[3] - detection.xyxy[1]).item()
        distance = focal_length * real_height / pixel_height
        return DistanceDetection(**asdict(detection), distance=distance)

    def predict(self, image: Image.Image, focal_length: float) -> DistanceDetection:
        detections: List[Detection] = predict_detection(model=self.detection_model, model_inp=image)
        distance_detections = [self._process_detection(detection=detection, focal_length=focal_length) for detection in detections]
        return distance_detections

    @classmethod
    def load(cls, height_model_path: Path, detection_model_path: Path) -> "DummyDistancePredictor":
        yolo_model = load_yolo_model(model_path=detection_model_path)
        height_model = json.load(open(height_model_path, "rb"))
        return cls(detection_model=yolo_model, height_model=height_model)


def main(args):
    ddp_model = DummyDistancePredictor.load(height_model_path=args.ddp_model_path, detection_model_path=args.detection_model_path)
    print("Models loaded...")

    image = Image.open(args.img_path)
    focal_length: float = get_focal_length(img_path=args.img_path)

    detections: List[DistanceDetection] = ddp_model.predict(
        image=image, focal_length=focal_length
    )
    print("Detections performed...")

    print("Detections:", detections)
    if args.out_path:
        img = draw_dist_detection_bbox(image=image, detections=detections)
        img.save(args.out_path)
        print(f"Saved to file: {args.out_path}")


if __name__ == "__main__":
    parser = ArgumentParser("Detection predictor")
    parser.add_argument("-detmp", "--detection-model-path", type=str, required=True, help="YOLO model .pt file path")
    parser.add_argument("-ddpmp", "--ddp-model-path", type=str, required=True, help="model.json file path")
    parser.add_argument("-ip", "--img-path", type=str, required=True, help=".png file path")
    parser.add_argument("-op", "--out-path", type=str, required=False, help=".png file path")
    args = parser.parse_args()
    main(args)
