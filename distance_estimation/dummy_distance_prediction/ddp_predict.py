import json
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from distance_estimation.detection.predict import Detection, load_yolo_model, predict_detection
from distance_estimation.dummy_distance_prediction.ddp_prepare import get_focal_length


@dataclass
class DistanceDetection(Detection):
    distance: torch.Tensor


class DummyDistancePredictor:

    def __init__(self, model: Dict[str, float]):
        self.model = model

    def predict(self, detection: Detection, focal_length: float) -> DistanceDetection:
        real_height = self.model[str(detection.class_idx.item())]
        pixel_height = detection.get_pixel_height()
        distance = focal_length * real_height / pixel_height
        return DistanceDetection(**asdict(detection), distance=distance)

    @classmethod
    def load(cls, model_path: Path) -> "DummyDistancePredictor":
        model = json.load(open(model_path, "rb"))
        return cls(model=model)


def draw_dist_detection_bbox(image: Image.Image, detections: List[DistanceDetection]) -> Image.Image:
    drawer = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=15)
    for detection in detections:
        bbox = detection.xyxy.tolist()
        drawer.rectangle(bbox, outline="red", width=3)
        text_on_bbox = f"{(detection.class_name).upper()}: {detection.distance:.02f} m"
        drawer.text(xy=(bbox[0], bbox[1] - 20), text=text_on_bbox, fill="red", font=font)
    return image


def predict_dummy_distance_prediction(ddp_model: DummyDistancePredictor, yolo_model: YOLO, model_inp: Path) -> List[DistanceDetection]:
    detections: List[Detection] = predict_detection(model=yolo_model, model_inp=model_inp)
    focal_length: float = get_focal_length(img_path=model_inp)
    distance_detections = [ddp_model.predict(detection=detection, focal_length=focal_length) for detection in detections]
    return distance_detections


def main(args):
    yolo_model = load_yolo_model(model_path=args.detection_model_path)
    ddp_model = DummyDistancePredictor.load(model_path=args.ddp_model_path)
    print("Models loaded...")

    image = Image.open(args.img_path)
    detections: List[DistanceDetection] = predict_dummy_distance_prediction(
        ddp_model=ddp_model, yolo_model=yolo_model, model_inp=args.img_path
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
