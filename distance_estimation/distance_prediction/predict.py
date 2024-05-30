from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from ultralytics import YOLO

from distance_estimation.depth_prediction.depth_anything.metric_depth.zoedepth.models.depth_model import DepthModel
from distance_estimation.depth_prediction.predict_depth_metric import load_depth_model, process_image
from distance_estimation.detection.predict import Detection, load_yolo_model, predict_detection
from distance_estimation.distance_prediction.helpers import DistanceDetection, draw_dist_detection_bbox
from distance_estimation.distance_prediction.strategies import bbox_depth


class DistancePredictor:

    def __init__(self, detection_model: YOLO, depth_model: DepthModel, strategy: str):
        self.detection_model = detection_model
        self.depth_model = depth_model
        self.strategy = strategy

    def predict(self, image_path: Path) -> List[DistanceDetection]:
        image = Image.open(image_path)

        detections: List[Detection] = predict_detection(model=self.detection_model, model_inp=image_path)
        depth_mask = process_image(model=self.depth_model, image=image)
        depth_array = np.array(depth_mask)

        return [
            DistanceDetection(**asdict(detection), distance=bbox_depth(depth_array, detection.xyxy, self.strategy))
            for detection in detections
        ]

    @classmethod
    def load(cls, depth_model_name: str, depth_model_path: Path, detection_model_path: Path, strategy: str) -> "DistancePredictor":
        yolo_model = load_yolo_model(model_path=detection_model_path)
        depth_model = load_depth_model(model_name=depth_model_name, pretrained_resource=depth_model_path)
        return cls(detection_model=yolo_model, depth_model=depth_model, strategy=strategy)


def main(args):
    predictor = DistancePredictor.load(
        depth_model_name=args.depth_model_name,
        depth_model_path=args.depth_model_path,
        detection_model_path=args.detection_model_path,
        strategy=args.strategy,
    )
    print("Models loaded...")

    detections = predictor.predict(image_path=args.img_path)
    print("Detections performed...")

    print("Detections:", detections)

    if args.out_path:
        img = draw_dist_detection_bbox(image=Image.open(args.img_path), detections=detections)
        img.save(args.out_path)
        print(f"Saved to {args.out_path}")


if __name__ == "__main__":
    parser = ArgumentParser("Detection predictor")
    parser.add_argument("-depmn", "--depth-model-name", type=str, default="zoedepth", help="Name of the model to test")
    parser.add_argument("-depmp", "--depth-model-path", type=str, required=True, help="Depth model .pt file path")
    parser.add_argument("-detmp", "--detection-model-path", type=str, required=True, help="YOLO model .pt file path")
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        default="bbox_median",
        help="Strategy name for concatenating bboxes with depth mask",
        choices=[
            "bbox_mean",
            "bbox_median",
            "bbox_min",
            "bbox_percentile",
            "center_mean",
            "center_median",
            "center_min",
            "center_percentile",
        ],
    )

    parser.add_argument("-ip", "--img-path", type=str, required=True, help=".png file path")
    parser.add_argument("-op", "--out-path", type=str, required=False, help=".png file path")
    args = parser.parse_args()
    main(args)
