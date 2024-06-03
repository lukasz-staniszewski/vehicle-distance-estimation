from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
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

    def __init__(self, detection_model: YOLO, depth_model: DepthModel, strategy: str, run_multithreaded: bool = False):
        self.detection_model = detection_model
        self.depth_model = depth_model
        self.strategy = strategy
        self.run_multithreaded = run_multithreaded

    def predict(self, image: Image.Image) -> List[DistanceDetection]:
        def _get_detections(model, image):
            return predict_detection(model=model, model_inp=image)

        def _get_depth_mask(model, image):
            return process_image(model=model, image=image)

        if self.run_multithreaded:
            with ThreadPoolExecutor(max_workers=2) as executor:
                detections_future = executor.submit(_get_detections, self.detection_model, image)
                depth_array_future = executor.submit(_get_depth_mask, self.depth_model, image)
                detections: List[Detection] = detections_future.result()
                depth_array: np.ndarray = depth_array_future.result()
        else:
            detections = _get_detections(model=self.detection_model, image=image)
            depth_array = _get_depth_mask(model=self.depth_model, image=image)

        return [
            DistanceDetection(**asdict(detection), distance=bbox_depth(depth_array, detection.xyxy, self.strategy))
            for detection in detections
        ]

    @classmethod
    def load(cls, vit_type: str, depth_model_path: Path, detection_model_path: Path, strategy: str, run_multithreaded: bool) -> "DistancePredictor":
        yolo_model = load_yolo_model(model_path=detection_model_path)
        depth_model = load_depth_model( pretrained_resource=depth_model_path, vit_encoder_type=vit_type)
        return cls(detection_model=yolo_model, depth_model=depth_model, strategy=strategy, run_multithreaded=run_multithreaded)


def main(args):
    predictor = DistancePredictor.load(
        vit_type=args.depth_vit_type,
        depth_model_path=args.depth_model_path,
        detection_model_path=args.detection_model_path,
        strategy=args.strategy,
        run_multithreaded=False
    )
    image = Image.open(args.img_path)
    print("Models loaded...")

    detections = predictor.predict(image=image)
    print("Detections performed...")

    print("Detections:", detections)

    if args.out_path:
        img = draw_dist_detection_bbox(image=Image.open(args.img_path), detections=detections)
        img.save(args.out_path)
        print(f"Saved to {args.out_path}")


if __name__ == "__main__":
    parser = ArgumentParser("Detection predictor")
    parser.add_argument("-depvt", "--depth-vit-type", type=str, choices=["small", "large", "big"], help="Type of ViT model for Depth Estimation")
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
