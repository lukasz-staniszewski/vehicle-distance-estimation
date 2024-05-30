from dataclasses import dataclass
from typing import List

import torch
from PIL import Image, ImageDraw, ImageFont

from distance_estimation.detection.predict import Detection


@dataclass
class DistanceDetection(Detection):
    distance: torch.Tensor


def draw_dist_detection_bbox(image: Image.Image, detections: List[DistanceDetection]) -> Image.Image:
    drawer = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=15)
    for detection in detections:
        bbox = detection.xyxy.tolist()
        drawer.rectangle(bbox, outline="red", width=3)
        text_on_bbox = f"{(detection.class_name).upper()}: {detection.distance:.02f} m"
        drawer.text(xy=(bbox[0], bbox[1] - 20), text=text_on_bbox, fill="red", font=font)
    return image
