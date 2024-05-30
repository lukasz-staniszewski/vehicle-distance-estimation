from dataclasses import dataclass
from typing import List

import torch
from PIL import Image, ImageDraw, ImageFont

from distance_estimation.detection.predict import Detection


@dataclass
class DistanceDetection(Detection):
    distance: torch.Tensor

    def __hash__(self):
        return hash((tuple(self.xyxy.tolist()), self.class_idx.item(), self.class_name, self.distance.item()))

    def __eq__(self, other):
        return (self.xyxy.tolist(), self.class_idx.item(), self.class_name, self.distance.item()) == (
            other.xyxy.tolist(),
            other.class_idx.item(),
            other.class_name,
            other.distance.item(),
        )


def draw_dist_detection_bbox(image: Image.Image, detections: List[DistanceDetection]) -> Image.Image:
    drawer = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=15)
    for detection in detections:
        bbox = detection.xyxy.tolist()
        drawer.rectangle(bbox, outline="red", width=3)
        text_on_bbox = f"{(detection.class_name).upper()}: {detection.distance:.02f} m"
        drawer.text(xy=(bbox[0], bbox[1] - 20), text=text_on_bbox, fill="red", font=font)
    return image
