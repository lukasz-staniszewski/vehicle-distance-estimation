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
    try:
        font = ImageFont.truetype("arial.ttf", size=15)
    except IOError:
        print("USING DEFAULT")
        font = ImageFont.load_default()

    for detection in detections:
        bbox = detection.xyxy.tolist()
        drawer.rectangle(bbox, outline="red", width=3)
        text_on_bbox = f"{(detection.class_name).upper()}: {detection.distance:.02f} m"

        text_bbox = drawer.textbbox((0, 0), text_on_bbox, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_position = (bbox[0], bbox[1] - text_height - 5)
        background_position = (
            text_position[0],
            text_position[1],
            text_position[0] + text_width + 1,
            text_position[1] + text_height + 4,
        )

        drawer.rectangle(background_position, fill="white")
        drawer.text(text_position, text=text_on_bbox, fill="red", font=font)

    return image
