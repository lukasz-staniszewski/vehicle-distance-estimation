from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T
from ultralytics import YOLO

from distance_estimation.detection.constants import KITTI_CLASS_NAMES


@dataclass
class Detection:
    xyxy: torch.Tensor
    class_idx: torch.Tensor
    class_name: str
    # img_size: Tuple[int, int]

    # def resize(self, new_size: Tuple[int, int]) -> None:
    #     kx = new_size[0] / self.img_size[0]
    #     ky = new_size[1] / self.img_size[1]
    #     self.xyxy = self.xyxy * torch.Tensor([kx, ky, kx, ky])
    #     self.img_size = new_size


def load_yolo_model(model_path: Path) -> YOLO:
    return YOLO(model=model_path, task="detect")


def predict(model: YOLO, model_inp: Path) -> List[Detection]:
    yolo_out = model(model_inp)

    dets = []
    img_detection = yolo_out[0].cpu()

    n_bboxes = img_detection.boxes.cls.shape[0]
    for bbox_idx in range(n_bboxes):
        cls_idx = img_detection.boxes.cls[bbox_idx].int()
        dets.append(Detection(xyxy=img_detection.boxes.xyxy[bbox_idx], class_idx=cls_idx, class_name=KITTI_CLASS_NAMES[cls_idx.item()]))

    return dets


def draw_bbox(image: Image.Image, detections: List[Detection]) -> Image.Image:
    drawer = ImageDraw.Draw(image)
    for detection in detections:
        bbox = detection.xyxy.tolist()
        drawer.rectangle(bbox, outline="red", width=3)
        drawer.text(xy=(bbox[0], bbox[1] - 10), text=detection.class_name, fill="red", font=ImageFont.load_default())
    return image


def main(args):
    yolo_model = load_yolo_model(model_path=args.model_path)
    print("Model loaded...")
    image = Image.open(args.img_path)
    print("Input loaded...")
    detections: List[Detection] = predict(model=yolo_model, model_inp=args.img_path)
    print("Detections performed...")
    print("Detections:", detections)
    if args.out_path:
        img = draw_bbox(image=image, detections=detections)
        img.save(args.out_path)
        print(f"Saved to file: {args.out_path}")


if __name__ == "__main__":
    parser = ArgumentParser("Detection predictor")
    parser.add_argument("-mp", "--model-path", type=str, required=True, help=".pt file path")
    parser.add_argument("-ip", "--img-path", type=str, required=True, help=".png file path")
    parser.add_argument("-op", "--out-path", type=str, required=False, help=".png file path")
    args = parser.parse_args()
    main(args)
