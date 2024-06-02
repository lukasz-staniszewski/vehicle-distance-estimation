import argparse

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from distance_estimation.depth_prediction.depth_anything.metric_depth.zoedepth.models.builder import build_model
from distance_estimation.depth_prediction.depth_anything.metric_depth.zoedepth.utils.config import get_config

# Global settings
DATASET = "kitti"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_depth_model(pretrained_resource, vit_encoder_type):
    config = get_config("zoedepth", "eval", DATASET)
    config.pretrained_resource = pretrained_resource
    config.vit_encoder_type = vit_encoder_type
    model = build_model(config).to(DEVICE)
    model.eval()
    return model


def process_image(model, image):
    original_width, original_height = image.size
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        pred = model(image_tensor, dataset=DATASET)
        if isinstance(pred, dict):
            pred = pred.get("metric_depth", pred.get("out"))
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        pred_resized = F.interpolate(pred, size=(original_height, original_width), mode="nearest")
        pred_resized = pred_resized[0, 0].cpu().numpy()
        return pred_resized


def save_image(depth_prediction, out_path):
    resized_pred = Image.fromarray(depth_prediction)
    resized_pred = resized_pred.convert("L")
    resized_pred.save(out_path)


def main(args):
    model = load_depth_model(args.pretrained_resource, args.vit_type)

    image = Image.open(args.img_in).convert("RGB")
    output = process_image(model, image)

    save_image(output, args.img_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pretrained_resource",
        type=str,
        default="local::./checkpoints/depth_anything_metric_depth_indoor.pt",
        help="Pretrained resource to use for fetching weights.",
    )
    parser.add_argument("--vit-type", type=str, choices=["small", "big", "large"])
    parser.add_argument("--img-in", type=str)
    parser.add_argument("--img-out", type=str, default="metric_depth.png")

    args = parser.parse_args()
    main(args)
