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


def load_depth_model(model_name, pretrained_resource):
    config = get_config(model_name, "eval", DATASET)
    config.pretrained_resource = pretrained_resource
    model = build_model(config).to(DEVICE)
    model.eval()
    return model


def process_image(model, image):
    original_width, original_height = image.size
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(DEVICE)

    pred = model(image_tensor, dataset=DATASET)
    if isinstance(pred, dict):
        pred = pred.get("metric_depth", pred.get("out"))
    elif isinstance(pred, (list, tuple)):
        pred = pred[-1]
    pred = pred.squeeze().detach().cpu()

    pred = pred.unsqueeze(0).unsqueeze(0)
    pred_resized = F.interpolate(pred, size=(original_height, original_width), mode="nearest")
    pred_resized = pred_resized[0, 0].numpy()
    return pred_resized


def save_image(depth_prediction, out_path):
    resized_pred = Image.fromarray(depth_prediction)
    resized_pred = resized_pred.convert("L")
    resized_pred.save(out_path)


def main(model_name, pretrained_resource, inp_path, out_path):
    model = load_depth_model(model_name, pretrained_resource)

    image = Image.open(inp_path).convert("RGB")
    output = process_image(model, image)

    save_image(output, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="zoedepth", help="Name of the model to test")
    parser.add_argument(
        "-p",
        "--pretrained_resource",
        type=str,
        default="local::./checkpoints/depth_anything_metric_depth_indoor.pt",
        help="Pretrained resource to use for fetching weights.",
    )
    parser.add_argument("--img-in", type=str)
    parser.add_argument("--img-out", type=str, default="metric_depth.png")

    args = parser.parse_args()
    main(args.model, args.pretrained_resource, args.img_in, args.img_out)
