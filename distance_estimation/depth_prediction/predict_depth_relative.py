import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from distance_estimation.depth_prediction.depth_anything.depth_anything.dpt import DepthAnything
from distance_estimation.depth_prediction.depth_anything.depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch_tf = Compose(
    [
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)


def load_model():
    return DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14").to(DEVICE).eval()


def path2tensor(path):
    raw_image = cv2.imread(path)
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

    h, w = image.shape[:2]

    image = torch_tf({"image": image})["image"]
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    return image, h, w


def predict(model, image, w, h):
    with torch.no_grad():
        depth = model(image)

    depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[0, 0]
    return depth.cpu()


def save_predict(depth, in_filename, outdir):
    os.makedirs(outdir, exist_ok=True)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.numpy().astype(np.uint8)
    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

    filename = os.path.basename(in_filename)
    cv2.imwrite(os.path.join(outdir, filename[: filename.rfind(".")] + "_depth.png"), depth)


def main(args):
    depth_anything = load_model()
    image, h, w = path2tensor(args.img_path)
    depth = predict(depth_anything, image, w, h)
    save_predict(depth, args.img_path, args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path", type=str)
    parser.add_argument("--outdir", type=str, default="./vis_depth")

    args = parser.parse_args()
    main(args)
