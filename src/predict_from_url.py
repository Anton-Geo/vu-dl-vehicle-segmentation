from __future__ import annotations

import argparse
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

from model import UNet


CLASS_COLORS = {
    0: (0, 0, 0),        # Black for background
    1: (255, 0, 0),      # Red for cars
    2: (0, 255, 0),      # green for buses
    3: (0, 0, 255),      # blue for trucks
}


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color

    return color_mask


def download_image_from_url(url: str) -> Image.Image:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    return Image.open(BytesIO(response.content)).convert("RGB")


def preprocess(image: Image.Image, size=(256, 256)) -> torch.Tensor:
    image = image.resize(size, Image.BILINEAR)
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))  # HWC -> CHW
    return torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)


def normalize_batch(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def build_deeplab(num_classes=4) -> torch.nn.Module:
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights)

    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = torch.nn.Conv2d(in_channels, num_classes, kernel_size=1)

    if model.aux_classifier is not None:
        aux_in_channels = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = torch.nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)

    return model


def load_models(device):
    # my custom U-Net
    unet = UNet(in_channels=3, num_classes=4).to(device)
    unet.load_state_dict(torch.load("models/custom_unet/best_unet.pth", map_location=device))
    unet.eval()

    # DeepLab
    deeplab = build_deeplab().to(device)
    deeplab.load_state_dict(
        torch.load("models/deeplabv3_mobilenet_pretrained/best_deeplabv3_mobilenet.pth",
                   map_location=device)
    )
    deeplab.eval()

    return unet, deeplab


def predict(unet, deeplab, image, device):
    x = preprocess(image).to(device)

    with torch.no_grad():
        # U-Net
        unet_logits = unet(x)
        unet_mask = torch.argmax(unet_logits, dim=1).squeeze().cpu().numpy()

        # DeepLab
        x_norm = normalize_batch(x, device)
        deeplab_out = deeplab(x_norm)["out"]
        deeplab_mask = torch.argmax(deeplab_out, dim=1).squeeze().cpu().numpy()

    return unet_mask, deeplab_mask


def visualize(image, unet_mask, deeplab_mask):
    image_np = np.array(image.resize((256, 256)))

    unet_color = mask_to_color(unet_mask)
    deeplab_color = mask_to_color(deeplab_mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].imshow(image_np)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(unet_color)
    axes[1].set_title("U-Net prediction")
    axes[1].axis("off")

    axes[2].imshow(deeplab_color)
    axes[2].set_title("DeepLab prediction")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Predict segmentation from image URL")
    parser.add_argument("url", type=str, help="Image URL")

    args = parser.parse_args()

    device = torch.device("cpu")

    print("Downloading image...")
    image = download_image_from_url(args.url)

    print("Loading models...")
    unet, deeplab = load_models(device)

    print("Running prediction...")
    unet_mask, deeplab_mask = predict(unet, deeplab, image, device)

    print("Visualizing...")
    visualize(image, unet_mask, deeplab_mask)


if __name__ == "__main__":
    main()
