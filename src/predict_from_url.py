from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image, ImageOps
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

from src.model import UNet


CLASS_COLORS = {
    0: (0, 0, 0),      # background
    1: (255, 0, 0),    # car
    2: (0, 255, 0),    # bus
    3: (0, 0, 255),    # truck
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNET_PATH = PROJECT_ROOT / "models" / "custom_unet" / "best_unet.pth"
DEEPLAB_PATH = PROJECT_ROOT / "models" / "deeplabv3_mobilenet_pretrained" / "best_deeplabv3_mobilenet.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "prediction_examples_url"


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color
    return color_mask


def download_image_from_url(url: str) -> Image.Image:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def resize_and_pad(image: Image.Image, size: tuple[int, int] = (256, 256)) -> Image.Image:
    target_w, target_h = size
    orig_w, orig_h = image.size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    image = image.resize((new_w, new_h), Image.BILINEAR)

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    return ImageOps.expand(image, border=(left, top, right, bottom), fill=0)


def preprocess(image: Image.Image, size: tuple[int, int] = (256, 256)) -> torch.Tensor:
    image = resize_and_pad(image, size)
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))
    return torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)


def normalize_batch(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def build_deeplab(num_classes: int = 4) -> torch.nn.Module:
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights)

    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = torch.nn.Conv2d(in_channels, num_classes, kernel_size=1)

    if model.aux_classifier is not None:
        aux_in_channels = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = torch.nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)

    return model


def load_models(device: torch.device):
    if not UNET_PATH.exists():
        raise FileNotFoundError(f"U-Net weights not found: {UNET_PATH}")
    if not DEEPLAB_PATH.exists():
        raise FileNotFoundError(f"DeepLab weights not found: {DEEPLAB_PATH}")

    unet = UNet(in_channels=3, num_classes=4).to(device)
    unet.load_state_dict(torch.load(UNET_PATH, map_location=device))
    unet.eval()

    deeplab = build_deeplab().to(device)
    deeplab.load_state_dict(torch.load(DEEPLAB_PATH, map_location=device))
    deeplab.eval()

    return unet, deeplab


def predict(unet, deeplab, image: Image.Image, device: torch.device, image_size: tuple[int, int]):
    x = preprocess(image, size=image_size).to(device)

    with torch.no_grad():
        unet_logits = unet(x)
        unet_mask = torch.argmax(unet_logits, dim=1).squeeze(0).cpu().numpy()

        x_norm = normalize_batch(x, device)
        deeplab_out = deeplab(x_norm)["out"]
        deeplab_mask = torch.argmax(deeplab_out, dim=1).squeeze(0).cpu().numpy()

    return x.squeeze(0).cpu(), unet_mask, deeplab_mask


def visualize_and_save(
    processed_image_tensor: torch.Tensor,
    unet_mask: np.ndarray,
    deeplab_mask: np.ndarray,
    save_path: Path | None,
    show: bool,
) -> None:
    image_np = processed_image_tensor.permute(1, 2, 0).numpy()
    image_np = np.clip(image_np, 0.0, 1.0)

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

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Predict segmentation from image URL")
    parser.add_argument("url", type=str, help="Image URL")
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256], help="Model input size")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "url_prediction_1.png"),
        help="Path to save visualization",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open matplotlib window")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Downloading image...")
    image = download_image_from_url(args.url)

    print("Loading models...")
    unet, deeplab = load_models(device)

    print("Running prediction...")
    processed_image_tensor, unet_mask, deeplab_mask = predict(
        unet,
        deeplab,
        image,
        device,
        image_size=tuple(args.image_size),
    )

    print("Visualizing...")
    visualize_and_save(
        processed_image_tensor,
        unet_mask,
        deeplab_mask,
        save_path=Path(args.output),
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
