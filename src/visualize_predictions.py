from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)

from src.dataset import OpenImagesSegmentationDataset
from src.model import UNet


CLASS_COLORS = {
    0: (0, 0, 0),      # background
    1: (255, 0, 0),    # car
    2: (0, 255, 0),    # bus
    3: (0, 0, 255),    # truck
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_INDEX = PROJECT_ROOT / "data" / "processed" / "test_index.json"
DEFAULT_UNET_PATH = PROJECT_ROOT / "models" / "custom_unet" / "best_unet.pth"
DEFAULT_DEEPLAB_PATH = PROJECT_ROOT / "models" / "deeplabv3_mobilenet_pretrained" / "best_deeplabv3_mobilenet.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "prediction_examples"


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color
    return color_mask


def normalize_batch(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def build_deeplab_model(num_classes: int = 4) -> torch.nn.Module:
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights)

    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = torch.nn.Conv2d(in_channels, num_classes, kernel_size=1)

    if model.aux_classifier is not None:
        aux_in_channels = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = torch.nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)

    return model


def load_models(device: torch.device, unet_path: Path, deeplab_path: Path):
    if not unet_path.exists():
        raise FileNotFoundError(f"U-Net weights not found: {unet_path}")
    if not deeplab_path.exists():
        raise FileNotFoundError(f"DeepLab weights not found: {deeplab_path}")

    unet_model = UNet(in_channels=3, num_classes=4).to(device)
    unet_model.load_state_dict(torch.load(unet_path, map_location=device))
    unet_model.eval()

    deeplab_model = build_deeplab_model(num_classes=4).to(device)
    deeplab_model.load_state_dict(torch.load(deeplab_path, map_location=device))
    deeplab_model.eval()

    return unet_model, deeplab_model


def main():
    parser = argparse.ArgumentParser(description="Visualize predictions for U-Net and DeepLabV3")
    parser.add_argument("--test-index", type=str, default=str(DEFAULT_TEST_INDEX))
    parser.add_argument("--unet-path", type=str, default=str(DEFAULT_UNET_PATH))
    parser.add_argument("--deeplab-path", type=str, default=str(DEFAULT_DEEPLAB_PATH))
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    random.seed(args.seed)

    dataset = OpenImagesSegmentationDataset(
        index_path=args.test_index,
        image_size=tuple(args.image_size),
        augment=False,
    )

    unet_model, deeplab_model = load_models(
        device,
        unet_path=Path(args.unet_path),
        deeplab_path=Path(args.deeplab_path),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_examples = min(args.num_examples, len(dataset))
    indices = random.sample(range(len(dataset)), k=num_examples)

    print(f"Saving {num_examples} examples to: {output_dir}")

    with torch.no_grad():
        for i, idx in enumerate(indices, start=1):
            image_tensor, true_mask_tensor = dataset[idx]
            image_batch = image_tensor.unsqueeze(0).to(device)

            unet_logits = unet_model(image_batch)
            unet_pred_mask_tensor = torch.argmax(unet_logits, dim=1).squeeze(0).cpu()

            deeplab_input = normalize_batch(image_batch, device)
            deeplab_outputs = deeplab_model(deeplab_input)
            deeplab_pred_mask_tensor = torch.argmax(deeplab_outputs["out"], dim=1).squeeze(0).cpu()

            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            image_np = np.clip(image_np, 0.0, 1.0)
            true_mask_np = true_mask_tensor.cpu().numpy()
            unet_pred_mask_np = unet_pred_mask_tensor.numpy()
            deeplab_pred_mask_np = deeplab_pred_mask_tensor.numpy()

            true_mask_color = mask_to_color(true_mask_np)
            unet_pred_mask_color = mask_to_color(unet_pred_mask_np)
            deeplab_pred_mask_color = mask_to_color(deeplab_pred_mask_np)

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            axes[0, 0].imshow(image_np)
            axes[0, 0].set_title("Original image")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(true_mask_color)
            axes[0, 1].set_title("Ground truth mask")
            axes[0, 1].axis("off")

            axes[1, 0].imshow(unet_pred_mask_color)
            axes[1, 0].set_title("Custom U-Net prediction")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(deeplab_pred_mask_color)
            axes[1, 1].set_title("DeepLabV3 prediction")
            axes[1, 1].axis("off")

            plt.tight_layout()
            save_path = output_dir / f"comparison_{i:02d}_idx_{idx}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
