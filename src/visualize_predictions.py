from __future__ import annotations

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


def main():
    device = torch.device("cpu")

    dataset = OpenImagesSegmentationDataset(
        index_path="data/processed/test_index.json",
        image_size=(256, 256),
        augment=False,
    )

    # My custom U-Net
    unet_model = UNet(in_channels=3, num_classes=4).to(device)
    unet_model.load_state_dict(torch.load(
        "models/custom_unet/best_unet.pth",
        map_location=device)
    )
    unet_model.eval()

    # Pretrained DeepLabV3 with fine-tuning
    deeplab_model = build_deeplab_model(num_classes=4).to(device)
    deeplab_model.load_state_dict(
        torch.load(
            "models/deeplabv3_mobilenet_pretrained/best_deeplabv3_mobilenet.pth",
            map_location=device,
        )
    )
    deeplab_model.eval()

    output_dir = Path("models/prediction_examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(len(dataset)), k=min(5, len(dataset)))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image_tensor, true_mask_tensor = dataset[idx]

            image_batch = image_tensor.unsqueeze(0).to(device)

            # U-Net prediction
            unet_logits = unet_model(image_batch)
            unet_pred_mask_tensor = torch.argmax(unet_logits, dim=1).squeeze(0).cpu()

            # DeepLab prediction
            deeplab_input = normalize_batch(image_batch, device)
            deeplab_outputs = deeplab_model(deeplab_input)
            deeplab_pred_mask_tensor = torch.argmax(deeplab_outputs["out"], dim=1).squeeze(0).cpu()

            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            true_mask_np = true_mask_tensor.cpu().numpy()
            unet_pred_mask_np = unet_pred_mask_tensor.cpu().numpy()
            deeplab_pred_mask_np = deeplab_pred_mask_tensor.cpu().numpy()

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
            axes[1, 1].set_title("Pretrained DeepLabV3 prediction")
            axes[1, 1].axis("off")

            plt.tight_layout()
            save_path = output_dir / f"comparison_{i + 1}.png"
            plt.savefig(save_path, dpi=150)
            plt.close(fig)

            print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
