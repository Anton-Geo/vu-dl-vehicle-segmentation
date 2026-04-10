from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import OpenImagesSegmentationDataset
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


def main():
    device = torch.device("cpu")

    dataset = OpenImagesSegmentationDataset(
        index_path="data/processed/test_index.json",
        image_size=(256, 256),
        augment=False,
    )

    model = UNet(in_channels=3, num_classes=4).to(device)
    model.load_state_dict(torch.load("models/best_unet.pth", map_location=device))
    model.eval()

    output_dir = Path("models/prediction_examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(len(dataset)), k=min(5, len(dataset)))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image_tensor, true_mask_tensor = dataset[idx]

            image_batch = image_tensor.unsqueeze(0).to(device)
            logits = model(image_batch)
            pred_mask_tensor = torch.argmax(logits, dim=1).squeeze(0).cpu()

            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            true_mask_np = true_mask_tensor.cpu().numpy()
            pred_mask_np = pred_mask_tensor.cpu().numpy()

            true_mask_color = mask_to_color(true_mask_np)
            pred_mask_color = mask_to_color(pred_mask_np)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(image_np)
            axes[0].set_title("Original image")
            axes[0].axis("off")

            axes[1].imshow(true_mask_color)
            axes[1].set_title("Ground truth mask")
            axes[1].axis("off")

            axes[2].imshow(pred_mask_color)
            axes[2].set_title("Predicted mask")
            axes[2].axis("off")

            plt.tight_layout()
            save_path = output_dir / f"example_{i + 1}.png"
            plt.savefig(save_path, dpi=150)
            plt.close(fig)

            print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
