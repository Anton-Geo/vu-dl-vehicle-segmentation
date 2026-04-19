from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import OpenImagesSegmentationDataset
from metrics import compute_segmentation_metrics, collect_predictions
from model import UNet


CLASS_NAMES = {
    0: "background",
    1: "car",
    2: "bus",
    3: "truck",
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = OpenImagesSegmentationDataset(
        index_path="data/processed/test_index.json",
        image_size=(256, 256),
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(in_channels=3, num_classes=4).to(device)
    model_path = Path("models/custom_unet/best_unet.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    y_true, y_pred = collect_predictions(model, test_loader, device)
    results = compute_segmentation_metrics(y_true, y_pred)

    print("\n=== Overall Metrics ===")
    print(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
    print(f"Macro Precision: {results['precision_macro']:.4f}")
    print(f"Macro Recall:    {results['recall_macro']:.4f}")
    print(f"Macro F1:        {results['f1_macro']:.4f}")

    print("\n=== Per-Class Metrics ===")
    for class_name, metrics in results["per_class"].items():
        print(
            f"{class_name:>10} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Support: {metrics['support']}"
        )

    output_path = Path("models/custom_unet/test_metrics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
