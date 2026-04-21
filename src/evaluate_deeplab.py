from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)

from src.dataset import OpenImagesSegmentationDataset
from src.metrics import compute_segmentation_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeepLabV3 model")

    parser.add_argument("--test-index", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256])

    return parser.parse_args()


def normalize_batch(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def build_model(num_classes: int = 4) -> torch.nn.Module:
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights)

    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = torch.nn.Conv2d(in_channels, num_classes, kernel_size=1)

    if model.aux_classifier is not None:
        aux_in_channels = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = torch.nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)

    return model


def collect_predictions_deeplab(model, loader, device):
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            images = normalize_batch(images, device)
            outputs = model(images)
            preds = torch.argmax(outputs["out"], dim=1)

            all_true.append(masks.cpu().numpy().reshape(-1))
            all_pred.append(preds.cpu().numpy().reshape(-1))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    return y_true, y_pred


def save_json(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = OpenImagesSegmentationDataset(
        index_path=args.test_index,
        image_size=tuple(args.image_size),
        augment=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(num_classes=4).to(device)
    model_path = Path(args.model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))

    y_true, y_pred = collect_predictions_deeplab(model, test_loader, device)
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

    output_path = output_dir / "test_metrics.json"
    save_json(results, output_path)

    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
