from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)

from dataset import OpenImagesSegmentationDataset


CLASS_NAMES = {
    0: "background",
    1: "car",
    2: "bus",
    3: "truck",
}


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


def evaluate_model(model, loader, device):
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


def main():
    device = torch.device("cpu")
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
    )

    model = build_model(num_classes=4).to(device)
    model_path = Path("models/deeplabv3_mobilenet_pretrained/best_deeplabv3_mobilenet.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    y_true, y_pred = evaluate_model(model, test_loader, device)

    pixel_accuracy = accuracy_score(y_true, y_pred)

    selected_labels = [1, 2, 3]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=selected_labels,
        average=None,
        zero_division=0,
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=selected_labels,
        average="macro",
        zero_division=0,
    )

    print("\n=== Overall Metrics ===")
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")

    print("\n=== Per-Class Metrics ===")
    for i, class_id in enumerate(selected_labels):
        print(
            f"{CLASS_NAMES[class_id]:>5} | "
            f"Precision: {precision[i]:.4f} | "
            f"Recall: {recall[i]:.4f} | "
            f"F1: {f1[i]:.4f} | "
            f"Support: {support[i]}"
        )

    print("\n=== Macro Average (car, bus, truck) ===")
    print(f"Precision: {precision_macro:.4f}")
    print(f"Recall:    {recall_macro:.4f}")
    print(f"F1:        {f1_macro:.4f}")

    results = {
        "pixel_accuracy": float(pixel_accuracy),
        "per_class": {
            CLASS_NAMES[class_id]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, class_id in enumerate(selected_labels)
        },
        "macro_avg_selected_classes": {
            "precision": float(precision_macro),
            "recall": float(recall_macro),
            "f1": float(f1_macro),
        },
    }

    output_path = Path("models/deeplabv3_mobilenet_pretrained/test_metrics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
