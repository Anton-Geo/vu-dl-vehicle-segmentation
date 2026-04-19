from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


CLASS_NAMES = {
    0: "background",
    1: "car",
    2: "bus",
    3: "truck",
}

SELECTED_LABELS = [1, 2, 3]


def collect_predictions(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            all_true.append(masks.cpu().numpy().reshape(-1))
            all_pred.append(preds.cpu().numpy().reshape(-1))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return y_true, y_pred


def compute_segmentation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    selected_labels: list[int] | tuple[int, ...] = SELECTED_LABELS,
) -> dict:
    pixel_accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(selected_labels),
        average=None,
        zero_division=0,
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(selected_labels),
        average="macro",
        zero_division=0,
    )

    per_class = {
        CLASS_NAMES[class_id]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, class_id in enumerate(selected_labels)
    }

    return {
        "pixel_accuracy": float(pixel_accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "per_class": per_class,
    }
