from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

from src.dataset import OpenImagesSegmentationDataset
from src.model import UNet


CLASS_NAMES = ["car", "bus", "truck"]
CLASS_IDS = [1, 2, 3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot confusion matrix for U-Net or DeepLabV3 on the test set."
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["unet", "deeplab"],
        help="Model architecture to load.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to model weights (.pth).",
    )
    parser.add_argument(
        "--test-index",
        default="data/processed/test_index.json",
        help="Path to test_index.json.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Image size used during evaluation.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional filename prefix. Defaults to model type.",
    )
    return parser.parse_args()


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


def build_model(model_type: str, model_path: Path, device: torch.device) -> torch.nn.Module:
    if model_type == "unet":
        model = UNet(in_channels=3, num_classes=4).to(device)
    elif model_type == "deeplab":
        model = build_deeplab_model(num_classes=4).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    all_true = []
    all_pred = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            if model_type == "deeplab":
                logits = model(normalize_batch(images, device))["out"]
            else:
                logits = model(images)

            preds = torch.argmax(logits, dim=1)

            all_true.append(masks.cpu().numpy().reshape(-1))
            all_pred.append(preds.cpu().numpy().reshape(-1))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    # Exclude background
    fg_mask = y_true != 0
    y_true = y_true[fg_mask]
    y_pred = y_pred[fg_mask]

    return y_true, y_pred


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0,
    )
    return cm, cm_norm


def annotate_heatmap(ax, data: np.ndarray, normalized: bool) -> None:
    threshold = data.max() / 2.0 if data.size else 0.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            color = "white" if value > threshold else "black"
            text = f"{value:.2f}" if normalized else f"{int(value)}"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)


def save_confusion_matrix(cm: np.ndarray, output_path: Path, title: str, normalized: bool) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")

    annotate_heatmap(ax, cm, normalized)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = OpenImagesSegmentationDataset(
        index_path=args.test_index,
        image_size=tuple(args.image_size),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args.model_type, model_path, device)
    y_true, y_pred = collect_predictions(model, loader, device, args.model_type)
    cm, cm_norm = compute_confusion(y_true, y_pred)

    prefix = args.prefix or args.model_type

    counts_path = output_dir / f"{prefix}_confusion_matrix_counts.png"
    norm_path = output_dir / f"{prefix}_confusion_matrix_normalized.png"

    save_confusion_matrix(
        cm,
        counts_path,
        f"{args.model_type.upper()} confusion matrix (vehicle classes, counts)",
        normalized=False,
    )
    save_confusion_matrix(
        cm_norm,
        norm_path,
        f"{args.model_type.upper()} confusion matrix (vehicle classes, normalized)",
        normalized=True,
    )

    print(f"Saved: {counts_path}")
    print(f"Saved: {norm_path}")


if __name__ == "__main__":
    main()
