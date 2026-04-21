from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)

from src.dataset import OpenImagesSegmentationDataset
from src.losses import ComboLoss
from src.metrics import collect_predictions, compute_segmentation_metrics


NUM_CLASSES = 4  # background, car, bus, truck


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLabV3 for segmentation")

    parser.add_argument("--train-index", type=str, required=True)
    parser.add_argument("--val-index", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=5e-4)

    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256])

    parser.add_argument("--freeze-backbone-epochs", type=int, default=8)
    parser.add_argument("--scheduler-factor", type=float, default=0.6)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-threshold", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)

    return parser.parse_args()


def normalize_batch(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def build_model(num_classes: int = NUM_CLASSES, freeze_backbone: bool = True) -> nn.Module:
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights)

    if not isinstance(model.classifier[-1], nn.Conv2d):
        raise TypeError("Unexpected DeepLab classifier structure")
    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    if model.aux_classifier is not None:
        if not isinstance(model.aux_classifier[-1], nn.Conv2d):
            raise TypeError("Unexpected DeepLab aux classifier structure")
        aux_in_channels = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = True


def compute_loss(outputs: dict[str, torch.Tensor], masks: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    loss = criterion(outputs["out"], masks)
    if "aux" in outputs and outputs["aux"] is not None:
        loss = loss + 0.4 * criterion(outputs["aux"], masks)
    return loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        images = normalize_batch(images, device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = compute_loss(outputs, masks, criterion)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            images = normalize_batch(images, device)
            outputs = model(images)
            loss = compute_loss(outputs, masks, criterion)

            running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


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

    import numpy as np
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return y_true, y_pred


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def make_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable_params, lr=lr)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = OpenImagesSegmentationDataset(
        index_path=args.train_index,
        image_size=tuple(args.image_size),
        augment=True,
    )
    val_dataset = OpenImagesSegmentationDataset(
        index_path=args.val_index,
        image_size=tuple(args.image_size),
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(num_classes=NUM_CLASSES, freeze_backbone=True).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024**2

    print("=" * 50)
    print("Training Setup")
    print(f"Device: {device}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Input size: (3, {args.image_size[0]}, {args.image_size[1]})")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Approx. model size: {model_size_mb:.2f} MB")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Freeze backbone epochs: {args.freeze_backbone_epochs}")
    print(f"Patience: {args.patience}")
    print(f"Min delta: {args.min_delta}")
    print("=" * 50)

    alpha = torch.tensor([0.05, 0.20, 1.10, 1.15], dtype=torch.float32).to(device)
    gamma = 2.0
    criterion = ComboLoss(
        alpha=alpha,
        gamma=gamma,
        focal_weight=0.7,
        dice_weight=0.3,
    )

    optimizer = make_optimizer(model, args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        threshold=args.scheduler_threshold,
        min_lr=args.min_lr,
    )

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    backbone_unfrozen = False

    config = {
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "optimizer": "Adam",
        "loss": "ComboLoss",
        "gamma": gamma,
        "alpha": alpha.tolist(),
        "focal_weight": 0.7,
        "dice_weight": 0.3,
        "selection_metric": "val_f1_macro",
        "patience": args.patience,
        "min_delta": args.min_delta,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_threshold": args.scheduler_threshold,
        "min_lr": args.min_lr,
    }
    save_json(config, output_dir / "train_config.json")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision_macro": [],
        "val_recall_macro": [],
        "val_f1_macro": [],
        "val_car_precision": [],
        "val_car_recall": [],
        "val_car_f1": [],
        "val_bus_precision": [],
        "val_bus_recall": [],
        "val_bus_f1": [],
        "val_truck_precision": [],
        "val_truck_recall": [],
        "val_truck_f1": [],
        "epoch_time_sec": [],
        "lr": [],
        "best_epoch": None,
        "best_val_f1_macro": None,
    }

    for epoch in range(args.epochs):
        if (not backbone_unfrozen) and (epoch == args.freeze_backbone_epochs):
            print("Unfreezing DeepLab backbone")
            unfreeze_backbone(model)
            optimizer = make_optimizer(model, args.lr * 0.1)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=args.scheduler_factor,
                patience=args.scheduler_patience,
                threshold=args.scheduler_threshold,
                min_lr=args.min_lr,
            )
            backbone_unfrozen = True

        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        y_true, y_pred = collect_predictions_deeplab(model, val_loader, device)
        val_metrics = compute_segmentation_metrics(y_true, y_pred)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        car_metrics = val_metrics["per_class"]["car"]
        bus_metrics = val_metrics["per_class"]["bus"]
        truck_metrics = val_metrics["per_class"]["truck"]

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["pixel_accuracy"])
        history["val_precision_macro"].append(val_metrics["precision_macro"])
        history["val_recall_macro"].append(val_metrics["recall_macro"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["epoch_time_sec"].append(epoch_time)
        history["lr"].append(current_lr)

        history["val_car_precision"].append(car_metrics["precision"])
        history["val_car_recall"].append(car_metrics["recall"])
        history["val_car_f1"].append(car_metrics["f1"])

        history["val_bus_precision"].append(bus_metrics["precision"])
        history["val_bus_recall"].append(bus_metrics["recall"])
        history["val_bus_f1"].append(bus_metrics["f1"])

        history["val_truck_precision"].append(truck_metrics["precision"])
        history["val_truck_recall"].append(truck_metrics["recall"])
        history["val_truck_f1"].append(truck_metrics["f1"])

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['pixel_accuracy']:.4f} | "
            f"Val Precision: {val_metrics['precision_macro']:.4f} | "
            f"Val Recall: {val_metrics['recall_macro']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f} | "
            f"Car F1: {car_metrics['f1']:.4f} | "
            f"Bus F1: {bus_metrics['f1']:.4f} | "
            f"Truck F1: {truck_metrics['f1']:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        current_val_f1 = val_metrics["f1_macro"]

        if current_val_f1 > best_val_f1 + args.min_delta:
            best_val_f1 = current_val_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            history["best_epoch"] = best_epoch
            history["best_val_f1_macro"] = best_val_f1

            torch.save(model.state_dict(), output_dir / "best_deeplabv3_mobilenet.pth")
            save_json(val_metrics, output_dir / "best_val_metrics.json")
            print("Saved best DeepLab model by macro F1")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

        save_json(history, output_dir / "training_history.json")

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print("Training finished")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
