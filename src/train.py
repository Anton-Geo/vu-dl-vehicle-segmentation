from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import OpenImagesSegmentationDataset
from src.losses import ComboLoss
from src.metrics import collect_predictions, compute_segmentation_metrics
from src.model import UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net for segmentation")

    parser.add_argument("--train-index", type=str, required=True)
    parser.add_argument("--val-index", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-4)

    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256])

    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

            logits = model(images)
            loss = criterion(logits, masks)
            running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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

    model = UNet(in_channels=3, num_classes=4).to(device)

    alpha = torch.tensor([0.05, 0.20, 1.10, 1.15], dtype=torch.float32).to(device)
    gamma = 2.0
    criterion = ComboLoss(
        alpha=alpha,
        gamma=gamma,
        focal_weight=0.7,
        dice_weight=0.3,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=1e-3,
        min_lr=1e-6,
    )

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

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
    }
    save_json(config, output_dir / "train_config.json")

    history = {
        "train_loss": [],
        "val_loss": [],

        "val_car_precision": [],
        "val_car_recall": [],
        "val_car_f1": [],

        "val_bus_precision": [],
        "val_bus_recall": [],
        "val_bus_f1": [],

        "val_truck_precision": [],
        "val_truck_recall": [],
        "val_truck_f1": [],

        "val_accuracy": [],
        "val_precision_macro": [],
        "val_recall_macro": [],
        "val_f1_macro": [],

        "epoch_time_sec": [],
        "best_epoch": None,
        "best_val_f1_macro": None,
    }

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        y_true, y_pred = collect_predictions(model, val_loader, device)
        val_metrics = compute_segmentation_metrics(y_true, y_pred)

        car_metrics = val_metrics["per_class"]["car"]
        bus_metrics = val_metrics["per_class"]["bus"]
        truck_metrics = val_metrics["per_class"]["truck"]

        epoch_time = time.time() - epoch_start

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["pixel_accuracy"])
        history["val_precision_macro"].append(val_metrics["precision_macro"])
        history["val_recall_macro"].append(val_metrics["recall_macro"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["epoch_time_sec"].append(epoch_time)

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

            torch.save(model.state_dict(), output_dir / "best_unet.pth")
            save_json(val_metrics, output_dir / "best_val_metrics.json")
            print("Saved best model by macro F1")
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
