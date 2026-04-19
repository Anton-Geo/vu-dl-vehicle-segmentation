from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import OpenImagesSegmentationDataset
from losses import FocalLoss
from metrics import collect_predictions, compute_segmentation_metrics
from model import UNet


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = OpenImagesSegmentationDataset(
        index_path="data/processed/train_index.json",
        image_size=(256, 256),
        augment=True,
    )
    val_dataset = OpenImagesSegmentationDataset(
        index_path="data/processed/val_index.json",
        image_size=(256, 256),
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(in_channels=3, num_classes=4).to(device)

    # Global pixel distribution
    # background: 3040482227 pixels (73.7 %)
    # car: 802699729 pixels (19.4 %)
    # bus: 145753989 pixels (3.5 %)
    # truck: 136332028 pixels (3.3 %)

    alpha = torch.tensor([0.05, 0.20, 1.10, 1.15], dtype=torch.float32)
    gamma = 2.0
    criterion = FocalLoss(alpha=alpha, gamma=gamma)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20
    best_val_f1 = -1.0
    best_epoch = -1

    models_dir = Path("models/custom_unet")
    models_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "image_size": [256, 256],
        "batch_size": 4,
        "num_epochs": num_epochs,
        "learning_rate": 1e-3,
        "optimizer": "Adam",
        "loss": "FocalLoss",
        "gamma": gamma,
        "alpha": alpha.tolist(),
        "selection_metric": "val_f1_macro",
    }
    save_json(config, models_dir / "train_config.json")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision_macro": [],
        "val_recall_macro": [],
        "val_f1_macro": [],
        "epoch_time_sec": [],
        "best_epoch": None,
        "best_val_f1_macro": None,
    }

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        y_true, y_pred = collect_predictions(model, val_loader, device)
        val_metrics = compute_segmentation_metrics(y_true, y_pred)

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["pixel_accuracy"])
        history["val_precision_macro"].append(val_metrics["precision_macro"])
        history["val_recall_macro"].append(val_metrics["recall_macro"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["epoch_time_sec"].append(epoch_time)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['pixel_accuracy']:.4f} | "
            f"Val Precision: {val_metrics['precision_macro']:.4f} | "
            f"Val Recall: {val_metrics['recall_macro']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch + 1
            history["best_epoch"] = best_epoch
            history["best_val_f1_macro"] = best_val_f1

            torch.save(model.state_dict(), models_dir / "best_unet.pth")
            save_json(val_metrics, models_dir / "best_val_metrics.json")
            print("Saved best model by macro F1")

        save_json(history, models_dir / "training_history.json")

    print("Training finished")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
