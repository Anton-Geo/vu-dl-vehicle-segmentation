from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import OpenImagesSegmentationDataset
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


def save_history(history: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def main():
    device = torch.device("cpu")
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
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(in_channels=3, num_classes=4).to(device)

    # I try to get empirically more penalties for background and car classes
    class_weights = torch.tensor([0.5, 1.0, 2.5, 3.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 15
    best_val_loss = float("inf")

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "epoch_time_sec": [],
    }

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_time_sec"].append(epoch_time)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), models_dir / "best_unet.pth")
            print("Saved best model")

        save_history(history, models_dir / "training_history.json")

    print("Training finished")


if __name__ == "__main__":
    main()
