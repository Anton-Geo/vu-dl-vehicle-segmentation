from __future__ import annotations

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

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


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

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")

    train_dataset = OpenImagesSegmentationDataset(
        index_path="data/processed/train_index.json",
        image_size=(256, 256),
    )

    val_dataset = OpenImagesSegmentationDataset(
        index_path="data/processed/val_index.json",
        image_size=(256, 256),
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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 2
    best_val_loss = float("inf")

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), models_dir / "best_unet.pth")
            print("Saved best model")

    print("Training finished")


if __name__ == "__main__":
    main()
