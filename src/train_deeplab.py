from __future__ import annotations

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


NUM_CLASSES = 4  # background, car, bus, truck


def normalize_batch(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def build_model(num_classes: int = NUM_CLASSES, freeze_backbone: bool = True) -> nn.Module:
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights)

    # Replace main classifier head
    if not isinstance(model.classifier[-1], nn.Conv2d):
        raise TypeError("Unexpected DeepLab classifier structure")
    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    # Replace auxiliary classifier head if present
    if model.aux_classifier is not None:
        if not isinstance(model.aux_classifier[-1], nn.Conv2d):
            raise TypeError("Unexpected DeepLab aux classifier structure")
        aux_in_channels = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model


def compute_loss(
    outputs: dict[str, torch.Tensor],
    masks: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
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

    model = build_model(num_classes=NUM_CLASSES, freeze_backbone=True).to(device)

    class_weights = torch.tensor([0.5, 1.0, 2.5, 3.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

    num_epochs = 16
    best_val_loss = float("inf")
    best_epoch = -1

    output_dir = Path("models/deeplabv3_mobilenet_pretrained")
    output_dir.mkdir(parents=True, exist_ok=True)

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
            best_epoch = epoch + 1

            torch.save(model.state_dict(), output_dir / "best_deeplabv3_mobilenet.pth")
            print("Saved best DeepLab model")

        save_history(history, output_dir / "training_history.json")

    print("Training finished")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
