from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OpenImagesSegmentationDataset(Dataset):
    def __init__(
        self,
        index_path: str | Path,
        image_size: tuple[int, int] = (256, 256),
        augment: bool = False,
    ) -> None:
        self.index_path = Path(index_path)
        self.image_size = image_size
        self.augment = augment

        with open(self.index_path, "r", encoding="utf-8") as f:
            self.records: list[dict[str, Any]] = json.load(f)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]

        image = Image.open(record["image_path"]).convert("RGB")
        orig_width, orig_height = image.size

        # Final mask: 0=background, 1=car, 2=bus, 3=truck
        final_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)

        for obj in record["objects"]:
            class_id = int(obj["class_id"])

            mask_img = Image.open(obj["mask_path"]).convert("L")
            obj_mask = np.array(mask_img)

            # We want obj_mask.shape == (orig_height, orig_width)
            if obj_mask.shape != (orig_height, orig_width):
                # Common issue: width/height swapped
                if obj_mask.shape == (orig_width, orig_height):
                    obj_mask = obj_mask.T
                else:
                    # Fallback: resize to match image size
                    mask_img = mask_img.resize((orig_width, orig_height), Image.NEAREST)
                    obj_mask = np.array(mask_img)

            obj_mask = obj_mask > 0
            final_mask[obj_mask] = class_id

        # Resize mask with nearest-neighbor to preserve class IDs
        mask_pil = Image.fromarray(final_mask)

        # Add some augmentation: horizontal flip with prob 0.5
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)

        # Resize image
        image = image.resize(self.image_size, Image.BILINEAR)
        mask_pil = mask_pil.resize(self.image_size, Image.NEAREST)

        # Convert to tensors
        image_np = np.array(image, dtype=np.float32) / 255.0
        image_np = np.transpose(image_np, (2, 0, 1))

        mask_np = np.array(mask_pil, dtype=np.int64)

        image_tensor = torch.tensor(image_np, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_np, dtype=torch.long)

        return image_tensor, mask_tensor
