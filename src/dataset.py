from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance
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

        # Add some augmentation
        if self.augment:
            # Horizontal flip
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)

            # # Small rotation
            # if random.random() < 0.3:
            #     angle = random.uniform(-10, 10)
            #     image = image.rotate(angle, resample=Image.BILINEAR)
            #     mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST)
            #
            # # Brightness
            # if random.random() < 0.3:
            #     factor = random.uniform(0.85, 1.15)
            #     image = ImageEnhance.Brightness(image).enhance(factor)
            #
            # # Contrast
            # if random.random() < 0.3:
            #     factor = random.uniform(0.85, 1.15)
            #     image = ImageEnhance.Contrast(image).enhance(factor)

        # Resize image (save proportion with padding)
        image, mask_pil = self._resize_and_pad(image, mask_pil)

        # Convert to tensors
        image_np = np.array(image, dtype=np.float32) / 255.0
        image_np = np.transpose(image_np, (2, 0, 1))

        mask_np = np.array(mask_pil, dtype=np.int64)

        image_tensor = torch.tensor(image_np, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_np, dtype=torch.long)

        return image_tensor, mask_tensor

    def _resize_and_pad(
            self,
            image: Image.Image,
            mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        target_w, target_h = self.image_size
        orig_w, orig_h = image.size

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))

        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        pad_w = target_w - new_w
        pad_h = target_h - new_h

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)
        mask = ImageOps.expand(mask, border=(left, top, right, bottom), fill=0)

        return image, mask
