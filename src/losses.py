from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor | list[float] | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        if alpha is None:
            self.alpha = None
        else:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha_tensor)

        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, C, H, W]
        targets: [B, H, W]
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        targets_unsq = targets.unsqueeze(1)

        log_pt = torch.gather(log_probs, dim=1, index=targets_unsq).squeeze(1)
        pt = torch.gather(probs, dim=1, index=targets_unsq).squeeze(1)

        focal_term = (1.0 - pt) ** self.gamma
        loss = -focal_term * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            loss = loss[valid_mask]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1e-6,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, C, H, W]
        targets: [B, H, W]
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        targets_for_one_hot = targets.clone()

        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index).unsqueeze(1)  # [B,1,H,W]
            targets_for_one_hot = targets_for_one_hot.clone()
            targets_for_one_hot[targets == self.ignore_index] = 0
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool).unsqueeze(1)

        targets_one_hot = F.one_hot(targets_for_one_hot, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_per_class.mean()

        return dice_loss


class ComboLoss(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor | list[float] | None = None,
        gamma: float = 2.0,
        focal_weight: float = 0.7,
        dice_weight: float = 0.3,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()

        total = focal_weight + dice_weight
        if total <= 0:
            raise ValueError("focal_weight + dice_weight must be > 0")

        self.focal_weight = focal_weight / total
        self.dice_weight = dice_weight / total

        self.focal = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction="mean",
            ignore_index=ignore_index,
        )
        self.dice = DiceLoss(
            smooth=1e-6,
            ignore_index=ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss
