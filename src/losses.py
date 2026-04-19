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

        log_probs = F.log_softmax(logits, dim=1)  # [B, C, H, W]
        probs = torch.exp(log_probs)  # [B, C, H, W]

        targets_unsq = targets.unsqueeze(1)  # [B, 1, H, W]

        log_pt = torch.gather(log_probs, dim=1, index=targets_unsq).squeeze(1)  # [B, H, W]
        pt = torch.gather(probs, dim=1, index=targets_unsq).squeeze(1)  # [B, H, W]

        focal_term = (1.0 - pt) ** self.gamma
        loss = -focal_term * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # [B, H, W]
            loss = alpha_t * loss

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            loss = loss[valid_mask]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
