from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.dropout(x)
        return self.conv2(x)


class AttentionGate(nn.Module):
    def __init__(self, g_channels: int, x_channels: int, inter_channels: int) -> None:
        super().__init__()

        self.W_g = nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False)
        self.W_x = nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False)

        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)

        psi = self.psi(g1 + x1)

        # residual gating: safer than x * psi
        return x * (1.0 + psi)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 4) -> None:
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 32, dropout=0.0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(32, 64, dropout=0.0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(64, 128, dropout=0.0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = DoubleConv(128, 256, dropout=0.1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(256, 512, dropout=0.3)

        # Attention only for deeper skip connections
        self.att4 = AttentionGate(g_channels=256, x_channels=256, inter_channels=128)
        self.att3 = AttentionGate(g_channels=128, x_channels=128, inter_channels=64)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512, 256, dropout=0.1)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128, dropout=0.1)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64, dropout=0.05)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32, dropout=0.0)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.up4(bottleneck)
        enc4_att = self.att4(g=dec4, x=enc4)
        dec4 = torch.cat([dec4, enc4_att], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.up3(dec4)
        enc3_att = self.att3(g=dec3, x=enc3)
        dec3 = torch.cat([dec3, enc3_att], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        logits = self.final_conv(dec1)
        return logits
