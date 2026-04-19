"""Small U-Net for pixel-level crack segmentation.

Binary segmentation (background vs crack), single-channel sigmoid output.
Designed to stay lightweight so it can train and infer on CPU beside the
existing ResNet18 classifier.

With ``base_channels=16`` the network has ~1.9M parameters and runs at
<500 ms per 384x384 image on a modern CPU — fast enough for a Streamlit
demo without a GPU.

Why a custom U-Net (rather than ``segmentation_models_pytorch`` or
Ultralytics YOLO-seg)?

- No extra third-party dependency; pure ``torch.nn``.
- License-clean (MIT) — Ultralytics is AGPL-3.0.
- Architecture fits in ~80 LoC, so it can be explained line-by-line in
  an interview setting.
- The task (binary, small images, limited training data) does not need a
  heavyweight pretrained backbone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

__all__ = [
    "CLASS_NAMES_SEG",
    "UNet",
    "build_unet",
    "load_seg_model",
    "count_parameters",
]

CLASS_NAMES_SEG = ["background", "crack"]


class _DoubleConv(nn.Module):
    """Two 3x3 conv + BN + ReLU blocks — the standard U-Net building block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Classic U-Net with 4 down-sampling stages and skip connections.

    Output is a single-channel logit map (no sigmoid); apply
    ``torch.sigmoid`` at inference time. This keeps training numerically
    stable with ``BCEWithLogitsLoss``.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        b = base_channels

        self.enc1 = _DoubleConv(in_channels, b)
        self.enc2 = _DoubleConv(b, b * 2)
        self.enc3 = _DoubleConv(b * 2, b * 4)
        self.enc4 = _DoubleConv(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _DoubleConv(b * 8, b * 16)

        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.dec4 = _DoubleConv(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(b * 2, b)

        self.out_conv = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


def build_unet(in_channels: int = 3, base_channels: int = 16) -> UNet:
    """Construct a fresh U-Net with random weights."""
    return UNet(in_channels=in_channels, base_channels=base_channels)


def load_seg_model(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    base_channels: int = 16,
) -> UNet:
    """Load a trained U-Net checkpoint in eval mode.

    The checkpoint may be a raw ``state_dict`` or a dict with a
    ``state_dict`` key (matches how ``src.seg_train`` saves).
    """
    model = build_unet(base_channels=base_channels)
    state = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
