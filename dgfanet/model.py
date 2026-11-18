"""DGFA-Net (report Section 4 and Appendix D) with switches for the ablations A0-A6.

Pipeline (Figure 1 of the report):
  1 input I_low -> 2 feature stem F0 -> 3 illumination / noise heads (M_L, M_N)
  -> 4 fusion (34 -> 32 ch) -> 5 encoder (F1, F2, F3) -> 6 landmark-guided facial
  attention at the bottleneck  F_guided = F3 * (1 + alpha * A_f)
  -> 7 decoder with skip connections -> 8 residual output  I_enh = clamp(I_low + dI).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


def head(c: int, act: nn.Module) -> nn.Sequential:
    """Conv1x1(c -> c/2), LeakyReLU, Conv1x1(c/2 -> 1), activation."""
    return nn.Sequential(nn.Conv2d(c, c // 2, 1), nn.LeakyReLU(0.1, True),
                         nn.Conv2d(c // 2, 1, 1), act)


class DGFANet(nn.Module):
    """Degradation-aware, Geometry-guided Face Enhancement Network.

    Ablation switches (report Section 6.3):
      use_degradation=False, use_attention=False  -> A0  Residual U-Net only
      use_degradation=True,  use_attention=False  -> A1  + illumination & noise heads
      use_degradation=True,  use_attention=True, use_gate=False -> A2  + landmark attention, no gate
      all True                                    -> A3  full DGFA-Net
      learnable_alpha=False                       -> A4  fixed alpha
    (A5 = A3 with landmarks detected on the raw dark image is an inference-time change;
     A6 = A3 trained without L_face is a training-time change.)
    """

    def __init__(self, c: int = 32, alpha: float = 0.5, use_degradation: bool = True,
                 use_attention: bool = True, use_gate: bool = True, learnable_alpha: bool = True):
        super().__init__()
        self.use_degradation = use_degradation
        self.use_attention = use_attention
        self.use_gate = use_gate

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)))

        # Stage 1: feature stem  -> F0
        self.stem = nn.Sequential(nn.Conv2d(3, c, 3, padding=1), nn.LeakyReLU(0.1, True),
                                  ResBlock(c), ResBlock(c))
        # Stage 2: degradation-aware heads + fusion (34 -> 32)
        if use_degradation:
            self.illum = head(c, nn.Sigmoid())
            self.noise = head(c, nn.Softplus())
            self.fuse = nn.Conv2d(c + 2, c, 1)
        # Stage 3: encoder
        self.enc1 = ResBlock(c)                                                   # 256x256, c
        self.down1 = nn.Conv2d(c, 2 * c, 3, stride=2, padding=1)
        self.enc2 = ResBlock(2 * c)                                               # 128x128, 2c
        self.down2 = nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1)
        self.enc3 = ResBlock(4 * c)                                               # 64x64, 4c (F3)
        # Stage 4: facial guidance
        if use_attention:
            self.att = nn.Sequential(nn.Conv2d(4 * c + 1, 1, 1), nn.Sigmoid())
        # Stage 5: decoder with skip connections
        self.up2 = nn.Conv2d(4 * c, 2 * c, 3, padding=1)
        self.dec2 = nn.Sequential(nn.Conv2d(4 * c, 2 * c, 1), ResBlock(2 * c))
        self.up1 = nn.Conv2d(2 * c, c, 3, padding=1)
        self.dec1 = nn.Sequential(nn.Conv2d(2 * c, c, 1), ResBlock(c))
        # Stage 6: residual output
        self.out = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, low: torch.Tensor, heat: torch.Tensor, face_flag: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """low: B,3,H,W in [0,1]; heat: B,1,H,W; face_flag: B,1,1,1 (1 = face found).

        H and W must be multiples of 4 (use utils.pad_to_multiple otherwise).
        Returns (clamped output, aux). aux['raw'] is the *unclamped* sum used by the loss.
        """
        f0 = self.stem(low)
        aux: Dict[str, torch.Tensor] = {}
        if self.use_degradation:
            ill, noi = self.illum(f0), self.noise(f0)
            f = self.fuse(torch.cat([f0, ill, noi], 1))
            aux["illum"], aux["noise"] = ill, noi
        else:
            f = f0
        e1 = self.enc1(f)
        e2 = self.enc2(self.down1(e1))
        f3 = self.enc3(self.down2(e2))
        if self.use_attention:
            h = F.interpolate(heat, size=f3.shape[-2:], mode="bilinear", align_corners=False)
            gate = face_flag if self.use_gate else torch.ones_like(face_flag)
            a = self.att(torch.cat([f3, h], 1)) * gate           # gated: no face -> A_f = 0
            g = f3 * (1 + self.alpha * a)                         # F_guided
            aux["attn"] = a
        else:
            g = f3
        d2 = F.interpolate(g, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([self.up2(d2), e2], 1))
        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([self.up1(d1), e1], 1))
        delta = self.out(d1)
        raw = low + delta
        aux["delta"] = delta
        aux["raw"] = raw
        return raw.clamp(0, 1), aux


# --------------------------------------------------------------------------- #
# ablation registry
# --------------------------------------------------------------------------- #
VARIANTS = {
    "A0": dict(desc="Residual U-Net only (no degradation heads, no guidance)",
               model=dict(use_degradation=False, use_attention=False), face_loss=True),
    "A1": dict(desc="A0 + illumination & noise heads",
               model=dict(use_degradation=True, use_attention=False), face_loss=True),
    "A2": dict(desc="A1 + landmark attention, no presence gate",
               model=dict(use_degradation=True, use_attention=True, use_gate=False), face_loss=True),
    "A3": dict(desc="A1 + landmark attention + presence gate (full DGFA-Net)",
               model=dict(use_degradation=True, use_attention=True, use_gate=True), face_loss=True),
    "A4": dict(desc="A3 with alpha fixed (0.5) instead of learnable",
               model=dict(use_degradation=True, use_attention=True, use_gate=True, learnable_alpha=False),
               face_loss=True),
    "A6": dict(desc="A3 trained without the landmark-weighted loss L_face",
               model=dict(use_degradation=True, use_attention=True, use_gate=True), face_loss=False),
}


def build_model(variant: str = "A3", c: int = 32) -> DGFANet:
    if variant not in VARIANTS:
        raise KeyError(f"Unknown variant '{variant}'. Choose from {sorted(VARIANTS)} (A5 is inference-only).")
    return DGFANet(c=c, **VARIANTS[variant]["model"])
