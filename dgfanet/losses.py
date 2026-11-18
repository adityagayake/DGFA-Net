"""Training objective (report Section 5):  L = L1 + 0.2 Lssim + 0.5 Lface + 0.01 Ltv.

L1 / Lssim / Lface act on the *unclamped* output (I_low + dI) so that gradients never vanish
at the clamp. Ltv is the mean |grad| of the illumination map M_L (or of dI for models that
have no illumination head).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _gauss_window(size: int = 11, sigma: float = 1.5, device=None, dtype=torch.float32) -> torch.Tensor:
    x = torch.arange(size, dtype=dtype, device=device) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim_torch(x: torch.Tensor, y: torch.Tensor, win: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM of two Bx3xHxW tensors in [0,1] (differentiable)."""
    c = x.shape[1]
    w = _gauss_window(win, sigma, x.device, x.dtype).expand(c, 1, win, win)
    pad = win // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=c)
    mu_y = F.conv2d(y, w, padding=pad, groups=c)
    sxx = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return m.mean()


def tv_mean(m: torch.Tensor) -> torch.Tensor:
    return (m[..., 1:, :] - m[..., :-1, :]).abs().mean() + (m[..., :, 1:] - m[..., :, :-1]).abs().mean()


def face_loss(raw: torch.Tensor, gt: torch.Tensor, heat: torch.Tensor, flag: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    """sum(H * |I_enh - I_gt|) / (sum(H) + eps), averaged over the batch; zero when no face."""
    err = (raw - gt).abs().mean(1, keepdim=True)                       # B,1,H,W
    num = (heat * err).flatten(1).sum(1)
    den = heat.flatten(1).sum(1) + eps
    per = num / den * flag.flatten()
    return per.mean()


class DGFALoss:
    def __init__(self, lam_ssim: float = 0.2, lam_face: float = 0.5, lam_tv: float = 0.01,
                 use_face_loss: bool = True):
        self.ls, self.lf, self.lt = lam_ssim, lam_face, lam_tv
        self.use_face = use_face_loss

    def __call__(self, aux: Dict[str, torch.Tensor], gt: torch.Tensor, heat_gt: torch.Tensor,
                 flag: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        raw = aux["raw"]
        l1 = (raw - gt).abs().mean()
        ls = 1.0 - ssim_torch(raw, gt)
        lf = face_loss(raw, gt, heat_gt, flag) if self.use_face else raw.new_zeros(())
        lt = tv_mean(aux["illum"] if "illum" in aux else aux["delta"])
        total = l1 + self.ls * ls + (self.lf * lf if self.use_face else 0.0) + self.lt * lt
        return total, dict(l1=float(l1.detach()), ssim=float(ls.detach()), face=float(lf.detach()), tv=float(lt.detach()))
