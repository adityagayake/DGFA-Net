"""Baselines.

  Input            : the dark image itself (lower bound)
  Gamma            : I ** (1/2.2)
  CLAHE            : contrast-limited adaptive histogram equalisation on the L channel
  Zero-DCE (re-impl): the unsupervised curve-estimation network of Guo et al. (CVPR 2020),
                      re-implemented here (7 conv layers, 8 curve iterations, 4 non-reference
                      losses) and trained on the *low-light inputs only* (no ground truth)
  Residual U-Net   : ablation A0 (supervised, no degradation heads / guidance)
  External         : evaluates outputs produced elsewhere (Retinex-Net, Low-FaceNet ...) that you
                     drop into external/<method>/<index>.png - see README

Retinex-Net and Low-FaceNet are NOT bundled (they need their own code / weights).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import count_params, pad_to_multiple, to_numpy, to_tensor, to_uint8


# --------------------------------------------------------------------------- #
# enhancer wrappers: __call__(low Bx3xHxW, heat Bx1xHxW, flag Bx1x1x1) -> Bx3xHxW in [0,1]
# --------------------------------------------------------------------------- #
class Enhancer:
    name = "enhancer"
    params = 0
    torch_model: Optional[nn.Module] = None

    def __call__(self, low, heat, flag):
        raise NotImplementedError


class Identity(Enhancer):
    name = "Input (dark)"

    def __call__(self, low, heat, flag):
        return low


class Gamma(Enhancer):
    def __init__(self, g: float = 2.2):
        self.g = g
        self.name = f"Gamma (1/{g:g})"

    def __call__(self, low, heat, flag):
        return low.clamp(1e-6, 1).pow(1.0 / self.g)


class CLAHE(Enhancer):
    name = "CLAHE"

    def __init__(self, clip: float = 3.0, grid: int = 8, pre_gamma: float = 1.0):
        self.cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        self.pre_gamma = pre_gamma

    def __call__(self, low, heat, flag):
        outs = []
        for b in range(low.shape[0]):
            im = to_uint8(to_numpy(low[b]))
            lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
            lab[..., 0] = self.cl.apply(lab[..., 0])
            outs.append(to_tensor(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0))
        return torch.stack(outs).to(low.device)


class TorchEnhancer(Enhancer):
    """Wraps a DGFANet-style model (low, heat, flag) -> (out, aux)."""

    def __init__(self, model: nn.Module, name: str):
        self.torch_model = model
        self.name = name
        self.params = count_params(model)

    @torch.no_grad()
    def __call__(self, low, heat, flag):
        self.torch_model.eval()
        x, hw = pad_to_multiple(low, 4)
        hp, _ = pad_to_multiple(heat, 4)
        out, _ = self.torch_model(x, hp, flag)
        return out[..., :hw[0], :hw[1]]


class External(Enhancer):
    """Outputs produced by another program. Files: <root>/<index:05d>.png (same size as the test input)."""

    def __init__(self, name: str, root: Path, indices):
        self.name, self.root = name, Path(root)
        self.files = [self.root / f"{i:05d}.png" for i in indices]
        self.ptr = 0
        self.params = float("nan")

    def available(self) -> bool:
        return all(f.exists() for f in self.files)

    def reset(self):
        self.ptr = 0

    def __call__(self, low, heat, flag):
        outs = []
        for _ in range(low.shape[0]):
            im = cv2.cvtColor(cv2.imread(str(self.files[self.ptr])), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if im.shape[:2] != tuple(low.shape[-2:]):
                im = cv2.resize(im, (low.shape[-1], low.shape[-2]))
            outs.append(to_tensor(im))
            self.ptr += 1
        return torch.stack(outs).to(low.device)


# --------------------------------------------------------------------------- #
# Zero-DCE (re-implementation, DCE-Net with 32 channels, ~79 k parameters)
# --------------------------------------------------------------------------- #
class ZeroDCENet(nn.Module):
    def __init__(self, c: int = 32, iters: int = 8):
        super().__init__()
        self.iters = iters
        conv = lambda i, o: nn.Conv2d(i, o, 3, padding=1)
        self.c1, self.c2, self.c3, self.c4 = conv(3, c), conv(c, c), conv(c, c), conv(c, c)
        self.c5, self.c6 = conv(2 * c, c), conv(2 * c, c)
        self.c7 = conv(2 * c, 3 * iters)

    def forward(self, x):
        r = F.relu
        x1 = r(self.c1(x)); x2 = r(self.c2(x1)); x3 = r(self.c3(x2)); x4 = r(self.c4(x3))
        x5 = r(self.c5(torch.cat([x3, x4], 1)))
        x6 = r(self.c6(torch.cat([x2, x5], 1)))
        curves = torch.tanh(self.c7(torch.cat([x1, x6], 1)))
        out = x
        for a in torch.split(curves, 3, dim=1):
            out = out + a * out * (1 - out)                        # LE(x) = x + A x (1 - x)
        return out.clamp(0, 1), curves


class ZeroDCE(Enhancer):
    name = "Zero-DCE (re-impl.)"

    def __init__(self, net: ZeroDCENet):
        self.net = net
        self.torch_model = net
        self.params = count_params(net)

    @torch.no_grad()
    def __call__(self, low, heat, flag):
        self.net.eval()
        return self.net(low)[0]


def zero_dce_losses(x: torch.Tensor, enh: torch.Tensor, curves: torch.Tensor, E: float = 0.6):
    """Spatial consistency, exposure control, colour constancy, illumination smoothness."""
    # spatial consistency
    def pooled(t):
        return F.avg_pool2d(t.mean(1, keepdim=True), 4)
    ox, oe = pooled(x), pooled(enh)
    dv = lambda t: t[..., :-1, :] - t[..., 1:, :]
    dh = lambda t: t[..., :, :-1] - t[..., :, 1:]
    l_spa = ((dv(oe).abs() - dv(ox).abs()) ** 2).mean() + ((dh(oe).abs() - dh(ox).abs()) ** 2).mean()
    # exposure
    l_exp = ((F.avg_pool2d(enh.mean(1, keepdim=True), 16) - E) ** 2).mean()
    # colour constancy
    m = enh.mean((2, 3))
    r, g, b = m[:, 0], m[:, 1], m[:, 2]
    l_col = ((r - g) ** 2 + (r - b) ** 2 + (g - b) ** 2).mean()
    # illumination smoothness on the curve maps
    l_tv = ((curves[..., 1:, :] - curves[..., :-1, :]) ** 2).mean() + ((curves[..., :, 1:] - curves[..., :, :-1]) ** 2).mean()
    return 20 * l_spa + 10 * l_exp + 5 * l_col + 200 * l_tv, dict(spa=float(l_spa.detach()), exp=float(l_exp.detach()),
                                                                      col=float(l_col.detach()), tv=float(l_tv.detach()))
