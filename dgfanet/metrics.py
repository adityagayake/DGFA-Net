"""Evaluation metrics: PSNR, SSIM, optional LPIPS, landmark NME, face-detection rate and the
ITA skin-tone grouping used for the fairness analysis."""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np
import torch

from .landmarks import FaceLandmarker, box_from_points, brighten_for_detection, template_in_box
from .losses import ssim_torch


# --------------------------------------------------------------------------- #
# image quality
# --------------------------------------------------------------------------- #
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse < 1e-10 else float(10 * np.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    ta = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[None].float()
    tb = torch.from_numpy(np.ascontiguousarray(b.transpose(2, 0, 1)))[None].float()
    return float(ssim_torch(ta, tb))


class LPIPSMeter:
    """Optional LPIPS (needs `pip install lpips` and a one-off weight download). Disabled
    gracefully (returns NaN) when unavailable."""

    def __init__(self, enabled: bool = True, device="cpu"):
        self.net = None
        self.device = device
        self.status = "disabled"
        if not enabled:
            return
        try:
            import lpips                                              # noqa
            self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
            self.status = "lpips(alex)"
        except Exception as e:                                        # noqa
            self.status = f"unavailable ({type(e).__name__})"

    @torch.no_grad()
    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.net is None:
            return float("nan")
        t = lambda x: (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].float()
                       .to(self.device) * 2 - 1)
        return float(self.net(t(a), t(b)).item())


# --------------------------------------------------------------------------- #
# face analysis (detector wrapper; real detector or simulated proxy for procedural faces)
# --------------------------------------------------------------------------- #
class FaceAnalyzer:
    """Returns face landmarks for an image.

    simulate=False : uses the real `FaceLandmarker` (MediaPipe or cascade).
    simulate=True  : *proxy detector for procedural cartoon faces* (a real detector is unreliable
                     on them). It uses the known ground-truth landmarks, succeeds only if the face
                     region is bright enough / has enough contrast and perturbs the points in
                     proportion to the measured noise. It is deterministic per image.
    """

    def __init__(self, landmarker: Optional[FaceLandmarker], simulate: bool = False):
        self.lm = landmarker
        self.simulate = simulate

    @property
    def name(self) -> str:
        return "simulated-detector" if self.simulate else (self.lm.name if self.lm else "none")

    def detect(self, img: np.ndarray, gt_pts: Optional[np.ndarray] = None, brighten: bool = False
               ) -> Optional[np.ndarray]:
        """img float32 HxWx3 [0,1]; returns (K,2) points or None."""
        if not self.simulate:
            r = self.lm.detect_float(img, brighten=brighten) if self.lm else None
            return None if r is None else r["points"]
        if gt_pts is None:
            return None
        x = brighten_for_detection(img) if brighten else img
        h, w = x.shape[:2]
        x0, y0, x1, y1 = box_from_points(gt_pts, hw=(h, w))
        sl = x[int(y0):int(y1) + 1, int(x0):int(x1) + 1]
        if sl.size == 0:
            return None
        lum = sl.mean(-1)
        mean, std = float(lum.mean()), float(lum.std())
        hf = float((lum - cv2.GaussianBlur(lum, (0, 0), 1.2)).std())
        if mean < 0.10 or std < 0.025 or hf > 0.12:
            return None
        rng = np.random.default_rng(int(abs(float(x.sum())) * 1000) % (2 ** 32))
        jit = (0.004 + 0.8 * hf) * max(h, w)
        return (gt_pts + rng.normal(0, jit, gt_pts.shape)).astype(np.float32)


def nme(pred: np.ndarray, gt: np.ndarray) -> float:
    """Normalised mean error: mean point distance / sqrt(GT face-box area)."""
    x0, y0, x1, y1 = box_from_points(gt)
    norm = max(1.0, float(np.sqrt(max(1.0, (x1 - x0) * (y1 - y0)))))
    return float(np.linalg.norm(pred - gt, axis=1).mean() / norm)


# --------------------------------------------------------------------------- #
# ITA skin tone (Chardon et al.)
# --------------------------------------------------------------------------- #
def face_core_crop(img: np.ndarray, pts: Optional[np.ndarray]) -> np.ndarray:
    h, w = img.shape[:2]
    if pts is None:
        return img[h // 3:2 * h // 3, w // 3:2 * w // 3]
    x0, y0, x1, y1 = box_from_points(pts, hw=(h, w))
    cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)
    # lower-centre cheeks/forehead region of the face box (avoids hair and eyes)
    a, b = int(max(0, cy - 0.15 * bh)), int(min(h, cy + 0.25 * bh))
    c, d = int(max(0, cx - 0.30 * bw)), int(min(w, cx + 0.30 * bw))
    crop = img[a:b, c:d]
    return crop if crop.size else img[h // 3:2 * h // 3, w // 3:2 * w // 3]


def ita_degrees(img: np.ndarray, pts: Optional[np.ndarray]) -> float:
    from skimage.color import rgb2lab
    crop = face_core_crop(img, pts)
    lab = rgb2lab(np.clip(crop, 0, 1).astype(np.float64))
    L, b = float(np.median(lab[..., 0])), float(np.median(lab[..., 2]))
    return float(np.degrees(np.arctan2(L - 50.0, b if abs(b) > 1e-3 else 1e-3)))


def ita_group(ita: float) -> str:
    """Three coarse groups: light (ITA > 41), intermediate (10 < ITA <= 41), dark (ITA <= 10)."""
    if np.isnan(ita):
        return "unknown"
    return "light" if ita > 41 else ("intermediate" if ita > 10 else "dark")


def luminance(img: np.ndarray) -> float:
    return float(img.mean())
