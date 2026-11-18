"""Evaluation of enhancers on an EvalSet / PairedDataset; ablation helpers; profiling."""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baselines import Enhancer, External
from .metrics import FaceAnalyzer, LPIPSMeter, nme, psnr, ssim
from .utils import count_macs, sync, to_numpy


def _mean(v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


@torch.no_grad()
def evaluate(enh: Enhancer, ds, az: FaceAnalyzer, device: torch.device, lp: Optional[LPIPSMeter] = None,
             bs: int = 8, keep: Optional[Sequence[int]] = None) -> Dict:
    """Returns dict(per=[per-sample dicts], preds={index: float image}). Per-sample keys:
    psnr, ssim, lpips, nme, detected (None if the sample has no GT face)."""
    if isinstance(enh, External):
        enh.reset()
    per: List[Dict] = []
    preds: Dict[int, np.ndarray] = {}
    keep = set(keep or [])
    i = 0
    for b in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0):
        low, heat, flag = (b[k].to(device) for k in ("low", "heat", "flag"))
        out = enh(low, heat, flag).clamp(0, 1)
        for k in range(out.shape[0]):
            pred, gt = to_numpy(out[k]), ds.gts[i]
            m = ds.meta[i]
            r = dict(i=i, kind=m["kind"], group=m["group"], noise=m["noise"], dark=m["dark"], ita=m["ita"],
                     skin=m["skin"], psnr=psnr(pred, gt), ssim=ssim(pred, gt),
                     lpips=lp(pred, gt) if lp is not None else float("nan"), nme=float("nan"), detected=None)
            if m["gt_pts"] is not None:
                det = az.detect(pred, m["gt_pts"], brighten=False)
                r["detected"] = det is not None
                if det is not None and len(det) == len(m["gt_pts"]):
                    r["nme"] = nme(det, m["gt_pts"])
            per.append(r)
            if i in keep:
                preds[i] = pred
            i += 1
    return dict(per=per, preds=preds)


def summarize(per: List[Dict], kind: Optional[str] = None, mask=None) -> Dict[str, float]:
    rows = [r for r in per if (kind is None or r["kind"] == kind) and (mask is None or mask(r))]
    det = [r["detected"] for r in rows if r["detected"] is not None]
    return dict(n=len(rows), psnr=_mean([r["psnr"] for r in rows]), ssim=_mean([r["ssim"] for r in rows]),
                lpips=_mean([r["lpips"] for r in rows]), nme=_mean([r["nme"] for r in rows]),
                detect=float(np.mean(det)) if det else float("nan"))


def split_masks(per: List[Dict]) -> Dict[str, callable]:
    """Subsets used by the hypothesis checks: noisy / clean (median split of noise level), darkest third."""
    faces = [r for r in per if r["kind"] == "face"]
    if not faces:
        return {}
    nz = np.median([r["noise"] for r in faces])
    d33 = np.percentile([r["dark"] for r in faces], 33.3)
    return {"noisy": lambda r: r["noise"] >= nz, "clean": lambda r: r["noise"] < nz,
            "darkest": lambda r: r["dark"] <= d33}


@torch.no_grad()
def profile(enh: Enhancer, device: torch.device, size: int = 256, reps: int = 5) -> Dict[str, float]:
    """Parameters, GMACs and ms/image (batch 1, at size x size)."""
    x = torch.rand(1, 3, size, size, device=device) * 0.3
    h = torch.zeros(1, 1, size, size, device=device)
    f = torch.ones(1, 1, 1, 1, device=device)
    gm = float("nan")
    m = enh.torch_model
    if m is not None:
        try:
            gm = count_macs(m, x, h, f) / 1e9 if hasattr(m, "use_attention") else count_macs(m, x) / 1e9
        except Exception:
            pass
    elif not isinstance(enh, External):
        gm = 0.0
    ms = float("nan")
    if not isinstance(enh, External):
        try:
            enh(x, h, f)
            sync(device)
            t0 = time.time()
            for _ in range(reps):
                enh(x, h, f)
            sync(device)
            ms = (time.time() - t0) / reps * 1000
        except Exception:
            pass
    return dict(params=float(enh.params) / 1e6 if enh.params == enh.params else float("nan"), gmacs=gm, ms=ms)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def bootstrap_ci(x, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Mean and (1-alpha) percentile-bootstrap confidence interval."""
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], dtype=np.float64)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_stats(a, b) -> Dict[str, float]:
    """Paired comparison of two per-image metric lists (a = ours, b = other): mean difference with CI,
    win rate and a two-sided Wilcoxon signed-rank p-value (NaN if scipy is unavailable)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[ok]
    if len(d) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), win=float("nan"), p=float("nan"), n=0)
    m, lo, hi = bootstrap_ci(d)
    p = float("nan")
    try:
        from scipy.stats import wilcoxon
        if len(d) >= 6 and np.any(d != 0):
            p = float(wilcoxon(d).pvalue)
    except Exception:
        pass
    return dict(mean=m, lo=lo, hi=hi, win=float(np.mean(d > 0)), p=p, n=int(len(d)))
