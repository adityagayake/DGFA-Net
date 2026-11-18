"""Figures (matplotlib, Agg backend: no display needed)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402

from .landmarks import box_from_points                               # noqa: E402
from .utils import ensure_dir                                        # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a6bd6", "#4a4a4a"]
INK, MUTED = "#222222", "#6b6b6b"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#bbbbbb", "axes.labelcolor": INK, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})


def _save(fig, path):
    ensure_dir(Path(path).parent)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def image_grid(columns: Sequence[str], rows: Sequence[Sequence[np.ndarray]], path, captions=None,
               row_labels: Optional[Sequence[str]] = None, title: str = "", cell: float = 1.7):
    """rows[r][c] is an HxWx3 (or HxW heat-map) image; captions[r][c] a short string under each cell."""
    R, C = len(rows), len(columns)
    fig, axes = plt.subplots(R, C, figsize=(cell * C, cell * R * (1.12 if captions else 1.0) + 0.5), squeeze=False)
    for r in range(R):
        for c in range(C):
            ax = axes[r][c]
            im = rows[r][c]
            if im is None:
                ax.axis("off")
                continue
            if im.ndim == 2:
                lo, hi = float(np.min(im)), float(np.max(im))
                ax.imshow(im, cmap="magma", vmin=lo, vmax=hi if hi - lo > 1e-6 else lo + 1.0)   # each map auto-scaled
            else:
                ax.imshow(np.clip(im, 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(columns[c], fontsize=8.5, color=INK)
            if captions and captions[r][c]:
                ax.set_xlabel(captions[r][c], fontsize=7.5, color=MUTED, labelpad=2)
            if c == 0 and row_labels:
                ax.set_ylabel(row_labels[r], fontsize=8, color=MUTED)
    if title:
        fig.suptitle(title, fontsize=10, y=1.0)
    fig.tight_layout(w_pad=0.2, h_pad=0.6)
    return _save(fig, path)


def bar_chart(labels: Sequence[str], values: Sequence[float], path, title: str, ylabel: str,
              highlight: Optional[Sequence[int]] = None, fmt: str = "{:.2f}", ylim=None):
    fig, ax = plt.subplots(figsize=(max(4.5, 0.95 * len(labels) + 1.5), 3.6))
    colors = [PALETTE[1] if (highlight and i in highlight) else PALETTE[0] for i in range(len(labels))]
    vals = [0 if (v is None or np.isnan(v)) else v for v in values]
    bars = ax.bar(range(len(labels)), vals, color=colors, width=0.62)
    for b, v, raw in zip(bars, vals, values):
        txt = "n/a" if (raw is None or np.isnan(raw)) else fmt.format(raw)
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), txt, ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)
    return _save(fig, path)


def grouped_bars(groups: Sequence[str], series: Dict[str, Sequence[float]], path, title: str, ylabel: str,
                 fmt: str = "{:.1f}"):
    fig, ax = plt.subplots(figsize=(max(5.0, 1.8 * len(groups) + 2), 3.8))
    n = len(series)
    w = 0.8 / max(1, n)
    for k, (name, vals) in enumerate(series.items()):
        x = np.arange(len(groups)) + (k - (n - 1) / 2) * w
        v = [0 if np.isnan(a) else a for a in vals]
        bars = ax.bar(x, v, w * 0.92, label=name, color=PALETTE[k % len(PALETTE)])
        for b, a in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), "n/a" if np.isnan(a) else fmt.format(a),
                    ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=min(n, 3), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)
    return _save(fig, path)


def curves(history: Dict[str, List[Sequence[float]]], path, title: str = "Training curves"):
    """history[name] = list of (epoch, train_loss, val_psnr)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, (name, h) in enumerate(history.items()):
        h = np.array(h, dtype=float)
        if h.ndim != 2 or h.shape[1] < 3:
            continue
        a.plot(h[:, 0], h[:, 1], color=PALETTE[k % len(PALETTE)], lw=1.6, label=name)
        b.plot(h[:, 0], h[:, 2], color=PALETTE[k % len(PALETTE)], lw=1.6, label=name)
    a.set_title("training loss", loc="left", fontsize=9); a.set_xlabel("epoch")
    b.set_title("validation PSNR (dB)", loc="left", fontsize=9); b.set_xlabel("epoch")
    for ax in (a, b):
        ax.grid(color="#eeeeee")
    b.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    return _save(fig, path)


def overlay_points(img: np.ndarray, pts: Optional[np.ndarray], color=(1.0, 0.85, 0.0)) -> np.ndarray:
    out = np.clip(img, 0, 1).copy()
    if pts is None:
        return out
    h, w = out.shape[:2]
    r = max(1, int(round(min(h, w) / 128)))
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        out[max(0, yi - r):min(h, yi + r + 1), max(0, xi - r):min(w, xi + r + 1)] = color
    return out


def crop_by_box(img: np.ndarray, pts: Optional[np.ndarray], part: str) -> np.ndarray:
    """Eyes or mouth crop located relative to the face box (works for any landmark backend)."""
    h, w = img.shape[:2]
    if pts is None:
        return img
    x0, y0, x1, y1 = box_from_points(pts, hw=(h, w))
    bw, bh = x1 - x0, y1 - y0
    fy = (0.18, 0.46) if part == "eyes" else (0.62, 0.95)
    a, b = int(y0 + fy[0] * bh), int(y0 + fy[1] * bh)
    c, d = int(x0), int(x1)
    return img[max(0, a):max(a + 2, b), max(0, c):max(c + 2, d)]


def pad_to(img: np.ndarray, hw) -> np.ndarray:
    """Resize (nearest) a crop to a fixed hw so that crops of different methods line up."""
    import cv2
    return cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_CUBIC)


def boxplot(labels: Sequence[str], data: Sequence[Sequence[float]], path, title: str, ylabel: str,
            highlight: Optional[int] = None):
    fig, ax = plt.subplots(figsize=(max(5, 1.0 * len(labels) + 1.5), 3.8))
    clean = [[v for v in d if not np.isnan(v)] or [0.0] for d in data]
    bp = ax.boxplot(clean, patch_artist=True, showmeans=True, widths=0.55,
                    medianprops=dict(color=INK, lw=1.4), meanprops=dict(marker="D", markerfacecolor="white",
                    markeredgecolor=INK, markersize=4), flierprops=dict(marker=".", markersize=3, alpha=0.5))
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(PALETTE[1] if i == highlight else PALETTE[0])
        b.set_alpha(0.85)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    for i, d in enumerate(clean):
        ax.text(i + 1, np.median(d), f"{np.median(d):.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)
    return _save(fig, path)


def scatter_pair(x, y, groups, xlabel: str, ylabel: str, path, title: str):
    """Per-image comparison of two methods; points above the diagonal favour y."""
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    x, y = np.asarray(x, float), np.asarray(y, float)
    names = ["light", "intermediate", "dark"]
    for k, g in enumerate(names):
        m = np.array([gg == g for gg in groups])
        if m.any():
            ax.scatter(x[m], y[m], s=22, color=PALETTE[k], label=f"{g} (n={int(m.sum())})", alpha=0.85,
                       edgecolor="white", linewidth=0.6)
    rest = np.array([gg not in names for gg in groups])
    if rest.any():
        ax.scatter(x[rest], y[rest], s=22, color=PALETTE[6], label="other", alpha=0.8)
    lo, hi = np.nanmin(np.r_[x, y]) - 0.5, np.nanmax(np.r_[x, y]) + 0.5
    ax.plot([lo, hi], [lo, hi], color="#999999", lw=1, ls="--")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.text(0.04, 0.95, f"above diagonal: {int(np.sum(y > x))}/{len(x)} images", transform=ax.transAxes, fontsize=8, va="top")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(color="#eeeeee")
    return _save(fig, path)


def binned_lines(xbins: Sequence[str], series: Dict[str, Sequence[float]], path, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for k, (name, v) in enumerate(series.items()):
        ax.plot(range(len(xbins)), v, marker="o", lw=1.8, color=PALETTE[k % len(PALETTE)], label=name)
    ax.set_xticks(range(len(xbins)))
    ax.set_xticklabels(xbins, fontsize=8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(color="#eeeeee")
    return _save(fig, path)


def summary_panel(names: Sequence[str], metrics: Dict[str, Sequence[float]], path, highlight: Optional[int] = None,
                  title: str = "Baselines vs DGFA-Net"):
    """2x2 (or n) small bar panels, one per metric: {metric title: values aligned with names}."""
    n = len(metrics)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, 3.3 * rows), squeeze=False)
    for ax, (t, vals) in zip(axes.ravel(), metrics.items()):
        v = [0 if (a is None or np.isnan(a)) else a for a in vals]
        colors = [PALETTE[1] if k == highlight else PALETTE[0] for k in range(len(names))]
        bars = ax.bar(range(len(names)), v, color=colors, width=0.65)
        for b, a in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), "n/a" if (a is None or np.isnan(a)) else f"{a:.2f}" if abs(a) >= 1 else f"{a:.3f}",
                    ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        ax.set_title(t, fontsize=9, loc="left")
        ax.grid(axis="y", color="#eeeeee")
        ax.set_axisbelow(True)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, path)
