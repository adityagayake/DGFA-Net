"""Instant demo (no training, no download): the non-learned *pipeline prototype* of report Section 7.1
versus the gamma-correction baseline on one real face photograph.

    python demo_prototype.py                       # built-in scikit-image portrait
    python demo_prototype.py --image my_face.jpg   # your own well-lit photo (it is darkened for you)

The prototype mirrors the stages of DGFA-Net without learning: smoothed max-channel illumination map,
noise map from local high-frequency energy (sets the denoising strength), illumination-adaptive gain and
face emphasis (1 + alpha*H) from a detector heat-map (alpha = 0.25). Outputs of both methods are
brightness-matched to the ground truth, as in the report. Numbers can differ from Table 7.1 (different
detector build / random noise); this is an illustration, not evidence of general performance.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from dgfanet.degradation import synthesize_low_light
from dgfanet.landmarks import FaceLandmarker, box_from_points, make_heatmap, sigma_for
from dgfanet.metrics import psnr, ssim
from dgfanet.utils import ensure_dir, text_table


def match_brightness(x, ref):
    return np.clip(x * (ref.mean() / max(x.mean(), 1e-6)), 0, 1)


def prototype(low, heat, alpha=0.25):
    h, w = low.shape[:2]
    ill = cv2.GaussianBlur(low.max(-1), (0, 0), max(h, w) / 40)                     # illumination map M_L
    hf = low.mean(-1) - cv2.GaussianBlur(low.mean(-1), (0, 0), 1.5)
    noise = np.sqrt(cv2.GaussianBlur(hf ** 2, (0, 0), max(h, w) / 60))             # noise map M_N
    strength = np.clip(noise / (noise.max() + 1e-6) * 2.0, 0.35, 1.0)[..., None]   # denoise more where noisy
    u8 = (np.clip(low, 0, 1) * 255).astype(np.uint8)
    den = cv2.fastNlMeansDenoisingColored(u8, None, 8, 8, 5, 15).astype(np.float32) / 255.0
    low_d = strength * den + (1 - strength) * low
    ill_n = ill / (ill.max() + 1e-6)
    expo = 1.0 / (2.0 + 0.8 * (1.0 - ill_n))[..., None]                             # illumination-adaptive gain:
    out = np.clip(low_d, 1e-6, 1) ** expo                                           # darker areas get a stronger lift
    out = np.clip(out * (1 + alpha * heat[..., None]), 0, 1)                        # facial emphasis
    return out, ill, noise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=None)
    ap.add_argument("--beta", type=float, default=0.6)
    ap.add_argument("--gamma", type=float, default=2.4)
    ap.add_argument("--out", default="outputs/prototype_demo")
    ap.add_argument("--landmarks", default="auto")
    a = ap.parse_args()

    if a.image:
        im = cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB)
    else:
        from skimage import data
        im = data.astronaut()
    s = min(im.shape[:2])
    gt = im[(im.shape[0] - s) // 2:(im.shape[0] + s) // 2, (im.shape[1] - s) // 2:(im.shape[1] + s) // 2]
    gt = cv2.resize(gt, (384, 384), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    low, par = synthesize_low_light(gt, np.random.default_rng(0), beta=a.beta, gamma=a.gamma, peak=120, sigma=0.015)

    lm = FaceLandmarker(a.landmarks, verbose=True)
    det = lm.detect_float(low, brighten=True)
    if det is None:
        print("[warn] no face detected - the prototype runs without face emphasis (gate = 0).")
    heat = make_heatmap(None if det is None else det["points"], gt.shape[:2], sigma_for(384))
    if det is not None:
        # a face box -> a filled soft blob is more robust than 68 template dots for this demo
        x0, y0, x1, y1 = det["box"]
        yy, xx = np.mgrid[0:384, 0:384]
        blob = np.exp(-(((xx - (x0 + x1) / 2) / ((x1 - x0) / 2 + 1)) ** 2 + ((yy - (y0 + y1) / 2) / ((y1 - y0) / 2 + 1)) ** 2) * 1.2)
        heat = np.maximum(heat, blob.astype(np.float32))

    base = match_brightness(np.clip(low, 1e-6, 1) ** (1 / 2.2), gt)
    ours, ill, noise = prototype(low, heat)
    ours = match_brightness(ours, gt)

    def crop(x):
        if det is None:
            return x
        x0, y0, x1, y1 = [int(v) for v in det["box"]]
        return x[max(0, y0):y1, max(0, x0):x1]

    rows = []
    for name, img in (("Low-light input", low), ("Baseline (gamma)", base), ("Ours (pipeline prototype)", ours)):
        fp = psnr(crop(img), crop(gt)) if det is not None else float("nan")
        fs = ssim(crop(img), crop(gt)) if det is not None and min(crop(gt).shape[:2]) >= 11 else float("nan")
        rows.append([name, f"{psnr(img, gt):.2f}", f"{ssim(img, gt):.3f}", f"{fp:.2f}", f"{fs:.3f}"])
    print()
    print(text_table(["Output", "PSNR (dB)", "SSIM", "Face PSNR (dB)", "Face SSIM"], rows,
                     f"Demo image, beta={par['beta']}, gamma={par['gamma']} (higher is better)"))
    print("Baseline = gamma correction. Both outputs are brightness-matched to the ground truth.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = ensure_dir(a.out)
    fig, ax = plt.subplots(2, 5, figsize=(14, 6))
    imgs = [("Ground truth", gt), ("Low-light input", low), ("Baseline (gamma)", base), ("Ours (prototype)", ours)]
    for k, (t, x) in enumerate(imgs):
        ax[0, k].imshow(np.clip(x, 0, 1)); ax[0, k].set_title(t, fontsize=10)
        ax[1, k].imshow(np.clip(crop(x), 0, 1))
    ax[0, 4].imshow(ill, cmap="magma"); ax[0, 4].set_title("illumination map", fontsize=10)
    ax[1, 4].imshow(noise, cmap="magma"); ax[1, 4].set_title("noise map", fontsize=10)
    for x in ax.ravel():
        x.set_xticks([]); x.set_yticks([])
    ax[1, 0].set_ylabel("face zoom")
    fig.suptitle("Pipeline prototype vs gamma baseline (no trained weights)", y=0.99)
    fig.tight_layout()
    fig.savefig(Path(out) / "prototype_demo.png", dpi=130)
    print(f"\nFigure saved to {Path(out) / 'prototype_demo.png'}")


if __name__ == "__main__":
    main()
