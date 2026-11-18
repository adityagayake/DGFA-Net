"""Train DGFA-Net (and optionally the learned baselines / other ablation variants).

    python train.py --preset smoke
    python train.py --preset quick --data data/faces            # Kaggle faces, GPU
    python train.py --preset quick --with-baselines              # also Residual U-Net (A0) and Zero-DCE
    python train.py --variants A1 A2 A4 A6                       # extra ablation variants
Weights are stored in outputs/<run>/checkpoints/ and re-used by evaluate.py / enhance_image.py.
"""
import argparse

from dgfanet.pipeline import add_common_args, get_variant, get_zerodce, setup
from dgfanet.utils import banner

if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--with-baselines", action="store_true", help="also train Residual U-Net (A0) and Zero-DCE")
    ap.add_argument("--variants", nargs="*", default=[], help="extra variants to train, e.g. A1 A2 A4 A6")
    a = ap.parse_args()
    R = setup(a, need_lpips=False)
    banner("Training", "-")
    todo = ["A3"] + (["A0"] if a.with_baselines else []) + [v for v in a.variants if v not in ("A3",)]
    for v in dict.fromkeys(todo):
        get_variant(R, v, R.cfg["epochs"])
    if a.with_baselines and not a.skip_zerodce:
        get_zerodce(R)
    banner(f"Training finished in {R.sw.pretty()} | weights in {R.ckpt}")
    print("Next:  python evaluate.py with the same --preset/--data options.")
