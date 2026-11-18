"""Run the complete pipeline: baselines + DGFA-Net + ablations + fairness + figures + tables.

    python run_all.py --preset smoke      # 1-3 min check
    python run_all.py --preset cpu        # laptop CPU
    python run_all.py --preset quick      # GPU
"""
import argparse

from dgfanet.pipeline import add_common_args, run_all

if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    run_all(a, ablations=not a.skip_ablations, figures=not a.no_figures)
