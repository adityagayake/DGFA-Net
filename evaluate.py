"""Evaluate trained weights: prints the baseline-vs-DGFA-Net table (and ablations if all weights exist).

    python evaluate.py --preset smoke           # after: python train.py --preset smoke --with-baselines
Use the same --preset / --data / --size options that you trained with.
"""
import argparse

from dgfanet.pipeline import add_common_args, run_all

if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--with-ablations", action="store_true", help="also evaluate A1,A2,A4,A6,A5 (needs their weights)")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    run_all(a, ablations=a.with_ablations, figures=not a.no_figures, eval_only=True)
