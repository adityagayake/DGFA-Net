"""Ablation study A0-A6 + hypothesis checks H1-H3 (report Tables 7.3 and Section 7.4).

    python run_ablations.py --preset smoke
Trains A0, A1, A2, A3, A4, A6 (A5 re-uses A3 with landmarks detected on the raw dark image).
"""
import argparse

from dgfanet.pipeline import add_common_args, run_all

if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    a.skip_zerodce = True
    run_all(a, ablations=True, figures=not a.no_figures)
