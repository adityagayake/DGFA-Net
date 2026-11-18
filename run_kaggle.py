"""ONE COMMAND for the real project run: Kaggle faces -> baselines -> DGFA-Net -> ablations -> figures -> REPORT.md.

    python run_kaggle.py                    # GPU: quick preset (128 px); CPU: cpu preset (64 px)
    python run_kaggle.py --preset full      # report protocol (256 px, 200 epochs; many hours on a GPU)
    python run_kaggle.py --faces-slug owner/dataset-name      # choose the Kaggle dataset yourself
    python run_kaggle.py --faces-from "C:\\Users\\me\\Downloads\\celeba_hq"    # dataset already downloaded in the browser

It REFUSES to continue with synthetic faces: if no real face photos are available it stops and tells you why.
Needs a Kaggle API token (see README, Section 4) unless you use --faces-from or already have data/faces.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from dgfanet.data import find_images
from dgfanet.pipeline import add_common_args, run_all

if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--faces-slug", default=None, help="Kaggle dataset 'owner/name'")
    ap.add_argument("--faces-from", default=None, help="folder of face photos you downloaded manually")
    ap.add_argument("--lol-slug", default=None)
    ap.add_argument("--lol-from", default=None)
    ap.add_argument("--max-images", type=int, default=5000, help="how many photos to take from the dataset")
    ap.add_argument("--skip-ablations", action="store_true")
    a = ap.parse_args()

    have = Path("data/faces").exists() and len(find_images("data/faces", 50)) >= 50
    if a.faces_slug or a.faces_from or not have:
        cmd = [sys.executable, "download_datasets.py", "--max-images", str(a.max_images)]
        if a.faces_slug:
            cmd += ["--faces-slug", a.faces_slug]
        if a.faces_from:
            cmd += ["--faces-from", a.faces_from]
        if a.lol_slug:
            cmd += ["--lol-slug", a.lol_slug]
        if a.lol_from:
            cmd += ["--lol-from", a.lol_from]
        subprocess.call(cmd)
    n = len(find_images("data/faces", 100)) if Path("data/faces").exists() else 0
    if n < 50:
        sys.exit("\nSTOP: no real face photos in data/faces (need at least 50).\n"
                 "Fix the Kaggle token / dataset name (README Section 4), or download a face dataset from kaggle.com in the\n"
                 "browser, unzip it and run:  python run_kaggle.py --faces-from <unzipped folder>")
    a.data = "data/faces"
    run_all(a, ablations=not a.skip_ablations)
