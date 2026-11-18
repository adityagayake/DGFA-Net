"""ONE COMMAND for the real project run:  Kaggle faces -> baselines -> DGFA-Net -> ablations -> tables + figures.

    python run_project.py                    # GPU: preset quick (128 px);  no GPU: preset cpu
    python run_project.py --preset full      # report protocol (256 px, 200 epochs; many hours)
    python run_project.py --faces-from "C:\\Users\\me\\Downloads\\celeba_hq"   # dataset downloaded by hand

Steps: (1) make sure data/faces holds REAL face photos (downloads them from Kaggle if missing),
(2) run the whole pipeline on them, (3) print where RESULTS.md and the figures are.
It refuses to fall back to procedural faces; use `run_all.py --data synthetic` for that.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from dgfanet.data import find_images


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--faces-from", default=None, help="folder with face photos you downloaded manually")
    ap.add_argument("--max-images", type=int, default=5000)
    ap.add_argument("--skip-download", action="store_true")
    a, rest = ap.parse_known_args()

    have = len(find_images("data/faces")) if Path("data/faces").exists() else 0
    if have < 200 and not a.skip_download:
        cmd = [sys.executable, "download_datasets.py", "--max-images", str(a.max_images)]
        if a.faces_from:
            cmd += ["--faces-from", a.faces_from]
        print("[project] data/faces is empty -> downloading real face photos ...")
        subprocess.call(cmd)
        have = len(find_images("data/faces")) if Path("data/faces").exists() else 0
    if have < 200:
        print("\n[project] STOP: no real face dataset available in data/faces.\n"
              "  * Create a Kaggle token (kaggle.com > Settings > API > Create New Token -> ~/.kaggle/access_token)\n"
              "    and run again, OR download a CelebA-HQ / FFHQ dataset from kaggle.com in the browser, unzip it and run:\n"
              "    python run_project.py --faces-from <unzipped folder>")
        sys.exit(1)
    print(f"[project] {have} real face photos in data/faces")
    try:
        import mediapipe  # noqa: F401
    except Exception:
        print("[project] tip: `pip install mediapipe` gives true facial landmarks (otherwise a face-box template is used).")
    cmd = [sys.executable, "run_all.py", "--data", "data/faces"] + rest
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
