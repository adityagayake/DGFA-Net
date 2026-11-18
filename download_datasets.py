"""Download face datasets from Kaggle and prepare them for the pipeline.

    python download_datasets.py                       # faces + LOL, default Kaggle datasets
    python download_datasets.py --faces-only --max-images 5000
    python download_datasets.py --faces-slug <owner>/<dataset> --lol-slug <owner>/<dataset>
    python download_datasets.py --faces-from C:\\Users\\me\\Downloads\\celeba_hq   # already downloaded manually

Result:  data/faces/*.jpg   (clean face photos, centre-cropped to 256x256, at most --max-images)
         data/lol/...       (LOL-style low/high pairs, used as an extra *real* low-light test)

The script first asks the Kaggle search API for CelebA-HQ / FFHQ / LOL datasets (so it does not depend on a
hard-coded name), then falls back to a built-in candidate list. Needs `pip install kaggle kagglehub`.

Kaggle needs an API token: kaggle.com -> Settings -> API -> "Create New Token" -> save the current
access_token file as ~/.kaggle/access_token (Windows: C:\\Users\\<you>\\.kaggle\\access_token), or set
KAGGLE_USERNAME / KAGGLE_KEY.
If a download fails the script tells you what to do; the pipeline still runs on procedural faces without it.
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from dgfanet.data import find_images, find_paired, load_rgb

# Candidate Kaggle datasets (tried in order). Dataset names on Kaggle change; override with --faces-slug / --lol-slug.
FACE_SLUGS = ["badasstechie/celebahq-resized-256x256", "jessicali9530/celeba-dataset",
              "xhlulu/flickrfaceshq-dataset-nvidia-resized-256px", "lamsimon/celebahq"]
LOL_SLUGS = ["soumikrakshit/lol-dataset", "tanhyml/lol-v2-dataset"]


def rank_search_results(results, must_contain, max_gb: float = 6.0):
    """Pick plausible datasets from a Kaggle search result list (objects with .ref/.title/.total_bytes/.vote_count)."""
    ranked = []
    for d in results:
        ref = str(getattr(d, "ref", "") or "")
        title = str(getattr(d, "title", "") or "")
        text = (ref + " " + title).lower()
        if not any(k in text for k in must_contain):
            continue
        size = getattr(d, "total_bytes", None) or getattr(d, "totalBytes", None)
        try:
            gb = float(size) / 1e9
        except (TypeError, ValueError):
            gb = 0.0
        if gb > max_gb:
            continue
        votes = getattr(d, "vote_count", None) or getattr(d, "voteCount", 0) or 0
        ranked.append((-(float(votes) if str(votes).replace(".", "").isdigit() else 0.0), gb, ref))
    ranked.sort()
    return [r[2] for r in ranked if r[2]]


def search_kaggle(query: str, must_contain, limit: int = 8):
    """Ask the Kaggle API for datasets matching `query` (needs `pip install kaggle` + token). Returns slugs."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        res = api.dataset_list(search=query)
        return rank_search_results(list(res), must_contain)[:limit]
    except Exception as e:
        print(f"[search] Kaggle search unavailable ({type(e).__name__}: {str(e)[:100]})")
        return []


def kaggle_download(slug: str) -> Path:
    try:
        import kagglehub
    except ImportError:
        raise RuntimeError("kagglehub is not installed: pip install kagglehub")
    return Path(kagglehub.dataset_download(slug))


def prepare_faces(src: Path, dst: Path, max_images: int, size: int, seed: int = 0) -> int:
    files = find_images(src)
    if not files:
        raise RuntimeError(f"no images under {src}")
    rng = np.random.default_rng(seed)
    if len(files) > max_images:
        files = [files[i] for i in sorted(rng.choice(len(files), max_images, replace=False))]
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for k, f in enumerate(files):
        try:
            im = load_rgb(f, size)
        except Exception:
            continue
        cv2.imwrite(str(dst / f"{k:05d}.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    return n


def prepare_lol(src: Path, dst: Path) -> int:
    pairs = find_paired(src)
    if not pairs:
        raise RuntimeError(f"no low/high folders found under {src}")
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "low").mkdir(parents=True)
    (dst / "high").mkdir(parents=True)
    for k, (lo, hi) in enumerate(pairs[:200]):
        shutil.copy(lo, dst / "low" / f"{k:04d}{lo.suffix}")
        shutil.copy(hi, dst / "high" / f"{k:04d}{hi.suffix}")
    return min(len(pairs), 200)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--faces-only", action="store_true")
    ap.add_argument("--lol-only", action="store_true")
    ap.add_argument("--faces-slug", default=None)
    ap.add_argument("--lol-slug", default=None)
    ap.add_argument("--faces-from", default=None, help="folder of face photos you downloaded yourself")
    ap.add_argument("--lol-from", default=None, help="LOL folder you downloaded yourself")
    ap.add_argument("--max-images", type=int, default=5000)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    ok = True

    if not a.lol_only:
        done = False
        if a.faces_from:
            n = prepare_faces(Path(a.faces_from), Path("data/faces"), a.max_images, a.size)
            print(f"[faces] prepared {n} images from {a.faces_from} -> data/faces")
            done = True
        else:
            cands = [a.faces_slug] if a.faces_slug else (
                search_kaggle("celeba hq 256", ("celeba", "celebahq")) + search_kaggle("ffhq 256", ("ffhq", "flickr-faces", "flickrfaces"))
                + FACE_SLUGS)
            cands = list(dict.fromkeys(cands))
            for slug in cands:
                try:
                    print(f"[faces] trying kaggle dataset '{slug}' ...")
                    p = kaggle_download(slug)
                    n = prepare_faces(p, Path("data/faces"), a.max_images, a.size)
                    if n < 200:
                        raise RuntimeError(f"only {n} images found - not a usable face dataset")
                    print(f"[faces] OK: {n} images -> data/faces   (source: {slug})")
                    done = True
                    break
                except Exception as e:
                    print(f"[faces] '{slug}' failed: {type(e).__name__}: {str(e)[:150]}")
        if not done:
            ok = False
            print("\n[faces] Could not fetch a face dataset automatically. Options:\n"
                  "  1. Set up the Kaggle token (see --help) and re-run, or pass --faces-slug owner/dataset\n"
                  "  2. Download any face dataset in the browser (e.g. search Kaggle for 'CelebA-HQ' or 'FFHQ'),\n"
                  "     unzip it and run:  python download_datasets.py --faces-from <unzipped folder>\n"
                  "  3. Do nothing: the pipeline runs on procedural faces (--data synthetic).")

    if not a.faces_only:
        done = False
        if a.lol_from:
            try:
                print(f"[lol] prepared {prepare_lol(Path(a.lol_from), Path('data/lol'))} pairs -> data/lol")
                done = True
            except Exception as e:
                print(f"[lol] failed: {e}")
        else:
            cands = [a.lol_slug] if a.lol_slug else (search_kaggle("lol dataset low light", ("lol", "low-light", "lowlight", "low light"))
                                                      + LOL_SLUGS)
            for slug in list(dict.fromkeys(cands)):
                try:
                    print(f"[lol] trying kaggle dataset '{slug}' ...")
                    p = kaggle_download(slug)
                    print(f"[lol] OK: {prepare_lol(p, Path('data/lol'))} pairs -> data/lol   (source: {slug})")
                    done = True
                    break
                except Exception as e:
                    print(f"[lol] '{slug}' failed: {type(e).__name__}: {str(e)[:150]}")
        if not done:
            print("[lol] LOL is optional (only used for the extra real low-light test). Skipped.")

    print("\nNext:  python run_all.py --preset quick --data data/faces")
    sys.exit(0)


if __name__ == "__main__":
    main()
