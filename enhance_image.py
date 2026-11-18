"""Enhance your own low-light photos with a trained DGFA-Net (and compare with gamma correction).

    python enhance_image.py --input my_dark_face.jpg
    python enhance_image.py --input folder_of_photos/ --ckpt outputs/quick_faces_128px/checkpoints/A3.pt

Saves  <name>_enhanced.png  and a side-by-side  <name>_compare.png  into --out (default outputs/enhanced).
If the photo contains no detectable face the presence gate switches the facial guidance off (it is still enhanced).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from dgfanet.data import find_images
from dgfanet.landmarks import FaceLandmarker, make_heatmap, sigma_for
from dgfanet.model import build_model
from dgfanet.utils import ensure_dir, get_device, pad_to_multiple, to_numpy, to_tensor, to_uint8


def find_ckpt(path):
    if path:
        return Path(path)
    cands = sorted(Path("outputs").glob("*/checkpoints/A3.pt"), key=lambda p: p.stat().st_mtime) if Path("outputs").exists() else []
    if not cands:
        raise SystemExit("No trained weights found. Run  python train.py --preset smoke  first (or pass --ckpt).")
    return cands[-1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="outputs/enhanced")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--landmarks", default="auto")
    ap.add_argument("--max-side", type=int, default=768, help="downscale larger images (speed / memory)")
    a = ap.parse_args()

    dev = get_device(a.device)
    ck = find_ckpt(a.ckpt)
    state = torch.load(ck, map_location=dev)
    model = build_model(state.get("variant", "A3")).to(dev)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[enhance] weights: {ck}")
    lm = FaceLandmarker(a.landmarks)
    out_dir = ensure_dir(a.out)
    files = find_images(a.input)
    if not files:
        raise SystemExit(f"No images found at {a.input}")
    for f in files:
        im = cv2.imread(str(f))
        if im is None:
            print(f"[skip] cannot read {f}")
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        if max(h, w) > a.max_side:
            sc = a.max_side / max(h, w)
            im = cv2.resize(im, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        low = im.astype(np.float32) / 255.0
        det = lm.detect_float(low, brighten=True)
        heat = make_heatmap(None if det is None else det["points"], low.shape[:2], sigma_for(min(low.shape[:2])))
        x, hw = pad_to_multiple(to_tensor(low)[None].to(dev), 4)
        hp, _ = pad_to_multiple(torch.from_numpy(heat)[None, None].to(dev), 4)
        flag = torch.full((1, 1, 1, 1), 0.0 if det is None else 1.0, device=dev)
        with torch.no_grad():
            y, _ = model(x, hp, flag)
        y = to_numpy(y[..., :hw[0], :hw[1]])
        gam = np.clip(low, 1e-6, 1) ** (1 / 2.2)
        cv2.imwrite(str(out_dir / f"{f.stem}_enhanced.png"), cv2.cvtColor(to_uint8(y), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{f.stem}_compare.png"),
                    cv2.cvtColor(to_uint8(np.concatenate([low, gam, y], 1)), cv2.COLOR_RGB2BGR))
        print(f"[enhance] {f.name}: face {'found' if det is not None else 'NOT found (gate=0)'} -> {out_dir / (f.stem + '_compare.png')}")
