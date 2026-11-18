"""Datasets.

Sources of *clean* face images (the low-light version is synthesised on the fly, report Sec. 3):
  * 'synthetic' : procedural faces with exact landmarks (offline, no download)
  * a folder    : any folder of face photos (CelebA-HQ, FFHQ ... from Kaggle, see README)
Real paired low-light data (LOL-style folders with low/ and high/) is supported by `PairedDataset`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation import noise_level, synthesize_low_light
from .landmarks import FaceLandmarker, make_heatmap, sigma_for
from .metrics import FaceAnalyzer, ita_degrees, ita_group
from .synthetic_faces import generate_faces, generate_scenes
from .utils import to_tensor

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# file discovery
# --------------------------------------------------------------------------- #
def find_images(root, limit: Optional[int] = None) -> List[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file())
    return files[:limit] if limit else files


def find_paired(root) -> List[Tuple[Path, Path]]:
    """LOL / LOL-v2 style: finds directories called low|Low|input and high|Normal|gt|target next to
    each other and pairs images by file name (case-insensitive, 'low' / 'normal' prefixes ignored)."""
    root = Path(root)
    low_names, high_names = {"low", "input", "lowlight", "low_light"}, {"high", "normal", "gt", "target", "well_lit"}
    pairs: List[Tuple[Path, Path]] = []
    for d in sorted([root] + [p for p in root.rglob("*") if p.is_dir()]):
        kids = {k.name.lower(): k for k in d.iterdir() if k.is_dir()}
        lows = [kids[n] for n in kids if n in low_names]
        highs = [kids[n] for n in kids if n in high_names]
        if lows and highs:
            def key(p: Path) -> str:
                s = p.stem.lower()
                for t in ("low", "normal", "high", "_", "-"):
                    s = s.replace(t, "")
                return s
            hmap = {key(p): p for p in find_images(highs[0])}
            for p in find_images(lows[0]):
                if key(p) in hmap:
                    pairs.append((p, hmap[key(p)]))
    return pairs


def load_rgb(path, size: int) -> np.ndarray:
    """Read an image, centre-crop to a square and resize -> uint8 size x size x 3."""
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(f"cannot read {path}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    im = im[y0:y0 + s, x0:x0 + s]
    return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA if s > size else cv2.INTER_CUBIC)


# --------------------------------------------------------------------------- #
# clean face collection
# --------------------------------------------------------------------------- #
class FaceData:
    """In-memory clean images + clean ground-truth landmarks (cached on disk)."""

    def __init__(self, imgs: np.ndarray, pts: List[Optional[np.ndarray]], skin: Optional[np.ndarray],
                 source: str, simulate_detector: bool):
        self.imgs, self.pts, self.skin, self.source = imgs, pts, skin, source
        self.simulate = simulate_detector                       # True -> procedural faces
        self.size = imgs.shape[1]

    def __len__(self):
        return len(self.imgs)


def build_face_data(source: str, size: int, n_images: int, landmarker: FaceLandmarker,
                    cache_dir="data/cache", seed: int = 0, verbose: bool = True) -> FaceData:
    """source = 'synthetic' or a path to a folder of face photos."""
    if source == "synthetic":
        imgs, pts, _, skin = generate_faces(n_images, size, seed=seed)
        return FaceData(imgs, [p for p in pts], skin, "synthetic", True)

    files = find_images(source)
    if not files:
        raise FileNotFoundError(f"No images found under {source}")
    rng = np.random.default_rng(seed)
    if len(files) > n_images:
        files = [files[i] for i in sorted(rng.choice(len(files), n_images, replace=False))]
    key = hashlib.md5((str(Path(source).resolve()) + f"|{size}|{len(files)}|{seed}|{landmarker.name}"
                       ).encode()).hexdigest()[:10]
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    cfile = cdir / f"faces_{key}.npz"
    if cfile.exists():
        try:
            z = np.load(cfile, allow_pickle=False)
            pts_arr = z["pts"]
            if verbose:
                print(f"[data] loaded cache {cfile.name} ({len(z['imgs'])} images)")
            return FaceData(z["imgs"], [None if np.isnan(p).any() else p for p in pts_arr], None,
                            str(source), False)
        except Exception:                                              # corrupt cache -> rebuild
            pass
    imgs, pts, K = [], [], None
    it = files
    try:
        from tqdm import tqdm
        it = tqdm(files, desc="reading + landmarks", leave=False)
    except Exception:
        pass
    for f in it:
        try:
            im = load_rgb(f, size)
        except Exception:
            continue
        r = landmarker.detect(im)
        p = None if r is None else r["points"]
        if p is not None and K is None:
            K = len(p)
        imgs.append(im)
        pts.append(p)
    if not imgs:
        raise RuntimeError(f"None of the images under {source} could be read.")
    K = K or 68
    arr = np.stack([p if p is not None else np.full((K, 2), np.nan, np.float32) for p in
                    [q if (q is None or len(q) == K) else None for q in pts]]).astype(np.float32)
    imgs = np.stack(imgs)
    np.savez_compressed(cfile, imgs=imgs, pts=arr)
    if verbose:
        found = int(sum(not np.isnan(a).any() for a in arr))
        print(f"[data] {len(imgs)} images, face found in {found} ({100 * found / len(imgs):.0f}%) [{landmarker.name}]")
    return FaceData(imgs, [None if np.isnan(p).any() else p for p in arr], None, str(source), False)


def split_indices(n: int, seed: int = 0, frac=(0.8, 0.1, 0.1)) -> Dict[str, np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(n * frac[0]), int(n * (frac[0] + frac[1]))
    return {"train": np.sort(idx[:a]), "val": np.sort(idx[a:b]), "test": np.sort(idx[b:])}


# --------------------------------------------------------------------------- #
# the PyTorch datasets
# --------------------------------------------------------------------------- #
def _pack(low, gt, heat, flag) -> Dict[str, torch.Tensor]:
    return dict(low=to_tensor(low), gt=to_tensor(gt), heat=torch.from_numpy(heat)[None].float(),
                flag=torch.tensor([float(flag)]).view(1, 1, 1))


class TrainSet(Dataset):
    """Random synthetic degradation each time an item is read; GT (clean) landmarks give the heatmap."""

    def __init__(self, data: FaceData, idx: Sequence[int], seed: int = 0, hflip: bool = True,
                 epoch_len: Optional[int] = None):
        self.d, self.idx, self.hflip = data, np.asarray(idx), hflip
        self.rng = np.random.default_rng(seed)
        self.sigma = sigma_for(data.size)
        self.n = epoch_len or len(self.idx)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        j = int(self.idx[i % len(self.idx)])
        gt = self.d.imgs[j].astype(np.float32) / 255.0
        pts = self.d.pts[j]
        pts = None if pts is None else pts.copy()
        if self.hflip and self.rng.random() < 0.5:
            gt = gt[:, ::-1].copy()
            if pts is not None:
                pts[:, 0] = (gt.shape[1] - 1) - pts[:, 0]
        low, _ = synthesize_low_light(gt, self.rng)
        heat = make_heatmap(pts, gt.shape[:2], self.sigma)
        return _pack(low, gt, heat, pts is not None)


class EvalSet(Dataset):
    """Deterministic low-light versions of held-out images.

    det_mode : 'brighten' detection on I_low**0.45 (DGFA-Net default / A3)
               'raw'      detection on the raw dark image (A5)
               'gt'       clean-image landmarks
               'none'     no landmarks at all
    """

    def __init__(self, data: FaceData, idx: Sequence[int], analyzer: FaceAnalyzer, det_mode: str = "brighten",
                 seed: int = 123, faces: bool = True, scenes: Optional[np.ndarray] = None,
                 fixed_params: Optional[Dict[str, float]] = None):
        self.d, self.az, self.det_mode = data, analyzer, det_mode
        self.sigma = sigma_for(data.size)
        self.size = data.size
        self.lows, self.gts, self.heats, self.flags, self.meta = [], [], [], [], []
        items: List[Tuple[np.ndarray, Optional[np.ndarray], int, str]] = []
        if faces:
            for j in idx:
                items.append((data.imgs[int(j)], data.pts[int(j)], int(j), "face"))
        if scenes is not None:
            for k, s in enumerate(scenes):
                items.append((s, None, -1 - k, "scene"))
        for n, (img, pts, j, kind) in enumerate(items):
            rng = np.random.default_rng([seed, j + 10 ** 6, n])
            gt = img.astype(np.float32) / 255.0
            low, par = synthesize_low_light(gt, rng, **(fixed_params or {}))
            det = self._detect(low, gt, pts, kind)
            self.lows.append(low)
            self.gts.append(gt)
            self.heats.append(make_heatmap(det, gt.shape[:2], self.sigma))
            self.flags.append(det is not None)
            ita = ita_degrees(gt, pts) if kind == "face" else float("nan")
            skin = int(data.skin[j]) if (data.skin is not None and kind == "face") else -1
            self.meta.append(dict(idx=j, kind=kind, gt_pts=pts, params=par, noise=noise_level(par),
                                  dark=float(low.mean()), ita=ita, group=ita_group(ita) if kind == "face" else "n/a",
                                  skin=skin, det_pts=det))

    def _detect(self, low, gt, pts, kind) -> Optional[np.ndarray]:
        m = self.det_mode
        if m == "none" or kind == "scene" and self.d.simulate:
            return None
        if m == "gt":
            return pts
        return self.az.detect(low, pts, brighten=(m == "brighten"))

    def __len__(self):
        return len(self.lows)

    def __getitem__(self, i):
        return dict(low=to_tensor(self.lows[i]), gt=to_tensor(self.gts[i]),
                    heat=torch.from_numpy(self.heats[i])[None].float(),
                    flag=torch.tensor([float(self.flags[i])]).view(1, 1, 1))

    def with_mode(self, det_mode: str) -> "EvalSet":
        """Same low-light images, different landmark source (cheap re-detection)."""
        new = object.__new__(EvalSet)
        new.__dict__.update(self.__dict__)
        new.det_mode = det_mode
        new.heats, new.flags = [], []
        for low, gt, m in zip(self.lows, self.gts, self.meta):
            det = new._detect(low, gt, m["gt_pts"], m["kind"])
            new.heats.append(make_heatmap(det, gt.shape[:2], self.sigma))
            new.flags.append(det is not None)
        new.meta = [dict(m, det_pts=None) for m in self.meta]
        return new


class PairedDataset(Dataset):
    """Real paired low/high images (LOL-style). No synthetic degradation is applied."""

    def __init__(self, pairs: Sequence[Tuple[Path, Path]], size: int, landmarker: FaceLandmarker,
                 limit: Optional[int] = None):
        self.sigma = sigma_for(size)
        self.lows, self.gts, self.heats, self.flags, self.meta = [], [], [], [], []
        for lo, hi in list(pairs)[:limit]:
            try:
                low = load_rgb(lo, size).astype(np.float32) / 255.0
                gt = load_rgb(hi, size).astype(np.float32) / 255.0
            except Exception:
                continue
            r = landmarker.detect_float(low, brighten=True)
            det = None if r is None else r["points"]
            self.lows.append(low)
            self.gts.append(gt)
            self.heats.append(make_heatmap(det, (size, size), self.sigma))
            self.flags.append(det is not None)
            self.meta.append(dict(idx=len(self.meta), kind="paired", gt_pts=None, params={}, noise=float("nan"),
                                  dark=float(low.mean()), ita=float("nan"), group="n/a", skin=-1, det_pts=det,
                                  name=Path(lo).name))
        self.size = size

    def __len__(self):
        return len(self.lows)

    __getitem__ = EvalSet.__getitem__


def make_scenes(n: int, size: int, seed: int = 7) -> np.ndarray:
    return generate_scenes(n, size, seed=seed)
