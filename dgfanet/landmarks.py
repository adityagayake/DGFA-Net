"""Facial landmark providers and Gaussian heatmaps (report Section 4.4).

The landmark detector is only a *pre-processing* step: it is run offline and is never
trained with the network. Backends (selected with --landmarks):

  mediapipe : MediaPipe Face Mesh (true landmarks; `pip install mediapipe`). Works with both
              the legacy `mp.solutions` API and the newer Tasks API (which downloads the
              small `face_landmarker.task` model once into data/models/).
  cascade   : face box from an OpenCV Haar cascade (or, if the OpenCV build ships no
              cascade files, scikit-image's bundled LBP cascade) + a canonical 68-point
              template placed in the box. This is an approximation (box-derived
              pseudo-landmarks), needs no download and always works. `haar` is an alias.
  oracle    : landmarks are supplied by the caller (used for the synthetic demo faces,
              where the exact landmarks are known by construction).
  auto      : mediapipe if it imports and passes a self-test, otherwise cascade.

Every backend returns the same thing: {"points": (N,2) float32 [x,y] in pixels,
"box": (x0,y0,x1,y1)} or None when no face is found.
"""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# canonical 68-point template in unit-box coordinates
# --------------------------------------------------------------------------- #
def _ellipse(cx, cy, rx, ry, n, a0=0.0, a1=2 * np.pi, endpoint=False):
    t = np.linspace(a0, a1, n, endpoint=endpoint)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)


def build_template() -> np.ndarray:
    """68 points laid out like the iBUG 68 scheme (jaw, brows, nose, eyes, mouth)."""
    th = np.pi * (1 - np.linspace(0, 1, 17))
    jaw = np.stack([0.5 + 0.5 * np.cos(th), 0.40 + 0.60 * np.sin(th)], 1)               # 17
    lb = np.stack([np.linspace(0.12, 0.42, 5), 0.30 - 0.05 * np.sin(np.linspace(0, np.pi, 5))], 1)   # 5
    rb = np.stack([np.linspace(0.58, 0.88, 5), 0.30 - 0.05 * np.sin(np.linspace(0, np.pi, 5))], 1)   # 5
    bridge = np.stack([np.full(4, 0.5), np.linspace(0.34, 0.56, 4)], 1)                  # 4
    nose = np.stack([np.linspace(0.38, 0.62, 5), 0.66 + 0.02 * np.sin(np.linspace(0, np.pi, 5))], 1)  # 5
    le = _ellipse(0.27, 0.42, 0.11, 0.04, 6)                                             # 6
    re = _ellipse(0.73, 0.42, 0.11, 0.04, 6)                                             # 6
    mo = _ellipse(0.5, 0.82, 0.20, 0.07, 12)                                             # 12
    mi = _ellipse(0.5, 0.82, 0.13, 0.03, 8)                                              # 8
    pts = np.concatenate([jaw, lb, rb, bridge, nose, le, re, mo, mi], 0).astype(np.float32)
    assert pts.shape == (68, 2), pts.shape
    return pts


TEMPLATE_68 = build_template()


def template_in_box(box, tilt_deg: float = 0.0) -> np.ndarray:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    p = (TEMPLATE_68 - 0.5)
    if tilt_deg:
        a = np.deg2rad(tilt_deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
        p = p @ R.T
    return (p * np.array([w, h], np.float32) + np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)).astype(np.float32)


def box_from_points(pts: np.ndarray, pad: float = 0.0, hw=None):
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    w, h = x1 - x0, y1 - y0
    x0, x1 = x0 - pad * w, x1 + pad * w
    y0, y1 = y0 - pad * h, y1 + pad * h
    if hw is not None:
        H, W = hw
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W - 1, x1), min(H - 1, y1)
    return (float(x0), float(y0), float(x1), float(y1))


# --------------------------------------------------------------------------- #
# heatmap
# --------------------------------------------------------------------------- #
def make_heatmap(points: Optional[np.ndarray], hw, sigma: float) -> np.ndarray:
    """H(x,y) = max_k exp(-((x-xk)^2 + (y-yk)^2) / (2 sigma^2)); zeros when points is None.

    Returns float32 (H, W). Separable and vectorised: O(K*(H+W) + K*H*W) memory-light.
    """
    H, W = hw
    out = np.zeros((H, W), np.float32)
    if points is None or len(points) == 0:
        return out
    xs = np.arange(W, dtype=np.float32)[None, :]
    ys = np.arange(H, dtype=np.float32)[None, :]
    gx = np.exp(-((xs - points[:, :1]) ** 2) / (2 * sigma ** 2))        # K,W
    gy = np.exp(-((ys - points[:, 1:2]) ** 2) / (2 * sigma ** 2))       # K,H
    # chunk over K to bound memory for 468-point meshes
    for s in range(0, len(points), 64):
        blk = gy[s:s + 64, :, None] * gx[s:s + 64, None, :]
        out = np.maximum(out, blk.max(0))
    return out


def sigma_for(size: int) -> float:
    """sigma = 5 px at 256x256 (report); scaled for other resolutions."""
    return max(1.5, 5.0 * size / 256.0)


def brighten_for_detection(low: np.ndarray, gamma: float = 0.45) -> np.ndarray:
    """Detection is run on I_low ** 0.45 (report Section 4.4); the network still gets I_low."""
    return np.power(np.clip(low, 0, 1), gamma)


# --------------------------------------------------------------------------- #
# detectors
# --------------------------------------------------------------------------- #
class _CascadeBackend:
    """OpenCV Haar cascade; falls back to scikit-image's LBP cascade if no cascade file exists."""

    def __init__(self):
        self.cascade = self.cascade_alt = self.lbp = None
        self.name = "cascade"
        try:
            base = cv2.data.haarcascades
            c1 = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
            c2 = cv2.CascadeClassifier(base + "haarcascade_frontalface_alt2.xml")
            if not c1.empty():
                self.cascade = c1
                self.cascade_alt = c2 if not c2.empty() else None
                self.name = "cascade(opencv-haar)"
        except Exception:                                           # noqa
            pass
        if self.cascade is None:
            from skimage import data as skdata
            from skimage.feature import Cascade
            self.lbp = Cascade(skdata.lbp_frontal_face_cascade_filename())
            self.name = "cascade(skimage-lbp)"

    def _boxes_lbp(self, gray: np.ndarray, min_side: int):
        h, w = gray.shape
        scale = 1.0
        if max(h, w) > 400:                                          # keep the pure-python-ish cascade fast
            scale = 400.0 / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        hh, ww = gray.shape
        ms = max(20, int(min_side * scale))
        mx = min(hh, ww)
        if mx <= ms:
            return []
        det = self.lbp.detect_multi_scale(img=gray, scale_factor=1.2, step_ratio=1,
                                          min_size=(ms, ms), max_size=(mx, mx))
        out = []
        for d in det:
            out.append((d["c"] / scale, d["r"] / scale, d["width"] / scale, d["height"] / scale))
        return out

    def detect(self, rgb_u8: np.ndarray) -> Optional[Dict]:
        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)                                # helps on dark inputs
        h, w = gray.shape
        min_side = max(20, int(0.18 * min(h, w)))
        if self.cascade is not None:
            boxes = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4,
                                                  minSize=(min_side, min_side))
            if len(boxes) == 0 and self.cascade_alt is not None:
                boxes = self.cascade_alt.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,
                                                          minSize=(min_side, min_side))
        else:
            boxes = self._boxes_lbp(gray, min_side)
        if len(boxes) == 0:
            return None
        x, y, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
        box = (float(x), float(y), float(x + bw), float(y + bh))
        return {"points": template_in_box(box), "box": box}


MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")


def _subset_indices(fm_module=None) -> np.ndarray:
    """Indices of the contour/eye/brow/lip/nose points of the 468-point Face Mesh."""
    idx = set()
    if fm_module is not None:
        for name in ("FACEMESH_FACE_OVAL", "FACEMESH_LEFT_EYE", "FACEMESH_RIGHT_EYE",
                     "FACEMESH_LEFT_EYEBROW", "FACEMESH_RIGHT_EYEBROW", "FACEMESH_LIPS",
                     "FACEMESH_NOSE"):
            for a, b in getattr(fm_module, name, []):
                idx.add(a)
                idx.add(b)
    return np.array(sorted(idx), dtype=np.int64) if idx else np.arange(0, 468, 4, dtype=np.int64)


class _MediaPipeBackend:
    """MediaPipe Face Mesh via the legacy `solutions` API or the newer Tasks API."""
    name = "mediapipe"

    def __init__(self, model_dir: str = "data/models"):
        import mediapipe as mp                                       # may raise ImportError
        self.api = None
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            fm = mp.solutions.face_mesh
            self.mesh = fm.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False,
                                    min_detection_confidence=0.3)
            self.idx = _subset_indices(fm)
            self.api = "solutions"
            self.mesh.process(np.zeros((64, 64, 3), np.uint8))       # self-test
        else:
            from pathlib import Path
            import urllib.request
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            model = Path(model_dir) / "face_landmarker.task"
            if not model.exists():
                model.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(MODEL_URL, str(model))     # one-off ~4 MB download
            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model)),
                running_mode=vision.RunningMode.IMAGE, num_faces=1,
                min_face_detection_confidence=0.3, min_face_presence_confidence=0.3)
            self.task = vision.FaceLandmarker.create_from_options(opts)
            self.mp = mp
            self.idx = _subset_indices(None)
            self.api = "tasks"
            self.task.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                      data=np.zeros((64, 64, 3), np.uint8)))   # self-test
        self.name = f"mediapipe({self.api})"

    def detect(self, rgb_u8: np.ndarray) -> Optional[Dict]:
        h, w = rgb_u8.shape[:2]
        rgb = np.ascontiguousarray(rgb_u8)
        if self.api == "solutions":
            res = self.mesh.process(rgb)
            if not res.multi_face_landmarks:
                return None
            lm = res.multi_face_landmarks[0].landmark
        else:
            res = self.task.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
            if not res.face_landmarks:
                return None
            lm = res.face_landmarks[0]
        allp = np.array([[p.x * w, p.y * h] for p in lm], np.float32)
        pts = allp[self.idx[self.idx < len(allp)]]
        return {"points": pts.astype(np.float32), "box": box_from_points(allp, hw=(h, w))}


class _OracleBackend:
    name = "oracle"

    def detect(self, rgb_u8):                                        # pragma: no cover
        raise RuntimeError("The oracle backend does not detect; landmarks come from the data source.")


class FaceLandmarker:
    """Unified landmark / face-detection interface used everywhere in the project."""

    def __init__(self, backend: str = "auto", verbose: bool = True, model_dir: str = "data/models"):
        backend = (backend or "auto").lower()
        if backend == "haar":
            backend = "cascade"
        self.impl = None
        self.note = ""
        if backend in ("auto", "mediapipe"):
            try:
                self.impl = _MediaPipeBackend(model_dir)
            except Exception as e:                                    # noqa
                self.note = f"MediaPipe unavailable ({type(e).__name__}: {str(e)[:70]})"
                if backend == "mediapipe":
                    raise RuntimeError(self.note + ". See README (optional MediaPipe setup).") from e
        if self.impl is None and backend in ("auto", "cascade"):
            self.impl = _CascadeBackend()
        if backend == "oracle":
            self.impl = _OracleBackend()
        if self.impl is None:
            raise ValueError(f"Unknown landmark backend '{backend}' (use auto|mediapipe|cascade|oracle)")
        self.name = self.impl.name
        if verbose:
            extra = (f"  [{self.note}; using face-box + 68-point template instead]" if self.note else "")
            print(f"[landmarks] backend = {self.name}{extra}")

    @property
    def is_oracle(self) -> bool:
        return self.impl.name == "oracle"

    @property
    def is_template(self) -> bool:
        """True when the points are box-derived pseudo-landmarks (not a real landmark model)."""
        return self.name.startswith("cascade")

    def detect(self, rgb_u8: np.ndarray) -> Optional[Dict]:
        if self.is_oracle:
            return None
        try:
            return self.impl.detect(rgb_u8)
        except Exception:                                             # a failing detector must never crash a run
            return None

    def detect_float(self, img: np.ndarray, brighten: bool = False) -> Optional[Dict]:
        x = brighten_for_detection(img) if brighten else img
        return self.detect((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))
