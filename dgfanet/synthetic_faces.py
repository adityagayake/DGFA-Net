"""Procedurally generated cartoon-style faces with *exact* 68-point landmarks.

Purpose: an offline fallback so that the whole pipeline (training, baselines, ablations,
figures) runs end-to-end with no internet and no Kaggle account. The faces are NOT
photographs; numbers obtained on them are only a functional test of the pipeline. Use the
Kaggle datasets (see README) for meaningful results.

The faces cover a range of skin tones (Fitzpatrick-like I-VI), hair, tilt, size and
position, so that the ITA skin-tone fairness analysis has something to work on.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .landmarks import template_in_box

# approximate skin colours (RGB 0-255) from light to dark
SKIN_TONES = np.array([
    (255, 224, 200), (240, 200, 170), (225, 176, 135),
    (198, 140, 100), (150, 98, 66), (105, 68, 46), (72, 46, 32)], np.float32)
HAIR_TONES = np.array([
    (20, 14, 10), (45, 30, 20), (90, 60, 35), (150, 110, 60), (200, 170, 110), (120, 120, 120)], np.float32)
CLOTH_TONES = np.array([
    (180, 60, 60), (60, 100, 170), (70, 140, 90), (200, 170, 60), (110, 70, 140), (230, 230, 230), (50, 50, 55)],
    np.float32)


def _c(color):
    return tuple(float(v) for v in color)


def _poly(img, pts, color, ss):
    cv2.fillPoly(img, [np.round(pts * ss).astype(np.int32)], _c(color), lineType=cv2.LINE_AA)


def _line(img, pts, color, thick, ss, closed=False):
    cv2.polylines(img, [np.round(pts * ss).astype(np.int32)], closed, _c(color),
                  max(1, int(round(thick * ss))), lineType=cv2.LINE_AA)


def generate_face(size: int, rng: np.random.Generator, skin_idx: int | None = None
                  ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float], int]:
    """Returns (rgb float32 HxWx3 in [0,1], points (68,2) float32 in pixels, face box, skin index)."""
    ss = 2                                                         # supersampling for anti-aliasing
    S = size * ss
    skin_idx = int(rng.integers(0, len(SKIN_TONES))) if skin_idx is None else int(skin_idx)
    skin = SKIN_TONES[skin_idx] * rng.uniform(0.92, 1.05, 3).clip(0.85, 1.08) / 255.0
    skin = np.clip(skin, 0, 1)
    hair = HAIR_TONES[rng.integers(0, len(HAIR_TONES))] / 255.0
    cloth = CLOTH_TONES[rng.integers(0, len(CLOTH_TONES))] / 255.0

    # ---- background: soft two-colour gradient + a few blobs
    c1, c2 = rng.uniform(0.25, 0.9, 3), rng.uniform(0.25, 0.9, 3)
    yy = np.linspace(0, 1, S, dtype=np.float32)[:, None, None]
    img = (c1 * (1 - yy) + c2 * yy) * np.ones((S, S, 3), np.float32)
    img = np.ascontiguousarray(img.astype(np.float32))
    for _ in range(int(rng.integers(0, 4))):
        col = tuple(float(v) for v in rng.uniform(0.2, 0.95, 3))
        cv2.circle(img, (int(rng.uniform(0, S)), int(rng.uniform(0, S))), int(rng.uniform(0.05, 0.2) * S),
                   col, -1, lineType=cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 0.01 * S)

    # ---- face geometry
    fw = rng.uniform(0.30, 0.40) * size                           # face-box width  (px, final res)
    fh = fw * rng.uniform(1.18, 1.35)
    cx = size * 0.5 + rng.uniform(-0.06, 0.06) * size
    cy = size * 0.47 + rng.uniform(-0.05, 0.05) * size
    tilt = float(rng.uniform(-12, 12))
    box = (cx - fw / 2, cy - fh / 2, cx + fw / 2, cy + fh / 2)
    pts = template_in_box(box, tilt)                              # exact landmarks, final-res pixels
    P = pts * ss

    # neck + shoulders
    ncx = cx + rng.uniform(-0.01, 0.01) * size
    neck = np.array([[ncx - 0.22 * fw, cy + 0.2 * fh], [ncx + 0.22 * fw, cy + 0.2 * fh],
                     [ncx + 0.25 * fw, cy + 0.75 * fh], [ncx - 0.25 * fw, cy + 0.75 * fh]], np.float32)
    _poly(img, neck, tuple(skin * 0.85), ss)
    sh_y = cy + 0.62 * fh
    shoulders = np.array([[0, size], [0, sh_y + 0.1 * fh], [ncx - 0.4 * fw, sh_y], [ncx + 0.4 * fw, sh_y],
                          [size, sh_y + 0.1 * fh], [size, size]], np.float32)
    _poly(img, shoulders, tuple(cloth), ss)

    # hair (behind the head)
    hair_rx, hair_ry = 0.62 * fw, 0.62 * fh
    cv2.ellipse(img, (int(cx * ss), int((cy - 0.05 * fh) * ss)), (int(hair_rx * ss), int(hair_ry * ss)),
                tilt, 0, 360, tuple(float(v) for v in hair), -1, lineType=cv2.LINE_AA)
    if rng.random() < 0.5:                                        # long hair falling beside the face
        side = np.array([[cx - 0.62 * fw, cy], [cx - 0.55 * fw, cy + 0.6 * fh], [cx + 0.55 * fw, cy + 0.6 * fh],
                         [cx + 0.62 * fw, cy]], np.float32)
        _poly(img, side, tuple(hair * 0.9), ss)

    # face polygon: jaw (17 pts) + forehead arc
    jaw = P[:17]
    t = np.linspace(0, np.pi, 18)[1:-1]
    x0, y0, x1, y1 = [v * ss for v in box]
    mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
    ang = np.deg2rad(tilt)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    fore = np.stack([0.5 * (x1 - x0) * np.cos(t), -0.40 * (y1 - y0) * np.sin(t) - 0.10 * (y1 - y0)], 1)
    fore = fore @ R.T + mid + np.array([0, -0.0], np.float32)
    # forehead arc spans from the right jaw end to the left jaw end
    fore = fore[::1]
    face = np.concatenate([jaw, fore], 0) / ss
    _poly(img, face, tuple(skin), ss)

    # shading: smooth directional light (cheek / forehead highlights) restricted to the face mask
    mask = np.zeros((S, S), np.uint8)
    cv2.fillPoly(mask, [np.round(face * ss).astype(np.int32)], 255, lineType=cv2.LINE_AA)
    gx = np.linspace(-1, 1, S, dtype=np.float32)[None, :]
    light = rng.uniform(-0.12, 0.12) * gx + rng.uniform(-0.06, 0.06)
    img = img + (mask[..., None] / 255.0) * light[..., None]

    # fringe / hairline
    fr = np.array([[cx - 0.55 * fw, cy - 0.20 * fh], [cx - 0.3 * fw, cy - 0.40 * fh],
                   [cx + 0.3 * fw, cy - 0.40 * fh], [cx + 0.55 * fw, cy - 0.20 * fh],
                   [cx + 0.3 * fw, cy - 0.28 * fh], [cx - 0.3 * fw, cy - 0.28 * fh]], np.float32)
    if rng.random() < 0.7:
        _poly(img, fr, tuple(hair), ss)

    # ---- facial features (drawn exactly at the landmark positions)
    dark = tuple(float(v) for v in np.clip(skin * 0.35, 0, 1))
    brow_c = tuple(float(v) for v in np.clip(hair * 0.8, 0, 0.5))
    lip = tuple(float(v) for v in np.clip(skin * np.array([0.85, 0.55, 0.55]) + 0.04, 0, 1))
    for sl in (slice(17, 22), slice(22, 27)):
        _line(img, P[sl] / ss, brow_c, 0.022 * fw, ss)
    for sl in (slice(36, 42), slice(42, 48)):
        eye = P[sl] / ss
        _poly(img, eye, (0.95, 0.95, 0.95), ss)
        c = eye.mean(0)
        cv2.circle(img, (int(c[0] * ss), int(c[1] * ss)), int(0.055 * fw * ss),
                   (0.12, 0.08, 0.05), -1, lineType=cv2.LINE_AA)
        _line(img, eye, dark, 0.012 * fw, ss, closed=True)
    _line(img, P[27:31] / ss, tuple(float(v) for v in np.clip(skin * 0.8, 0, 1)), 0.016 * fw, ss)
    _line(img, P[31:36] / ss, dark, 0.016 * fw, ss)
    _poly(img, P[48:60] / ss, lip, ss)
    _poly(img, P[60:68] / ss, (0.18, 0.05, 0.06), ss)
    _line(img, P[48:60] / ss, tuple(float(v) for v in np.clip(skin * 0.45, 0, 1)), 0.008 * fw, ss, closed=True)

    img = np.clip(img, 0, 1)
    out = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    # mild skin / sensor texture so that images are not perfectly flat
    out = out + rng.normal(0, 0.004, out.shape).astype(np.float32)
    return np.clip(out, 0, 1).astype(np.float32), pts.astype(np.float32), tuple(float(v) for v in box), skin_idx


def generate_faces(n: int, size: int = 256, seed: int = 0):
    """Deterministic list of n faces: (imgs uint8 NxHxWx3, pts NxKx2, boxes Nx4, skin idx N)."""
    rng = np.random.default_rng(seed)
    imgs, pts, boxes, sk = [], [], [], []
    for i in range(n):
        im, p, b, s = generate_face(size, rng, skin_idx=i % len(SKIN_TONES) if i < 7 * 2 else None)
        imgs.append((im * 255 + 0.5).astype(np.uint8))
        pts.append(p)
        boxes.append(b)
        sk.append(s)
    return np.stack(imgs), np.stack(pts), np.array(boxes, np.float32), np.array(sk)


def generate_scene(size: int, rng: np.random.Generator) -> np.ndarray:
    """A face-free procedural scene (gradient + shapes) used to test that the presence gate
    leaves non-face images alone (hypothesis H2). float32 HxWx3 in [0,1]."""
    S = size
    c1, c2 = rng.uniform(0.2, 0.9, 3), rng.uniform(0.2, 0.9, 3)
    yy = np.linspace(0, 1, S, dtype=np.float32)[:, None, None]
    img = np.ascontiguousarray(((c1 * (1 - yy) + c2 * yy) * np.ones((S, S, 3))).astype(np.float32))
    for _ in range(int(rng.integers(4, 10))):
        col = tuple(float(v) for v in rng.uniform(0.05, 1.0, 3))
        kind = int(rng.integers(0, 3))
        x, y = int(rng.uniform(0, S)), int(rng.uniform(0, S))
        if kind == 0:
            cv2.circle(img, (x, y), int(rng.uniform(0.04, 0.2) * S), col, -1, lineType=cv2.LINE_AA)
        elif kind == 1:
            w, h = int(rng.uniform(0.05, 0.3) * S), int(rng.uniform(0.05, 0.3) * S)
            cv2.rectangle(img, (x, y), (min(S - 1, x + w), min(S - 1, y + h)), col, -1, lineType=cv2.LINE_AA)
        else:
            cv2.line(img, (x, y), (int(rng.uniform(0, S)), int(rng.uniform(0, S))), col,
                     int(rng.uniform(1, 6)), lineType=cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    return np.clip(img + rng.normal(0, 0.004, img.shape), 0, 1).astype(np.float32)


def generate_scenes(n: int, size: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([(generate_scene(size, rng) * 255 + 0.5).astype(np.uint8) for _ in range(n)])
