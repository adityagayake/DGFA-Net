"""Small shared helpers (seeding, devices, tensor/array conversion, tables)."""
from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# reproducibility / device
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    """'auto' -> CUDA if available else CPU. 'cpu', 'cuda', 'mps' are honoured."""
    pref = (pref or "auto").lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available - falling back to CPU.")
        return torch.device("cpu")
    if pref == "mps":
        mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        if not mps_ok:
            print("[warn] MPS requested but not available - falling back to CPU.")
            return torch.device("cpu")
    return torch.device(pref)


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# array <-> tensor
# --------------------------------------------------------------------------- #
def to_tensor(img: np.ndarray) -> torch.Tensor:
    """float32 HxWx3 in [0,1]  ->  float32 3xHxW tensor."""
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """3xHxW (or 1x3xHxW) tensor -> float32 HxWx3 clipped to [0,1]."""
    if t.dim() == 4:
        t = t[0]
    return t.detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def pad_to_multiple(x: torch.Tensor, m: int = 4):
    """Reflect-pad a BxCxHxW tensor so H and W are multiples of m. Returns (x, (h, w))."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph == 0 and pw == 0:
        return x, (h, w)
    mode = "reflect" if (ph < h and pw < w) else "replicate"
    return F.pad(x, (0, pw, 0, ph), mode=mode), (h, w)


# --------------------------------------------------------------------------- #
# model statistics
# --------------------------------------------------------------------------- #
def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


@torch.no_grad()
def count_macs(model: nn.Module, *inputs) -> int:
    """Multiply-accumulates of all Conv2d layers for one forward pass (bias ignored)."""
    total = [0]
    hooks = []

    def hook(mod, inp, out):
        k = mod.kernel_size[0] * mod.kernel_size[1]
        total[0] += int(out.shape[-1] * out.shape[-2] * out.shape[1] * (mod.in_channels // mod.groups) * k) * int(out.shape[0])

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    try:
        model(*inputs)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return total[0]


# --------------------------------------------------------------------------- #
# pretty printing / saving
# --------------------------------------------------------------------------- #
def fmt(v, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    try:
        if np.isnan(v):
            return "n/a"
    except TypeError:
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.{nd}f}"


def text_table(headers: Sequence[str], rows: Iterable[Sequence], title: str = "") -> str:
    """ASCII-only table (safe for any Windows console)."""
    rows = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = []
    if title:
        out.append(title)
    out.append(sep)
    out.append("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    out.append(sep)
    for r in rows:
        out.append("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)) + " |")
    out.append(sep)
    return "\n".join(out)


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence]) -> str:
    rows = [[str(c) for c in r] for r in rows]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def save_csv(path, headers: Sequence[str], rows: Iterable[Sequence]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(headers))
        for r in rows:
            w.writerow(list(r))


def save_json(path, obj) -> None:
    ensure_dir(Path(path).parent)

    def default(o):
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=default)


def banner(msg: str, char: str = "=") -> None:
    line = char * max(8, len(msg) + 4)
    print("\n" + line)
    print(f"  {msg}")
    print(line, flush=True)


class Stopwatch:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0

    def pretty(self) -> str:
        s = int(self.elapsed())
        return f"{s // 60}m{s % 60:02d}s"


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
