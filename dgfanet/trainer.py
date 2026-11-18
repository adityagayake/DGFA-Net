"""Training loops for DGFA-Net variants (supervised) and the Zero-DCE baseline (unsupervised)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baselines import ZeroDCENet, zero_dce_losses
from .losses import DGFALoss
from .model import VARIANTS, DGFANet, build_model
from .utils import count_params, ensure_dir, save_csv, sync


def _psnr_batch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = ((pred.clamp(0, 1) - gt) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return float((10 * torch.log10(1.0 / mse)).mean())


@torch.no_grad()
def validate(model: DGFANet, val_ds, device, bs: int = 8) -> float:
    model.eval()
    tot, n = 0.0, 0
    for b in DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0):
        low, gt, heat, flag = (b[k].to(device) for k in ("low", "gt", "heat", "flag"))
        out, _ = model(low, heat, flag)
        tot += _psnr_batch(out, gt) * low.shape[0]
        n += low.shape[0]
    return tot / max(1, n)


def train_variant(variant: str, train_ds, val_ds, device: torch.device, out_dir, epochs: int, batch_size: int,
                  lr: float = 2e-4, width: int = 32, lam=(0.2, 0.5, 0.01), force: bool = False,
                  log: Callable = print) -> DGFANet:
    """Trains (or loads) a DGFA-Net variant. Best-validation weights are returned."""
    out_dir = ensure_dir(out_dir)
    ck = out_dir / f"{variant}.pt"
    model = build_model(variant, width).to(device)
    if ck.exists() and not force:
        try:
            state = torch.load(ck, map_location=device)
            if state.get("epochs") == epochs and state.get("n_train") == len(train_ds):
                model.load_state_dict(state["model"])
                log(f"  [{variant}] loaded trained weights from {ck} (use --retrain to retrain)")
                return model
        except Exception:
            pass
    spec = VARIANTS[variant]
    crit = DGFALoss(*lam, use_face_loss=spec["face_loss"])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=len(train_ds) > batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(loader)))
    best, best_state, hist = -1.0, None, []
    t0 = time.time()
    log(f"  [{variant}] {spec['desc']} | {count_params(model) / 1e6:.3f} M params | {epochs} epochs x {len(loader)} steps")
    for ep in range(1, epochs + 1):
        model.train()
        run, nb = 0.0, 0
        for b in loader:
            low, gt, heat, flag = (b[k].to(device) for k in ("low", "gt", "heat", "flag"))
            _, aux = model(low, heat, flag)
            loss, parts = crit(aux, gt, heat, flag)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            run += float(loss.detach())
            nb += 1
        vp = validate(model, val_ds, device)
        hist.append((ep, run / max(1, nb), vp))
        if vp > best:
            best = vp
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep == epochs or ep % max(1, epochs // 10) == 0:
            log(f"    epoch {ep:3d}/{epochs}  train loss {run / max(1, nb):.4f}  val PSNR {vp:.2f} dB  "
                f"[{(time.time() - t0) / 60:.1f} min]")
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "epochs": epochs, "n_train": len(train_ds), "variant": variant,
                "best_val_psnr": best}, ck)
    save_csv(out_dir / f"{variant}_history.csv", ["epoch", "train_loss", "val_psnr"], hist)
    log(f"  [{variant}] best val PSNR {best:.2f} dB -> {ck.name}")
    return model


def train_zerodce(train_ds, device: torch.device, out_dir, epochs: int, batch_size: int, lr: float = 1e-4,
                  force: bool = False, log: Callable = print) -> ZeroDCENet:
    out_dir = ensure_dir(out_dir)
    ck = out_dir / "ZeroDCE.pt"
    net = ZeroDCENet().to(device)
    if ck.exists() and not force:
        try:
            state = torch.load(ck, map_location=device)
            if state.get("epochs") == epochs and state.get("n_train") == len(train_ds):
                net.load_state_dict(state["model"])
                log(f"  [Zero-DCE] loaded trained weights from {ck}")
                return net
        except Exception:
            pass
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=len(train_ds) > batch_size)
    log(f"  [Zero-DCE] unsupervised (no ground truth) | {count_params(net) / 1e6:.3f} M params | {epochs} epochs")
    t0 = time.time()
    hist = []
    for ep in range(1, epochs + 1):
        net.train()
        run, nb = 0.0, 0
        for b in loader:
            low = b["low"].to(device)
            enh, curves = net(low)
            loss, _ = zero_dce_losses(low, enh, curves)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run += float(loss.detach())
            nb += 1
        hist.append((ep, run / max(1, nb)))
        if ep == 1 or ep == epochs or ep % max(1, epochs // 10) == 0:
            log(f"    epoch {ep:3d}/{epochs}  loss {run / max(1, nb):.4f}  [{(time.time() - t0) / 60:.1f} min]")
    torch.save({"model": net.state_dict(), "epochs": epochs, "n_train": len(train_ds)}, ck)
    save_csv(out_dir / "ZeroDCE_history.csv", ["epoch", "loss"], hist)
    return net


def load_variant(variant: str, ckpt_dir, device: torch.device, width: int = 32) -> Optional[DGFANet]:
    ck = Path(ckpt_dir) / f"{variant}.pt"
    if not ck.exists():
        return None
    m = build_model(variant, width).to(device)
    m.load_state_dict(torch.load(ck, map_location=device)["model"])
    return m.eval()


def load_zerodce(ckpt_dir, device: torch.device) -> Optional[ZeroDCENet]:
    ck = Path(ckpt_dir) / "ZeroDCE.pt"
    if not ck.exists():
        return None
    n = ZeroDCENet().to(device)
    n.load_state_dict(torch.load(ck, map_location=device)["model"])
    return n.eval()
