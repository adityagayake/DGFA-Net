"""End-to-end pipeline: data -> baselines -> DGFA-Net -> ablations -> fairness -> figures -> tables."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from . import viz
from .baselines import CLAHE, External, Gamma, Identity, TorchEnhancer, ZeroDCE
from .config import PRESETS
from .data import (EvalSet, FaceData, PairedDataset, TrainSet, build_face_data, find_images, find_paired,
                   make_scenes, split_indices)
from .evaluation import bootstrap_ci, evaluate, paired_stats, profile, split_masks, summarize
from .landmarks import FaceLandmarker, make_heatmap, sigma_for
from .metrics import FaceAnalyzer, LPIPSMeter
from .model import VARIANTS
from .trainer import load_variant, load_zerodce, train_variant, train_zerodce
from .utils import (banner, ensure_dir, fmt, get_device, markdown_table, save_csv, save_json, set_seed,
                    text_table, to_numpy, to_tensor, Stopwatch)

ABL_NAMES = {"A0": "Residual U-Net", "A1": "+ degradation heads", "A2": "+ attention, no gate",
             "A3": "+ gate (full DGFA-Net)", "A4": "fixed alpha", "A5": "detect on raw dark image",
             "A6": "no L_face"}


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #
def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    auto = "quick" if torch.cuda.is_available() else "cpu"
    p.add_argument("--preset", default=auto, choices=list(PRESETS),
                   help="smoke | cpu | quick | full  (default: %(default)s = quick on GPU, cpu otherwise)")
    p.add_argument("--data", default="auto",
                   help="auto | synthetic | <folder of face images>. auto = data/faces if it exists, else synthetic")
    p.add_argument("--lol-dir", default="auto", help="folder with LOL-style low/high pairs (auto = look in data/lol)")
    p.add_argument("--landmarks", default="auto", choices=["auto", "mediapipe", "cascade", "haar"])
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--out", default="outputs", help="output root folder")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=None, help="override image size")
    p.add_argument("--n-images", type=int, default=None, help="override number of clean images")
    p.add_argument("--epochs", type=int, default=None, help="override epochs (all models)")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--retrain", action="store_true", help="ignore saved weights and train again")
    p.add_argument("--no-lpips", action="store_true", help="skip LPIPS")
    p.add_argument("--skip-zerodce", action="store_true")
    return p


def resolve_config(args) -> Dict:
    c = dict(PRESETS[args.preset])
    for k_arg, k in (("size", "size"), ("n_images", "n_images"), ("epochs", "epochs"),
                     ("batch", "batch")):
        v = getattr(args, k_arg, None)
        if v:
            c[k] = v
    if c["size"] % 4:
        raise SystemExit("--size must be a multiple of 4")
    return c


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #
class Run:
    pass


def setup(args, need_lpips: bool = True) -> Run:
    R = Run()
    R.args, R.cfg = args, resolve_config(args)
    set_seed(args.seed)
    R.device = get_device(args.device)
    R.sw = Stopwatch()
    src_tag = "synthetic"
    data_arg = args.data
    if data_arg == "auto":
        data_arg = "data/faces" if (Path("data/faces").exists() and find_images("data/faces", 1)) else "synthetic"
    elif data_arg != "synthetic" and not Path(data_arg).exists():
        raise SystemExit(f"--data folder not found: {data_arg}\n"
                         "Download a dataset with  python download_datasets.py  or use --data synthetic")
    if data_arg != "synthetic":
        src_tag = Path(data_arg).resolve().name
    R.run_dir = ensure_dir(Path(args.out) / f"{args.preset}_{src_tag}_{R.cfg['size']}px")
    R.ckpt = ensure_dir(R.run_dir / "checkpoints")
    R.fig = ensure_dir(R.run_dir / "figures")
    R.tab = ensure_dir(R.run_dir / "tables")
    R._log = open(R.run_dir / "log.txt", "a", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        R._log.write(str(msg) + "\n")
        R._log.flush()

    R.log = log
    banner(f"DGFA-Net | preset={args.preset} | data={src_tag} | device={R.device} | {R.cfg['size']} px")
    R.log(f"  {PRESETS[args.preset]['desc']}")
    R.log(f"  outputs -> {R.run_dir}")

    R.lm = FaceLandmarker(args.landmarks, model_dir="data/models")
    R.data = build_face_data(data_arg, R.cfg["size"], R.cfg["n_images"], R.lm, cache_dir="data/cache", seed=args.seed)
    R.az = FaceAnalyzer(R.lm, simulate=R.data.simulate)
    if R.data.simulate:
        R.log("  NOTE: procedural faces -> landmark detection uses a simulated proxy detector (see README).")
    elif R.lm.is_template:
        R.log("  NOTE: landmark backend is the face-box template (approximate landmarks); install mediapipe for real ones.")
    R.split = split_indices(len(R.data), args.seed)
    R.log(f"  images: {len(R.data)} (train {len(R.split['train'])} / val {len(R.split['val'])} / test {len(R.split['test'])})")
    R.train_ds = TrainSet(R.data, R.split["train"], seed=args.seed)
    val_idx = R.split["val"][:64]
    R.val_ds = EvalSet(R.data, val_idx, R.az, "brighten", seed=999)
    scenes = make_scenes(R.cfg["n_scenes"], R.cfg["size"], seed=args.seed + 7)
    R.test = EvalSet(R.data, R.split["test"], R.az, "brighten", seed=123, scenes=scenes)
    R.n_faces = sum(m["kind"] == "face" for m in R.test.meta)
    R.log(f"  test set: {R.n_faces} face images + {len(R.test) - R.n_faces} face-free scenes "
          f"(low-light synthesised with fixed seeds)")
    R.lp = LPIPSMeter(not args.no_lpips and need_lpips, "cpu")
    R.log(f"  LPIPS: {R.lp.status}")
    # optional real paired low-light data
    R.lol = None
    lol_dir = args.lol_dir
    if lol_dir == "auto":
        lol_dir = "data/lol" if Path("data/lol").exists() else None
    if lol_dir and Path(lol_dir).exists():
        pairs = find_paired(lol_dir)
        if pairs:
            R.lol = PairedDataset(pairs, R.cfg["size"], R.lm, limit=60)
            R.log(f"  real low-light pairs found: {len(R.lol)} (from {lol_dir})")
        else:
            R.log(f"  [warn] no low/high pairs found under {lol_dir}")
    return R


# --------------------------------------------------------------------------- #
# model access
# --------------------------------------------------------------------------- #
def get_variant(R: Run, name: str, epochs: int, eval_only: bool = False):
    if eval_only:
        m = load_variant(name, R.ckpt, R.device)
        if m is None:
            raise SystemExit(f"No trained weights for {name} in {R.ckpt}. Run  python train.py  first "
                             f"(with the same --preset/--data).")
        return m
    return train_variant(name, R.train_ds, R.val_ds, R.device, R.ckpt, epochs, R.cfg["batch"],
                         force=R.args.retrain, log=R.log)


def get_zerodce(R: Run, eval_only: bool = False):
    if eval_only:
        n = load_zerodce(R.ckpt, R.device)
        if n is None:
            raise SystemExit(f"No Zero-DCE weights in {R.ckpt}. Run  python train.py --with-baselines  first.")
        return n
    return train_zerodce(R.train_ds, R.device, R.ckpt, R.cfg["epochs"], R.cfg["batch"], force=R.args.retrain, log=R.log)


def external_methods(R: Run) -> List[External]:
    out = []
    root = Path("external")
    if root.exists():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            e = External(d.name, d, range(len(R.test)))
            if e.available():
                out.append(e)
                R.log(f"  external method found: {d.name}")
            else:
                R.log(f"  [skip] external/{d.name}: expected {len(R.test)} files named 00000.png ... (see README)")
    return out


def predict(enh, ds, idxs, device) -> Dict[int, np.ndarray]:
    out = {}
    for i in idxs:
        b = ds[i]
        low, heat, flag = (b[k][None].to(device) for k in ("low", "heat", "flag"))
        if isinstance(enh, External):
            enh.ptr = int(i)
        out[int(i)] = to_numpy(enh(low, heat, flag)[0])
    return out


def export_test_inputs(R: Run):
    import cv2
    d = ensure_dir(R.run_dir / "test_inputs")
    g = ensure_dir(R.run_dir / "test_gt")
    for i in range(len(R.test)):
        cv2.imwrite(str(d / f"{i:05d}.png"), cv2.cvtColor((R.test.lows[i] * 255 + .5).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(g / f"{i:05d}.png"), cv2.cvtColor((R.test.gts[i] * 255 + .5).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return d


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def _row(name, s, prof=None):
    return [name,
            fmt(prof["params"], 3) if prof else "-", fmt(prof["gmacs"], 2) if prof else "-",
            fmt(prof["ms"], 1) if prof else "-",
            fmt(s["psnr"], 2), fmt(s["ssim"], 3), fmt(s["lpips"], 3), fmt(s["nme"], 3),
            "n/a" if np.isnan(s["detect"]) else f"{100 * s['detect']:.0f}%"]


MAIN_HEAD = ["Method", "Params(M)", "GMACs", "ms/img", "PSNR", "SSIM", "LPIPS", "NME", "Detect"]


def show_table(R: Run, name: str, headers, rows, title: str):
    R.log("\n" + text_table(headers, rows, title))
    save_csv(R.tab / f"{name}.csv", headers, rows)
    with open(R.tab / f"{name}.md", "w", encoding="utf-8") as f:
        f.write(f"**{title}**\n\n" + markdown_table(headers, rows) + "\n")


# --------------------------------------------------------------------------- #
# the stages
# --------------------------------------------------------------------------- #
def run_comparison(R: Run, eval_only: bool = False) -> Dict:
    banner("1/4  Baselines and DGFA-Net: training", "-")
    R.log("  Training the two learned baselines and the full model on the same data ...")
    models = {}
    models["A0"] = get_variant(R, "A0", R.cfg["epochs"], eval_only)
    models["A3"] = get_variant(R, "A3", R.cfg["epochs"], eval_only)
    zd = None if R.args.skip_zerodce else get_zerodce(R, eval_only)

    methods = [Identity(), Gamma(2.2), CLAHE()]
    if zd is not None:
        methods.append(ZeroDCE(zd))
    methods.append(TorchEnhancer(models["A0"], "Residual U-Net (A0)"))
    methods += external_methods(R)
    methods.append(TorchEnhancer(models["A3"], "DGFA-Net (ours)"))

    banner("2/4  Evaluation on the held-out test faces", "-")
    results, profs = {}, {}
    for m in methods:
        results[m.name] = evaluate(m, R.test, R.az, R.device, R.lp)
        profs[m.name] = profile(m, R.device, R.cfg["size"])
    rows = []
    for m in methods:
        s = summarize(results[m.name]["per"], "face")
        rows.append(_row(m.name, s, profs[m.name]))
    show_table(R, "main_comparison", MAIN_HEAD, rows,
               f"TABLE A - Baselines vs DGFA-Net ({R.n_faces} test faces, {R.cfg['size']}px; NME/Detect = on the enhanced image)")
    R.log("  PSNR/SSIM/LPIPS: mean over test faces. NME: landmark error / sqrt(face-box area), lower is better.\n"
          "  Detect: % of faces still detected after enhancement. GMACs/ms: one 256x256-equivalent pass at the run size.")
    return dict(methods=methods, results=results, profs=profs, models=models, zd=zd)


def run_ablation(R: Run, C: Dict, eval_only: bool = False) -> Dict:
    banner("3/4  Ablation study (A0-A6) and hypothesis checks", "-")
    models = dict(C["models"])
    for v in ("A1", "A2", "A4", "A6"):
        models[v] = get_variant(R, v, R.cfg["epochs"], eval_only)
    # every variant (incl. A0 and A3 from Table A) uses the same data and the same epoch budget
    res, masks = {}, None
    for v in ("A0", "A1", "A2", "A3", "A4", "A6"):
        res[v] = evaluate(TorchEnhancer(models[v], v), R.test, R.az, R.device, R.lp)
    # A5: the A3 weights, landmarks detected on the raw dark image (inference-time change only)
    raw = R.test.with_mode("raw")
    res["A5"] = evaluate(TorchEnhancer(models["A3"], "A5"), raw, R.az, R.device, R.lp)
    detect_raw = float(np.mean(raw.flags[:R.n_faces])) if R.n_faces else float("nan")
    detect_br = float(np.mean(R.test.flags[:R.n_faces])) if R.n_faces else float("nan")

    masks = split_masks(res["A3"]["per"])
    head = ["ID", "Variant", "PSNR", "SSIM", "LPIPS", "NME", "Detect", "PSNR noisy", "PSNR clean",
            "PSNR darkest", "PSNR face-free"]
    rows, S = [], {}
    for v in ("A0", "A1", "A2", "A3", "A4", "A5", "A6"):
        per = res[v]["per"]
        s = summarize(per, "face")
        sn = summarize(per, "face", masks.get("noisy"))["psnr"] if masks else float("nan")
        sc = summarize(per, "face", masks.get("clean"))["psnr"] if masks else float("nan")
        sd = summarize(per, "face", masks.get("darkest"))["psnr"] if masks else float("nan")
        sf = summarize(per, "scene")["psnr"]
        S[v] = dict(s, noisy=sn, clean=sc, darkest=sd, scene=sf)
        rows.append([v, ABL_NAMES[v], fmt(s["psnr"], 2), fmt(s["ssim"], 3), fmt(s["lpips"], 3), fmt(s["nme"], 3),
                     "n/a" if np.isnan(s["detect"]) else f"{100 * s['detect']:.0f}%",
                     fmt(sn, 2), fmt(sc, 2), fmt(sd, 2), fmt(sf, 2)])
    show_table(R, "ablation", head, rows, "TABLE B - Ablation study (report Table 7.3)")
    R.log(f"  A5 landmark detection rate on the dark input: {100 * detect_raw:.0f}% "
          f"(vs {100 * detect_br:.0f}% on the brightened input used by A3)")

    # hypotheses (report Section 7.4)
    def verdict(ok, *vals):
        if any(v is None or np.isnan(v) for v in vals):
            return "cannot be assessed (missing data, e.g. no face landmarks)"
        return "SUPPORTED in this run" if ok else "NOT supported in this run"
    g_noisy = S["A1"]["noisy"] - S["A0"]["noisy"]
    g_clean = S["A1"]["clean"] - S["A0"]["clean"]
    h1 = g_noisy > g_clean
    h2 = S["A3"]["nme"] < S["A1"]["nme"]
    h2b = abs(S["A3"]["scene"] - S["A1"]["scene"]) < 0.3
    h3 = S["A5"]["darkest"] < S["A3"]["darkest"]
    hyp = [
        ["H1", "A1 improves over A0 mainly on noisy inputs",
         f"gain noisy {S['A1']['noisy'] - S['A0']['noisy']:+.2f} dB vs clean {S['A1']['clean'] - S['A0']['clean']:+.2f} dB",
         verdict(h1, g_noisy, g_clean)],
        ["H2", "A3 improves NME vs A1, ~no PSNR change on face-free images",
         f"NME {fmt(S['A3']['nme'])} vs {fmt(S['A1']['nme'])}; face-free PSNR {S['A3']['scene'] - S['A1']['scene']:+.2f} dB",
         verdict(h2 and h2b, S["A3"]["nme"], S["A1"]["nme"], S["A3"]["scene"], S["A1"]["scene"])],
        ["H3", "A5 worse than A3 on the darkest samples",
         f"darkest-third PSNR {fmt(S['A5']['darkest'], 2)} vs {fmt(S['A3']['darkest'], 2)}", verdict(h3, S["A5"]["darkest"], S["A3"]["darkest"])]]
    show_table(R, "hypotheses", ["ID", "Hypothesis", "Observed", "Verdict"], hyp,
               "TABLE C - Hypothesis checks (report Section 7.4)")
    R.log(f"  Caution: with only {R.n_faces} test faces (and short training) differences of a few tenths of a dB are\n"
          "  within run-to-run noise. Treat verdicts from small presets as a functional check, not as evidence.")
    return dict(res=res, S=S, models=models, hyp=hyp, detect_raw=detect_raw, detect_br=detect_br)


def run_fairness(R: Run, C: Dict, A: Optional[Dict]) -> Dict:
    groups = ["light", "intermediate", "dark"]
    names = ["Gamma (1/2.2)", "Residual U-Net (A0)", "DGFA-Net (ours)"]
    out = {}
    rows = []
    for n in names:
        per = C["results"][n]["per"]
        vals = [summarize(per, "face", lambda r, g=g: r["group"] == g) for g in groups]
        out[n] = vals
        gap = np.nanmax([v["psnr"] for v in vals]) - np.nanmin([v["psnr"] for v in vals]) if any(v["n"] for v in vals) else float("nan")
        rows.append([n] + [f"{fmt(v['psnr'], 2)} (n={v['n']})" for v in vals] + [fmt(gap, 2)])
    show_table(R, "fairness", ["Method"] + [f"PSNR {g}" for g in groups] + ["gap (dB)"], rows,
               "TABLE D - Skin-tone fairness: PSNR by ITA group (smaller gap = fairer)")
    return out


def run_stats(R: Run, C: Dict) -> Dict:
    """Mean +- 95% bootstrap CI per method and paired comparison of DGFA-Net against every other method."""
    names = [m.name for m in C["methods"]]
    ours = "DGFA-Net (ours)"
    face = lambda n, k: [r[k] for r in C["results"][n]["per"] if r["kind"] == "face"]
    rows, out = [], {}
    for n in names:
        pm, plo, phi = bootstrap_ci(face(n, "psnr"))
        sm, slo, shi = bootstrap_ci(face(n, "ssim"))
        row = [n, f"{pm:.2f} [{plo:.2f}, {phi:.2f}]", f"{sm:.3f} [{slo:.3f}, {shi:.3f}]"]
        if n == ours:
            row += ["-", "-", "-"]
        else:
            d = paired_stats(face(ours, "psnr"), face(n, "psnr"))
            row += [f"{d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]", f"{100 * d['win']:.0f}%",
                    "n/a" if np.isnan(d["p"]) else (f"{d['p']:.1e}" if d["p"] < 1e-3 else f"{d['p']:.3f}")]
            out[n] = d
        rows.append(row)
    show_table(R, "statistics", ["Method", "PSNR mean [95% CI]", "SSIM mean [95% CI]",
                                 "DGFA-Net minus method: dPSNR [95% CI]", "DGFA wins", "Wilcoxon p"], rows,
               f"TABLE F - Statistics over {R.n_faces} test faces (bootstrap CI; paired comparison vs DGFA-Net)")
    R.log("  dPSNR > 0 means DGFA-Net is better; the CI excludes 0 when the difference is reliable (p < 0.05 = significant).")
    return out


def run_real(R: Run, C: Dict) -> Optional[Dict]:
    if R.lol is None:
        return None
    banner("Real low-light pairs (generalisation check, mostly non-face scenes)", "-")
    az_real = FaceAnalyzer(R.lm, simulate=False)
    rows, res = [], {}
    for m in C["methods"]:
        if isinstance(m, External):
            continue
        res[m.name] = evaluate(m, R.lol, az_real, R.device, None)
        s = summarize(res[m.name]["per"])
        rows.append([m.name, fmt(s["psnr"], 2), fmt(s["ssim"], 3)])
    show_table(R, "real_lowlight", ["Method", "PSNR", "SSIM"], rows,
               f"TABLE E - Real paired low-light images ({len(R.lol)} pairs; models trained on this run's face data only)")
    return res


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(R: Run, C: Dict, A: Optional[Dict], F: Optional[Dict], real: Optional[Dict]) -> List[str]:
    banner("4/4  Figures", "-")
    out: List[str] = []
    size, dev = R.cfg["size"], R.device
    faces = [i for i, m in enumerate(R.test.meta) if m["kind"] == "face"]
    scenes = [i for i, m in enumerate(R.test.meta) if m["kind"] == "scene"]
    m_by = {m.name: m for m in C["methods"]}
    dgfa, a0 = m_by["DGFA-Net (ours)"], m_by["Residual U-Net (A0)"]

    # --- dataset / degradation / landmark figures
    k = min(16, len(R.data))
    sel = np.linspace(0, len(R.data) - 1, k).astype(int)
    rows = [[R.data.imgs[j].astype(np.float32) / 255 for j in sel[r * 8:(r + 1) * 8]] for r in range(k // 8)]
    if rows and len(rows[0]) == 8:
        out.append(viz.image_grid([f"#{j}" for j in sel[:8]], rows, R.fig / "01_dataset_samples.png",
                                  title="Clean training images (ground truth)", cell=1.3))
    from .degradation import synthesize_low_light
    base = R.data.imgs[int(R.split["test"][0])].astype(np.float32) / 255
    betas, gams = (0.2, 0.4, 0.6, 0.8), (1.5, 2.25, 3.0)
    rows, caps = [], []
    for g in gams:
        rr, cc = [], []
        for b in betas:
            low, _ = synthesize_low_light(base, np.random.default_rng(5), beta=b, gamma=g, peak=120, sigma=0.015)
            rr.append(low); cc.append(f"mean {low.mean():.2f}")
        rows.append(rr); caps.append(cc)
    out.append(viz.image_grid([f"beta={b}" for b in betas], rows, R.fig / "02_degradation_model.png", caps,
                              [f"gamma={g}" for g in gams], "Synthetic low-light model I_low = clip(N(beta * I^gamma))", cell=1.7))

    sel_f = faces[:: max(1, len(faces) // 5)][:5]
    rows, caps = [], []
    for i in sel_f:
        m = R.test.meta[i]
        low = R.test.lows[i]
        dark_det = R.az.detect(low, m["gt_pts"], brighten=False)
        br_det = R.az.detect(low, m["gt_pts"], brighten=True)
        rows.append([low, np.power(low, 0.45), viz.overlay_points(np.power(low, 0.45), br_det), R.test.heats[i]])
        caps.append(["dark input", "I_low^0.45", "landmarks" if br_det is not None else "NO face found",
                     "heatmap fed to net" if br_det is not None else "empty heatmap (gate=0)"])
        _ = dark_det
    out.append(viz.image_grid(["input (dark)", "brightened for detection", "detected landmarks", "heatmap H"],
                              rows, R.fig / "03_landmark_detection.png", caps,
                              title="Landmark guidance: detect on the brightened copy, enhance the original", cell=1.8))

    # --- qualitative sheets
    order = sorted(faces, key=lambda i: R.test.meta[i]["ita"] if not np.isnan(R.test.meta[i]["ita"]) else 0)
    per_sheet = 6
    n_sheets = min(R.cfg["sheets"], max(1, int(np.ceil(len(order) / per_sheet))))
    want = min(len(order), n_sheets * per_sheet)
    chosen = [order[int(round(x))] for x in np.linspace(0, len(order) - 1, want)] if want else []
    cols_m = [m for m in C["methods"] if not isinstance(m, External) and m.name != "Input (dark)"]
    preds = {m.name: predict(m, R.test, chosen, dev) for m in cols_m}
    # --- three large showcase comparisons (one per skin-tone group where available)
    per_o = C["results"]["DGFA-Net (ours)"]["per"]
    pick = []
    for g in ("light", "intermediate", "dark"):
        cand = sorted([i for i in faces if R.test.meta[i]["group"] == g], key=lambda i: per_o[i]["psnr"])
        if cand:
            pick.append(cand[len(cand) // 2])                         # the median-PSNR face of the group
    for i in faces[:: max(1, len(faces) // 3)]:
        if len(pick) >= 3:
            break
        if i not in pick:
            pick.append(i)
    pick = pick[:3]
    if pick:
        pk = {m.name: predict(m, R.test, pick, dev) for m in cols_m}
        rows, caps = [], []
        for i in pick:
            r_, c_ = [R.test.lows[i]], [f"input, mean={R.test.lows[i].mean():.2f}"]
            for m in cols_m:
                q = C["results"][m.name]["per"][i]
                r_.append(pk[m.name][i]); c_.append(f"{q['psnr']:.1f} dB / SSIM {q['ssim']:.2f}")
            r_.append(R.test.gts[i]); c_.append("ground truth")
            rows.append(r_); caps.append(c_)
        out.append(viz.image_grid(["Input (dark)"] + [m.name for m in cols_m] + ["Ground truth"], rows,
                                  R.fig / "00_showcase_3_comparisons.png", caps,
                                  [R.test.meta[i]["group"] + " skin" for i in pick],
                                  title="Three test faces: baselines vs DGFA-Net vs ground truth (median-PSNR face of each skin-tone group)",
                                  cell=2.3))
    for s in range(n_sheets):
        part = chosen[s * per_sheet:(s + 1) * per_sheet]
        if not part:
            continue
        rows, caps = [], []
        for i in part:
            gt = R.test.gts[i]
            r_, c_ = [R.test.lows[i]], [f"mean={R.test.lows[i].mean():.2f}"]
            for m in cols_m:
                p = preds[m.name][i]
                r_.append(p)
                c_.append(f"{C['results'][m.name]['per'][i]['psnr']:.1f} dB")
            r_.append(gt); c_.append("ground truth")
            rows.append(r_); caps.append(c_)
        names = ["Input (dark)"] + [m.name for m in cols_m] + ["Ground truth"]
        out.append(viz.image_grid(names, rows, R.fig / f"04_examples_{s + 1:02d}.png", caps,
                                  [f"{R.test.meta[i]['group']}" for i in part],
                                  title=f"Test faces, sheet {s + 1}/{n_sheets} (rows ordered light -> dark skin tone)", cell=1.55))

    # --- internal maps of DGFA-Net
    model = C["models"]["A3"]
    part = chosen[:: max(1, len(chosen) // 6)][:6] if chosen else []
    rows, caps = [], []
    model.eval()
    for i in part:
        b = R.test[i]
        low, heat, flag = (b[k_][None].to(dev) for k_ in ("low", "heat", "flag"))
        with torch.no_grad():
            out_t, aux = model(low, heat, flag)
        up = torch.nn.functional.interpolate(aux["attn"], size=low.shape[-2:], mode="bilinear", align_corners=False)
        noi = aux["noise"][0, 0].cpu().numpy()
        rows.append([R.test.lows[i], R.test.heats[i], aux["illum"][0, 0].cpu().numpy(), noi / max(noi.max(), 1e-6),
                     up[0, 0].cpu().numpy(), to_numpy(out_t[0]), R.test.gts[i]])
        caps.append(["", f"gate={int(flag.item())}", f"mean {aux['illum'].mean().item():.2f}",
                     f"mean {noi.mean():.2f}", f"mean {up.mean().item():.2f}", "", ""])
    if rows:
        out.append(viz.image_grid(["input", "landmark heatmap H", "illumination M_L", "noise M_N (rescaled)",
                                   "facial attention A_f", "DGFA-Net output", "ground truth"], rows,
                                  R.fig / "05_internal_maps.png", caps,
                                  title="Internal maps of DGFA-Net (each map auto-scaled; captions give the mean)", cell=1.55))

    # --- eyes / mouth crops
    for part_name in ("eyes", "mouth"):
        rows, caps = [], []
        for i in chosen[:: max(1, len(chosen) // 4)][:4]:
            pts = R.test.meta[i]["gt_pts"]
            if pts is None:
                continue
            ims = [R.test.lows[i], preds["Gamma (1/2.2)"][i], preds["Residual U-Net (A0)"][i], preds["DGFA-Net (ours)"][i],
                   R.test.gts[i]]
            crops = [viz.pad_to(viz.crop_by_box(im, pts, part_name), (40, 96)) for im in ims]
            rows.append(crops)
            caps.append(["", "", "", "", ""])
        if rows:
            out.append(viz.image_grid(["input", "Gamma", "Residual U-Net", "DGFA-Net", "ground truth"], rows,
                                      R.fig / f"06_crops_{part_name}.png", caps, title=f"Facial detail: {part_name}", cell=2.0))

    # --- face-free behaviour (gate)
    if scenes and A is not None:
        sc = scenes[:4]
        pa1 = predict(TorchEnhancer(A["models"]["A1"], "A1"), R.test, sc, dev)
        pa3 = predict(TorchEnhancer(A["models"]["A3"], "A3"), R.test, sc, dev)
        rows = [[R.test.lows[i], pa1[i], pa3[i], R.test.gts[i]] for i in sc]
        caps = [["", f"{A['res']['A1']['per'][i]['psnr']:.1f} dB", f"{A['res']['A3']['per'][i]['psnr']:.1f} dB", ""] for i in sc]
        out.append(viz.image_grid(["input", "A1 (no guidance)", "A3 (full)", "ground truth"], rows,
                                  R.fig / "07_face_free_scenes.png", caps,
                                  title="Images without a face: the presence gate keeps the output unchanged (H2)", cell=1.8))

    # --- failure cases
    per3 = C["results"]["DGFA-Net (ours)"]["per"]
    worst = [r["i"] for r in sorted([r for r in per3 if r["kind"] == "face"], key=lambda r: r["psnr"])[:6]]
    if worst:
        pw = predict(dgfa, R.test, worst, dev)
        rows = [[R.test.lows[i], pw[i], R.test.gts[i], R.test.heats[i]] for i in worst]
        caps = [[f"mean={R.test.lows[i].mean():.2f}", f"{per3[i]['psnr']:.1f} dB",
                 "", "gate=1" if R.test.flags[i] else "no face found"] for i in worst]
        out.append(viz.image_grid(["input", "DGFA-Net", "ground truth", "heatmap"], rows,
                                  R.fig / "08_failure_cases.png", caps,
                                  title="Lowest-PSNR test faces (failure analysis)", cell=1.7))

    # --- charts
    names = [m.name for m in C["methods"]]
    ps = [summarize(C["results"][n]["per"], "face")["psnr"] for n in names]
    out.append(viz.bar_chart(names, ps, R.fig / "09_psnr_comparison.png", "PSNR on test faces (higher is better)", "PSNR (dB)",
                             highlight=[names.index("DGFA-Net (ours)")]))
    S_ = {n: summarize(C["results"][n]["per"], "face") for n in names}
    out.append(viz.summary_panel(names, {"PSNR (dB) - higher is better": [S_[n]["psnr"] for n in names],
                                         "SSIM - higher is better": [S_[n]["ssim"] for n in names],
                                         "Landmark NME - lower is better": [S_[n]["nme"] for n in names],
                                         "Face still detected (%)": [100 * S_[n]["detect"] for n in names]},
                                 R.fig / "00_summary_panel.png", highlight=names.index("DGFA-Net (ours)")))
    ns = [summarize(C["results"][n]["per"], "face")["nme"] for n in names]
    out.append(viz.bar_chart(names, ns, R.fig / "10_nme_comparison.png", "Landmark NME after enhancement (lower is better)",
                             "NME", highlight=[names.index("DGFA-Net (ours)")], fmt="{:.3f}"))
    if A is not None:
        ids = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
        out.append(viz.bar_chart(ids, [A["S"][v]["psnr"] for v in ids], R.fig / "11_ablation_psnr.png",
                                 "Ablation: PSNR", "PSNR (dB)", highlight=[3]))
        out.append(viz.bar_chart(ids, [A["S"][v]["nme"] for v in ids], R.fig / "12_ablation_nme.png",
                                 "Ablation: landmark NME (lower is better)", "NME", highlight=[3], fmt="{:.3f}"))
    if F:
        groups = ["light", "intermediate", "dark"]
        out.append(viz.grouped_bars(groups, {n: [v["psnr"] for v in F[n]] for n in F}, R.fig / "13_skin_tone_fairness.png",
                                    "PSNR by skin-tone group (ITA)", "PSNR (dB)"))
    hist = {}
    for v in ("A0", "A1", "A2", "A3", "A4", "A6"):
        f = R.ckpt / f"{v}_history.csv"
        if f.exists():
            try:
                hist[v] = np.genfromtxt(f, delimiter=",", skip_header=1).reshape(-1, 3).tolist()
            except Exception:
                pass
    if hist:
        out.append(viz.curves(hist, R.fig / "14_training_curves.png"))

    # --- statistics charts
    face_idx = faces
    labels = [n for n in names if n != "Input (dark)"]
    out.append(viz.boxplot(labels, [[C["results"][n]["per"][i]["psnr"] for i in face_idx] for n in labels],
                           R.fig / "16_psnr_distribution.png", "Per-image PSNR distribution (box = quartiles, diamond = mean)",
                           "PSNR (dB)", highlight=labels.index("DGFA-Net (ours)")))
    a0n = "Residual U-Net (A0)"
    out.append(viz.scatter_pair([C["results"][a0n]["per"][i]["psnr"] for i in face_idx],
                                [C["results"]["DGFA-Net (ours)"]["per"][i]["psnr"] for i in face_idx],
                                [R.test.meta[i]["group"] for i in face_idx], "Residual U-Net (A0) PSNR (dB)",
                                "DGFA-Net PSNR (dB)", R.fig / "17_dgfa_vs_a0_per_image.png",
                                "Per-image PSNR: DGFA-Net vs Residual U-Net"))
    dark = np.array([R.test.meta[i]["dark"] for i in face_idx])
    edges = np.quantile(dark, [0, 1 / 3, 2 / 3, 1.0])
    bins = np.clip(np.searchsorted(edges, dark, side="right") - 1, 0, 2)
    series = {}
    for n in [m for m in names]:
        v = np.array([C["results"][n]["per"][i]["psnr"] for i in face_idx])
        series[n] = [float(v[bins == b].mean()) if np.any(bins == b) else float("nan") for b in range(3)]
    blab = [f"darkest\n(mean<{edges[1]:.2f})", f"middle\n({edges[1]:.2f}-{edges[2]:.2f})", f"brightest\n(mean>{edges[2]:.2f})"]
    out.append(viz.binned_lines(blab, series, R.fig / "18_psnr_vs_darkness.png", "PSNR by input darkness",
                                "input brightness tercile", "PSNR (dB)"))

    # --- real low-light pairs
    if R.lol is not None and len(R.lol):
        idx = list(range(min(6, len(R.lol))))
        cm = [m for m in C["methods"] if not isinstance(m, External) and m.name != "Input (dark)"]
        pr = {m.name: predict(m, R.lol, idx, dev) for m in cm}
        rows = [[R.lol.lows[i]] + [pr[m.name][i] for m in cm] + [R.lol.gts[i]] for i in idx]
        out.append(viz.image_grid(["input"] + [m.name for m in cm] + ["ground truth"], rows,
                                  R.fig / "15_real_lowlight_pairs.png", None,
                                  title="Real low-light pairs (mostly not faces; trained on this run's face data only)", cell=1.5))
    R.log(f"  wrote {len(out)} figures -> {R.fig}")
    return out


def save_all(R: Run, C: Dict, A: Optional[Dict], F, real):
    summ = {"preset": R.args.preset, "size": R.cfg["size"], "n_test_faces": R.n_faces,
            "landmark_backend": R.az.name, "device": str(R.device),
            "main": {n: dict(summarize(C["results"][n]["per"], "face"), **C["profs"][n]) for n in C["results"]}}
    if A:
        summ["ablation"] = A["S"]
        summ["hypotheses"] = A["hyp"]
    save_json(R.tab / "results.json", summ)
    save_json(R.tab / "per_sample_DGFA-Net.json", C["results"]["DGFA-Net (ours)"]["per"])
    R.log(f"  tables saved -> {R.tab}")



def write_results_md(R: Run):
    """RESULTS.md: every table + the figure list of this run, ready to paste into the report / slides."""
    parts = [f"# DGFA-Net results ({R.run_dir.name})\n",
             f"* data: `{R.data.source}` ({len(R.data)} images, {R.cfg['size']} px; test faces: {R.n_faces})\n"
             f"* landmark backend: `{R.az.name}`; device: `{R.device}`; preset: `{R.args.preset}`; seed: {R.args.seed}\n"]
    for name in ("main_comparison", "statistics", "ablation", "hypotheses", "fairness", "real_lowlight"):
        f = R.tab / f"{name}.md"
        if f.exists():
            parts.append(f.read_text(encoding="utf-8"))
    parts.append("## Figures\n")
    for f in sorted(R.fig.glob("*.png")):
        parts.append(f"![{f.stem}](figures/{f.name})\n")
    (R.run_dir / "RESULTS.md").write_text("\n".join(parts), encoding="utf-8")


def write_report(R: Run, C: Dict, A: Optional[Dict]):
    """Auto-generated REPORT.md: tables + figures + factual takeaways of THIS run (for presenting)."""
    S = {n: summarize(C["results"][n]["per"], "face") for n in C["results"]}
    ours, a0 = S["DGFA-Net (ours)"], S["Residual U-Net (A0)"]
    classical = [n for n in S if n not in ("DGFA-Net (ours)", "Residual U-Net (A0)", "Input (dark)")]
    best_classical = max(classical, key=lambda n: S[n]["psnr"]) if classical else None
    real = not R.data.simulate
    L = [f"# DGFA-Net results - {R.run_dir.name}", "",
         f"* data: **{'REAL face photographs' if real else 'procedural (synthetic) faces'}** - {len(R.data)} images, "
         f"{R.n_faces} held-out test faces, size {R.cfg['size']} px, preset `{R.args.preset}`",
         f"* landmark backend: `{R.az.name}`; device: `{R.device}`; run time {R.sw.pretty()}", "",
         "## Key numbers (test faces)", "",
         f"* DGFA-Net: PSNR **{ours['psnr']:.2f} dB**, SSIM **{ours['ssim']:.3f}**, NME {fmt(ours['nme'])}",
         f"* Residual U-Net (A0): PSNR {a0['psnr']:.2f} dB, SSIM {a0['ssim']:.3f}, NME {fmt(a0['nme'])}"]
    if best_classical:
        b = S[best_classical]
        L.append(f"* best classical/unsupervised baseline ({best_classical}): PSNR {b['psnr']:.2f} dB, SSIM {b['ssim']:.3f} "
                 f"-> DGFA-Net gain {ours['psnr'] - b['psnr']:+.2f} dB")
    L.append(f"* DGFA-Net vs Residual U-Net: {ours['psnr'] - a0['psnr']:+.2f} dB PSNR, {ours['ssim'] - a0['ssim']:+.3f} SSIM "
             "(this is the effect of the degradation heads + landmark guidance)")
    L += ["", "## Figures", "", "![summary](figures/00_summary_panel.png)", "",
          "![examples](figures/04_examples_01.png)", "", "![maps](figures/05_internal_maps.png)", ""]
    for f in ("04_examples_02.png", "04_examples_03.png", "04_examples_04.png", "06_crops_eyes.png", "06_crops_mouth.png",
              "07_face_free_scenes.png", "08_failure_cases.png", "11_ablation_psnr.png", "13_skin_tone_fairness.png",
              "14_training_curves.png", "15_real_lowlight_pairs.png"):
        if (R.fig / f).exists():
            L += [f"![{f}](figures/{f})", ""]
    L += ["## Tables", ""]
    for name in ("main_comparison", "ablation", "hypotheses", "fairness", "real_lowlight"):
        f = R.tab / f"{name}.md"
        if f.exists():
            L += [f.read_text(encoding="utf-8"), ""]
    with open(R.run_dir / "REPORT.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    R.log(f"  summary report -> {R.run_dir / 'REPORT.md'}")


def run_all(args, ablations: bool = True, figures: bool = True, eval_only: bool = False) -> Run:
    R = setup(args)
    C = run_comparison(R, eval_only)
    A = run_ablation(R, C, eval_only) if ablations else None
    F = run_fairness(R, C, A)
    run_stats(R, C)
    real = run_real(R, C)
    if figures:
        make_figures(R, C, A, F, real)
    save_all(R, C, A, F, real)
    write_report(R, C, A)
    export_test_inputs(R)
    write_results_md(R)
    banner(f"Done in {R.sw.pretty()}  |  results in {R.run_dir}")
    R.C, R.A = C, A
    return R
