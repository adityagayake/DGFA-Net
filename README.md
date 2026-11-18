# DGFA-Net: Degradation-Aware, Landmark-Guided Low-Light Face Enhancement

A complete, runnable implementation of the project report **"DGFA-Net: A Lightweight Degradation-Aware,
Landmark-Guided Network for Low-Light Face Image Enhancement"** (Computer Vision 2026-27, Netaji Subhas
University of Technology, Prof. Ankush Jain; Saloni Singh and Aditya Gayake).

One command trains the proposed network **and** the baselines on the same data, evaluates them on the same
held-out low-light faces and **prints both result tables**, plus the ablation study, hypothesis checks, a
skin-tone fairness analysis and ~15 figures of examples.

```powershell
python run_all.py --preset smoke      # 1-3 min functional check
python run_all.py --preset cpu --data data/faces --device cpu --landmarks cascade --n-images 2000 --size 64 --epochs 14 --batch 16 --seed 0 --no-lpips --out .\outputs\final_cpu_2000_64px_seed0_20261118
```

The verified real-data experiment in this repository uses 2,000 CelebA-HQ photographs, CPU PyTorch,
64x64 training inputs, 14 epochs, seed 0, and the cascade landmark backend. Its results are in
[`outputs/final_cpu_2000_64px_seed0_20261118/cpu_faces_64px/`](outputs/final_cpu_2000_64px_seed0_20261118/cpu_faces_64px/).
Step-by-step VS Code instructions and a suggested running order for presenting: **[PRESENTING.md](PRESENTING.md)**.

> **Read this first - what the numbers mean.** The repository contains a verified real-data CPU experiment
> using 2,000 CelebA-HQ photographs; its measured results are documented in [Section 9](#9-verified-real-data-experiment)
> and the linked output directory. The project can also run on **procedural cartoon faces with exact landmarks**
> as an offline functional fallback, but those results say nothing about real photos. See [Honest notes](#12-honest-notes-and-limitations)
> for every place where this implementation approximates the report.

---

## Table of contents

1. [Quick start (VS Code)](#1-quick-start-vs-code)
2. [What the program prints](#2-what-the-program-prints)
3. [The pipeline, stage by stage](#3-the-pipeline-stage-by-stage)
4. [Datasets (Kaggle) and data preparation](#4-datasets-kaggle-and-data-preparation)
5. [Presets and command reference](#5-presets-and-command-reference)
6. [Model: DGFA-Net architecture](#6-model-dgfa-net-architecture)
7. [Baselines, ablations and hypotheses](#7-baselines-ablations-and-hypotheses)
8. [Metrics](#8-metrics)
9. [Example results](#9-example-results)
10. [Output folder layout](#10-output-folder-layout)
11. [Project structure](#11-project-structure)
12. [Honest notes and limitations](#12-honest-notes-and-limitations)
13. [Troubleshooting](#13-troubleshooting)
14. [References, license, authors](#14-references-license-authors)

---

## 1. Quick start (VS Code)

Requirements: Python 3.10-3.13. A GPU is optional. This project was verified with Python 3.12.10 and
CPU-only PyTorch 2.15.0.

```powershell
# 1. open the folder in VS Code (File > Open Folder > DGFA-Net) and open a terminal (Ctrl+`)
py -3.12 -m venv .venv-dgfa
.\.venv-dgfa\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install pytest mediapipe six

# 2. check the model matches the report (0.716 M params, 12.0 GMACs at 256x256)
python dgfa_net.py

# 3. instant demo of the report's prototype vs the gamma baseline (no training, no download)
python demo_prototype.py

# 4. the whole pipeline on a tiny budget (1-3 minutes)
python run_all.py --preset smoke --device cpu --landmarks cascade --no-lpips
```

In VS Code you can also press **Run and Debug (Ctrl+Shift+D)** and pick one of the ready-made launch
configurations in `.vscode/launch.json` (sanity check, demo, smoke test, CPU run, GPU run, dataset download).
Select the `.venv` interpreter when VS Code asks (Ctrl+Shift+P > *Python: Select Interpreter*).

Results appear in `outputs/<preset>_<data>_<size>px/` (see [Section 10](#10-output-folder-layout)).

---

## 2. What the program prints

`run_all.py` prints, in this order (everything is also saved to `log.txt`, CSV and Markdown):

| Printed table | Content |
|---|---|
| **TABLE A** | **Baselines vs DGFA-Net**: Input (dark), Gamma, CLAHE, Zero-DCE (re-implementation), Residual U-Net (A0), any external method you add, and **DGFA-Net (ours)**. Columns: Params(M), GMACs, ms/img, PSNR, SSIM, LPIPS, NME, Detect. (report Table 7.2) |
| **TABLE B** | **Ablation A0-A6** with extra columns for noisy / clean / darkest-third / face-free subsets (report Table 7.3) |
| **TABLE C** | **Hypotheses H1-H3** of report Section 7.4, each marked supported / not supported / cannot be assessed |
| **TABLE D** | **Skin-tone fairness**: PSNR per ITA group (light / intermediate / dark) and the gap between groups |
| **TABLE E** | *(only if `data/lol` exists)* the same methods on real paired low-light photos (generalisation check) |

and writes these figures (in `figures/`):

| File | Shows |
|---|---|
| `01_dataset_samples.png` | clean training faces |
| `02_degradation_model.png` | the synthetic low-light model for beta in {0.2..0.8} x gamma in {1.5..3.0} |
| `03_landmark_detection.png` | detection on the brightened copy, landmarks and the heat-map fed to the network |
| `04_examples_XX.png` | **test faces, every method side by side**, with per-image PSNR, ordered light to dark skin |
| `05_internal_maps.png` | illumination map M_L, noise map M_N, facial attention A_f, output |
| `06_crops_eyes.png`, `06_crops_mouth.png` | zoomed facial detail per method |
| `07_face_free_scenes.png` | images without a face: effect of the presence gate (H2) |
| `08_failure_cases.png` | the lowest-PSNR test faces |
| `09`-`13_*.png` | PSNR / NME comparison, ablation charts, skin-tone fairness chart |
| `14_training_curves.png` | loss and validation PSNR per variant |
| `15_real_lowlight_pairs.png` | real low-light pairs (if LOL data present) |

---

## 3. The pipeline, stage by stage

```
 clean face image I_gt  (Kaggle folder or procedural face)
        |
        |  (offline, once)  landmark detection on the CLEAN image -> cached clean-GT landmarks  (data/cache/*.npz)
        v
 [1] Synthetic degradation  I_low = clip(N(beta * I_gt^gamma)),  beta~U[0.2,0.8], gamma~U[1.5,3.0]
        N = Poisson (signal-dependent) + Gaussian read noise                       dgfanet/degradation.py
        |
        +--> training:   heat-map H from the clean-GT landmarks, flag = face found
        +--> evaluation: face detector run on I_low^0.45 (brightened copy) -> H, flag;  A5: on the raw dark image
        v                                                                          dgfanet/landmarks.py
 [2] DGFA-Net(I_low, H, flag)                                                      dgfanet/model.py
        stem -> illumination head M_L (sigmoid) + noise head M_N (softplus) -> fusion (34->32 ch)
        -> encoder (256 / 128 / 64 px) -> landmark-guided facial attention at the 64x64 bottleneck
           F_guided = F3 * (1 + alpha * A_f),   A_f = sigmoid(conv1x1([F3, H])) * flag     (presence gate)
        -> decoder with skip connections -> residual output  I_enh = clamp(I_low + dI)
        v
 [3] Loss  L = L1 + 0.2 L_ssim + 0.5 L_face + 0.01 L_tv  (on the unclamped output)  dgfanet/losses.py
        L_face = sum(H * |I_enh - I_gt|) / (sum(H) + eps), zero when no face;  L_tv = mean |grad M_L|
        Adam 2e-4, cosine decay, grad-clip 5, best-validation checkpoint             dgfanet/trainer.py
        v
 [4] Baselines (same data, same test set): Gamma, CLAHE, Zero-DCE, Residual U-Net     dgfanet/baselines.py
        v
 [5] Evaluation on held-out faces + face-free scenes                                 dgfanet/evaluation.py
        PSNR, SSIM, LPIPS, landmark NME, face-detection rate, params/GMACs/ms, ITA groups   dgfanet/metrics.py
        v
 [6] Ablations A0-A6, hypotheses H1-H3, fairness table, figures, CSV/Markdown tables  dgfanet/pipeline.py, viz.py
```

Design decisions that follow the report:

* **Landmarks are offline pre-processing**, never trained with the network. Training uses the cached landmarks of
  the clean image; evaluation uses landmarks *detected on the brightened dark input* (the realistic case).
* **The network never sees beta, gamma or the noise level** - it must infer the degradation (the two heads).
* **Presence gate**: with no face found the attention is multiplied by 0, so the model reduces to the A1 network.
* **Every variant and baseline sees identical data and the same epoch budget.**

---

## 4. Datasets (Kaggle) and data preparation

The pipeline needs *well-lit* face photos; the low-light version is synthesised on the fly (report Section 3), so
any face dataset works. Recommended (all free on Kaggle):

| Purpose | Kaggle dataset to look for | Notes |
|---|---|---|
| Training / testing faces | **CelebA-HQ** (e.g. a 256x256 version) or **FFHQ 256x256** | faces, diverse skin tones; the script samples up to `--max-images` |
| Real low-light check (optional) | **LOL dataset** (Low-Light paired) | real low/high pairs, mostly non-face scenes; gives Table E |

**Automatic download**

```bash
# one-off: create a Kaggle API token (kaggle.com > Settings > API > Create New Token)
# current Kaggle CLI format: save the token at C:\Users\<you>\.kaggle\access_token
python download_datasets.py                  # faces + LOL  -> data/faces, data/lol
python download_datasets.py --faces-only --max-images 2000 --size 256
python download_datasets.py --faces-only --faces-slug badasstechie/celebahq-resized-256x256 --max-images 2000 --size 256
```

The script tries a few candidate Kaggle dataset names and reports which worked. **Kaggle dataset names change and
the defaults in `download_datasets.py` could not be verified from the build environment** - if none works,
pick any CelebA-HQ / FFHQ dataset on kaggle.com and either pass its `owner/name` with `--faces-slug`, or download
and unzip it in the browser and run

```bash
python download_datasets.py --faces-from "C:\Users\you\Downloads\celeba_hq"
python download_datasets.py --lol-from   "C:\Users\you\Downloads\lol_dataset"     # optional
```

**Use it**

```bash
python run_kaggle.py --preset cpu --data data/faces       # real-data wrapper; does not download when data/faces exists
python run_all.py --preset cpu --data data/faces          # run using an already prepared dataset
```

**Data details**

* Images are centre-cropped to a square and resized to the preset size; split 80/10/10 (train/val/test, seeded).
* Test low-light images are synthesised once with fixed seeds, so all methods see **exactly the same inputs**.
* A set of **face-free procedural scenes** is appended to the test set to check the presence gate (H2).
* **Landmark backends** (`--landmarks`): `mediapipe` (true landmarks, `pip install mediapipe`; the newer Tasks API
  downloads a 4 MB model once), `cascade` (face box from an OpenCV Haar / scikit-image LBP cascade + a canonical
  68-point template - approximate, always available), `auto` (mediapipe if it works, else cascade).
* The first run on a folder caches images + landmarks in `data/cache/`.
* **No dataset?** `--data synthetic` (the default when `data/faces` is empty) uses procedural faces.

---

## 5. Presets and command reference

| Preset | Image size | Clean images | Epochs (each model) | Intended hardware |
|---|---|---|---|---|
| `smoke` | 64 | 64 | 3 | any CPU, 1-3 min (checks only) |
| `cpu` | 64 | 400 by default | 14 | laptop CPU |
| `quick` | 128 | 3000 | 40 | consumer GPU, ~1-2 h |
| `full` | 256 | 6000 | 200 | the report protocol (256 px, 200 epochs, batch 16); hours per model, GPU required |

Default `--preset` is `quick` if CUDA is available, otherwise `cpu`. Any preset value can be overridden with
`--size`, `--n-images`, `--epochs`, `--batch`.

| Script | Purpose |
|---|---|
| `python run_kaggle.py` | **real-data project run**: Kaggle download + everything below; refuses synthetic faces |
| `python run_all.py` | everything: train, evaluate, ablations, fairness, figures, tables (`--skip-ablations`, `--no-figures`) |
| `python train.py` | train DGFA-Net (A3); `--with-baselines` adds A0 + Zero-DCE; `--variants A1 A2 A4 A6` |
| `python evaluate.py` | evaluate saved weights, print Table A (+ D, E); `--with-ablations` |
| `python run_ablations.py` | Table B and C only (trains A0-A4, A6; A5 reuses A3) |
| `python enhance_image.py --input photo.jpg` | enhance **your own** dark photos / folder with trained weights |
| `python demo_prototype.py` | the report's non-learned prototype vs gamma on a real face photo (instant) |
| `python download_datasets.py` | Kaggle download + preparation |
| `python dgfa_net.py` | print parameter / MAC counts of all variants |
| `python -m pytest -q` | unit tests |

Common options: `--data`, `--lol-dir`, `--landmarks`, `--device {auto,cpu,cuda,mps}`, `--seed`, `--out`,
`--n-images`, `--size`, `--epochs`, `--batch`, `--retrain` (ignore saved weights), `--no-lpips`,
`--skip-zerodce`, and `--skip-ablations`.
Saved weights are re-used automatically when preset/data/epochs are unchanged.

**Adding Retinex-Net, Low-FaceNet or any other method.** They need their own code/weights and are *not bundled*.
After a run, `outputs/<run>/test_inputs/00000.png ...` contains the exact low-light test inputs. Run your method on
them and save the results as `external/<MethodName>/00000.png, 00001.png, ...` (same names). The next run of
`evaluate.py`/`run_all.py` adds the method to Table A automatically.

---

## 6. Model: DGFA-Net architecture

| Stage | Layers | Output |
|---|---|---|
| Feature stem | Conv3x3(3->32), LeakyReLU, 2 x ResBlock(32) | F0, 256x256x32 |
| Degradation heads | each: Conv1x1(32->16), LeakyReLU, Conv1x1(16->1); sigmoid (M_L) / softplus (M_N) | M_L, M_N |
| Fusion | Conv1x1(34->32) on [F0, M_L, M_N] | 256x256x32 |
| Encoder | ResBlock(32) -> stride-2 conv -> ResBlock(64) -> stride-2 conv -> ResBlock(128) | F1, F2, F3 (64x64x128) |
| Facial guidance | A_f = sigmoid(Conv1x1([F3, H_64])) x flag;  F_guided = F3 x (1 + alpha A_f), alpha learnable (init 0.5) | 64x64x128 |
| Decoder | bilinear up + conv + skip (F2, F1) + ResBlock, twice | 256x256x32 |
| Output | Conv3x3(32->3) = dI;  I_enh = clamp(I_low + dI) | 256x256x3 |

`python dgfa_net.py` prints **715,912 parameters (0.716 M)** and **12.00 GMACs** per 256x256 image, matching the
report (the two heads and the attention add < 1.3 k parameters). The heads have no direct supervision; they are
trained only through the final loss (plus the TV prior on M_L), so inspect `05_internal_maps.png` before
interpreting them as "illumination" and "noise".

---

## 7. Baselines, ablations and hypotheses

**Baselines**

| Method | Type | Implementation |
|---|---|---|
| Input (dark) | lower bound | identity |
| Gamma | classical | I^(1/2.2) |
| CLAHE | classical | contrast-limited AHE on the L channel (clip 3, 8x8 tiles) |
| Zero-DCE | learned, **unsupervised** | **re-implementation** (7 conv layers, 8 curve iterations, 4 non-reference losses, 79 k params), trained on the low-light inputs only |
| Residual U-Net (A0) | learned, supervised | DGFA-Net without heads and guidance |
| Retinex-Net, Low-FaceNet | published methods | **not bundled** - plug in via `external/` (Section 5) |

**Ablations** (report Table 7.3)

| ID | Variant |
|---|---|
| A0 | Residual U-Net only |
| A1 | A0 + illumination / noise heads |
| A2 | A1 + landmark attention, **no** presence gate |
| A3 | A1 + attention + gate (**full DGFA-Net**) |
| A4 | A3 with fixed alpha (0.5) |
| A5 | A3 weights, landmarks detected on the **raw dark** image (inference-time change) |
| A6 | A3 trained **without** L_face |

**Hypotheses** (report Section 7.4), evaluated automatically in Table C:
H1 - A1 improves over A0 mainly on noisy inputs (noisy vs clean half, split by the sample's noise level);
H2 - A3 improves landmark NME over A1 with little PSNR change on face-free images (|delta| < 0.3 dB);
H3 - A5 is worse than A3 on the darkest third of the samples.
With few test images these are noisy; the program says so and prints "cannot be assessed" when data are missing.

---

## 8. Metrics

* **PSNR / SSIM** - mean over test faces against the well-lit ground truth (SSIM: 11x11 Gaussian window, sigma 1.5).
* **LPIPS** - optional (`pip install lpips`; needs a one-off AlexNet weight download). Shown as `n/a` if unavailable.
* **NME** - landmarks detected on the *enhanced* image vs the ground-truth landmarks, mean point distance divided by
  sqrt(face-box area). Lower is better. (With the `cascade` backend the points come from a template inside the
  detected box, so NME mainly measures how well the face box is preserved.)
* **Detect** - share of faces that are still detected after enhancement.
* **Params / GMACs / ms** - parameters, conv multiply-accumulates for one 256x256 image, and latency (batch 1, run size).
* **ITA fairness** - Individual Typology Angle of the face core (median CIELAB L*, b*) of the ground truth; groups
  light (ITA > 41), intermediate (10..41), dark (<= 10); the table reports PSNR per group and the gap.

---

## 9. Verified real-data experiment

The verified final CPU experiment used the complete prepared dataset:

* Dataset: `badasstechie/celebahq-resized-256x256`, 2,000 real face photographs.
* Split: 1,600 train / 200 validation / 200 test faces, seed 0.
* Training: 64x64, 14 epochs, batch 16, CPU-only PyTorch 2.15.0+cpu.
* Landmark backend: `cascade(opencv-haar)`, with approximate box-derived landmarks.
* Preprocessing detected faces in 1,954/2,000 images (98%).
* LPIPS: skipped (`n/a`) because the optional package and model weights were not installed.
* Runtime: 78m22s on the AMD Ryzen 7 8840U.
* Output: [`outputs/final_cpu_2000_64px_seed0_20261118/cpu_faces_64px/`](outputs/final_cpu_2000_64px_seed0_20261118/cpu_faces_64px/)

**TABLE A - Baselines vs DGFA-Net (200 held-out test faces)**

| Method | PSNR | SSIM | NME | Detect | ms/img |
|---|---:|---:|---:|---:|---:|
| Input (dark) | 9.40 | 0.261 | 0.025 | 97% | 0.0 |
| Gamma (1/2.2) | 15.83 | 0.567 | 0.024 | 95% | 0.0 |
| CLAHE | 13.15 | 0.523 | 0.034 | 93% | 0.2 |
| Zero-DCE (re-implementation) | 11.07 | 0.396 | 0.037 | 73% | 4.9 |
| Residual U-Net (A0) | **22.29** | **0.798** | 0.026 | 96% | 13.3 |
| DGFA-Net (A3) | 22.02 | 0.784 | **0.025** | 96% | 14.3 |

On this experiment, A0 was the best model by PSNR and SSIM. DGFA-Net A3 was 0.27 dB lower in PSNR
and 0.015 lower in SSIM, while having slightly better NME. This is a measured result, not a claim that
the proposed architecture is superior. The result is consistent with the preliminary smoke test and
suggests that the degradation heads and landmark guidance were not beneficial under this short,
64px CPU training configuration.

**TABLE B - Ablation study**

| ID | Variant | PSNR | SSIM | NME | Detect |
|---|---|---:|---:|---:|---:|
| A0 | Residual U-Net | **22.29** | **0.798** | 0.026 | 96% |
| A1 | + degradation heads | 22.10 | 0.783 | 0.025 | 95% |
| A2 | + attention, no gate | 21.95 | 0.781 | 0.026 | 96% |
| A3 | + gate (full DGFA-Net) | 22.02 | 0.784 | **0.025** | 96% |
| A4 | fixed alpha | 21.97 | 0.781 | 0.025 | 96% |
| A5 | raw-dark landmark detection | 22.02 | 0.784 | 0.025 | 95% |
| A6 | no L_face | 22.19 | 0.786 | 0.026 | 96% |

The pipeline reported H1, H2, and H3 as not supported in this run. The available 200 test faces
and short CPU training schedule do not establish that differences of a few tenths of a dB are robust.
The bootstrap/Wilcoxon comparison for A3 versus A0 gave a measured PSNR difference of -0.27 dB
with 95% CI [-0.35, -0.20] and p=3.2e-11; this is statistically detectable for this fixed synthetic
degradation/test set, but does not establish generalisation to other datasets or degradation models.

**Fairness diagnostic**

The measured PSNR gaps were 0.33 dB for A0 and 0.32 dB for A3 across the coarse ITA groups.
This is a diagnostic only, not a fairness certification. The full per-image results, tables, figures,
failure cases, and training curves are in the final output directory.


---

## 10. Output folder layout

```
outputs/<experiment-root>/<preset>_<data>_<size>px/
  log.txt                      everything that was printed
  checkpoints/                 A0.pt ... A6.pt, ZeroDCE.pt, *_history.csv  (re-used on the next run)
  tables/                      main_comparison, ablation, hypotheses, fairness, real_lowlight (.csv + .md), results.json
  figures/                     01_... 15_*.png
  REPORT.md                    generated factual run report
  RESULTS.md                   generated tables and figure references
  test_inputs/, test_gt/       the exact low-light inputs / ground truths (for external methods)
```

The verified experiment is under
`outputs/final_cpu_2000_64px_seed0_20261118/cpu_faces_64px/`. It contains seven checkpoints,
seven training-history CSV files, 21 PNG figures, per-sample results, and generated reports.

---

## 11. Project structure

```
DGFA-Net/
  run_all.py  train.py  evaluate.py  run_ablations.py  enhance_image.py
  demo_prototype.py  download_datasets.py  dgfa_net.py
  dgfanet/
    model.py          DGFA-Net + ablation switches
    degradation.py    synthetic low-light model (beta, gamma, Poisson + Gaussian noise)
    landmarks.py      landmark backends, 68-point template, Gaussian heat-maps
    synthetic_faces.py  procedural faces with exact landmarks (offline fallback)
    data.py           datasets, Kaggle-folder / LOL discovery, caching, splits
    losses.py         L1 + SSIM + face-weighted + TV
    baselines.py      Gamma, CLAHE, Zero-DCE, external-output hook
    metrics.py        PSNR, SSIM, LPIPS, NME, detection, ITA
    trainer.py  evaluation.py  pipeline.py  viz.py  config.py  utils.py
  tests/test_basic.py
  data/  external/  assets/  .vscode/
```

---

## 12. Honest notes and limitations

* **Synthetic faces are not faces.** The default offline data are cartoon faces; results on them only show that the
  code works. Use Kaggle photos for real conclusions.
* **Proxy detector for procedural faces.** A real detector is unreliable on cartoons, so for them detection is
  *simulated* from the known landmarks (succeeds only if the face region is bright/contrasty enough, jitter grows with
  noise). This makes the NME/Detect/A5/H2/H3 columns on procedural data qualitative. On real photos the real
  detector is used.
* **Landmarks.** Without MediaPipe the "landmarks" are a template placed in a detected face box (approximate). The
  OpenCV/scikit-image cascade finds only clear, roughly frontal faces; at 64 px it misses many, which lowers the
  detection rate for *every* method.
* **Zero-DCE is a re-implementation**, trained briefly on this dataset; it is not the authors' released model.
* **Retinex-Net and Low-FaceNet are not included** (see the external hook).
* **Hand-set values**: lambda weights (0.2 / 0.5 / 0.01), alpha init 0.5, Haar/LBP parameters follow the report's
  starting values and were not tuned.
* **Prototype demo** (`demo_prototype.py`) re-implements the described non-learned prototype; its numbers need not
  equal report Table 7.1 (different noise draw / detector build).
* **Statistics.** The verified CPU run has 200 held-out faces, but its 64px resolution and 14-epoch schedule
  are a practical laptop configuration rather than the report's 256px/200-epoch protocol. The reported
  confidence intervals and tests describe this fixed experiment; they do not prove generalisation.
* **Final CPU result.** On the verified 2,000-image run, A0 outperformed A3 by 0.27 dB PSNR and 0.015 SSIM.
  This is reported unchanged. Longer training, higher resolution, a true landmark backend, and independent
  datasets are needed before drawing architectural conclusions.
* **LPIPS.** LPIPS was intentionally skipped in the verified CPU run because it is optional and its weights
  were not installed. The corresponding cells are `n/a`, not fabricated values.
* **Fairness.** ITA groups come from synthetic skin colours or an automatic skin-region crop; this is a coarse
  diagnostic, not a fairness certification. Report per-group numbers, not only averages.
* **Intended use.** Face enhancement can aid or harm (identification, surveillance). Use responsibly and respect
  dataset licences (CelebA-HQ / FFHQ / LOL have their own terms).

---

## 13. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: dgfanet` | run scripts from the project root (the folder containing `run_all.py`) and use `python -m pytest` rather than a different global `pytest` executable |
| Out of memory on GPU | `--batch 8` or a smaller `--size` |
| Training too slow | `--preset smoke/cpu`, or reduce `--epochs`, `--n-images` |
| `No trained weights` in evaluate.py | run `python train.py --with-baselines` with the same `--preset/--data` first |
| Kaggle download fails (403/404) | check `C:\Users\<you>\.kaggle\access_token`, accept the dataset's terms on kaggle.com, or use `--faces-from` |
| `MediaPipe unavailable` message | harmless; the face-box template is used. For real landmarks: `pip install mediapipe` (needs internet once for the model) |
| `LPIPS: unavailable` | optional; `pip install lpips` (needs internet once) or ignore |
| Garbled characters on Windows | not expected (output is ASCII); if so run `chcp 65001` |
| Results differ between runs | seeds are fixed (`--seed`), but GPU kernels are not bit-exact |

---

## 14. References, license, authors

* Project report: *DGFA-Net: A Lightweight Degradation-Aware, Landmark-Guided Network for Low-Light Face Image
  Enhancement*, Computer Vision 2026-27 (the report cites the related work in full).
* C. Guo et al., *Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement*, CVPR 2020 (Zero-DCE).
* C. Wei et al., *Deep Retinex Decomposition for Low-Light Enhancement*, BMVC 2018 (LOL dataset, Retinex-Net).
* S. Chardon et al., *Skin colour typology and suntanning pathways*, 1991 (ITA).
* R. Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*, CVPR 2018 (LPIPS).

Released under the MIT License (see `LICENSE`). Authors: Saloni Singh, Aditya Gayake - Netaji Subhas University
of Technology.
