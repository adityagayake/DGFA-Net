# Running and presenting DGFA-Net (VS Code)

## A. One-time setup (10 minutes)
1. Install Python 3.10+ and VS Code (with the *Python* extension). Open the `DGFA-Net` folder: **File > Open Folder**.
2. Terminal (**Ctrl+`**):
   ```
py -3.12 -m venv .venv-dgfa
.\.venv-dgfa\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
   ```
NVIDIA GPU? install the matching CUDA build of PyTorch instead. Optional extras for real facial landmarks and
LPIPS: `python -m pip install -r requirements-optional.txt`.
3. **Kaggle token** (needed once, for the real data): kaggle.com > your profile > *Settings* > *API* > *Create New Token*.
   Save the downloaded token in the current Kaggle CLI format to `C:\Users\<you>\.kaggle\access_token`
   (macOS/Linux: `~/.kaggle/access_token`).

## B. Run the real experiment (one command)
```powershell
python run_all.py --preset cpu --data data/faces --device cpu --landmarks cascade --n-images 2000 --size 64 --epochs 14 --batch 16 --seed 0 --no-lpips --out .\outputs\final_cpu_2000_64px_seed0_20261118
```
* GPU: use the `quick` preset. CPU only: use the `cpu` preset. The verified 2,000-image CPU run took 78m22s.
  The report protocol is `--preset full` and is not practical on a CPU-only laptop.
* If the dataset name fails: download any **CelebA-HQ** or **FFHQ** dataset on kaggle.com in the browser, unzip it and run
  `python run_kaggle.py --faces-from "C:\path\to\unzipped_folder"`.
* The run **stops** instead of using synthetic faces if no real photos are found.
* Everything lands in `outputs/<preset>_faces_<size>px/` - open **`REPORT.md`** there (VS Code: right-click > *Open Preview*).

## C. What to show your professor (suggested order, ~10 min)
| Step | Show | File (in `outputs/.../figures/`) |
|---|---|---|
| 1. Problem + data | real faces and how we darken them | `01_dataset_samples.png`, `02_degradation_model.png` |
| 2. Pipeline idea | landmark detection on the brightened copy, heat-map | `03_landmark_detection.png` |
| 3. **Baselines vs ours (3+ outputs)** | every method side by side with PSNR under each image | `04_examples_01.png` (also 02-04) |
| 4. Numbers | PSNR / SSIM / NME / detection panel + the printed **Table A** | `00_summary_panel.png`, terminal / `tables/main_comparison.md` |
| 5. What the network learned | illumination, noise, attention maps | `05_internal_maps.png` |
| 6. Detail | eyes and mouth crops | `06_crops_eyes.png`, `06_crops_mouth.png` |
| 7. Ablation + hypotheses | which part helps (Tables B, C) | `11_ablation_psnr.png`, `tables/ablation.md`, `tables/hypotheses.md` |
| 8. Fairness | PSNR per skin-tone group (Table D) | `13_skin_tone_fairness.png` |
| 9. Limits | failure cases, face-free images | `08_failure_cases.png`, `07_face_free_scenes.png` |

Be ready to say honestly: the hypotheses are tested, not assumed - `tables/hypotheses.md` says "supported" or "not supported"
for **your** run, and `README.md` Section 12 lists the approximations (Zero-DCE is a re-implementation; Retinex-Net and
Low-FaceNet are not bundled; without MediaPipe the landmarks come from a face-box template).

## D. Live demo options
* Instant (no training): `python demo_prototype.py`
* Your own dark photo with the trained model: `python enhance_image.py --input photo.jpg` (writes a side-by-side image)
* Re-print the tables without retraining: `python evaluate.py --preset quick --data data/faces --with-ablations`
  (use the same options as the training run; weights are cached in `outputs/.../checkpoints/`).

## E. If something goes wrong
* `ModuleNotFoundError`: run from the folder that contains `run_all.py`, with the venv active.
* Kaggle 403/404: accept the dataset's terms on kaggle.com while logged in, check `access_token`, or use `--faces-from`.
* Too slow / out of memory: `--preset cpu`, or `--batch 8`, `--size 64`, `--epochs 10`.
