DGFA-Net: Degradation-Aware, Landmark-Guided Low-Light Face Enhancement
Research implementation and empirical evaluation | Computer Vision 2026–27 | Netaji Subhas University of Technology

DGFA-Net investigates whether explicitly modelling low-light degradation and using facial-landmark-guided attention can improve low-light face enhancement. This repository contains the proposed network, classical and learned baselines, ablation variants, evaluation pipeline, and a verified CPU experiment on real CelebA-HQ photographs.
Main empirical finding: In the completed 64×64, 14-epoch CPU experiment, the Residual U-Net baseline (A0) achieved higher PSNR and SSIM than the full DGFA-Net variant (A3). A3 achieved slightly lower NME. The current experiment therefore does not demonstrate that the proposed components improve overall reconstruction quality. This negative result is retained rather than obscured.

Authors: Saloni Singh and Aditya Gayake
Supervisor: Prof. Ankush Jain
Research at a glance
Item	Experimental setup
Dataset	2,000 real CelebA-HQ photographs
Split	1,600 train / 200 validation / 200 test
Input resolution	64 × 64
Training	14 epochs per learned model; batch size 16
Random seed	0
Hardware	CPU; AMD Ryzen 7 8840U
PyTorch	2.15.0+cpu
Landmark backend	OpenCV Haar cascade with approximate box-derived 68-point template
Total pipeline runtime	78 min 22 sec
LPIPS	Not run; optional dependency and weights were not installed


The low-light inputs in this experiment were generated synthetically from well-lit photographs. The results should not be interpreted as validation on a real paired low-light benchmark.
1. Research question and motivation
Low-light face images can suffer from reduced illumination, noise, loss of local contrast, and degradation of facial details. A generic enhancement model may improve global brightness without preserving identity-relevant structures.

This project evaluates a lightweight, degradation-aware, landmark-guided design with three central ideas:
1. Degradation estimation: predict illumination and noise maps from the input rather than receiving degradation parameters directly.
2. Landmark-guided attention: use a facial landmark heat-map to guide feature refinement around facial structure.
3. Face-presence gating: suppress landmark-guided attention when a face is not detected.
The central research question is whether these additions improve enhancement quality and facial-structure preservation compared with a simpler supervised baseline under a controlled evaluation protocol.
2. Proposed pipeline
Well-lit face photographs
          |
          v
Synthetic low-light degradation
  exposure + gamma + Poisson/Gaussian noise
          |
          +------------------------------+
          |                              |
          v                              v
Brightened-input face detection     Low-light image
and landmark heat-map                    |
          |                              |
          +---------------+--------------+
                          |
                          v
                    DGFA-Net
                          |
          +---------------+----------------+
          |                                |
          v                                v
 Illumination map M_L                Noise map M_N
          |                                |
          +---------------+----------------+
                          |
                   Feature fusion
                          |
                       Encoder
                          |
            Landmark-guided attention
                 with presence gate
                          |
                       Decoder
                          |
                          v
                Enhanced face image
                          |
                          v
             PSNR / SSIM / NME / detection
                          |
                          v
       Baselines + ablations + analysis + figures
Synthetic degradation model
The implementation generates low-light observations using exposure scaling, gamma transformation, and noise:
\[
I_{\text{low}} =
\operatorname{clip}\left(
\beta I_{\text{gt}}^\gamma + N
\right)
\]
where \(\beta\) is sampled from \(U[0.2,0.8]\), \(\gamma\) from \(U[1.5,3.0]\), and \(N\) combines signal-dependent Poisson noise and Gaussian read noise.

The same fixed test inputs are used for all compared methods, enabling paired comparisons on the held-out split.
DGFA-Net components
- A feature stem extracts shallow image features.
- Two degradation heads estimate an illumination map \(M_L\) and noise map \(M_N\).
- A fusion layer combines the image features and estimated maps.
- An encoder-decoder predicts a residual enhancement.
- Landmark-guided attention refines bottleneck features.
- A face-presence flag gates the attention when a face is unavailable.
The attention formulation in the implementation is:
\[
A_f = \sigma(\operatorname{Conv}_{1\times1}([F_3,H]))\cdot flag
\]
\[
F_{\text{guided}} = F_3(1+\alpha A_f)
\]
The output is a residual correction:
\[
I_{\text{enh}} = \operatorname{clamp}(I_{\text{low}}+\Delta I)
\]
Training objective
The implemented objective combines reconstruction, structural similarity, face-region weighting, and illumination-map smoothness:
\[
L = L_1 + 0.2L_{\text{SSIM}} + 0.5L_{\text{face}} + 0.01L_{\text{TV}}
\]
The model is trained with Adam, an initial learning rate of \(2\times10^{-4}\), cosine decay, gradient clipping, and best-validation checkpoint selection.
3. Baselines and ablation design
Baselines
Method	Role
Input (dark)	Unprocessed lower-bound reference
Gamma correction	Classical brightness correction
CLAHE	Classical local contrast enhancement
Zero-DCE	Re-implementation of a learned zero-reference baseline
Residual U-Net (A0)	Supervised architectural baseline
DGFA-Net (A3)	Full proposed configuration


Zero-DCE in this repository is a re-implementation, not the original authors' released checkpoint. Retinex-Net and Low-FaceNet are not bundled; the repository provides an external-method integration route.
Ablation variants
Variant	Configuration
A0	Residual U-Net baseline
A1	A0 + illumination/noise degradation heads
A2	A1 + landmark attention, without presence gate
A3	A1 + landmark attention + presence gate (full DGFA-Net)
A4	A3 with fixed \(\alpha=0.5\)
A5	A3 weights, but landmarks detected on the raw dark input
A6	A3 trained without the face-weighted loss


A0, A1, A2, A3, A4, A6 and Zero-DCE completed in the verified run. A5 is an inference-time evaluation variant using A3 weights.
4. Quantitative results
Baselines versus the proposed model
Results below are from 200 held-out test faces.
Method	PSNR ↑ (dB)	SSIM ↑	NME ↓	Detection ↑	Latency (ms/image)
Input (dark)	9.40	0.261	0.025	97%	0.0
Gamma (1/2.2)	15.83	0.567	0.024	95%	0.0
CLAHE	13.15	0.523	0.034	93%	0.2
Zero-DCE (re-implementation)	11.07	0.396	0.037	73%	4.9
Residual U-Net (A0)	22.29	0.798	0.026	96%	13.3
DGFA-Net (A3)	22.02	0.784	0.025	96%	14.3


Interpretation
- A0 achieved the best PSNR and SSIM in this experiment.
- A3 was 0.27 dB lower in PSNR and 0.014 lower in SSIM than A0 (rounded table values).
- A3 achieved slightly lower NME than A0, while both had a 96% face-detection rate.
- A3's measured latency was 14.3 ms/image versus 13.3 ms/image for A0 in this run.
- These results do not support a claim that the full DGFA-Net improves overall image reconstruction quality under this configuration.
Ablation results
Variant	Description	PSNR ↑	SSIM ↑	NME ↓	Detection ↑
A0	Residual U-Net	22.29	0.798	0.026	96%
A1	+ degradation heads	22.10	0.783	0.025	95%
A2	+ attention, no gate	21.95	0.781	0.026	96%
A3	+ presence gate (full DGFA-Net)	22.02	0.784	0.025	96%
A4	Fixed alpha	21.97	0.781	0.025	96%
A5	Raw-dark landmark detection	22.02	0.784	0.025	95%
A6	Without face loss	22.19	0.786	0.026	96%


The pipeline reported H1, H2 and H3 as not supported in this run. These outcomes should be read in the context of the small held-out set and short, low-resolution training schedule.
Paired comparison: A3 versus A0
- PSNR difference (A3 − A0): −0.27 dB
- 95% confidence interval: [−0.35, −0.20]
- Wilcoxon p-value: 3.2 × 10⁻¹¹
The difference is statistically detectable for this fixed test set and synthetic degradation procedure. It does not establish generalisation to other datasets, resolutions, training schedules, or real-world low-light conditions.
Fairness diagnostic
The measured PSNR gaps between the coarse ITA groups were 0.33 dB for A0 and 0.32 dB for A3. This is a diagnostic from the current pipeline, not a fairness certification.
5. Qualitative results and output figures
The full experiment generated 21 figures. To make the figures visible on GitHub, copy the generated PNGs from the local output directory into assets/results/research_figures/ in the repository. The outputs/ directory is intentionally ignored by Git, so embedding images directly from it will not work in the published README.

After copying the figures, the following embeds will render when the corresponding files exist:
Dataset and degradation
Clean dataset samples	Synthetic degradation model
	


Landmark guidance and internal representations

Enhancement comparisons and facial details

Eye-region crops	Mouth-region crops
	


Failure cases and training behaviour

Figure file note: The exact figure filenames can vary by run. If a filename above differs from the generated file, use the actual filename from the experiment's figures/ directory and update the corresponding Markdown image path. Do not create or substitute illustrative images for measured outputs.

6. Reproducibility
Environment
The verified experiment used Python 3.12.10 and CPU-only PyTorch 2.15.0. A GPU is optional; the verified laptop run was CPU-only.
Install
py -3.12 -m venv .venv-dgfa
.\.venv-dgfa\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install pytest mediapipe six
Run a quick smoke test
python run_all.py --preset smoke --device cpu --landmarks cascade --no-lpips
Reproduce the verified experiment
This command uses the prepared data/faces directory. It does not require another Kaggle download if the data are already present.
python run_all.py --preset cpu --data data/faces --device cpu --landmarks cascade --n-images 2000 --size 64 --epochs 14 --batch 16 --seed 0 --no-lpips --out .\outputs\final_cpu_2000_64px_seed0_20261118
Tests and model inspection
python -m pytest -q
python dgfa_net.py
7. Repository outputs
The verified run is stored locally at:
outputs/final_cpu_2000_64px_seed0_20261118/cpu_faces_64px/
It contains:
- checkpoints/ — trained model checkpoints for A0, A1, A2, A3, A4, A6 and Zero-DCE.
- figures/ — 21 generated PNG figures.
- tables/ — CSV, JSON and Markdown result tables.
- REPORT.md and RESULTS.md — generated experiment reports.
- log.txt — run log.
- Per-sample evaluation outputs and training histories.
The dataset, model checkpoints, caches, and complete run outputs are excluded from Git by .gitignore to keep the source repository manageable. Selected result figures should be copied into assets/results/research_figures/ if they are to appear in the GitHub README.
8. Limitations and interpretation
1. Resolution and training budget: the verified run uses 64×64 inputs and 14 epochs, not the report's 256×256/200-epoch protocol.
2. Synthetic low-light data: degradation is generated from well-lit CelebA-HQ photographs. The experiment does not establish performance on real paired low-light face photographs.
3. Approximate landmarks: the OpenCV Haar cascade detects face boxes; the 68 points are template-derived rather than individually detected facial landmarks. NME is therefore a limited structural proxy.
4. LPIPS unavailable: LPIPS was not computed. Its entries should remain n/a; no values are inferred.
5. Baseline result: A0 outperformed A3 on PSNR and SSIM. The proposed components require further investigation and stronger experiments before any superiority claim.
6. External baselines: Retinex-Net and Low-FaceNet are not included. Zero-DCE is a project re-implementation rather than the original released model.
7. Fairness analysis: the ITA grouping is a coarse diagnostic and should not be presented as a fairness certification.
8. Generalisability: one seed, one dataset source, one resolution and one short training schedule are insufficient for broad claims.
Next research steps
- Repeat experiments with multiple seeds and report mean, standard deviation and confidence intervals.
- Train at higher resolution and with a longer schedule on suitable GPU hardware.
- Use a robust facial-landmark detector and report landmark failure rates separately.
- Evaluate on a real paired low-light dataset such as LOL where face-relevant paired samples are available.
- Investigate degradation-head calibration, attention maps, loss weights and the A0/A3 performance gap.
- Add independently sourced implementations/checkpoints for additional published baselines under their respective licences.
9. Project structure
DGFA-Net/
├── dgfanet/
│   ├── model.py
│   ├── degradation.py
│   ├── landmarks.py
│   ├── data.py
│   ├── losses.py
│   ├── baselines.py
│   ├── metrics.py
│   ├── trainer.py
│   ├── evaluation.py
│   ├── pipeline.py
│   └── viz.py
├── tests/
├── assets/
│   └── results/
│       └── research_figures/
├── data/                 # local data; ignored by Git
├── outputs/              # local checkpoints and experiment outputs; ignored by Git
├── run_all.py
├── train.py
├── evaluate.py
├── run_ablations.py
├── download_datasets.py
├── enhance_image.py
├── README.md
└── requirements.txt
References
- Guo et al., “Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement,” CVPR 2020.
- Wei et al., “Deep Retinex Decomposition for Low-Light Enhancement,” BMVC 2018.
- Zhang et al., “The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,” CVPR 2018.
- Project report: DGFA-Net: A Lightweight Degradation-Aware, Landmark-Guided Network for Low-Light Face Image Enhancement, Computer Vision 2026–27.
