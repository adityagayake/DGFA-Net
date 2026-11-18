**TABLE A - Baselines vs DGFA-Net (40 test faces, 64px; NME/Detect = on the enhanced image)**

| Method | Params(M) | GMACs | ms/img | PSNR | SSIM | LPIPS | NME | Detect |
|---|---|---|---|---|---|---|---|---|
| Input (dark) | 0.000 | 0.00 | 0.0 | 8.31 | 0.239 | n/a | 0.183 | 52% |
| Gamma (1/2.2) | 0.000 | 0.00 | 0.1 | 15.58 | 0.504 | n/a | 0.209 | 92% |
| CLAHE | 0.000 | 0.00 | 0.2 | 11.28 | 0.327 | n/a | 0.254 | 68% |
| Zero-DCE (re-impl.) | 0.079 | 0.32 | 8.8 | 13.93 | 0.390 | n/a | 0.299 | 30% |
| Residual U-Net (A0) | 0.714 | 0.74 | 14.1 | 20.21 | 0.667 | n/a | 0.162 | 95% |
| DGFA-Net (ours) | 0.716 | 0.75 | 20.1 | 19.52 | 0.653 | n/a | 0.158 | 95% |
