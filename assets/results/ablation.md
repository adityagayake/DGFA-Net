**TABLE B - Ablation study (report Table 7.3)**

| ID | Variant | PSNR | SSIM | LPIPS | NME | Detect | PSNR noisy | PSNR clean | PSNR darkest | PSNR face-free |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 | Residual U-Net | 20.21 | 0.667 | n/a | 0.162 | 95% | 20.23 | 20.19 | 19.19 | 20.85 |
| A1 | + degradation heads | 19.51 | 0.650 | n/a | 0.154 | 95% | 19.30 | 19.72 | 18.01 | 20.26 |
| A2 | + attention, no gate | 20.07 | 0.656 | n/a | 0.155 | 95% | 20.06 | 20.08 | 19.05 | 20.77 |
| A3 | + gate (full DGFA-Net) | 19.52 | 0.653 | n/a | 0.158 | 95% | 19.50 | 19.55 | 18.36 | 19.74 |
| A4 | fixed alpha | 19.67 | 0.651 | n/a | 0.155 | 95% | 19.39 | 19.95 | 18.26 | 20.53 |
| A5 | detect on raw dark image | 19.51 | 0.654 | n/a | 0.159 | 95% | 19.48 | 19.55 | 18.29 | 19.74 |
| A6 | no L_face | 20.51 | 0.669 | n/a | 0.149 | 95% | 20.46 | 20.57 | 19.33 | 21.37 |
