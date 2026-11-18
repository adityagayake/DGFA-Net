"""Fast unit tests:  python -m pytest -q   (from the project root)"""
import numpy as np
import torch

from dgfanet.degradation import synthesize_low_light
from dgfanet.landmarks import TEMPLATE_68, make_heatmap
from dgfanet.losses import ssim_torch
from dgfanet.model import DGFANet, VARIANTS, build_model
from dgfanet.utils import count_params


def test_param_count_matches_report():
    assert abs(count_params(DGFANet()) - 715_912) < 5


def test_forward_shapes_and_residual_range():
    m = DGFANet().eval()
    x = torch.rand(2, 3, 64, 64)
    y, aux = m(x, torch.zeros(2, 1, 64, 64), torch.zeros(2, 1, 1, 1))
    assert y.shape == x.shape and float(y.min()) >= 0 and float(y.max()) <= 1
    assert aux["attn"].shape == (2, 1, 16, 16)


def test_gate_zero_means_no_attention():
    m = DGFANet().eval()
    _, aux = m(torch.rand(1, 3, 64, 64), torch.rand(1, 1, 64, 64), torch.zeros(1, 1, 1, 1))
    assert float(aux["attn"].abs().max()) == 0.0


def test_all_variants_build():
    for v in VARIANTS:
        build_model(v)


def test_degradation_darkens():
    gt = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    low, p = synthesize_low_light(gt, np.random.default_rng(1))
    assert low.mean() < gt.mean() and 0.2 <= p["beta"] <= 0.8 and 1.5 <= p["gamma"] <= 3.0


def test_heatmap_and_template():
    assert TEMPLATE_68.shape == (68, 2)
    h = make_heatmap(TEMPLATE_68 * 63, (64, 64), 2.0)
    assert h.shape == (64, 64) and float(h.max()) > 0.9
    assert float(make_heatmap(None, (8, 8), 2.0).max()) == 0.0


def test_ssim_identity():
    x = torch.rand(1, 3, 32, 32)
    assert abs(float(ssim_torch(x, x)) - 1.0) < 1e-5
