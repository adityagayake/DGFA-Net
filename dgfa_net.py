"""Quick sanity check of the model definition (report Appendix D).

    python dgfa_net.py

Prints the output shapes, the parameter count (report: ~0.716 M) and the MACs per
256x256 image (report: ~12.0 GMACs).
"""
import torch

from dgfanet.model import DGFANet, build_model, VARIANTS
from dgfanet.utils import count_macs, count_params

if __name__ == "__main__":
    m = DGFANet()
    x = torch.rand(2, 3, 256, 256)
    y, aux = m(x, torch.zeros(2, 1, 256, 256), torch.zeros(2, 1, 1, 1))
    print("output:", tuple(y.shape), "| illum:", tuple(aux["illum"].shape),
          "| noise:", tuple(aux["noise"].shape), "| attn:", tuple(aux["attn"].shape))
    print("params:", count_params(m), f"({count_params(m) / 1e6:.3f} M)")
    macs = count_macs(m, x[:1], torch.zeros(1, 1, 256, 256), torch.zeros(1, 1, 1, 1))
    print(f"MACs per 256x256 image: {macs / 1e9:.2f} G")
    print("\nAblation variants:")
    for k in VARIANTS:
        print(f"  {k}: {count_params(build_model(k)) / 1e6:.3f} M params - {VARIANTS[k]['desc']}")
