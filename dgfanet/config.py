"""Run presets. `full` follows the protocol of the report (256 px, 200 epochs, batch 16)."""
from __future__ import annotations

PRESETS = {
    # name   : image size, #clean images, #face-free scenes, epochs (every model, incl. the ablations),
    #          batch, #example sheets
    "smoke": dict(size=64, n_images=64, n_scenes=6, epochs=3, batch=8, sheets=1,
                  desc="~1-3 min on a CPU: checks that everything runs (numbers are meaningless)"),
    "cpu":   dict(size=64, n_images=400, n_scenes=24, epochs=14, batch=16, sheets=2,
                  desc="~15-40 min on a laptop CPU: small synthetic run"),
    "quick": dict(size=128, n_images=3000, n_scenes=60, epochs=40, batch=16, sheets=3,
                  desc="~1-2 h on a consumer GPU"),
    "full":  dict(size=256, n_images=6000, n_scenes=100, epochs=200, batch=16, sheets=4,
                  desc="report protocol: 256 px, 200 epochs (hours per model; GPU required)"),
}
