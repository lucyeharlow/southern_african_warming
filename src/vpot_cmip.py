"""
October 200 hPa velocity potential and divergence: CMIP6 AMIP and historical multi-model means
(main figure; supplementary).

Run:  python velocity_potential_cmip6.py [compute | plot]
FIGURES results/velocity_potential/chi200_cmip6_october.png (velocity potential)
         results/velocity_potential/div200_cmip6_october.png (divergence)
The calculation and the figure layout are in velocity_potential.py (compute covers every
dataset; it runs here if this figure's data files are missing).
"""
from __future__ import annotations

import sys

import vpot as vp

NAMES = list(vp.CMIP)

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "compute" or (what == "all" and vp.missing(NAMES)):
        vp.compute()
    if what in ("plot", "all"):
        vp.plot_group(NAMES, "cmip6", "CMIP6 multi-model means")
