"""
Sensitivity of the omega - cloud fraction relation to the omega level: the detrended
correlation of omega_cf_correlation.py (omega at one level against cloud fraction at every
level of the column, 5-degree bands 10-40E, land, October) repeated with omega at LEVELS.

Run:  python omega_cf_sensitivity.py [compute | plot] [ERA5 MERRA-2 JRA-3Q]

FIGURES  results/omega_cf_sensitivity/
  omega_levels_cf_<dataset>.png  rows = omega level, columns = bands: detrended r(omega_p, CF);
                                 black dots: significant (FDR q = 0.10 per panel); black mark
                                 on the left edge: the omega level of the row
  omega_levels_cf_summary.png    per band: % of the section (area-weighted) with significant
                                 r, and the area-weighted mean r, against the omega level, one
                                 line per dataset
TABLE    omega_levels_cf_summary.csv
DATA FILES  data/omega_cf/omega<level>_cf_<dataset>.nc (shared with omega_cf_correlation.py,
  so omega300 is computed once)
"""
from __future__ import annotations

import calendar
import sys

import numpy as np
import pandas as pd
import xarray as xr

import config
import omega_cf_corr as oc
import plotstyle as ps
import stats

LEVELS = [200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 850.0]
OUT = "omega_cf_sensitivity"


def missing(names):
    return [n for n in names if not all(oc.data_file(n, l).exists() for l in LEVELS)]


def compute(names=oc.NAMES):
    oc.compute(names, LEVELS)


def summary(names):
    rows = []
    for name in names:
        for lev in LEVELS:
            f = oc.data_file(name, lev)
            if not f.exists():
                continue
            ds = xr.load_dataset(f)
            for b in ds.band.values:
                o = ds.sel(band=b)
                valid = o.r.notnull()
                rows.append({"dataset": name, "omega_level_hPa": lev, "band": str(b),
                             "pct_section_sig": stats.pct_area(o.sig.astype(bool), valid),
                             "mean_r": float(stats.area_mean(o.r.where(valid), dims=("lev", "lat"))),
                             "pct_section_r_negative_sig": stats.pct_area(
                                 o.sig.astype(bool) & (o.r < 0), valid)})
    df = pd.DataFrame(rows)
    path = config.RESULTS / OUT / "omega_levels_cf_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.3f")
    print(f"Saved: {path}", flush=True)
    return df


def plot_dataset(name):
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.pyplot as plt
    res = {l: xr.load_dataset(oc.data_file(name, l)) for l in LEVELS
           if oc.data_file(name, l).exists()}
    if not res:
        return
    fs, W = config.FS, config.WIDTH["full"]
    levs = list(res)
    nb = len(oc.BANDS)
    left, right, gx, gy = 0.75, 0.08, 0.1, 0.12
    pw = (W - left - right - (nb - 1) * gx) / nb
    ph = 0.82
    top = 0.62
    tops = [top + i * (ph + gy) for i in range(len(levs))]
    cb_top = tops[-1] + ph + 0.42
    H = cb_top + 0.7 + (0.2 if oc.layered(name) else 0)
    fig, at = ps.figure("full", H)
    rnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    k = 0
    for ri, lev in enumerate(levs):
        ds = res[lev]
        ps.text(fig, 0.1, tops[ri] + ph / 2, f"ω$_{{{lev:.0f}}}$", rotation=90, ha="center",
                va="center", fontsize=fs["label"], fontweight="bold")
        for bi in range(nb):
            o = ds.isel(band=bi)
            ax = at(left + bi * (pw + gx), tops[ri], pw, ph)
            oc.section(ax, o, "r", norm=rnorm, show=o.sig.astype(bool).values,
                       valid=np.isfinite(o.r.values), omega_lev=lev)
            if ri == 0:
                ax.set_title(str(o.band.values), fontsize=fs["title"], fontweight="bold", pad=3)
            ax.tick_params(labelleft=(bi == 0), labelbottom=(ri == len(levs) - 1), length=2,
                           pad=1.5)
            ax.set_yticks([200, 500, 800])
            if bi == 0:
                ax.set_ylabel("hPa", fontsize=fs["small"])
            ps.letter(ax, k)
            k += 1
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=rnorm, cmap="RdBu_r"),
                      cax=at(left + 0.6, cb_top, W - left - right - 1.2, 0.08),
                      orientation="horizontal", ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label(f"Detrended r(ω at the level of the row, CF); dots: significant (FDR q = "
                 f"{config.ALPHA_FDR:.2f}); brown: surface pressure", fontsize=fs["label"])
    cb.ax.tick_params(labelsize=fs["tick"], length=2)
    if oc.layered(name):
        ps.text(fig, W / 2, H - 0.08, "JRA-3Q cloud: its high (90-500 hPa), middle (500-850) "
                "and low (850-1100) cloud covers, each drawn over its pressure range",
                ha="center", fontsize=fs["small"], color="0.3")
    ps.text(fig, W / 2, 0.06, f"{ps.label(name)} {calendar.month_name[oc.MONTH]} "
            f"{next(iter(res.values())).attrs['years']}: detrended r(ω at each level, cloud "
            "fraction), land", ha="center", va="top", fontsize=fs["suptitle"], fontweight="bold")
    ps.save(fig, config.RESULTS / OUT / f"omega_levels_cf_{name.replace('-', '')}.png")


def plot_summary(df):
    if df.empty:
        return
    fs, W = config.FS, config.WIDTH["full"]
    bands = list(dict.fromkeys(df.band))
    nb = len(bands)
    ncol = 3
    nrow = int(np.ceil(nb / ncol))
    left, right, gx, gy, pw_h = 0.62, 0.1, 0.35, 0.5, 1.15
    pw = (W - left - right - (ncol - 1) * gx) / ncol
    blocks = [("pct_section_sig", "Section with significant r (%)"),
              ("mean_r", "Area-weighted mean r")]
    top = 0.65
    H = top + len(blocks) * nrow * (pw_h + gy) + 0.55
    fig, at = ps.figure("full", H)
    k = 0
    names = list(dict.fromkeys(df.dataset))
    for bl, (col, ylab) in enumerate(blocks):
        for i, b in enumerate(bands):
            rr, cc = divmod(i, ncol)
            t0 = top + (bl * nrow + rr) * (pw_h + gy)
            ax = at(left + cc * (pw + gx), t0, pw, pw_h)
            for n in names:
                d = df[(df.dataset == n) & (df.band == b)].sort_values("omega_level_hPa")
                ax.plot(d[col], d.omega_level_hPa, color=ps.colour(n), lw=1.3, marker="o",
                        ms=3, label=ps.label(n))
            ax.axhline(oc.OMEGA_LEV, color="0.6", lw=0.6, ls=":")
            if col == "mean_r":
                ax.axvline(0, color="k", lw=0.5)
            ax.set_ylim(max(LEVELS) + 50, min(LEVELS) - 50)
            ax.set_yticks(LEVELS[::2] if len(LEVELS) > 5 else LEVELS)
            ax.grid(alpha=0.3, lw=0.4)
            ax.set_title(f"{b}", fontsize=fs["label"], fontweight="bold", pad=2)
            if cc == 0:
                ax.set_ylabel("ω level (hPa)", fontsize=fs["small"])
            ax.set_xlabel(ylab, fontsize=fs["small"])
            ps.letter(ax, k)
            k += 1
    ps.dataset_legend(fig, names, loc="upper center", ncol=len(names),
                      bbox_to_anchor=(0.5, 1 - 0.12 / H))
    ps.save(fig, config.RESULTS / OUT / "omega_levels_cf_summary.png")


def plot(names=oc.NAMES):
    ps.setup()
    df = summary(names)
    for n in names:
        plot_dataset(n)
    plot_summary(df)


if __name__ == "__main__":
    args = sys.argv[1:]
    what = next((a for a in args if a in ("compute", "plot")), "all")
    names = [a for a in args if a in oc.NAMES] or oc.NAMES
    todo = names if what == "compute" else missing(names)
    if what in ("compute", "all") and todo:
        compute(todo)
    if what in ("plot", "all"):
        plot(names)
