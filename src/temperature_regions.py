"""
Monthly 2 m temperature trends over land: Southern Africa compared with the tropics and the
whole globe, as area-weighted boxplots of the grid-cell trends (ERA5, MERRA-2, JRA-3Q, NOAA,
CRUTEM5).

Run:  python temperature_regions.py            compute (if the data files are missing) and plot
      python temperature_regions.py compute    re-read the raw data and rewrite the data files
      python temperature_regions.py plot       plot from the data files only

FIGURE  results/temperature_regions/t2m_trends_regions.png
  a) Southern Africa (10-40E, 35-10S)   b) tropics (23.5S-23.5N)   c) globe; land only.
  One box per dataset and month: box = area-weighted IQR of the cell trends, line =
  area-weighted median, whiskers = most extreme cell within 1.5 IQR. The same y axis in all
  three panels, so the regions can be compared directly.

METHOD  as temperature_trends.py: native grids, land cells (Natural Earth 50 m land fraction
  >= 0.5) with a value in every year (config.YEARS), OLS trend per decade, area weighting
  (cos latitude). Southern Africa is taken from temperature_trends.py's data files when they
  exist (the same numbers as that figure); otherwise it is cut from the global field. The %
  of area significant (FDR q = 0.10 over the land cells of each region and month) is in the
  CSV. NOAA and CRUTEM5 are anomalies (their trends are the same as for absolute values);
  CRUTEM5 has no value in many tropical cells, so it covers less of the tropics (n_cells in
  the CSV).

DATA FILES  data/temperature_regions/t2m_regions_<dataset>.nc: box statistics (region, month)
OUTPUTS     results/temperature_regions/t2m_trends_regions.png, t2m_trends_regions.csv
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd
import xarray as xr

import config
import masks
import plotstyle as ps
import readers as rd
import stats
import temperature_trends as tt

NAMES = ["ERA5", "MERRA-2", "JRA-3Q", "NOAA", "CRUTEM5"]
REGIONS = [("southern_africa", "Southern Africa (10-40°E, 35-10°S)"),
           ("tropics_band", "Tropics (23.5°S-23.5°N)"),
           ("global", "Globe")]
STATS = ["q1", "med", "q3", "whislo", "whishi", "mean", "n_cells", "pct_sig"]
OUT = "temperature_regions"


def data_file(name):
    return config.DATA / OUT / f"t2m_regions_{name.replace('-', '')}.nc"


def load_global(name, month):
    """(year, lat, lon) global field for one calendar month, degC (NOAA, CRUTEM5 anomalies)."""
    raw, mo = config.RAW, [month]
    if name == "ERA5":
        da = rd.single_file(raw["era5"] / "global_2m_temperature.nc", "t2m", "global",
                            offset=-273.15, months=mo)
    elif name == "MERRA-2":
        da = rd.merra2("instM_2d_asm_Nx", "T2M", "global", offset=-273.15, months=mo)
    elif name == "JRA-3Q":
        da = rd.jra3q("", "tmp2m-hgt-an-gauss", "global", offset=-273.15, months=mo)
    elif name == "NOAA":
        path = sorted(glob.glob(str(raw["noaa"] / "NOAAGlobalTemp_*.nc")))[-1]
        da = rd.single_file(path, "anom", "global", fill=1e3, months=mo)
    elif name == "CRUTEM5":
        da = rd.single_file(raw["crutem"], "tas", "global", fill=1e3, months=mo)
    else:
        raise KeyError(name)
    return da.sel(month=month)


def summarise(trend, p):
    """Area-weighted box statistics of a (lat, lon) trend field, and % of area significant."""
    b = stats.box_stats(trend)
    if b is None:
        return {k: np.nan for k in STATS} | {"n_cells": 0}
    sig = stats.fdr(p)
    return {"q1": b["q1"], "med": b["med"], "q3": b["q3"], "whislo": b["whislo"],
            "whishi": b["whishi"], "mean": b["mean"], "n_cells": int(trend.notnull().sum()),
            "pct_sig": stats.pct_area(sig, trend.notnull())}


def compute(names=NAMES):
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    for name in names:
        rd.log(f"--- {name} ---")
        sa_file = tt.data_file(name)
        sa = xr.load_dataset(sa_file) if sa_file.exists() else None
        if sa is not None:
            rd.log(f"  Southern Africa from {sa_file.name}")
        out = np.full((len(REGIONS), 12, len(STATS)), np.nan)
        land = None
        for m in range(1, 13):
            da = load_global(name, m)
            if land is None:
                land = masks.land_mask(da)
            da = stats.select_years(da.where(land).rename(f"{name} month {m}"),
                                    config.YEARS[name])
            st = stats.ols_stats(da)
            for ri, (reg, _) in enumerate(REGIONS):
                if reg == "southern_africa" and sa is not None:
                    tr, p = sa.trend.sel(month=m), sa.p.sel(month=m)
                else:
                    tr, p = rd.cut(st.trend, reg), rd.cut(st.p, reg)
                out[ri, m - 1] = [summarise(tr, p)[k] for k in STATS]
            rd.log(f"  month {m:2d}: median trend SA {out[0, m - 1, 1]:+.2f}, tropics "
                   f"{out[1, m - 1, 1]:+.2f}, globe {out[2, m - 1, 1]:+.2f} degC/decade")
        ds = xr.Dataset({"box": (("region", "month", "stat"), out)},
                        coords={"region": [r for r, _ in REGIONS], "month": np.arange(1, 13),
                                "stat": STATS})
        ds.attrs = {"dataset": name, "years": f"{config.YEARS[name][0]}-{config.YEARS[name][1]}",
                    "units": "degC per decade (n_cells: count; pct_sig: % of area)",
                    "southern_africa_from": sa_file.name if sa is not None else "global field"}
        ds.to_netcdf(data_file(name))
        rd.log(f"  saved {data_file(name)}")


def plot():
    res = {n: xr.load_dataset(data_file(n)) for n in NAMES if data_file(n).exists()}
    if not res:
        raise SystemExit("no data files: run 'python temperature_regions.py compute' first")
    rows = [{"dataset": n, "years": d.attrs["years"], "region": str(r), "month": int(m),
             **{s: float(d.box.sel(region=r, month=m, stat=s)) for s in STATS}}
            for n, d in res.items() for r in d.region.values for m in d.month.values]
    csv = config.RESULTS / OUT / "t2m_trends_regions.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"Saved: {csv}", flush=True)

    ps.setup()
    fs, W = config.FS, config.WIDTH["full"]
    left, right, ph, gap, top = 0.62, 0.15, 1.35, 0.42, 0.62
    H = top + 3 * ph + 2 * gap + 0.35
    fig, at = ps.figure("full", H)
    names = list(res)
    years = {n: ps.year_list(res[n].attrs["years"]) for n in names}
    lo = min(float(d.box.sel(stat="whislo").min()) for d in res.values())
    hi = max(float(d.box.sel(stat="whishi").max()) for d in res.values())
    pad = 0.05 * (hi - lo)
    axes = []
    for ri, (reg, title) in enumerate(REGIONS):
        ax = at(left, top + ri * (ph + gap), W - left - right, ph)
        boxes = {}
        for n in names:
            b = res[n].box.sel(region=reg)
            boxes[n] = {int(m): {k: float(b.sel(month=m, stat=k)) for k in
                                 ("q1", "med", "q3", "whislo", "whishi")} | {"fliers": []}
                        for m in b.month.values if np.isfinite(float(b.sel(month=m, stat="med")))}
        ps.monthly_boxplot_stats(ax, boxes, names, "Trend, area-weighted\n(°C decade$^{-1}$)")
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{title}, land", fontsize=fs["title"], fontweight="bold", pad=3)
        ps.letter(ax, ri)
        axes.append(ax)
    ps.dataset_legend(axes[0], names, years, loc="lower center", bbox_to_anchor=(0.5, 1.13),
                      ncol=3, handlelength=1.2, columnspacing=1.2, borderaxespad=0.2)
    ps.save(fig, config.RESULTS / OUT / "t2m_trends_regions.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "compute" or (what == "all" and not all(data_file(n).exists() for n in NAMES)):
        compute()
    if what in ("plot", "all"):
        plot()
