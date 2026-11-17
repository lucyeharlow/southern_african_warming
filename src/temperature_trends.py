"""
Southern Africa (10-40E, 35-10S) 2 m temperature trends from six datasets.

Run:  python temperature_trends.py            compute (if the data files are missing) and plot
      python temperature_trends.py compute    re-read the raw data and rewrite the data files
      python temperature_trends.py plot       plot from the data files only

DATASETS (years from config.YEARS)
  ERA5     global_2m_temperature.nc                      t2m (K)
  MERRA-2  instM_2d_asm_Nx                               T2M (K)
  JRA-3Q   *tmp2m-hgt-an-gauss*.nc (6-hourly, monthly files)
  CHIRTS   chirts/{Tmax,Tmin}/{Tmax,Tmin}.YYYY.nc        Tmean = (Tmax + Tmin) / 2
  NOAA     NOAAGlobalTemp_*.nc                           anom (degC anomalies)
  CRUTEM5  CRUTEM.5.1.0.0.anomalies.nc                   tas (degC anomalies)

METHOD (shared conventions: stats.py, masks.py)
  Native grids; land cells only (Natural Earth 50 m land fraction >= 0.5); a cell is used
  for a month only if it has a value in every year; anomalies from the 1991-2020 monthly
  climatology; OLS trend per decade with the lag-1 effective sample size (n_eff <= n);
  Benjamini-Hochberg FDR at 0.10 over the land cells of each map; boxplots, regional series
  and % of area significant area-weighted (cos latitude).

DATA FILES  data/temperature/t2m_<dataset>.nc
  trend, p, n_eff, sig (month, lat, lon); October regional anomaly series (year) and its trend.
OUTPUTS     results/temperature/
  t2m_trends_southern_africa.png    (a) monthly boxplots, (b)-(g) October maps, (h) series
  t2m_trends_southern_africa.csv    area-weighted summary per dataset and month
"""
from __future__ import annotations

import calendar
import glob
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr

import config
import masks
import plotstyle as ps
import readers as rd
import stats

REGION = "southern_africa"
MONTH = 10
ORDER = ["ERA5", "MERRA-2", "JRA-3Q", "NOAA", "CHIRTS", "CRUTEM5"]     # map order
SUMMARY = ["ERA5", "MERRA-2", "JRA-3Q", "NOAA", "CHIRTS"]             # boxplots and series
OUT = "temperature"


def data_file(name):
    return config.DATA / OUT / f"t2m_{name.replace('-', '').replace(' ', '_')}.nc"


# ---------------------------------------------------------------- reading
def load_chirts():
    root = config.RAW["chirts"]

    def read(pair, y):
        with xr.open_dataset(pair[0]) as a, xr.open_dataset(pair[1]) as b:
            tx = rd.clean(rd.cut(rd.standardise(a["Tmax"]), REGION), 1e3)
            tn = rd.clean(rd.cut(rd.standardise(b["Tmin"]), REGION), 1e3)
            tx, tn = xr.align(tx, tn, join="inner")
            return ((tx + tn) / 2).resample(time="1MS").mean().load()

    files = {}
    for p in sorted(glob.glob(str(root / "Tmax" / "Tmax.*.nc"))):
        y = int(re.search(r"Tmax\.(\d{4})\.nc$", p)[1])
        q = root / "Tmin" / f"Tmin.{y}.nc"
        if q.exists():
            files[y] = (p, str(q))
    return rd.from_months(rd.per_month(rd.cached(f"chirts_tmean_{REGION}", files, read)))


def load(name):
    """(year, month, lat, lon) in degC (anomalies for NOAA and CRUTEM5), over REGION."""
    raw = config.RAW
    if name == "ERA5":
        return rd.single_file(raw["era5"] / "global_2m_temperature.nc", "t2m", REGION,
                              offset=-273.15)
    if name == "MERRA-2":
        return rd.merra2("instM_2d_asm_Nx", "T2M", REGION, offset=-273.15)
    if name == "JRA-3Q":
        return rd.jra3q("", "tmp2m-hgt-an-gauss", REGION, offset=-273.15)
    if name == "CHIRTS":
        return load_chirts()
    if name == "NOAA":
        path = sorted(glob.glob(str(raw["noaa"] / "NOAAGlobalTemp_*.nc")))[-1]
        return rd.single_file(path, "anom", REGION, fill=1e3)
    if name == "CRUTEM5":
        return rd.single_file(raw["crutem"], "tas", REGION, fill=1e3)
    raise KeyError(name)


# ---------------------------------------------------------------- compute
def compute():
    out_dir = config.DATA / OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ORDER:
        rd.log(f"--- {name} ---")
        da = load(name)
        da = da.where(masks.land_mask(da))
        da = stats.select_years(da.rename(name), config.YEARS[name])
        anom = stats.anomalies(da)
        st = stats.ols_stats(anom)                                       # (month, lat, lon)
        sig = xr.concat([stats.fdr(st.p.sel(month=m)) for m in st.month], "month")
        series = stats.regional_series(anom.sel(month=MONTH))
        sst = stats.ols_stats(series)
        ds = xr.Dataset({"trend": st.trend.astype("float32"), "p": st.p.astype("float32"),
                         "n_eff": st.n_eff.astype("float32"), "sig": sig.astype("int8"),
                         "series": series})
        ds.attrs = {"dataset": name, "years": f"{da.year.values[0]}-{da.year.values[-1]}",
                    "units": "degC per decade", "series_month": MONTH,
                    "series_trend": float(sst.trend), "series_p": float(sst.p),
                    "series_n_eff": float(sst.n_eff), "series_n": float(sst.n)}
        enc = {v: {"zlib": True, "complevel": 4} for v in ("trend", "p", "n_eff")}
        ds.to_netcdf(data_file(name), encoding=enc)
        o = ds.sel(month=MONTH)
        rd.log(f"  {ds.attrs['years']}: October area-weighted median trend "
               f"{stats.wmedian(o.trend):+.3f} degC/decade, "
               f"{stats.pct_area(o.sig.astype(bool), o.trend.notnull()):.0f} % of area significant; "
               f"regional {float(sst.trend):+.3f}, {stats.p_text(float(sst.p))}")
        rd.log(f"  saved {data_file(name)}")


# ---------------------------------------------------------------- plot
def read_all():
    out = {}
    for name in ORDER:
        f = data_file(name)
        if f.exists():
            with xr.open_dataset(f) as ds:
                out[name] = ds.load()
    return out


def summary_csv(res):
    rows = []
    for name, ds in res.items():
        for m in ds.month.values:
            o = ds.sel(month=m)
            rows.append({"dataset": name, "years": ds.attrs["years"], "month": int(m),
                         **{f"{k}_degC_per_decade" if k != "n_cells" else k: v
                            for k, v in stats.describe(o.trend).items()},
                         "median_n_eff": stats.wmedian(o.n_eff),
                         "pct_area_sig_fdr": stats.pct_area(o.sig.astype(bool), o.trend.notnull())})
    path = config.RESULTS / OUT / "t2m_trends_southern_africa.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved: {path}", flush=True)


def plot():
    import cartopy.crs as ccrs
    res = read_all()
    if not res:
        raise SystemExit("no data files: run 'python temperature_trends.py compute' first")
    summary_csv(res)
    ps.setup()
    fs = config.FS
    years = {}
    for n in res:
        y0, y1 = (int(y) for y in res[n].attrs["years"].split("-"))
        years[n] = list(range(y0, y1 + 1))
    W = config.WIDTH["full"]
    left, right = 0.62, 0.15

    # geometry (inches from the top)
    box_top, box_h = 0.55, 1.2
    gap, ncol = 0.24, 3
    mw = (W - left - right - (ncol - 1) * gap) / ncol
    r = config.REGIONS[REGION]
    mh = mw * (r["lat"][1] - r["lat"][0]) / (r["lon"][1] - r["lon"][0])
    head = 0.4                                             # dataset name + years
    map_top = box_top + box_h + 0.62 + head
    row_step = mh + 0.2 + head
    cb_top = map_top + 2 * row_step - head + 0.0
    ts_top, ts_h = cb_top + 0.8, 1.05
    H = ts_top + ts_h + 1.0
    fig, at = ps.figure("full", H)

    # (a) monthly boxplots
    ax = at(left, box_top, W - left - right, box_h)
    names = [n for n in SUMMARY if n in res]
    ps.monthly_boxplot(ax, {n: {int(m): res[n].trend.sel(month=m) for m in res[n].month.values}
                            for n in names}, names,
                       "Trend, area-weighted\n(°C decade$^{-1}$)")
    ps.letter(ax, 0)
    ps.dataset_legend(ax, names, years, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                      ncol=3, handlelength=1.2, columnspacing=1.2, borderaxespad=0.2)

    # (b)-(g) October maps
    maps = [n for n in ORDER if n in res]
    norm = ps.sym_norm([res[n].trend.sel(month=MONTH) for n in maps])
    ps.text(fig, W / 2, map_top - head - 0.08, f"{calendar.month_name[MONTH]} trends",
            ha="center", fontsize=fs["title"], fontweight="bold")
    for i, name in enumerate(maps):
        rr, cc = divmod(i, ncol)
        x0, t0 = left + cc * (mw + gap), map_top + rr * row_step
        o = res[name].sel(month=MONTH)
        axm = at(x0, t0, mw, mh, projection=ccrs.PlateCarree())
        ps.map_panel(axm, o.trend, norm, sig=o.sig.astype(bool).values, region=REGION,
                     left=(cc == 0), bottom=True)
        ps.letter(axm, i + 1)
        ps.text(fig, x0 + mw / 2, t0 - 0.2, ps.label(name), ha="center",
                fontsize=fs["title"], fontweight="bold", color=ps.colour(name))
        ps.text(fig, x0 + mw / 2, t0 - 0.03, ps.span(years[name]), ha="center",
                fontsize=fs["small"], color="0.25")
    cb = ps.colourbar(fig, at(left + 0.6, cb_top, W - left - right - 1.2, 0.1), norm, "RdBu_r",
                      f"{calendar.month_name[MONTH]} trend (°C decade$^{{-1}}$); "
                      f"dots: significant (FDR q = {config.ALPHA_FDR:.2f})")

    # (h) October regional anomaly series
    ax = at(left, ts_top, W - left - right, ts_h)
    for name in names:
        s, a = res[name].series, res[name].attrs
        ax.plot(s.year, s, color=ps.colour(name), lw=1.2,
                label=f"{ps.label(name)}: {a['series_trend']:+.2f}, {stats.p_text(a['series_p'])}, "
                      f"n$_{{eff}}$ = {a['series_n_eff']:.0f}")
    ax.axhline(0, color="black", lw=0.6, ls="--", zorder=0)
    ax.set_xlim(min(y[0] for y in years.values()), max(y[-1] for y in years.values()))
    ax.set_ylabel(f"{calendar.month_abbr[MONTH]} anomaly (°C)")
    ax.grid(alpha=0.3, linewidth=0.4)
    ps.letter(ax, len(maps) + 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False,
              handlelength=1.2, columnspacing=1.0,
              title="Regional trend (°C decade$^{-1}$)", title_fontsize=config.FS["legend"])
    ps.text(fig, W / 2, ts_top - 0.05,
            f"{calendar.month_name[MONTH]} area-weighted land anomaly "
            f"(from {config.BASELINE[0]}-{config.BASELINE[1]})",
            ha="center", fontsize=fs["title"], fontweight="bold")
    ps.save(fig, config.RESULTS / OUT / "t2m_trends_southern_africa.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("compute", "all") and (what == "compute" or not all(
            data_file(n).exists() for n in ORDER)):
        compute()
    if what in ("plot", "all"):
        plot()