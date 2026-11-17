"""
October relative humidity (RH) and specific humidity (q) trends at 850 hPa and an upper level
(UPPER, 200 hPa) over sub-Saharan Africa (10-40E, 35S-18N, as the SW and cloud figures):
ERA5, MERRA-2 and JRA-3Q.

Run:  python humidity_trends.py            compute (if the data files are missing) and plot
      python humidity_trends.py compute    re-read the raw data
      python humidity_trends.py plot       plot from the data files only

FIGURES  results/humidity/ (file names end in _<TAG>)
  humidity_boxplots_<TAG>.png   one panel per variable (RH 850, q 850, RH UPPER, q UPPER):
      monthly boxplots of the cell trends over REGION (land, above ground), area-weighted
      (box = weighted IQR, line = weighted median, whiskers = most extreme cell within 1.5
      IQR), the three reanalyses side by side
  humidity_trends_<level>hPa_october_<TAG>.png   one per level: rows RH and q, columns =
      datasets: October trend maps (black dots: significant, FDR q = 0.10 over the land cells
      of each map; one colour scale per row, brown = drier, green = moister; dashed: the
      Southern Africa box); underneath, the October REGION-mean anomaly series of RH and q
      with all three datasets (anomalies from config.BASELINE, 1991-2020) and each dataset's
      OLS trend and p
  humidity_trends_<TAG>.csv     area-weighted summary per variable, dataset and month

METHOD (stats.py, masks.py): native grids; land cells (Natural Earth 50 m land fraction >= 0.5)
  above the ground (MERRA-2 climatological surface pressure > level; matters at 850 hPa over
  the plateau) with a value in every year; anomalies from the 1991-2020 monthly climatology;
  OLS trend per decade, lag-1 n_eff <= n; Benjamini-Hochberg FDR per map.
UNITS  RH in % (trend: % points per decade); q in g kg-1 at 850 hPa and mg kg-1 at and above
  Q_MG_BELOW hPa, where it is about a thousand times smaller.
CAUTION at the upper level the reanalyses define RH differently in cold air (ERA5 over water
  above 0 C, over ice below -23 C, blended between; MERRA-2 and JRA-3Q their own conventions),
  so their RH levels (less so their trends) are not directly comparable there; q is.

DATA (as rh_decomposition.py, read over the same area so its caches are reused)
  ERA5     global/{relative_humidity, specific_humidity}/global_<var>_YYYY.nc: r, q
  MERRA-2  instM_3d_asm_Np RH (fraction -> %), QV
  JRA-3Q   rh-pres-an-ll125, spfh-pres-an-ll125 (monthly files); a level missing from the
           files is reported and its panel left empty
DATA FILES  data/humidity/<var><level>_<dataset>.nc: trend, p, n_eff, sig (month, lat, lon);
  October box-mean anomaly series (year) and its trend, p, n_eff (attributes)
"""
from __future__ import annotations

import calendar
import sys

import numpy as np
import pandas as pd
import xarray as xr

import config
import masks
import plotstyle as ps
import readers as rd
import stats

REGION = "africa_10_40"           # sub-Saharan Africa: maps, boxplots and series
READ = REGION                     # area read ("africa_south" reuses rh_decomposition.py's
                                  # caches, but only with a REGION south of the equator)
TAG = "subsaharan"                # ends the data, figure and table names
BOX = "southern_africa"           # dashed on the maps
MONTH = 10
UPPER = 200                       # upper level (hPa); 250 also works
LEVELS = [850, UPPER]
NAMES = ["ERA5", "MERRA-2", "JRA-3Q"]
OUT = "humidity"
CMAP = "BrBG"                     # brown: drier, green: moister
Q_MG_BELOW = 400                  # q in mg kg-1 at and above this level (hPa)
VARS = [("rh", 850), ("q", 850), ("rh", UPPER), ("q", UPPER)]


def data_file(var, lev, name):
    return config.DATA / OUT / f"{var}{lev}_{name.replace('-', '')}_{TAG}.nc"


def q_scale(lev):
    """(factor from g kg-1, unit) for q at this level."""
    return (1000.0, "mg kg$^{-1}$") if lev <= Q_MG_BELOW else (1.0, "g kg$^{-1}$")


def unit(var, lev):
    return "%" if var == "rh" else q_scale(lev)[1]


def label(var, lev, short=False):
    name = "RH" if var == "rh" else "$q$"
    return f"{lev} hPa {name}" if short else f"{lev} hPa {'relative' if var == 'rh' else 'specific'} humidity"


# ---------------------------------------------------------------- reading
def load(name, lev):
    """{'rh': %, 'q': g kg-1} (year, month, lat, lon) at one level over READ."""
    if name == "ERA5":
        d = config.RAW["era5"] / "global"
        f = lambda folder, v, sc: rd.era5_yearly(d / folder, f"global_{folder}_*.nc", v, READ,
                                                 scale=sc, lev=lev)
        out = {"rh": f("relative_humidity", "r", 1.0), "q": f("specific_humidity", "q", 1000.0)}
    elif name == "MERRA-2":
        f = lambda v, sc: rd.merra2("instM_3d_asm_Np", v, READ, scale=sc, lev=lev)
        out = {"rh": f("RH", 100.0), "q": f("QV", 1000.0)}
    else:
        f = lambda v, sc: rd.jra3q(v, v, READ, scale=sc, lev=lev)
        out = {"rh": f("rh-pres-an-ll125", 1.0), "q": f("spfh-pres-an-ll125", 1000.0)}
    return {k: v.drop_vars("lev", errors="ignore") for k, v in out.items()}


# ---------------------------------------------------------------- compute
def compute():
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    psc = rd.surface_pressure_clim(READ)
    for lev in LEVELS:
        for name in NAMES:
            rd.log(f"--- {name}, {lev} hPa ---")
            try:
                fields = load(name, lev)
            except Exception as e:
                rd.log(f"  [WARN] {name} {lev} hPa left out: {type(e).__name__}: {e}")
                continue
            for var, da in fields.items():
                da = rd.cut(da, REGION)
                ps_ = psc.interp(lat=da.lat, lon=da.lon, kwargs={"fill_value": None})
                keep = masks.land_mask(da)
                if config.MASK_BELOW_GROUND:
                    keep = keep & (ps_ > lev)
                if var == "q":
                    da = da * q_scale(lev)[0]
                da = da.where(keep).where(da > 0)
                da = stats.select_years(da.rename(f"{name} {var}{lev}"), config.YEARS[name])
                anom = stats.anomalies(da)
                st = stats.ols_stats(anom)                                   # (month, lat, lon)
                sig = xr.concat([stats.fdr(st.p.sel(month=m)) for m in st.month], "month")
                series = stats.regional_series(anom.sel(month=MONTH))
                sst = stats.ols_stats(series)
                ds = xr.Dataset({"trend": st.trend.astype("float32"),
                                 "p": st.p.astype("float32"),
                                 "n_eff": st.n_eff.astype("float32"), "sig": sig.astype("int8"),
                                 "series": series})
                ds.attrs = {"dataset": name, "var": var, "level_hPa": lev,
                            "units": unit(var, lev).replace("$", ""),
                            "years": f"{config.YEARS[name][0]}-{config.YEARS[name][1]}",
                            "series_trend": float(sst.trend), "series_p": float(sst.p),
                            "series_n_eff": float(sst.n_eff)}
                ds.to_netcdf(data_file(var, lev, name),
                             encoding={v: {"zlib": True} for v in ("trend", "p", "n_eff")})
                o = ds.sel(month=MONTH)
                rd.log(f"  {var}{lev}: October median trend {stats.wmedian(o.trend):+.3g} "
                       f"{ds.attrs['units']} per decade, "
                       f"{stats.pct_area(o.sig.astype(bool), o.trend.notnull()):.0f} % of area "
                       f"significant; box mean {float(sst.trend):+.3g}, "
                       f"{stats.p_text(float(sst.p))}")


# ---------------------------------------------------------------- plot
def read_all():
    res = {}
    for var, lev in VARS:
        for n in NAMES:
            if data_file(var, lev, n).exists():
                res[(var, lev, n)] = xr.load_dataset(data_file(var, lev, n))
    return res


def summary_csv(res):
    rows = []
    for (var, lev, n), ds in res.items():
        for m in ds.month.values:
            o = ds.sel(month=m)
            rows.append({"variable": var, "level_hPa": lev, "dataset": n,
                         "units_per_decade": ds.attrs["units"], "years": ds.attrs["years"],
                         "month": int(m), **stats.describe(o.trend),
                         "pct_area_sig_fdr": stats.pct_area(o.sig.astype(bool), o.trend.notnull())})
    path = config.RESULTS / OUT / f"humidity_trends_{TAG}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.4g")
    print(f"Saved: {path}", flush=True)


def plot_boxplots(res):
    fs, W = config.FS, config.WIDTH["full"]
    left, right = 0.75, 0.1
    top, bh, gy = 0.95, 1.15, 0.38
    H = top + 4 * bh + 3 * gy + 0.45
    fig, at = ps.figure("full", H)
    names = [n for n in NAMES if any(k[2] == n for k in res)]
    years = {n: ps.year_list(next(d for k, d in res.items() if k[2] == n).attrs["years"])
             for n in names}
    for i, (var, lev) in enumerate(VARS):
        ax = at(left, top + i * (bh + gy), W - left - right, bh)
        here = [n for n in names if (var, lev, n) in res]
        if here:
            ps.monthly_boxplot(ax, {n: {int(m): res[(var, lev, n)].trend.sel(month=m)
                                        for m in res[(var, lev, n)].month.values} for n in here},
                               here, f"Trend\n({unit(var, lev)} decade$^{{-1}}$)")
        ax.axhline(0, color="k", lw=0.5, zorder=0)
        ax.set_title(label(var, lev), loc="left", fontsize=fs["title"], fontweight="bold", pad=3)
        ax.tick_params(labelbottom=(i == len(VARS) - 1))
        ps.letter(ax, i)
        if i == 0:
            ps.dataset_legend(fig, names, years, loc="upper center",
                              bbox_to_anchor=(0.5, 1 - 0.3 / H), ncol=3, handlelength=1.2,
                              columnspacing=1.0, frameon=False)
    ps.text(fig, W / 2, 0.06, "Humidity trends over sub-Saharan Africa (land, area-weighted)",
            ha="center", va="top", fontsize=fs["suptitle"], fontweight="bold")
    ps.save(fig, config.RESULTS / OUT / f"humidity_boxplots_{TAG}.png")


def plot_maps(res, lev):
    """One level: rows RH and q, columns datasets; the two anomaly series underneath."""
    import cartopy.crs as ccrs
    from matplotlib.lines import Line2D
    fs, W = config.FS, config.WIDTH["full"]
    vars_ = [("rh", lev), ("q", lev)]
    left, right, gx, gy = 0.62, 0.08, 0.1, 0.16
    cbw = 0.72                                        # vertical colour bar beside each row
    r = config.REGIONS[REGION]
    aspect = (r["lat"][1] - r["lat"][0]) / (r["lon"][1] - r["lon"][0])
    pw = min((W - left - right - cbw - 2 * gx) / 3, 2.55 / aspect)  # keep within a page
    mh = pw * aspect
    left = left + (W - left - right - cbw - 3 * pw - 2 * gx) / 2     # centre the maps
    head = 0.4
    top = 0.5 + head
    tops = [top + i * (mh + gy) for i in range(len(vars_))]
    ts_top = tops[-1] + mh + 0.8
    th, tgx = 0.95, 0.5
    tl = 0.62                                         # series panels use the full width
    tw = (W - tl - right - tgx) / 2
    H = ts_top + th + 0.62
    fig, at = ps.figure("full", H)
    mid = left + (3 * pw + 2 * gx) / 2
    ps.text(fig, mid, 0.36, f"{calendar.month_name[MONTH]} trends", ha="center",
            fontsize=fs["title"], fontweight="bold")
    ps.text(fig, mid, 0.53, f"dots: significant (FDR q = {config.ALPHA_FDR:.2f}); dashed: "
            "Southern Africa (10-40°E, 35-10°S)", ha="center", fontsize=fs["small"] - 1,
            color="0.3")
    for ci, n in enumerate(NAMES):
        d = next((res[k] for k in res if k[2] == n and k[1] == lev), None)
        x0 = left + ci * (pw + gx)
        ps.text(fig, x0 + pw / 2, top - 0.2, ps.label(n), ha="center", fontsize=fs["title"],
                fontweight="bold", color=ps.colour(n))
        if d is not None:
            ps.text(fig, x0 + pw / 2, top - 0.03, d.attrs["years"], ha="center",
                    fontsize=fs["small"], color="0.25")
    k = 0
    yt = [-30, -15, 0, 15]
    for ri, (var, lv) in enumerate(vars_):
        t0 = tops[ri]
        row = [res[(var, lv, n)].trend.sel(month=MONTH) for n in NAMES if (var, lv, n) in res]
        norm = ps.sym_norm(row or [[1.0]], cmap=CMAP)
        ps.text(fig, left - 0.55, t0 + mh / 2, label(var, lv, short=True), rotation=90,
                ha="center", va="center", fontsize=fs["label"], fontweight="bold")
        for ci, n in enumerate(NAMES):
            ax = at(left + ci * (pw + gx), t0, pw, mh, projection=ccrs.PlateCarree())
            last = ri == len(vars_) - 1
            if (var, lv, n) in res:
                o = res[(var, lv, n)].sel(month=MONTH)
                ps.map_panel(ax, o.trend, norm, cmap=CMAP, sig=o.sig.astype(bool).values,
                             region=REGION, left=(ci == 0), bottom=last, box=BOX,
                             xticks=[15, 30], yticks=yt)
            else:
                ps.map_frame(ax, REGION, left=(ci == 0), bottom=last, xticks=[15, 30],
                             yticks=yt)
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                        va="center", fontsize=fs["small"], color="0.4")
            ps.letter(ax, k)
            k += 1
        ps.colourbar(fig, at(left + 3 * pw + 2 * gx + 0.12, t0 + 0.1, 0.08, mh - 0.2), norm,
                     CMAP, f"{unit(var, lv)} decade$^{{-1}}$", orientation="vertical", nticks=5)

    ps.text(fig, tl + (W - tl - right) / 2, ts_top - 0.45, f"{calendar.month_name[MONTH]} "
            f"anomaly over sub-Saharan Africa (from {config.BASELINE[0]}-{config.BASELINE[1]})",
            ha="center", fontsize=fs["title"], fontweight="bold")
    for i, (var, lv) in enumerate(vars_):
        ax = at(tl + i * (tw + tgx), ts_top, tw, th)
        lines = []
        for n in NAMES:
            d = res.get((var, lv, n))
            if d is None:
                continue
            ax.plot(d.year, d.series, color=ps.colour(n), lw=1.1)
            lines.append((n, d.attrs))
        ax.axhline(0, color="k", lw=0.5, ls="--", zorder=0)
        ax.grid(alpha=0.3, lw=0.4)
        ax.set_title(label(var, lv, short=True), loc="left", fontsize=fs["small"] + 0.5,
                     fontweight="bold", pad=2)
        ax.set_ylabel(unit(var, lv), fontsize=fs["small"])
        ax.tick_params(labelsize=fs["tick"] - 0.5)
        for j, (n, a) in enumerate(lines):           # trend per decade and p, top right
            ax.text(0.99, 0.97 - 0.12 * j, f"{a['series_trend']:+.2g} decade$^{{-1}}$, "
                    f"{stats.p_text(a['series_p'])}", transform=ax.transAxes, ha="right",
                    va="top", fontsize=fs["small"] - 1.5, color=ps.colour(n),
                    fontweight="bold")
        ps.letter(ax, k)
        k += 1
    fig.legend(handles=[Line2D([], [], color=ps.colour(n), lw=1.4, label=ps.label(n))
                        for n in NAMES], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.5)
    ps.text(fig, W / 2, 0.06, f"Relative and specific humidity trends at {lev} hPa", ha="center",
            va="top", fontsize=fs["suptitle"], fontweight="bold")
    ps.save(fig, config.RESULTS / OUT /
            f"humidity_trends_{lev}hPa_{calendar.month_name[MONTH].lower()}_{TAG}.png")


def plot():
    res = read_all()
    if not res:
        raise SystemExit("no data files: run 'python humidity_trends.py compute' first")
    ps.setup()
    summary_csv(res)
    plot_boxplots(res)
    for lev in LEVELS:
        plot_maps(res, lev)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    missing = not all(data_file(v, l, n).exists() for v, l in VARS for n in NAMES)
    if what == "compute" or (what == "all" and missing):
        compute()
    if what in ("plot", "all"):
        plot()
