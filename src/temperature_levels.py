"""
October air-temperature trends on every pressure level over Southern Africa (10-40E, 35-10S):
ERA5, MERRA-2 and JRA-3Q. (temperature_levels_monthly.py draws the same figures for every
month from the same data files.)

Run:  python temperature_levels.py            compute (if the data files are missing) and plot
      python temperature_levels.py compute    re-read the raw data and rewrite the data files
      python temperature_levels.py plot       plot from the data files only

FIGURES  results/temperature_levels/<month>/
  ta_levels_maps_<month>_<dataset>.png  one figure per dataset: a trend map for every level
           in LEVELS (top left 100 hPa ... bottom right 1000 hPa); one colour scale shared by
           the three datasets' figures; black dots: significant (FDR q = 0.10 over the land
           cells of each map); white: below the ground or sea
  ta_levels_profile_<month>.png  a) box-mean trend profiles of the three datasets (filled
           markers: p < 0.05 for the regional series); b-d) latitude-pressure sections of the
           trend of the 10-40E land mean (dots: FDR q = 0.10 over each section)
  ta_levels_summary.csv  box-mean trend and p per dataset, month and level

METHOD (stats.py, masks.py, as humidity_trends.py and rh_decomposition.py)
  Native grids; land cells (Natural Earth 50 m land fraction >= 0.5) above the ground
  (MERRA-2 climatological surface pressure of that month > level) with a value in every
  year; OLS trend per decade, lag-1 n_eff <= n; Benjamini-Hochberg FDR per map. The box mean
  is the area-weighted (cos latitude) mean of those cells, so near the surface it covers
  only the lower-lying cells (850 hPa leaves out the highveld; 1000 hPa only the coasts).
UNITS  °C (= K) per decade.

DATA (the temperature folders of rh_decomposition.py)
  ERA5     global/temperature/global_temperature_YYYY.nc: t
  MERRA-2  instM_3d_asm_Np T
  JRA-3Q   JRA3Q_T_SOURCE: "grib" = anl_p125_tmp 6-hourly GRIB (config.RAW['jra3q_grib']),
           "netcdf" = tmp-pres-an-ll125
  Levels in LEVELS that a dataset does not have are reported and left out of its panels.
DATA FILES  data/temperature_levels/ta_levels_<dataset>.nc
  trend, p, sig (month, lev, lat, lon); box_trend, box_p (month, lev); sec_trend, sec_p,
  sec_sig (month, lev, lat)
"""
from __future__ import annotations

import calendar
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

import config
import masks
import plotstyle as ps
import readers as rd
import stats

READ = BOX = "southern_africa"          # area read, mapped and averaged
MONTH = 10
NAMES = ["ERA5", "MERRA-2", "JRA-3Q"]
LEVELS = [100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]   # hPa, map order
JRA3Q_T_SOURCE = "grib"
OUT = "temperature_levels"
CMAP = "RdBu_r"
MAP_COLS = 4
TICKS = dict(xticks=[15, 25, 35], yticks=[-30, -20])
P_PROFILE = 0.05                        # filled profile markers
UNIT = "°C decade$^{-1}$"


def data_file(name):
    return config.DATA / OUT / f"ta_levels_{name.replace('-', '')}.nc"


def fig_dir(month):
    return config.RESULTS / OUT / calendar.month_name[month].lower()


# ---------------------------------------------------------------- reading
def _levels_of(path, engine=None):
    kw = {}
    if engine == "cfgrib":
        kw = dict(engine="cfgrib", backend_kwargs={
            "indexpath": "", "filter_by_keys": {"typeOfLevel": "isobaricInhPa"}})
    with xr.open_dataset(path, **kw) as ds:
        da = rd.standardise(ds[list(ds.data_vars)[0]] if engine == "cfgrib" else
                            ds[[v for v in ds.data_vars if "lev" in rd.standardise(ds[v]).dims][0]])
        return set(float(v) for v in np.atleast_1d(da.lev.values))


def available(name):
    """The levels of LEVELS that the dataset's files hold (all of LEVELS if it cannot tell)."""
    try:
        if name == "ERA5":
            f = sorted(glob.glob(str(config.RAW["era5"] / "global/temperature/global_temperature_*.nc")))[0]
            have = _levels_of(f)
        elif name == "MERRA-2":
            have = _levels_of(next(iter(rd.merra2_files("instM_3d_asm_Np").values())))
        elif JRA3Q_T_SOURCE == "grib":
            d = str(config.RAW["jra3q_grib"] / "anl_p125_tmp")
            f = sorted(x for x in os.listdir(d) if x.startswith("anl_p125_tmp."))[0]
            have = _levels_of(os.path.join(d, f), "cfgrib")
        else:
            have = _levels_of(next(iter(rd.jra3q_files("tmp-pres-an-ll125").values())))
    except Exception as e:
        rd.log(f"  [WARN] {name}: could not list the levels ({type(e).__name__}: {e}); "
               "trying all of LEVELS")
        return list(LEVELS)
    use = [l for l in LEVELS if any(abs(l - h) < 0.5 for h in have)]
    miss = [l for l in LEVELS if l not in use]
    if miss:
        rd.log(f"  [WARN] {name}: no {', '.join(map(str, miss))} hPa in the files; left out")
    return use


def load(name):
    """T (K): (year, month, lev, lat, lon) over READ at the dataset's levels of LEVELS."""
    lev = available(name)
    if name == "ERA5":
        d = config.RAW["era5"] / "global/temperature"
        T = rd.era5_yearly(d, "global_temperature_*.nc", "t", READ, lev=lev)
    elif name == "MERRA-2":
        T = rd.merra2("instM_3d_asm_Np", "T", READ, lev=lev)
    elif JRA3Q_T_SOURCE == "grib":
        T = rd.jra3q_grib(str(config.RAW["jra3q_grib"] / "anl_p125_tmp"), "anl_p125_tmp",
                          READ, lev=lev, years=config.YEARS["JRA-3Q"])
    else:
        T = rd.jra3q("tmp-pres-an-ll125", "tmp-pres-an-ll125", READ, lev=lev)
    if "lev" not in T.dims:                                    # a single level
        T = T.expand_dims(lev=[float(lev[0])])
    T = T.assign_coords(lev=np.round(T.lev.values.astype(float)))
    return T.transpose("year", "month", "lev", "lat", "lon")


# ---------------------------------------------------------------- compute
def compute():
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    psc = rd.surface_pressure_clim(READ)
    for name in NAMES:
        rd.log(f"--- {name} ---")
        T = load(name)
        keep = masks.land_mask(T)
        if config.MASK_BELOW_GROUND:
            ps_ = psc.interp(lat=T.lat, lon=T.lon, kwargs={"fill_value": None})
            keep = keep & (T.lev <= ps_)
        T = stats.select_years(T.where(keep).rename(f"{name} T"), config.YEARS[name])
        T = T.astype("float64")
        st = stats.ols_stats(T)
        sig = stats.fdr_maps(st.p, over=("lat", "lon"))
        box = stats.ols_stats(stats.regional_series(rd.cut(T, BOX)))
        sec = stats.ols_stats(stats.complete(T).mean("lon"))
        sec_sig = stats.fdr_maps(sec.p, over=("lev", "lat"))
        ds = xr.Dataset({"trend": st.trend.astype("float32"), "p": st.p.astype("float32"),
                         "sig": sig, "box_trend": box.trend, "box_p": box.p,
                         "sec_trend": sec.trend.astype("float32"), "sec_p": sec.p.astype("float32"),
                         "sec_sig": sec_sig})
        ds.attrs = {"dataset": name, "units": "degC per decade",
                    "years": f"{config.YEARS[name][0]}-{config.YEARS[name][1]}"}
        ds.to_netcdf(data_file(name), encoding={k: {"zlib": True} for k in ("trend", "p", "sig")})
        o = ds.sel(month=MONTH)
        txt = ", ".join(f"{int(l)}: {float(t):+.2f}{'*' if float(p) < P_PROFILE else ''}"
                        for l, t, p in zip(o.lev.values, o.box_trend.values, o.box_p.values))
        rd.log(f"  {calendar.month_abbr[MONTH]} box means (°C/decade, * p < {P_PROFILE}): {txt}")


def read_all():
    res = {n: xr.load_dataset(data_file(n)) for n in NAMES if data_file(n).exists()}
    if not res:
        raise SystemExit("no data files: run 'python temperature_levels.py compute' first")
    return res


def summary_csv(res):
    rows = [{"dataset": n, "years": d.attrs["years"], "month": int(m), "level_hPa": int(l),
             "box_trend_degC_per_decade": float(d.box_trend.sel(month=m, lev=l)),
             "box_p": float(d.box_p.sel(month=m, lev=l)),
             "pct_box_significant": stats.pct_area(d.sig.sel(month=m, lev=l),
                                                   d.trend.sel(month=m, lev=l).notnull())}
            for n, d in res.items() for m in d.month.values for l in d.lev.values]
    p = config.RESULTS / OUT / "ta_levels_summary.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False, float_format="%.4f")
    print(f"Saved: {p}", flush=True)


# ---------------------------------------------------------------- plot
def norm_for(res, months):
    """One colour scale: the 98th percentile of |trend| over the datasets, levels and months."""
    return ps.sym_norm([d.trend.sel(month=m).values for d in res.values() for m in months],
                       cmap=CMAP)


def plot_maps(res, month, norm):
    import cartopy.crs as ccrs
    fs, W = config.FS, config.WIDTH["full"]
    r = config.REGIONS[BOX]
    left, right, gap, vgap = 0.45, 0.1, 0.08, 0.12
    pw = (W - left - right - (MAP_COLS - 1) * gap) / MAP_COLS
    ph = pw * (r["lat"][1] - r["lat"][0]) / (r["lon"][1] - r["lon"][0])
    rows = int(np.ceil(len(LEVELS) / MAP_COLS))
    top = 0.55
    H = top + rows * (ph + vgap) + 0.75
    for name, d in res.items():
        fig, at = ps.figure("full", H)
        ps.text(fig, W / 2, top - 0.12,
                f"{calendar.month_name[month]} temperature trends, {ps.label(name)} "
                f"({d.attrs['years']})", ha="center", fontsize=fs["title"], fontweight="bold",
                color=ps.colour(name))
        for i, lev in enumerate(LEVELS):
            rr, cc = divmod(i, MAP_COLS)
            ax = at(left + cc * (pw + gap), top + rr * (ph + vgap), pw, ph,
                    projection=ccrs.PlateCarree())
            if lev in d.lev.values:
                o = d.sel(month=month, lev=lev)
                ps.map_panel(ax, o.trend, norm, CMAP, o.sig.values, BOX, left=(cc == 0),
                             bottom=(rr == rows - 1), **TICKS)
                if o.trend.notnull().sum() == 0:
                    ax.text(0.5, 0.5, "below ground", transform=ax.transAxes, ha="center",
                            va="center", fontsize=fs["small"], color="0.4")
            else:
                ps.map_frame(ax, BOX, left=(cc == 0), bottom=(rr == rows - 1), **TICKS)
                ax.text(0.5, 0.5, "level not in the files", transform=ax.transAxes,
                        ha="center", va="center", fontsize=fs["small"], color="0.4")
            ax.text(0.98, 0.04, f"{lev} hPa", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=fs["small"], fontweight="bold", zorder=10,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
            ps.letter(ax, i)
        cb_top = top + rows * (ph + vgap) + 0.12
        ps.colourbar(fig, at(left + 0.6, cb_top, W - left - right - 1.2, 0.1), norm, CMAP,
                     f"Trend ({UNIT}); dots: significant (FDR q = 0.10)")
        ps.save(fig, fig_dir(month) / f"ta_levels_maps_{calendar.month_abbr[month].lower()}_"
                                      f"{name.replace('-', '')}.png")


def pressure_axis(ax, label=True):
    ax.set_yscale("log")
    ax.set_ylim(max(LEVELS) + 25, min(LEVELS) - 5)
    ax.set_yticks(LEVELS)
    ax.set_yticklabels(["" if l == 925 else str(l) for l in LEVELS])   # crowded at 925
    ax.minorticks_off()
    ax.tick_params(labelsize=config.FS["tick"], length=2, labelleft=label)
    if label:
        ax.set_ylabel("Pressure (hPa)", fontsize=config.FS["label"])


def plot_profile(res, month, sec_norm, xlim=None):
    from matplotlib.lines import Line2D
    fs, W = config.FS, config.WIDTH["full"]
    left, gap, right = 0.55, 0.12, 0.1
    pw0 = 1.55
    pw = (W - left - pw0 - 0.45 - 2 * gap - right) / 3
    top, h = 0.5, 2.6
    H = top + h + 1.45
    fig, at = ps.figure("full", H)
    # a) box-mean profiles
    ax = at(left, top, pw0, h)
    for name, d in res.items():
        o = d.sel(month=month)
        lev, t, p = o.lev.values, o.box_trend.values, o.box_p.values
        ax.plot(t, lev, color=ps.colour(name), lw=1.2, zorder=3)
        s = p < P_PROFILE
        ax.scatter(t[s], lev[s], s=16, color=ps.colour(name), edgecolor="black", lw=0.4, zorder=4)
        ax.scatter(t[~s], lev[~s], s=16, facecolor="white", edgecolor=ps.colour(name), lw=0.9,
                   zorder=4)
    ax.axvline(0, color="black", lw=0.6)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(alpha=0.3, lw=0.4)
    pressure_axis(ax)
    ax.set_xlabel(f"Box-mean trend\n({UNIT})", fontsize=fs["label"])
    ax.set_title("10-40°E, 35-10°S, land", fontsize=fs["small"], pad=3)
    ps.letter(ax, 0)
    h_ = [Line2D([], [], color=ps.colour(n), lw=1.2, marker="o", markersize=3.5,
                 markeredgecolor="black", markeredgewidth=0.4,
                 label=f"{ps.label(n)} ({d.attrs['years']})") for n, d in res.items()]
    h_ += [Line2D([], [], ls="none", marker="o", markersize=3.5, markerfacecolor="white",
                  markeredgecolor="0.3", label=f"p ≥ {P_PROFILE}")]
    fig.legend(handles=h_, loc="upper left", bbox_to_anchor=(0.05 / W, 1 - (top + h + 0.62) / H),
               ncol=1, frameon=False, fontsize=fs["small"], handlelength=1.6, labelspacing=0.3)
    # b-d) latitude-pressure sections of the 10-40E land mean
    x0 = left + pw0 + 0.45
    for i, name in enumerate(NAMES):
        ax = at(x0 + i * (pw + gap), top, pw, h)
        ax.set_facecolor("0.85")                                # below the ground / no land
        if name in res:
            o = res[name].sel(month=month)
            lat, lev = o.lat.values, o.lev.values
            ax.pcolormesh(lat, lev, o.sec_trend.values, cmap=CMAP, norm=sec_norm,
                          shading="nearest", rasterized=True, zorder=1)
            ps.section_dots(ax, lat, lev, o.sec_sig.values, np.isfinite(o.sec_trend.values))
            ax.set_title(f"{ps.label(name)}", fontsize=fs["small"], fontweight="bold",
                         color=ps.colour(name), pad=3)
        r = config.REGIONS[BOX]
        ax.set_xlim(*r["lat"])
        ax.set_xticks([-30, -20])
        ax.set_xticklabels(["30°S", "20°S"])
        pressure_axis(ax, label=False)
        ps.letter(ax, i + 1)
        if i == 1:
            ax.set_xlabel("Latitude (10-40°E land mean)", fontsize=fs["label"])
    ps.colourbar(fig, at(x0 + 0.2, top + h + 0.55, 3 * pw + 2 * gap - 0.4, 0.1), sec_norm, CMAP,
                 f"Trend ({UNIT}); dots: FDR q = 0.10")
    ps.text(fig, W / 2, top - 0.36, f"{calendar.month_name[month]} temperature trends by level",
            ha="center", fontsize=fs["title"], fontweight="bold")
    ps.save(fig, fig_dir(month) / f"ta_levels_profile_{calendar.month_abbr[month].lower()}.png")


def profile_xlim(res, months):
    v = [abs(float(x)) for d in res.values() for m in months
         for x in d.box_trend.sel(month=m).values if np.isfinite(x)]
    lim = max(v + [0.05]) * 1.1
    return -lim, lim


def plot_month(res, month, norm=None, sec_norm=None, xlim=None):
    plot_maps(res, month, norm or norm_for(res, [month]))
    sec_norm = sec_norm or ps.sym_norm([d.sec_trend.sel(month=month).values for d in res.values()],
                                       cmap=CMAP)
    plot_profile(res, month, sec_norm, xlim)


def plot():
    ps.setup()
    res = read_all()
    summary_csv(res)
    plot_month(res, MONTH)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "compute" or (what != "plot" and not all(data_file(n).exists() for n in NAMES)):
        compute()
    plot()
