"""
October vertical velocity (omega) trends and regime change over the whole globe, ERA5, MERRA-2
and JRA-3Q: the global counterpart of omega_trends.py (35S-20N, 10W-60E), for comparison.
omega > 0 is subsidence.

Run:  python omega_trends_global.py [compute | plot] [level ...]      (hPa; default: LEVELS)
      As a SLURM array over levels, e.g. --array=0-11: task i does LEVELS[i].
      Each level is read, computed and saved on its own (a global ERA5 field at all levels is
      too large to hold and cache at once).

FIGURES  results/omega_global/omega_global_<level>hPa.png
  Rows = datasets; column 1 October omega trend (Pa s-1 per decade; one colour scale for all
  datasets at that level); column 2 regime of every cell (stats.regimes, as omega_trends.py),
  shaded by |trend| (full colour at the colour-bar limit). Black dots: significant (FDR q =
  0.10 over all cells of each map, land and ocean). Dashed: the omega_trends.py region and
  the Southern Africa box.
TABLE    results/omega_global/omega_global_summary.csv, per level and dataset, for the globe,
  land, ocean, the tropics (23.5S-23.5N), the omega_trends.py region and Southern Africa:
  % of area significant, area-weighted mean trend, and the % of the significant area in each
  regime; plus the share of "new" regimes confirmed by a Pettitt test.

DATA (native grids; October; config.YEARS) as omega_trends.py: ERA5 global_vertical_velocity_
  YYYY.nc w; MERRA-2 instM_3d_asm_Np OMEGA; JRA-3Q vvel-pres-an-ll125 (netCDF) or the GRIB
  analyses (JRA3Q_SOURCE). Levels below the MERRA-2 climatological October surface are masked.
DATA FILES  data/omega_global/omega_<dataset>_<level>.nc: trend, p, sig, regime, pettitt
"""
from __future__ import annotations

import calendar
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
import omega_trends as ot

REGION = "global"
MONTH = 10
OUT = "omega_global"
NAMES = ["ERA5", "MERRA-2", "JRA-3Q"]
LEVELS = ot.LEVELS
COMPARE = [("globe", None), ("tropics", "tropics_band"), ("africa_region", "omega"),
           ("southern_africa", "southern_africa")]


def data_file(name, lev):
    return config.DATA / OUT / f"omega_{name.replace('-', '')}_{lev:04.0f}.nc"


def load(name, lev):
    """(year, lat, lon) October omega at one level, Pa s-1."""
    if name == "ERA5":
        da = rd.era5_yearly(config.RAW["era5"] / "global" / "vertical_velocity",
                            "global_vertical_velocity_*.nc", "w", REGION, lev=lev, months=[MONTH])
    elif name == "MERRA-2":
        da = rd.merra2("instM_3d_asm_Np", "OMEGA", REGION, lev=lev, months=[MONTH])
    elif ot.JRA3Q_SOURCE == "grib":
        da = rd.jra3q_grib(str(config.RAW["jra3q_grib"] / "anl_p125_vvel"), "anl_p125_vvel",
                           REGION, lev=lev, months=[MONTH], years=config.YEARS["JRA-3Q"])
    else:
        da = rd.jra3q(ot.JRA3Q_FOLDER, "vvel-pres-an-ll125", REGION, lev=lev, months=[MONTH])
    return da.sel(month=MONTH).squeeze(drop=True)


def compute(levels=LEVELS):
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    psc = rd.surface_pressure_clim(REGION, months=[MONTH]).sel(month=MONTH)
    for lev in levels:
        for name in NAMES:
            rd.log(f"--- {name}, {lev:.0f} hPa ---")
            da = load(name, lev)
            ps_ = psc.interp(lat=da.lat, lon=da.lon, kwargs={"fill_value": None})
            da = da.where(lev <= ps_) if config.MASK_BELOW_GROUND else da
            da = stats.select_years(da.rename(f"{name} {lev:.0f} hPa"), config.YEARS[name])
            st = stats.ols_stats(da)
            sig = stats.fdr(st.p)
            ds = xr.Dataset({"trend": st.trend.astype("float32"), "p": st.p.astype("float32"),
                             "sig": sig.astype("int8"),
                             "regime": stats.regimes(da, st.trend).astype("int8"),
                             "pettitt": stats.pettitt(da).astype("int8")})
            ds.attrs = {"dataset": name, "month": MONTH, "level_hPa": float(lev),
                        "units": "Pa s-1 per decade",
                        "years": f"{config.YEARS[name][0]}-{config.YEARS[name][1]}"}
            ds.to_netcdf(data_file(name, lev),
                         encoding={v: {"zlib": True, "complevel": 4} for v in ("trend", "p")})
            rd.log(f"  {stats.pct_area(sig, st.trend.notnull()):.0f} % of the globe significant; "
                   f"saved {data_file(name, lev).name}")


# ---------------------------------------------------------------- statistics table
def summary(levels):
    rows = []
    for lev in levels:
        for name in NAMES:
            f = data_file(name, lev)
            if not f.exists():
                continue
            o = xr.load_dataset(f)
            land = masks.land_mask(o)
            areas = {"land": land, "ocean": ~land}
            for key, reg in COMPARE:
                if reg is None:
                    areas[key] = xr.ones_like(land)
                else:
                    r = config.REGIONS[reg]
                    areas[key] = ((o.lon >= r["lon"][0]) & (o.lon <= r["lon"][1])
                                  & (o.lat >= r["lat"][0]) & (o.lat <= r["lat"][1]))
            valid, sig = o.trend.notnull(), o.sig.astype(bool)
            for area, inside in areas.items():
                v = valid & inside
                row = {"level_hPa": float(lev), "dataset": name, "years": o.attrs["years"],
                       "area": area, "pct_area_sig": stats.pct_area(sig, v),
                       "mean_trend": float(stats.area_mean(o.trend.where(v)))}
                for k, key in enumerate(config.REGIMES):
                    row[f"pct_sig_area_{key}"] = stats.pct_area(o.regime == k, v & sig)
                new = o.regime.isin([config.REGIMES.index("new_sub"),
                                     config.REGIMES.index("new_asc")])
                row["pct_new_regime_pettitt"] = stats.pct_area(o.pettitt.astype(bool), v & new)
                rows.append(row)
    if not rows:
        return
    path = config.RESULTS / OUT / "omega_global_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.3g")
    print(f"Saved: {path}", flush=True)


# ---------------------------------------------------------------- plot
def outline(ax, region, **kw):
    import cartopy.crs as ccrs
    r = config.REGIONS[region]
    ax.plot([r["lon"][0], r["lon"][1], r["lon"][1], r["lon"][0], r["lon"][0]],
            [r["lat"][0], r["lat"][0], r["lat"][1], r["lat"][1], r["lat"][0]],
            transform=ccrs.PlateCarree(), zorder=6, **kw)


def plot(levels=LEVELS):
    import cartopy.crs as ccrs
    ps.setup()
    summary(levels)
    fs, W = config.FS, config.WIDTH["full"]
    cols = [config.REGIME_COLOURS[k_] for k_ in config.REGIMES]
    ticks = dict(xticks=[-120, -60, 0, 60, 120], yticks=[-60, -30, 0, 30, 60])
    left, right, gap, gy = 0.62, 0.1, 0.12, 0.3
    mw = (W - left - right - gap) / 2
    mh = mw / 2
    top = 0.62
    for lev in levels:
        res = {n: xr.load_dataset(data_file(n, lev)) for n in NAMES if data_file(n, lev).exists()}
        if not res:
            continue
        names = list(res)
        v = np.concatenate([np.abs(d.trend.values).ravel() for d in res.values()])
        vlim = float(np.nanpercentile(v, config.COLOUR_PERCENTILE))
        e = int(np.floor(np.log10(vlim))) if vlim > 0 else 0
        scale = 10.0 ** -e
        norm = ps.sym_norm([d.trend * scale for d in res.values()])
        vlim_s = norm.boundaries[-1]
        tops = [top + i * (mh + gy) for i in range(len(names))]
        cb_top = tops[-1] + mh + 0.4
        H = cb_top + 1.55
        fig, at = ps.figure("full", H)
        for ci, head in enumerate(["Trend", "Regime"]):
            ps.text(fig, left + ci * (mw + gap) + mw / 2, top - 0.25, head, ha="center",
                    fontsize=fs["title"], fontweight="bold")
        k = 0
        for ri, name in enumerate(names):
            o = res[name]
            ps.text(fig, 0.12, tops[ri] + mh / 2, f"{ps.label(name)}\n{o.attrs['years']}",
                    rotation=90, ha="center", va="center", fontsize=fs["label"],
                    fontweight="bold", color=ps.colour(name))
            sig = o.sig.astype(bool).values
            for ci in range(2):
                ax = at(left + ci * (mw + gap), tops[ri], mw, mh, projection=ccrs.PlateCarree())
                kw = dict(left=(ci == 0), bottom=(ri == len(names) - 1), **ticks)
                if ci == 0:
                    ps.map_panel(ax, o.trend * scale, norm, sig=sig, region=REGION, **kw)
                else:
                    strength = np.clip(np.abs(o.trend.values * scale) / vlim_s, 0, 1)
                    ps.categories(ax, o.lon, o.lat, o.regime.values, strength, cols,
                                  transform=ccrs.PlateCarree())
                    ps.sig_dots(ax, o.trend, sig)
                    ps.map_frame(ax, REGION, **kw)
                outline(ax, "southern_africa", color="k", lw=0.6)
                ps.text(fig, left + ci * (mw + gap) + mw, tops[ri] - 0.03,
                        f"{stats.pct_area(o.sig.astype(bool), o.trend.notnull()):.0f}% of area sig."
                        if ci == 0 else "", ha="right", fontsize=fs["small"], color="0.25")
                ps.letter(ax, k)
                k += 1
        ps.colourbar(fig, at(left + 0.5, cb_top, W - left - right - 1.0, 0.08), norm, "RdBu_r",
                     f"Trend (10$^{{{e}}}$ Pa s$^{{-1}}$ decade$^{{-1}}$); dots: significant (FDR q = "
                     f"{config.ALPHA_FDR:.2f})\nRegime colours reach full strength at the "
                     "colour-bar limit")
        ps.text(fig, W / 2, cb_top + 0.62, "Dashed: region of the Africa omega figure "
                "(10W-60E, 35S-20N); solid: Southern Africa box (10-40E, 35-10S)", ha="center",
                va="top", fontsize=fs["small"], color="0.25")
        fig.legend(handles=ps.regime_handles(), loc="upper center", ncol=3, frameon=False,
                   handlelength=1.2, columnspacing=1.0,
                   bbox_to_anchor=(0.5, 1 - (cb_top + 0.85) / H))
        ps.text(fig, W / 2, 0.06, f"{calendar.month_name[MONTH]} vertical velocity trends at "
                f"{lev:.0f} hPa, globe (omega > 0: subsidence)", ha="center", va="top",
                fontsize=fs["suptitle"], fontweight="bold")
        ps.save(fig, config.RESULTS / OUT / f"omega_global_{lev:04.0f}hPa.png")


def levels_from(args):
    levs = [float(a) for a in args if a.replace(".", "", 1).isdigit()]
    t = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    if not levs and t.isdigit() and int(t) < len(LEVELS):
        levs = [LEVELS[int(t)]]
    return levs or LEVELS


if __name__ == "__main__":
    what = next((a for a in sys.argv[1:] if a in ("compute", "plot")), "all")
    levs = levels_from(sys.argv[1:])
    missing = not all(data_file(n, l).exists() for n in NAMES for l in levs)
    if what == "compute" or (what == "all" and missing):
        compute(levs)
    if what in ("plot", "all"):
        plot(levs)
