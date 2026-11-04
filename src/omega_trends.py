"""
October vertical velocity (omega) trends and regime change, ERA5, MERRA-2 and JRA-3Q,
35S-20N and 10W-60E. omega > 0 is subsidence.

Run:  python omega_trends.py [compute | plot]

FIGURES  results/omega/omega_trends_<level>hPa.png, one per level in LEVELS
  Row 1  October omega trend (Pa s-1 per decade; one colour scale for all levels and datasets)
  Row 2  regime of every cell, shaded by |trend| (white at zero, full colour at the colour-bar
         limit): strengthening / weakening subsidence or ascent from the mean sign and the trend
         sign, or new subsidence / new ascent after a persistent sign change (config.MIN_PERSIST
         years); see stats.regimes.
  White dots: significant (FDR q = 0.10 over the cells of each map, land and ocean).
TABLE    results/omega/omega_trends_summary.csv: per level and dataset, % of area significant
         and, of the significant area, % in each regime; % of the Congo box with significant
         weakening ascent; % south of 10S with significant positive trends; ocean statistics
         and the share of "new" regimes confirmed by a Pettitt test.

DATA (native grids; October; config.YEARS)
  ERA5     global/vertical_velocity/global_vertical_velocity_YYYY.nc, w
  MERRA-2  instM_3d_asm_Np OMEGA (undefined below ground)
  JRA-3Q   JRA3Q_SOURCE = "netcdf": vvel-pres-an-ll125 monthly netCDF files (default);
           "grib": 6-hourly GRIB analyses anl_p125_vvel.YYYYMMDDHH (complete months only)
  Levels below the MERRA-2 climatological October surface are masked in every dataset.
DATA FILES  data/omega/omega_<dataset>.nc: trend, p, sig, regime, pettitt (lev, lat, lon)
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

REGION = "omega"
MONTH = 10
OUT = "omega"
NAMES = ["ERA5", "MERRA-2", "JRA-3Q"]
LEVELS = [100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 850.0, 925.0, 1000.0]
JRA3Q_SOURCE = "netcdf"          # "netcdf" (monthly files in RAW['jra3q']/JRA3Q_FOLDER) or
                                 # "grib" (6-hourly anl_p125_vvel.YYYYMMDDHH analyses)
JRA3Q_FOLDER = "*vvel-pres-an-ll125"   # folder under RAW['jra3q'] (a glob pattern is fine)
CONGO = dict(lon=(10.0, 32.0), lat=(-10.0, 20.0))
REF_LAT = -10.0                  # dotted line, and "south of" for the statistics


def data_file(name):
    return config.DATA / OUT / f"omega_{name.replace('-', '')}.nc"


def load(name):
    if name == "ERA5":
        da = rd.era5_yearly(config.RAW["era5"] / "global" / "vertical_velocity",
                            "global_vertical_velocity_*.nc", "w", REGION, lev=LEVELS,
                            months=[MONTH])
    elif name == "MERRA-2":
        da = rd.merra2("instM_3d_asm_Np", "OMEGA", REGION, lev=LEVELS, months=[MONTH])
    elif JRA3Q_SOURCE == "grib":
        da = rd.jra3q_grib(str(config.RAW["jra3q_grib"] / "anl_p125_vvel"), "anl_p125_vvel",
                           REGION, lev=LEVELS, months=[MONTH], years=config.YEARS["JRA-3Q"])
    else:
        da = rd.jra3q(JRA3Q_FOLDER, "vvel-pres-an-ll125", REGION, lev=LEVELS, months=[MONTH])
    return da.sel(month=MONTH)


def compute():
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    psc = rd.surface_pressure_clim(REGION)
    for name in NAMES:
        rd.log(f"--- {name} ---")
        da = load(name)
        da = da.where(rd.above_ground_on(da, psc, MONTH)).rename(name)
        da = stats.select_years(da, config.YEARS[name])
        st = stats.ols_stats(da)
        sig = stats.fdr_maps(st.p)
        ds = xr.Dataset({"trend": st.trend.astype("float32"), "p": st.p.astype("float32"),
                         "sig": sig.astype("int8"),
                         "regime": stats.regimes(da, st.trend).astype("int8"),
                         "pettitt": stats.pettitt(da).astype("int8")})
        ds.attrs = {"dataset": name, "month": MONTH, "units": "Pa s-1 per decade",
                    "years": f"{config.YEARS[name][0]}-{config.YEARS[name][1]}"}
        ds.to_netcdf(data_file(name), encoding={v: {"zlib": True} for v in ("trend", "p")})
        for lev in ds.lev.values:
            o = ds.sel(lev=lev)
            rd.log(f"  {lev:6.0f} hPa: {stats.pct_area(o.sig.astype(bool), o.trend.notnull()):4.0f}"
                   " % of area significant")


# ---------------------------------------------------------------- statistics table
def summary(res):
    rows = []
    for name, ds in res.items():
        land = masks.land_mask(ds)
        lat2 = ds.lat.broadcast_like(ds.trend.isel(lev=0))
        lon2 = ds.lon.broadcast_like(ds.trend.isel(lev=0))
        congo = ((lon2 >= CONGO["lon"][0]) & (lon2 <= CONGO["lon"][1])
                 & (lat2 >= CONGO["lat"][0]) & (lat2 <= CONGO["lat"][1]))
        south = lat2 < REF_LAT
        for lev in ds.lev.values:
            o = ds.sel(lev=lev)
            valid, sig = o.trend.notnull(), o.sig.astype(bool)
            row = {"level_hPa": float(lev), "dataset": name, "years": ds.attrs["years"],
                   "pct_area_sig": stats.pct_area(sig, valid),
                   "pct_congo_sig_weakening_ascent": stats.pct_area(
                       sig & (o.regime == config.REGIMES.index("wk_asc")), valid & congo),
                   "pct_south_sig_positive": stats.pct_area(sig & (o.trend > 0), valid & south),
                   "pct_ocean_sig": stats.pct_area(sig, valid & ~land)}
            for k, key in enumerate(config.REGIMES):
                row[f"pct_sig_area_{key}"] = stats.pct_area(o.regime == k, valid & sig)
                row[f"pct_sig_ocean_{key}"] = stats.pct_area(o.regime == k, valid & sig & ~land)
            new = o.regime.isin([config.REGIMES.index("new_sub"), config.REGIMES.index("new_asc")])
            row["pct_new_regime_pettitt"] = stats.pct_area(o.pettitt.astype(bool), valid & new)
            rows.append(row)
    path = config.RESULTS / OUT / "omega_trends_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.1f")
    print(f"Saved: {path}", flush=True)


# ---------------------------------------------------------------- plot
def plot():
    import cartopy.crs as ccrs
    ps.setup()
    res = {n: xr.load_dataset(data_file(n)) for n in NAMES if data_file(n).exists()}
    if not res:
        raise SystemExit("no data files: run compute first")
    summary(res)
    fs, W = config.FS, config.WIDTH["full"]
    pct = config.COLOUR_PERCENTILE
    v = np.concatenate([np.abs(d.trend.values).ravel() for d in res.values()])
    vlim = float(np.nanpercentile(v, pct))
    e = int(np.floor(np.log10(vlim)))
    scale = 10.0 ** -e
    norm = ps.sym_norm([d.trend * scale for d in res.values()])
    vlim_s = norm.boundaries[-1]
    left, right, gap, gy = 0.85, 0.1, 0.12, 0.26
    r = config.REGIONS[REGION]
    mw = (W - left - right - 2 * gap) / 3
    mh = mw * (r["lat"][1] - r["lat"][0]) / (r["lon"][1] - r["lon"][0])
    top = 0.55
    tops = [top, top + mh + gy]
    cb_top = tops[1] + mh + 0.32
    H = cb_top + 1.25
    for lev in LEVELS:
        fig, at = ps.figure("full", H)
        k = 0
        for ri, rlab in enumerate(["Trend", "Regime"]):
            ps.text(fig, 0.12, tops[ri] + mh / 2, rlab, rotation=90, ha="center", va="center",
                    fontsize=fs["label"], fontweight="bold")
            for ci, name in enumerate(NAMES):
                x0 = left + ci * (mw + gap)
                ax = at(x0, tops[ri], mw, mh, projection=ccrs.PlateCarree())
                if ri == 0:
                    ps.text(fig, x0 + mw / 2, top - 0.04, f"{ps.label(name)}" +
                            (f" ({res[name].attrs['years']})" if name in res else ""),
                            ha="center", fontsize=fs["title"], fontweight="bold",
                            color=ps.colour(name))
                if name not in res:
                    ps.map_frame(ax, REGION, left=(ci == 0), bottom=(ri == 1), xticks=[0, 20, 40],
                                 yticks=[-30, -15, 0, 15])
                    continue
                o = res[name].sel(lev=lev)
                sig = o.sig.astype(bool).values
                if ri == 0:
                    ps.map_panel(ax, o.trend * scale, norm, sig=sig, region=REGION,
                                 left=(ci == 0), bottom=False, xticks=[0, 20, 40],
                                 yticks=[-30, -15, 0, 15])
                else:
                    strength = np.clip(np.abs(o.trend.values * scale) / vlim_s, 0, 1)
                    ps.categories(ax, o.lon, o.lat, o.regime.values, strength,
                                  [config.REGIME_COLOURS[k_] for k_ in config.REGIMES],
                                  transform=ccrs.PlateCarree())
                    ps.sig_dots(ax, o.trend, sig)
                    ps.map_frame(ax, REGION, left=(ci == 0), xticks=[0, 20, 40],
                                 yticks=[-30, -15, 0, 15])
                ax.plot([r["lon"][0], r["lon"][1]], [REF_LAT, REF_LAT], color="k", lw=0.6, ls=":",
                        transform=ccrs.PlateCarree(), zorder=6)
                ps.letter(ax, k)
                k += 1
        ps.colourbar(fig, at(left + 0.5, cb_top, W - left - right - 1.0, 0.08), norm, "RdBu_r",
                     f"Trend (10$^{{{e}}}$ Pa s$^{{-1}}$ decade$^{{-1}}$); dots: significant (FDR q = "
                     f"{config.ALPHA_FDR:.2f})\nRegime colours reach full strength at the colour-bar limit")
        fig.legend(handles=ps.regime_handles(), loc="upper center", ncol=3, frameon=False,
                   handlelength=1.2, columnspacing=1.0,
                   bbox_to_anchor=(0.5, 1 - (cb_top + 0.7) / H))
        ps.text(fig, W / 2, 0.06, f"{calendar.month_name[MONTH]} vertical velocity trends at "
                f"{lev:.0f} hPa (omega > 0: subsidence)", ha="center", va="top",
                fontsize=fs["suptitle"], fontweight="bold")
        ps.save(fig, config.RESULTS / OUT / f"omega_trends_{lev:04.0f}hPa.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "compute" or (what == "all" and not all(data_file(n).exists() for n in NAMES)):
        compute()
    if what in ("plot", "all"):
        plot()