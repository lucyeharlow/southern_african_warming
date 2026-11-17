"""
Detrended correlation between omega at 300 hPa and a surface variable (2 m temperature,
downwelling shortwave) for ERA5, MERRA-2 and JRA-3Q, mapped over Southern Africa (10-40E,
35S-10N, land), with the colour coding of the omega-CF script.

Run:  python omega_surface_correlation.py [compute | plot] [--var t2m,sw] [month ...]
      As a SLURM array (--array=1-12) each task does month SLURM_ARRAY_TASK_ID; default October.

FIGURE  results/omega_surface/omega300_<var>_<MM>_<Mon>.png
  Rows = datasets; columns:
  1 trend of X (one colour scale for all datasets)
  2 detrended r(omega300, X)
  3 X decreasing, 4 X increasing: colour = omega300 regime (stats.regimes), strength = |r|,
    light grey = X changes the other way.
  Black dots: significant (FDR q = 0.10 over the land cells of each map). Dashed: the
  Southern Africa box (10-40E, 35-10S).
METHOD  stats.corr_and_p on the detrended series (slope t test, n_eff - 3 df); cells need a
  value in every year. X is put on the omega grid only where the two grids differ (JRA-3Q's
  Gaussian-grid T2m, linear interpolation); otherwise both stay on the native grid.
DATA    omega: ERA5 global_vertical_velocity_YYYY.nc (w), MERRA-2 instM_3d_asm_Np OMEGA,
        JRA-3Q vvel-pres-an-ll125. t2m: ERA5 global_2m_temperature.nc, MERRA-2 instM_2d_asm_Nx
        T2M, JRA-3Q tmp2m-hgt-an-gauss. sw: ERA5 ssrd, MERRA-2 SWGDN, JRA-3Q dswrf1have.
DATA FILES  data/omega_surface/omega300_<var>_<dataset>_<MM>.nc
"""
from __future__ import annotations

import calendar
import os
import sys

import numpy as np
import xarray as xr

import config
import masks
import plotstyle as ps
import readers as rd
import stats

REGION = "omega_sfc"
BOX = "southern_africa"           # dashed on every map (10-40E, 35-10S)
OMEGA_LEV = 300.0
OUT = "omega_surface"
NAMES = ["ERA5", "MERRA-2", "JRA-3Q"]
VARS = {"t2m": dict(short="T2m", long="2 m temperature", units="°C", down="T2m cooling",
                    up="T2m warming"),
        "sw": dict(short=r"SW$\downarrow$", long="surface downwelling SW", units="W m$^{-2}$",
                   down=r"SW$\downarrow$ decreasing", up=r"SW$\downarrow$ increasing")}


def data_file(var, name, m):
    return config.DATA / OUT / f"omega300_{var}_{name.replace('-', '')}_{m:02d}.nc"


def load(name, var, m):
    mo = [m]
    era = config.RAW["era5"]
    if name == "ERA5":
        w = rd.era5_yearly(era / "global" / "vertical_velocity", "global_vertical_velocity_*.nc",
                           "w", REGION, lev=OMEGA_LEV, months=mo)
        x = (rd.single_file(era / "global_2m_temperature.nc", "t2m", REGION, offset=-273.15,
                            months=mo) if var == "t2m" else
             rd.single_file(era / "global_surface_solar_radiation_downwards.nc", "ssrd", REGION,
                            scale=1 / 86400.0, months=mo))
    elif name == "MERRA-2":
        w = rd.merra2("instM_3d_asm_Np", "OMEGA", REGION, lev=OMEGA_LEV, months=mo)
        x = (rd.merra2("instM_2d_asm_Nx", "T2M", REGION, offset=-273.15, months=mo)
             if var == "t2m" else rd.merra2("tavgM_2d_rad_Nx", "SWGDN", REGION, months=mo))
    else:
        w = rd.jra3q("vvel-pres-an-ll125", "vvel-pres-an-ll125", REGION, lev=OMEGA_LEV, months=mo)
        x = (rd.jra3q("", "tmp2m-hgt-an-gauss", REGION, offset=-273.15, pad=1.5, months=mo)
             if var == "t2m" else
             rd.jra3q("dswrf1have-sfc-fc-ll125", "dswrf1have-sfc-fc-ll125", REGION, months=mo))
    w, x = w.sel(month=m).drop_vars("lev", errors="ignore"), x.sel(month=m)
    if not (np.array_equal(w.lat, x.lat) and np.array_equal(w.lon, x.lon)):
        rd.log(f"  {name} {var}: interpolated to the omega grid")
        x = x.interp(lat=w.lat, lon=w.lon, kwargs={"fill_value": None})
    return w, x


def compute(months=(10,), var_list=("t2m", "sw")):
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    for m in months:
        for var in var_list:
            for name in NAMES:
                rd.log(f"--- {name} omega300 vs {var}, {calendar.month_name[m]} ---")
                w, x = load(name, var, m)
                land = masks.land_mask(w)
                w = stats.select_years(w.where(land).rename(name), config.YEARS[name])
                x = stats.select_years(x.where(land).rename(name), config.YEARS[name])
                r, p = stats.corr_and_p(stats.detrend(w), stats.detrend(x))
                sig = stats.fdr(p)
                reg = stats.regimes(w, stats.ols_stats(w).trend)
                xt = stats.ols_stats(x)
                cat = xr.where((reg >= 0) & xt.trend.notnull(),
                               2 * reg + xr.where(xt.trend > 0, 0, 1), -1)
                ds = xr.Dataset({"r": r.astype("float32"), "p": p.astype("float32"),
                                 "sig": sig.astype("int8"), "cat": cat.astype("int8"),
                                 "x_trend": xt.trend.astype("float32"),
                                 "x_sig": stats.fdr(xt.p).astype("int8")})
                ds.attrs = {"dataset": name, "var": var, "month": m,
                            "years": f"{config.YEARS[name][0]}-{config.YEARS[name][1]}"}
                ds.to_netcdf(data_file(var, name, m))
                rd.log(f"  {stats.pct_area(sig, r.notnull()):.0f} % of land with r significant; "
                       f"area-weighted median r {stats.wmedian(r):+.2f}")


def plot(months=(10,), var_list=("t2m", "sw")):
    import cartopy.crs as ccrs
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt
    ps.setup()
    fs, W = config.FS, config.WIDTH["full"]
    cols = [config.REGIME_COLOURS[k] for k in config.REGIMES]
    reg = config.REGIONS[REGION]
    left, right, gx, gy = 0.85, 0.08, 0.08, 0.22
    pw = (W - left - right - 3 * gx) / 4
    ph = pw * (reg["lat"][1] - reg["lat"][0]) / (reg["lon"][1] - reg["lon"][0])
    top = 0.62
    for m in months:
        for var in var_list:
            v = VARS[var]
            res = {n: xr.load_dataset(data_file(var, n, m)) for n in NAMES
                   if data_file(var, n, m).exists()}
            if not res:
                continue
            names = list(res)
            tops = [top + i * (ph + gy) for i in range(len(names))]
            cb_top = tops[-1] + ph + 0.4
            H = cb_top + 1.6
            fig, at = ps.figure("full", H)
            tnorm = ps.sym_norm([d.x_trend for d in res.values()])
            rnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
            heads = [f"{v['short']} trend", f"Detrended r(ω$_{{300}}$, {v['short']})", v["down"], v["up"]]
            k = 0
            for ri, name in enumerate(names):
                d = res[name]
                ps.text(fig, 0.12, tops[ri] + ph / 2, f"{ps.label(name)}\n{d.attrs['years']}",
                        rotation=90, ha="center", va="center", fontsize=fs["label"],
                        fontweight="bold", color=ps.colour(name))
                cat = d.cat.values
                for ci in range(4):
                    x0 = left + ci * (pw + gx)
                    if ri == 0:
                        ps.text(fig, x0 + pw / 2, top - 0.05, heads[ci], ha="center",
                                fontsize=fs["small"], fontweight="bold")
                    ax = at(x0, tops[ri], pw, ph, projection=ccrs.PlateCarree())
                    kw = dict(region=REGION, left=(ci == 0), bottom=(ri == len(names) - 1),
                              xticks=[15, 30], box=BOX)
                    if ci == 0:
                        ps.map_panel(ax, d.x_trend, tnorm, sig=d.x_sig.astype(bool).values, **kw)
                    elif ci == 1:
                        ps.map_panel(ax, d.r, rnorm, sig=d.sig.astype(bool).values, **kw)
                    else:
                        want = (cat >= 0) & (cat % 2 == (1 if ci == 2 else 0))
                        ps.categories(ax, d.lon, d.lat, np.where(want, cat // 2, -1),
                                      np.clip(np.abs(d.r.values), 0, 1), cols,
                                      other=(cat >= 0) & ~want, transform=ccrs.PlateCarree())
                        ps.sig_dots(ax, d.r, d.sig.astype(bool).values & want)
                        ps.map_frame(ax, **kw)
                    ps.letter(ax, k)
                    k += 1
            ps.colourbar(fig, at(left, cb_top, 1.75, 0.08), tnorm, "RdBu_r",
                         f"{v['short']} trend ({v['units']} decade$^{{-1}}$)", nticks=5)
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=rnorm, cmap="RdBu_r"),
                              cax=at(left + 2.6, cb_top, 1.9, 0.08), orientation="horizontal",
                              ticks=[-1, -0.5, 0, 0.5, 1])
            cb.set_label("r = colour saturation corresponds to correlation "
                         "strength", fontsize=fs["label"])
            cb.ax.tick_params(labelsize=fs["tick"], length=2)
            h = ps.regime_handles() + [
                Patch(facecolor=config.C_OTHER, label=f"{v['short']} changes the other way"),
                ps.dot_handle(f"Significant (FDR q = {config.ALPHA_FDR:.2f})")]
            fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, handlelength=1.2,
                       columnspacing=1.0, bbox_to_anchor=(0.5, 1 - (cb_top + 0.7) / H),
                       title="ω$_{300}$ long-term change", title_fontsize=fs["legend"])
            ps.text(fig, W / 2, 0.06, f"{calendar.month_name[m]}: detrended correlation of ω at "
                    f"{OMEGA_LEV:.0f} hPa with {v['long']}", ha="center", va="top",
                    fontsize=fs["suptitle"], fontweight="bold")
            ps.save(fig, config.RESULTS / OUT / f"omega300_{var}_{m:02d}_{calendar.month_abbr[m]}.png")


def parse(argv):
    months = [int(a) for a in argv if a.isdigit()]
    if "all" in argv:
        months = list(range(1, 13))
    if not months and os.environ.get("SLURM_ARRAY_TASK_ID", "").isdigit():
        months = [int(os.environ["SLURM_ARRAY_TASK_ID"])]
    var_list = []
    for i, a in enumerate(argv):
        if a == "--var" and i + 1 < len(argv):
            var_list += argv[i + 1].split(",")
        elif a.startswith("--var="):
            var_list += a.split("=", 1)[1].split(",")
    return months or [10], var_list or list(VARS)


if __name__ == "__main__":
    what = next((a for a in sys.argv[1:] if a in ("compute", "plot")), "all")
    months, var_list = parse(sys.argv[1:])
    missing = not all(data_file(v, n, m).exists() for v in var_list for m in months for n in NAMES)
    if what == "compute" or (what == "all" and missing):
        compute(months, var_list)
    if what in ("plot", "all"):
        plot(months, var_list)