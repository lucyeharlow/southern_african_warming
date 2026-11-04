"""
Decomposition of relative-humidity trends over Southern Africa into a specific-humidity part
and a temperature (saturation) part, ERA5, MERRA-2 and JRA-3Q, at LEVELS (default 850 hPa).

    RH = q / q_s(T, p)   =>   ln RH = ln q - ln q_s(T)     (exact, in logarithmic units)
    RH trend =  trend(ln RH)
    q part   =  trend(ln q)                   moistening raises RH, drying lowers it
    T part   = -trend(ln q_s(T))              warming raises q_s and so lowers RH
    residual =  trend(ln RH) - q part - T part   (RH of monthly means is not q/q_s of monthly
                                                  means; the reanalysis's own saturation formula)
Trends are OLS per decade (stats.ols_stats) of the logarithms, shown as LOG_SCALE x ln per
decade: LOG_SCALE = 100 reads as % (relative) change per decade, 1 as ln units per decade.
q_s = eps e_s / (p - (1 - eps) e_s), e_s from the IFS formula (water above 0 C, ice below
-23 C, quadratic blend between; SATURATION = "water" for water only).

Run:  python rh_decomposition.py [compute | plot]

FIGURES  results/rh_decomposition/
  rh_decomposition_<level>hPa.png  a) monthly box means (10-40E, 35-10S, land, above ground,
           area-weighted) of the RH trend (diamonds) and its q and T parts (bars), one panel per
           dataset; b) October maps (10-40E, 35S-0: the box's longitudes, no Madagascar), rows
           RH trend | q part | T part
  rh_decomposition_residual_<level>hPa.png  the equations, monthly box-mean residual next to
           the RH trend, and October residual maps
  rh_decomposition_summary.csv
DATA (native grids; config.YEARS; below-ground cells masked with the MERRA-2 surface pressure)
  ERA5     global/{relative_humidity, specific_humidity, temperature}/..._YYYY.nc: r, q, t
  MERRA-2  instM_3d_asm_Np RH, QV, T
  JRA-3Q   rh-pres-an-ll125, spfh-pres-an-ll125 (6-hourly, monthly files); temperature from
           JRA3Q_T_SOURCE: "grib" = anl_p125_tmp 6-hourly GRIB, "netcdf" = tmp-pres-an-ll125
DATA FILES  data/rh_decomposition/rh_terms_<dataset>_<level>.nc: total, q, T, res
            (month, lat, lon) and their box means (month)
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

MAP, BOX = "africa_south", "southern_africa"     # MAP: area read and computed
PLOT_MAP = "africa_10_40_s"                       # area of the maps
LEVELS = [850]
MONTH = 10
OUT = "rh_decomposition"
NAMES = ["ERA5", "MERRA-2", "JRA-3Q"]
SATURATION = "mixed"
JRA3Q_T_SOURCE = "grib"
CMAP = "BrBG"
COL = {"q": "#1B9E77", "T": "#D95F02", "res": "#7F7F7F"}
TERMS = [("total", "RH trend"), ("q", "q part"), ("T", "T part")]
LOG_SCALE = 100.0                 # 100: 100 ln per decade (relative %); 1: ln per decade
UNIT = "% decade$^{-1}$ (100 Δln)" if LOG_SCALE == 100 else "ln units decade$^{-1}$"
UNIT_TXT = "100 ln per decade (relative % per decade)" if LOG_SCALE == 100 else "ln per decade"


def data_file(name, level):
    return config.DATA / OUT / f"rh_terms_{name.replace('-', '')}_{level}.nc"


# ---------------------------------------------------------------- physics
def e_sat(T):
    """Saturation vapour pressure (Pa), IFS formula; T in K."""
    ew = 611.21 * np.exp(17.502 * (T - 273.16) / (T - 32.19))
    if SATURATION == "water":
        return ew
    ei = 611.21 * np.exp(22.587 * (T - 273.16) / (T + 0.7))
    a = np.clip((T - 250.16) / (273.16 - 250.16), 0, 1) ** 2
    return a * ew + (1 - a) * ei


def q_sat(T, p_hpa):
    """Saturation specific humidity (g kg-1) at p (hPa)."""
    es = e_sat(T)
    return 1000.0 * 0.622 * es / (p_hpa * 100.0 - 0.378 * es)


# ---------------------------------------------------------------- reading
def load(name, level):
    """RH (%), q (g kg-1), T (K): (year, month, lat, lon) at one level."""
    if name == "ERA5":
        d = config.RAW["era5"] / "global"
        f = lambda folder, v, sc: rd.era5_yearly(d / folder, f"global_{folder}_*.nc", v, MAP,
                                                 scale=sc, lev=level)
        out = {"rh": f("relative_humidity", "r", 1.0), "q": f("specific_humidity", "q", 1000.0),
               "T": f("temperature", "t", 1.0)}
    elif name == "MERRA-2":
        f = lambda v, sc: rd.merra2("instM_3d_asm_Np", v, MAP, scale=sc, lev=level)
        out = {"rh": f("RH", 100.0), "q": f("QV", 1000.0), "T": f("T", 1.0)}
    else:
        f = lambda v, sc: rd.jra3q(v, v, MAP, scale=sc, lev=level)
        out = {"rh": f("rh-pres-an-ll125", 1.0), "q": f("spfh-pres-an-ll125", 1000.0)}
        out["T"] = (rd.jra3q_grib(str(config.RAW["jra3q_grib"] / "anl_p125_tmp"), "anl_p125_tmp",
                                  MAP, lev=level, years=config.YEARS["JRA-3Q"])
                    if JRA3Q_T_SOURCE == "grib"
                    else f("tmp-pres-an-ll125", 1.0))
    return {k: v.drop_vars("lev", errors="ignore") for k, v in out.items()}


# ---------------------------------------------------------------- compute
def compute():
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    psc = rd.surface_pressure_clim(MAP)
    for level in LEVELS:
        for name in NAMES:
            rd.log(f"--- {name}, {level} hPa ---")
            f = load(name, level)
            like = f["rh"]
            ps_ = psc.interp(lat=like.lat, lon=like.lon, kwargs={"fill_value": None})
            keep = masks.land_mask(like) & (ps_ > level) if config.MASK_BELOW_GROUND else \
                masks.land_mask(like)
            rh, q, T = (stats.select_years(f[k].where(keep).rename(f"{name} {k}"), config.YEARS[name])
                        for k in ("rh", "q", "T"))
            q, rh = q.where(q > 0), rh.where(rh > 0)
            trend = lambda da: stats.ols_stats(da).trend
            qs = q_sat(T, level)
            terms = {"total": LOG_SCALE * trend(np.log(rh)), "q": LOG_SCALE * trend(np.log(q)),
                     "T": -LOG_SCALE * trend(np.log(qs))}
            terms["res"] = terms["total"] - terms["q"] - terms["T"]
            check = xr.corr(trend(np.log(q / qs)), terms["total"], dim=["lat", "lon"])
            ds = xr.Dataset({k: v.astype("float32") for k, v in terms.items()})
            for k in list(terms):
                ds[f"box_{k}"] = stats.area_mean(rd.cut(terms[k], BOX))
            ds["T_trend"] = trend(T).astype("float32")
            ds["check_r"] = check
            ds.attrs = {"dataset": name, "level_hPa": level, "units": UNIT_TXT,
                        "log_scale": LOG_SCALE,
                        "years": f"{config.YEARS[name][0]}-{config.YEARS[name][1]}"}
            ds.to_netcdf(data_file(name, level),
                         encoding={k: {"zlib": True} for k in ("total", "q", "T", "res")})
            o = ds.sel(month=MONTH)
            rd.log(f"  October box means: RH {float(o.box_total):+.2f}, q part {float(o.box_q):+.2f}, "
                   f"T part {float(o.box_T):+.2f}, residual {float(o.box_res):+.2f} ({UNIT_TXT}); "
                   f"q/q_s(T) vs RH trend pattern r = {float(o.check_r):.2f}")


# ---------------------------------------------------------------- plot
def bars(fig, at, x0s, top, h, pw, res, keys, ylabel, lim):
    for i, name in enumerate(NAMES):
        ax = at(x0s[i], top, pw, h)
        if name in res:
            d = res[name]
            m = d.month.values
            w = 0.8 / len(keys)
            for j, k in enumerate(keys):
                ax.bar(m + (j - (len(keys) - 1) / 2) * w, d[f"box_{k}"], w, color=COL[k], zorder=3)
            ax.scatter(m, d.box_total, marker="D", s=9, color="black", zorder=4)
            ax.set_title(f"{ps.label(name)} ({d.attrs['years']})", fontsize=config.FS["small"],
                         fontweight="bold", color=ps.colour(name), pad=3)
        ax.axhline(0, color="black", lw=0.6)
        ax.set(xlim=(0.4, 12.6), ylim=(-lim, lim), xticks=ps.MONTHS)
        ax.set_xticklabels([c[0] for c in calendar.month_abbr[1:]])
        ax.grid(axis="y", alpha=0.3, lw=0.4)
        ax.tick_params(labelleft=(i == 0), length=2)
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=config.FS["small"])
        ps.letter(ax, i)


def lim_of(res, keys):
    v = [abs(float(x)) for d in res.values() for k in keys for x in d[f"box_{k}"].values
         if np.isfinite(x)]
    lim = max(v + [0.1]) * 1.15
    step = 0.5 if lim > 2 else 0.25 if lim > 1 else 0.1
    return float(np.ceil(lim / step) * step)


def plot():
    import cartopy.crs as ccrs
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ps.setup()
    fs, W = config.FS, config.WIDTH["full"]
    rows = []
    for level in LEVELS:
        res = {n: xr.load_dataset(data_file(n, level)) for n in NAMES if data_file(n, level).exists()}
        if not res:
            continue
        old = [n for n, d in res.items() if d.attrs.get("log_scale") != LOG_SCALE]
        if old:
            raise SystemExit(f"data files for {', '.join(old)} at {level} hPa are from the old "
                             "percentage-point version (or another LOG_SCALE): run "
                             "'python rh.py compute' first")
        for n, d in res.items():
            for m in d.month.values:
                o = d.sel(month=m)
                rows.append({"level_hPa": level, "dataset": n, "years": d.attrs["years"],
                             "month": int(m), **{f"box_{k}": float(o[f"box_{k}"])
                                                 for k in ("total", "q", "T", "res")},
                             "trend_corr_rhcalc_vs_rh": float(o.check_r)})
        left, right, gx = 0.85, 0.1, 0.12
        pw = (W - left - right - 2 * gx) / 3
        x0s = [left + i * (pw + gx) for i in range(3)]
        r = config.REGIONS[PLOT_MAP]
        aspect = (r["lat"][1] - r["lat"][0]) / (r["lon"][1] - r["lon"][0])
        mh = min(pw * aspect, 1.75)                  # keep the figure within a page
        pw_m = mh / aspect
        x0m = [left + (W - left - right - 3 * pw_m - 2 * gx) / 2 + i * (pw_m + gx)
               for i in range(3)]
        cut = lambda da: rd.cut(da, PLOT_MAP)

        # main figure
        bar_top, bar_h = 0.95, 1.3
        map_top = bar_top + bar_h + 0.6
        tops = [map_top + i * (mh + 0.12) for i in range(3)]
        cb_top = tops[-1] + mh + 0.32
        H = cb_top + 0.6
        fig, at = ps.figure("full", H)
        bars(fig, at, x0s, bar_top, bar_h, pw, res, ["q", "T"],
             f"Contribution to the trend\nof ln RH ({UNIT})", lim_of(res, ["total", "q", "T"]))
        fig.legend(handles=[Line2D([], [], marker="D", color="black", ls="", markersize=3,
                                   label="RH trend"),
                            Patch(facecolor=COL["q"], label="q part (above 0: moistening raises RH)"),
                            Patch(facecolor=COL["T"], label="T part (below 0: warming lowers RH)")],
                   loc="upper center", ncol=3, frameon=False, handlelength=1.2, columnspacing=1.0,
                   bbox_to_anchor=(0.5, 1 - 0.32 / H))
        norm = ps.sym_norm([cut(res[n][k].sel(month=MONTH)) for n in res for k, _ in TERMS],
                           cmap=CMAP)
        ps.text(fig, W / 2, map_top - 0.22, f"{calendar.month_name[MONTH]} at {level} hPa (land, "
                "above ground; brown = RH falls, green = RH rises)", ha="center",
                fontsize=fs["title"], fontweight="bold")
        k = 3
        for ri, (term, tlab) in enumerate(TERMS):
            ps.text(fig, x0m[0] - 0.62, tops[ri] + mh / 2, tlab, rotation=90, ha="center",
                    va="center",
                    fontsize=fs["label"], fontweight="bold",
                    color={"total": "black", "q": COL["q"], "T": COL["T"]}[term])
            for ci, name in enumerate(NAMES):
                ax = at(x0m[ci], tops[ri], pw_m, mh, projection=ccrs.PlateCarree())
                if name in res:
                    ps.map_panel(ax, cut(res[name][term].sel(month=MONTH)), norm, cmap=CMAP,
                                 region=PLOT_MAP, left=(ci == 0), bottom=(ri == 2), box=BOX,
                                 xticks=[15, 30])
                else:
                    ps.map_frame(ax, PLOT_MAP, left=(ci == 0), bottom=(ri == 2), xticks=[15, 30])
                ps.letter(ax, k)
                k += 1
        ps.colourbar(fig, at(left + 0.5, cb_top, W - left - right - 1.0, 0.08), norm, CMAP,
                     f"Trend of ln RH or contribution ({UNIT}); dashed: box of a)-c)")
        ps.text(fig, W / 2, 0.05, f"Relative humidity trend decomposition at {level} hPa",
                ha="center", va="top", fontsize=fs["suptitle"], fontweight="bold")
        ps.save(fig, config.RESULTS / OUT / f"rh_decomposition_{level}hPa.png")

        # residual figure
        eq = ["RH = q / q$_s$(T, p),  so  ln RH = ln q - ln q$_s$  (exact)",
              "trend(ln RH) = trend(ln q) + [-trend(ln q$_s$)] = q part + T part",
              "residual = trend(ln RH) - q part - T part"]
        H2 = 0.95 + 1.3 + 0.75 + mh + 0.95
        fig, at = ps.figure("full", H2)
        for i, line in enumerate(eq):
            ps.text(fig, 0.3, 0.42 + 0.2 * i, line, fontsize=fs["label"], va="top")
        bar_top = 1.3
        bars(fig, at, x0s, bar_top, 1.2, pw, res, ["res"], f"Box mean\n({UNIT})",
             lim_of(res, ["total", "res"]))
        rnorm = ps.sym_norm([cut(res[n].res.sel(month=MONTH)) for n in res], cmap=CMAP)
        mt = bar_top + 1.2 + 0.55
        for ci, name in enumerate(NAMES):
            ax = at(x0m[ci], mt, pw_m, mh, projection=ccrs.PlateCarree())
            if name in res:
                o = res[name].sel(month=MONTH)
                ps.map_panel(ax, cut(o.res), rnorm, cmap=CMAP, region=PLOT_MAP, left=(ci == 0),
                             box=BOX, xticks=[15, 30])
                ax.set_title(f"Oct box mean {float(o.box_res):+.2f} (RH {float(o.box_total):+.2f})",
                             fontsize=fs["small"], pad=2)
            ps.letter(ax, 3 + ci)
        ps.colourbar(fig, at(left + 0.5, mt + mh + 0.32, W - left - right - 1.0, 0.08), rnorm, CMAP,
                     f"{calendar.month_name[MONTH]} residual ({UNIT})")
        fig.legend(handles=[Line2D([], [], marker="D", color="black", ls="", markersize=3,
                                   label="RH trend (for scale)"),
                            Patch(facecolor=COL["res"], label="Residual")],
                   loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(1 - right / W, 1 - 0.95 / H2))
        ps.text(fig, W / 2, 0.05, f"Residual of the RH trend decomposition at {level} hPa",
                ha="center", va="top", fontsize=fs["suptitle"], fontweight="bold")
        ps.save(fig, config.RESULTS / OUT / f"rh_decomposition_residual_{level}hPa.png")
    if rows:
        path = config.RESULTS / OUT / "rh_decomposition_summary.csv"
        pd.DataFrame(rows).to_csv(path, index=False, float_format="%.4f")
        print(f"Saved: {path}", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "compute" or (what == "all" and not all(data_file(n, l).exists() for n in NAMES for l in LEVELS)):
        compute()
    if what in ("plot", "all"):
        plot()