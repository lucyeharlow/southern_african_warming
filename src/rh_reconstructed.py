"""
Reconstructed relative humidity from the monthly specific humidity and temperature,

    RH* = 100 q / q_s(T, p)

compared with each reanalysis's own RH, ERA5, MERRA-2 and JRA-3Q, at LEVELS (default 850 hPa).
RH* uses the same q, T, saturation formula (q_s, SATURATION), land and below-ground masks and
years as the RH decomposition (rh.py / rh_decomposition.py, imported from there), so for RH*
that decomposition is exact: ln RH* = ln q - ln q_s(T). The difference RH* - RH is what the
monthly averaging (RH of monthly means is not q/q_s of monthly means) and each reanalysis's
own saturation formula add.

Run:  python rh_reconstructed.py [compute | plot]

FIGURES  results/rh_reconstructed/
  rh_reconstructed_trends_<level>hPa.png
    a)-c) monthly box means (10-40E, 35-10S, land, above ground, area-weighted) of the RH
          trend and the RH* trend (% points per decade), one panel per dataset
    then October maps (10-40E, 35S-0): rows RH trend | RH* trend | RH* trend - RH trend
  rh_reconstructed_climatology_<level>hPa.png
    October maps: rows mean RH | mean RH* | RH* - RH (% RH)
  rh_reconstructed_rln_<level>hPa.png
    R_ln = 100 [trend(ln RH) - trend(ln RH*)]: a)-c) monthly box means of R_ln next to
    100 trend(ln RH) for scale; d)-f) October R_ln maps. Since ln RH* = ln q - ln q_s(T),
    R_ln is exactly the residual of the log decomposition in rh.py (LOG_SCALE = 100).
    Log trends are in 10^-2 ln units per decade (-2: the ln falls by 0.02 per decade).
  rh_reconstructed_summary.csv   box means per dataset and month: RH and RH* trends and
    means, their difference, 100 trend(ln RH), 100 trend(ln RH*), R_ln, the pattern
    correlation of the RH and RH* trend maps over the map area
DATA FILES  data/rh_reconstructed/rh_star_<dataset>_<level>.nc: rh_trend, rhs_trend, rh_mean,
    rhs_mean, ln_rh_trend, ln_rhs_trend, r_ln (month, lat, lon) and the box means (month)
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

try:
    import rh_decomposition as rhd
except ImportError:                    # the same script, saved as rh.py
    import rh as rhd

LEVELS = rhd.LEVELS
NAMES = rhd.NAMES
MONTH = rhd.MONTH
MAP, BOX, PLOT_MAP = rhd.MAP, rhd.BOX, rhd.PLOT_MAP
OUT = "rh_reconstructed"
COL = {"rh": "#4D4D4D", "rhs": "#2C7FB8"}


def data_file(name, level):
    return config.DATA / OUT / f"rh_star_{name.replace('-', '')}_{level}.nc"


# ---------------------------------------------------------------- compute
def compute():
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    psc = rd.surface_pressure_clim(MAP)
    for level in LEVELS:
        for name in NAMES:
            rd.log(f"--- {name}, {level} hPa ---")
            f = rhd.load(name, level)
            like = f["rh"]
            ps_ = psc.interp(lat=like.lat, lon=like.lon, kwargs={"fill_value": None})
            keep = masks.land_mask(like) & (ps_ > level) if config.MASK_BELOW_GROUND else \
                masks.land_mask(like)
            rh, q, T = (stats.select_years(f[k].where(keep).rename(f"{name} {k}"),
                                           config.YEARS[name]) for k in ("rh", "q", "T"))
            rhs = 100.0 * q.where(q > 0) / rhd.q_sat(T, level)          # RH* in %
            rhs = rhs.where(rh.notnull())                               # the same cells
            trend = lambda da: stats.ols_stats(da).trend
            ln_rh = 100.0 * trend(np.log(rh.where(rh > 0)))
            ln_rhs = 100.0 * trend(np.log(rhs.where(rhs > 0)))
            ds = xr.Dataset({"rh_trend": trend(rh), "rhs_trend": trend(rhs),
                             "rh_mean": rh.mean("year"), "rhs_mean": rhs.mean("year"),
                             "ln_rh_trend": ln_rh, "ln_rhs_trend": ln_rhs,
                             "r_ln": ln_rh - ln_rhs})
            ds = ds.astype("float32")
            for v in list(ds.data_vars):
                ds[f"box_{v}"] = stats.area_mean(rd.cut(ds[v], BOX))
            plot_area = lambda da: rd.cut(da, PLOT_MAP)
            ds["pattern_r"] = xr.corr(plot_area(ds.rh_trend), plot_area(ds.rhs_trend),
                                      dim=["lat", "lon"])
            ds.attrs = {"dataset": name, "level_hPa": level, "units": "% RH; trends % points "
                        "per decade; ln trends and r_ln 10^-2 ln units per decade",
                        "saturation": rhd.SATURATION,
                        "years": f"{config.YEARS[name][0]}-{config.YEARS[name][1]}"}
            ds.to_netcdf(data_file(name, level),
                         encoding={k: {"zlib": True} for k in ("rh_trend", "rhs_trend",
                                                                "rh_mean", "rhs_mean")})
            o = ds.sel(month=MONTH)
            rd.log(f"  October box means: RH trend {float(o.box_rh_trend):+.2f}, RH* trend "
                   f"{float(o.box_rhs_trend):+.2f} % points per decade; R_ln "
                   f"{float(o.box_r_ln):+.3f} (10^-2 ln units per decade); mean RH "
                   f"{float(o.box_rh_mean):.1f}, RH* {float(o.box_rhs_mean):.1f} %; trend "
                   f"pattern r = {float(o.pattern_r):.2f}")


# ---------------------------------------------------------------- plot
def map_rows(fig, at, res, rows, tops, x0m, pw_m, mh, k0, cmap, norms):
    import cartopy.crs as ccrs
    k = k0
    for ri, (lab, fn) in enumerate(rows):
        ps.text(fig, x0m[0] - 0.62, tops[ri] + mh / 2, lab, rotation=90, ha="center",
                va="center", fontsize=config.FS["label"], fontweight="bold")
        for ci, name in enumerate(NAMES):
            ax = at(x0m[ci], tops[ri], pw_m, mh, projection=ccrs.PlateCarree())
            if name in res:
                ps.map_panel(ax, rd.cut(fn(res[name].sel(month=MONTH)), PLOT_MAP), norms[ri],
                             cmap=cmap[ri], region=PLOT_MAP, left=(ci == 0),
                             bottom=(ri == len(rows) - 1), box=BOX, xticks=[15, 30])
            else:
                ps.map_frame(ax, PLOT_MAP, left=(ci == 0), bottom=(ri == len(rows) - 1),
                             xticks=[15, 30])
            if ri == 0:
                ax.set_title(f"{ps.label(name)}" + (f" ({res[name].attrs['years']})"
                                                     if name in res else ""),
                             fontsize=config.FS["small"], fontweight="bold",
                             color=ps.colour(name), pad=3)
            ps.letter(ax, k)
            k += 1


def geometry():
    W = config.WIDTH["full"]
    left, right, gx = 0.85, 0.1, 0.12
    pw = (W - left - right - 2 * gx) / 3
    x0s = [left + i * (pw + gx) for i in range(3)]
    r = config.REGIONS[PLOT_MAP]
    aspect = (r["lat"][1] - r["lat"][0]) / (r["lon"][1] - r["lon"][0])
    mh = min(pw * aspect, 1.75)
    pw_m = mh / aspect
    x0m = [left + (W - left - right - 3 * pw_m - 2 * gx) / 2 + i * (pw_m + gx) for i in range(3)]
    return W, left, right, pw, x0s, mh, pw_m, x0m


def plot():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt
    ps.setup()
    fs = config.FS
    rows_csv = []
    for level in LEVELS:
        res = {n: xr.load_dataset(data_file(n, level)) for n in NAMES
               if data_file(n, level).exists()}
        if not res:
            continue
        for n, d in res.items():
            for m in d.month.values:
                o = d.sel(month=m)
                rows_csv.append({"level_hPa": level, "dataset": n, "years": d.attrs["years"],
                                 "month": int(m),
                                 "box_rh_trend": float(o.box_rh_trend),
                                 "box_rhs_trend": float(o.box_rhs_trend),
                                 "box_trend_diff": float(o.box_rhs_trend - o.box_rh_trend),
                                 "box_rh_mean": float(o.box_rh_mean),
                                 "box_rhs_mean": float(o.box_rhs_mean),
                                 "box_mean_diff": float(o.box_rhs_mean - o.box_rh_mean),
                                 "box_100_trend_ln_rh": float(o.box_ln_rh_trend),
                                 "box_100_trend_ln_rhs": float(o.box_ln_rhs_trend),
                                 "box_r_ln": float(o.box_r_ln),
                                 "trend_pattern_r": float(o.pattern_r)})
        W, left, right, pw, x0s, mh, pw_m, x0m = geometry()

        # ---- trends figure
        bar_top, bar_h = 0.95, 1.25
        map_top = bar_top + bar_h + 0.65
        tops = [map_top + i * (mh + 0.12) for i in range(3)]
        cb_top = tops[-1] + mh + 0.32
        H = cb_top + 1.0
        fig, at = ps.figure("full", H)
        lim = max([abs(float(x)) for d in res.values() for v in ("box_rh_trend", "box_rhs_trend")
                   for x in d[v].values if np.isfinite(x)] + [0.1]) * 1.15
        for i, name in enumerate(NAMES):
            ax = at(x0s[i], bar_top, pw, bar_h)
            if name in res:
                d = res[name]
                m = d.month.values
                ax.bar(m - 0.2, d.box_rh_trend, 0.4, color=COL["rh"], zorder=3)
                ax.bar(m + 0.2, d.box_rhs_trend, 0.4, color=COL["rhs"], zorder=3)
                ax.set_title(f"{ps.label(name)} ({d.attrs['years']})", fontsize=fs["small"],
                             fontweight="bold", color=ps.colour(name), pad=3)
            ax.axhline(0, color="black", lw=0.6)
            ax.set(xlim=(0.4, 12.6), ylim=(-lim, lim), xticks=ps.MONTHS)
            ax.set_xticklabels([c[0] for c in calendar.month_abbr[1:]])
            ax.grid(axis="y", alpha=0.3, lw=0.4)
            ax.tick_params(labelleft=(i == 0), length=2)
            if i == 0:
                ax.set_ylabel("Box-mean trend\n(% points decade$^{-1}$)", fontsize=fs["small"])
            ps.letter(ax, i)
        fig.legend(handles=[Patch(facecolor=COL["rh"], label="RH (reanalysis)"),
                            Patch(facecolor=COL["rhs"], label="RH* = 100 q / q$_s$(T, p)")],
                   loc="upper center", ncol=2, frameon=False, handlelength=1.2,
                   bbox_to_anchor=(0.5, 1 - 0.32 / H))
        tnorm = ps.sym_norm([rd.cut(res[n][v].sel(month=MONTH), PLOT_MAP) for n in res
                             for v in ("rh_trend", "rhs_trend")], cmap=rhd.CMAP)
        dnorm = ps.sym_norm([rd.cut((res[n].rhs_trend - res[n].rh_trend).sel(month=MONTH),
                                    PLOT_MAP) for n in res], cmap="PuOr_r")
        ps.text(fig, W / 2, map_top - 0.22, f"{calendar.month_name[MONTH]} at {level} hPa "
                "(land, above ground)", ha="center", fontsize=fs["title"], fontweight="bold")
        map_rows(fig, at, res, [("RH trend", lambda o: o.rh_trend),
                                ("RH* trend", lambda o: o.rhs_trend),
                                ("RH* - RH", lambda o: o.rhs_trend - o.rh_trend)],
                 tops, x0m, pw_m, mh, 3, [rhd.CMAP, rhd.CMAP, "PuOr_r"],
                 [tnorm, tnorm, dnorm])
        half = (W - left - right) / 2
        ps.colourbar(fig, at(left + 0.1, cb_top, half - 0.4, 0.08), tnorm, rhd.CMAP,
                     "RH or RH* trend (% points decade$^{-1}$)", nticks=5)
        ps.colourbar(fig, at(left + half + 0.3, cb_top, half - 0.4, 0.08), dnorm, "PuOr_r",
                     "RH* trend - RH trend (% points decade$^{-1}$)", nticks=5)
        ps.text(fig, W / 2, 0.05, f"Reconstructed RH* = 100 q / q$_s$(T, p) and reanalysis RH "
                f"trends at {level} hPa", ha="center", va="top", fontsize=fs["suptitle"],
                fontweight="bold")
        ps.save(fig, config.RESULTS / OUT / f"rh_reconstructed_trends_{level}hPa.png")

        # ---- climatology figure
        map_top = 0.8
        tops = [map_top + i * (mh + 0.12) for i in range(3)]
        cb_top = tops[-1] + mh + 0.32
        H = cb_top + 0.75
        fig, at = ps.figure("full", H)
        vals = np.concatenate([np.ravel(rd.cut(res[n][v].sel(month=MONTH), PLOT_MAP))
                               for n in res for v in ("rh_mean", "rhs_mean")])
        vals = vals[np.isfinite(vals)]
        hi = float(np.ceil(np.percentile(vals, config.COLOUR_PERCENTILE) / 10) * 10) if vals.size else 100
        from matplotlib.colors import BoundaryNorm
        cnorm = BoundaryNorm(np.linspace(0, hi, 11), plt.get_cmap("YlGnBu").N, extend="max")
        bnorm = ps.sym_norm([rd.cut((res[n].rhs_mean - res[n].rh_mean).sel(month=MONTH),
                                    PLOT_MAP) for n in res], cmap="PuOr_r")
        map_rows(fig, at, res, [("Mean RH", lambda o: o.rh_mean),
                                ("Mean RH*", lambda o: o.rhs_mean),
                                ("RH* - RH", lambda o: o.rhs_mean - o.rh_mean)],
                 tops, x0m, pw_m, mh, 0, ["YlGnBu", "YlGnBu", "PuOr_r"], [cnorm, cnorm, bnorm])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=cnorm, cmap="YlGnBu"),
                          cax=at(left + 0.1, cb_top, half - 0.4, 0.08), orientation="horizontal",
                          extend="max")
        cb.set_label(f"{calendar.month_name[MONTH]} mean RH or RH* (%)", fontsize=fs["label"])
        cb.ax.tick_params(labelsize=fs["tick"], length=2)
        ps.colourbar(fig, at(left + half + 0.3, cb_top, half - 0.4, 0.08), bnorm, "PuOr_r",
                     "RH* - RH (% RH)", nticks=5)
        ps.text(fig, W / 2, 0.05, f"{calendar.month_name[MONTH]} mean RH and RH* at {level} hPa",
                ha="center", va="top", fontsize=fs["suptitle"], fontweight="bold")
        ps.save(fig, config.RESULTS / OUT / f"rh_reconstructed_climatology_{level}hPa.png")

        # ---- R_ln figure
        import cartopy.crs as ccrs
        eq = [r"R$_{ln}$ = 100 [trend(ln RH) - trend(ln RH*)]  (10$^{-2}$ ln units decade$^{-1}$),   RH* = 100 q / q$_s$(T, p)",
              "= the residual of the log decomposition (ln RH* = ln q - ln q$_s$ exactly):",
              "what the monthly averaging and the reanalysis's own saturation formula add"]
        bar_top, bar_h = 1.3, 1.2
        mt = bar_top + bar_h + 0.55
        H3 = mt + mh + 0.95
        fig, at = ps.figure("full", H3)
        for i, line in enumerate(eq):
            ps.text(fig, 0.3, 0.42 + 0.2 * i, line, fontsize=fs["label"], va="top")
        lim = max([abs(float(x)) for d in res.values() for v in ("box_r_ln", "box_ln_rh_trend")
                   for x in d[v].values if np.isfinite(x)] + [0.1]) * 1.15
        for i, name in enumerate(NAMES):
            ax = at(x0s[i], bar_top, pw, bar_h)
            if name in res:
                d = res[name]
                m = d.month.values
                ax.bar(m, d.box_r_ln, 0.6, color="#7F7F7F", zorder=3)
                ax.scatter(m, d.box_ln_rh_trend, marker="D", s=9, color="black", zorder=4)
                ax.set_title(f"{ps.label(name)} ({d.attrs['years']})", fontsize=fs["small"],
                             fontweight="bold", color=ps.colour(name), pad=3)
            ax.axhline(0, color="black", lw=0.6)
            ax.set(xlim=(0.4, 12.6), ylim=(-lim, lim), xticks=ps.MONTHS)
            ax.set_xticklabels([c[0] for c in calendar.month_abbr[1:]])
            ax.grid(axis="y", alpha=0.3, lw=0.4)
            ax.tick_params(labelleft=(i == 0), length=2)
            if i == 0:
                ax.set_ylabel("Box mean (10$^{-2}$ ln\nunits decade$^{-1}$)", fontsize=fs["small"])
            ps.letter(ax, i)
        fig.legend(handles=[Line2D([], [], marker="D", color="black", ls="", markersize=3,
                                   label="Trend of ln RH (for scale)"),
                            Patch(facecolor="#7F7F7F", label="R$_{ln}$")],
                   loc="upper right", ncol=2, frameon=False,
                   bbox_to_anchor=(1 - right / W, 1 - 0.95 / H3))
        rnorm = ps.sym_norm([rd.cut(res[n].r_ln.sel(month=MONTH), PLOT_MAP) for n in res],
                            cmap="PuOr_r")
        for ci, name in enumerate(NAMES):
            ax = at(x0m[ci], mt, pw_m, mh, projection=ccrs.PlateCarree())
            if name in res:
                o = res[name].sel(month=MONTH)
                ps.map_panel(ax, rd.cut(o.r_ln, PLOT_MAP), rnorm, cmap="PuOr_r",
                             region=PLOT_MAP, left=(ci == 0), box=BOX, xticks=[15, 30])
                ax.set_title(f"box {float(o.box_r_ln):+.2f}, ln RH "
                             f"{float(o.box_ln_rh_trend):+.2f}", fontsize=fs["small"], pad=2)
            else:
                ps.map_frame(ax, PLOT_MAP, left=(ci == 0), xticks=[15, 30])
            ps.letter(ax, 3 + ci)
        ps.colourbar(fig, at(left + 0.5, mt + mh + 0.32, W - left - right - 1.0, 0.08), rnorm,
                     "PuOr_r", f"{calendar.month_name[MONTH]} R$_{{ln}}$ (10$^{{-2}}$ ln units "
                     "decade$^{{-1}}$)")
        ps.text(fig, W / 2, 0.05, f"R$_{{ln}}$: RH trend not explained by q and T, {level} hPa",
                ha="center", va="top", fontsize=fs["suptitle"], fontweight="bold")
        ps.save(fig, config.RESULTS / OUT / f"rh_reconstructed_rln_{level}hPa.png")
    if rows_csv:
        path = config.RESULTS / OUT / "rh_reconstructed_summary.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_csv).to_csv(path, index=False, float_format="%.4f")
        print(f"Saved: {path}", flush=True)


def up_to_date(name, level):
    """The data file exists and was written by this version (it has R_ln)."""
    f = data_file(name, level)
    if not f.exists():
        return False
    with xr.open_dataset(f) as ds:
        return "r_ln" in ds


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    stale = not all(up_to_date(n, l) for n in NAMES for l in LEVELS)
    if what == "plot" and stale:
        raise SystemExit("data files missing or from the earlier version (no R_ln): run "
                         "'python rh_reconstructed.py compute' first")
    if what == "compute" or (what == "all" and stale):
        compute()
    if what in ("plot", "all"):
        plot()
