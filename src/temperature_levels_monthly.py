"""
Air-temperature trends on every pressure level over Southern Africa (10-40E, 35-10S) for
every month: ERA5, MERRA-2 and JRA-3Q. The same data files, method and figures as
temperature_levels.py (October), drawn for each month, plus a month-pressure summary.

Run:  python temperature_levels_monthly.py                 compute if needed, plot all months
      python temperature_levels_monthly.py compute         re-read the raw data first
      python temperature_levels_monthly.py plot [1 2 ...]  plot (these months only)

FIGURES  results/temperature_levels/<month>/  (as temperature_levels.py, one folder per month)
  ta_levels_maps_<mon>_<dataset>.png, ta_levels_profile_<mon>.png
         SAME_SCALE_ALL_MONTHS = True: one colour scale and one profile axis for all months
         (the 98th percentile over every month), so the months can be compared directly
results/temperature_levels/
  ta_levels_month_pressure.png  a-c) box-mean trend by month (x) and level (y), one panel per
         dataset (dots: p < 0.05 for the regional series; grey: no land above the ground)
  ta_levels_summary.csv
"""
from __future__ import annotations

import calendar
import sys

import numpy as np

import config
import plotstyle as ps
import temperature_levels as tl

SAME_SCALE_ALL_MONTHS = True
MONTHS = list(range(1, 13))


def plot_month_pressure(res):
    fs, W = config.FS, config.WIDTH["full"]
    left, gap, right = 0.6, 0.15, 0.1
    pw = (W - left - right - 2 * gap) / 3
    top, h = 0.6, 2.4
    H = top + h + 0.95
    fig, at = ps.figure("full", H)
    norm = ps.sym_norm([d.box_trend.values for d in res.values()], cmap=tl.CMAP)
    for i, name in enumerate(tl.NAMES):
        ax = at(left + i * (pw + gap), top, pw, h)
        ax.set_facecolor("0.85")
        if name in res:
            d = res[name][["box_trend", "box_p"]].transpose("lev", "month")
            d.attrs = res[name].attrs
            m, lev = d.month.values, d.lev.values
            ax.pcolormesh(m, lev, d.box_trend.values, cmap=tl.CMAP, norm=norm, shading="nearest",
                          rasterized=True, zorder=1)
            s = d.box_p.values < tl.P_PROFILE
            mm, ll = np.meshgrid(m, lev)
            ax.scatter(mm[s], ll[s], s=4, color=config.DOT_COLOUR, linewidths=0, zorder=3)
            ax.set_title(f"{ps.label(name)} ({d.attrs['years']})", fontsize=fs["small"],
                         fontweight="bold", color=ps.colour(name), pad=3)
        ax.set_xlim(0.5, 12.5)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels([c[0] for c in calendar.month_abbr[1:]])
        tl.pressure_axis(ax, label=(i == 0))
        ps.letter(ax, i)
    ps.colourbar(fig, at(left + 0.6, top + h + 0.4, W - left - right - 1.2, 0.1), norm, tl.CMAP,
                 f"Box-mean trend ({tl.UNIT}); dots: p < {tl.P_PROFILE}")
    ps.text(fig, W / 2, top - 0.4, "Temperature trends by month and level, 10-40°E, 35-10°S, land",
            ha="center", fontsize=fs["title"], fontweight="bold")
    ps.save(fig, config.RESULTS / tl.OUT / "ta_levels_month_pressure.png")


def plot(months):
    ps.setup()
    res = tl.read_all()
    tl.summary_csv(res)
    plot_month_pressure(res)
    if SAME_SCALE_ALL_MONTHS:
        norm = tl.norm_for(res, MONTHS)
        sec_norm = ps.sym_norm([d.sec_trend.values for d in res.values()], cmap=tl.CMAP)
        xlim = tl.profile_xlim(res, MONTHS)
    else:
        norm = sec_norm = xlim = None
    for m in months:
        print(f"--- {calendar.month_name[m]} ---", flush=True)
        tl.plot_month(res, m, norm, sec_norm, xlim)


if __name__ == "__main__":
    args = sys.argv[1:]
    what = args.pop(0) if args and args[0] in ("compute", "plot") else ""
    months = [int(a) for a in args] or MONTHS
    if what == "compute" or (what != "plot" and not all(tl.data_file(n).exists() for n in tl.NAMES)):
        tl.compute()
    plot(months)
