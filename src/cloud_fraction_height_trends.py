"""
Global cloud fraction trends for every calendar month, at each dataset's native resolution:
one figure per month, rows = datasets (ERA5, MERRA-2, JRA-3Q, CLARA-A3), columns = total,
high, middle and low cloud.

Run:  python cloud_fraction_trends_monthly.py [compute | plot] [month ...]   (no month: all 12)
      As a SLURM array (--array=1-12) each task does month SLURM_ARRAY_TASK_ID; run one month
      on its own first (e.g. 'compute 10') so the shared MERRA-2 surface pressure is cached.

Work is done one calendar month at a time so that native-resolution global fields fit in
memory: ERA5 0.25 deg, MERRA-2 0.5 x 0.625 deg, JRA-3Q 1.25 deg, CLARA-A3 0.25 deg.

FIGURES  results/cloud_fraction/monthly_<REGION>/cloud_fraction_trends_<MM>_<Mon>.png
  4 x 4 maps; black dots: significant (FDR q = config.ALPHA_FDR over the cells of each map,
  plotstyle.sig_dots as in the other figures); each column heading gives the layer as each
  dataset defines it. One colour
  scale for all months (SAME_SCALE_ALL_MONTHS). LAND_ONLY = False keeps the oceans (cloud
  changes over the ocean are half the global story); True masks to land as the regional
  cloud figures.
TABLE  results/cloud_fraction/monthly_<REGION>/cloud_fraction_trends_monthly.csv: per month,
  dataset and layer, the area-weighted mean trend and share of area significant, globally
  and over the Southern Africa box

YEARS  config.years(dataset, month) (ERA5 1980-2025, CLARA-A3 1982-2024).
LAYERS  as cloud_fraction_trends.py: ERA5 from the 3-D cloud cover (high < 400, middle
  400-700, low > 700 hPa, maximum overlap within a layer (cft.OVERLAP), random between
  layers; levels below the MERRA-2 climatological surface left out); MERRA-2 CLDTOT/HGH/MID/
  LOW; JRA-3Q its own covers (high 500-90, middle 850-500, low 1100-850 hPa); CLARA-A3 by
  cloud-top pressure (high < 440, middle 440-680, low > 680 hPa).
CACHE  ERA5 layers per year-month (processed_data/cache/era5_cloud_layers_<OVERLAP>_<REGION>/);
  the other datasets through readers.py's caches.
DATA FILES  data/cloud_fraction_monthly/<REGION>/cf_<layer>_<dataset>_<MM>.nc: trend, p, sig
"""
from __future__ import annotations

import calendar
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr

import cloud_fraction_trends as cft
import config
import masks
import plotstyle as ps
import readers as rd
import stats

REGION = "global"
LAND_ONLY = False
BOX = cft.BOX
NAMES = cft.NAMES
LAYERS = cft.LAYERS
SAME_SCALE_ALL_MONTHS = True
CMAP = "RdBu_r"                   # red: more cloud
TICKS = dict(xticks=[-120, 0, 120], yticks=[-60, 0, 60])
# column headings: the layer and how each dataset defines it
HEADS = {"total": ("Total cloud", []),
         "high": ("High cloud", ["ERA5, MERRA-2: < 400 hPa", "JRA-3Q: 500-90 hPa",
                                 "CLARA-A3: top < 440 hPa"]),
         "mid": ("Middle cloud", ["ERA5, MERRA-2: 400-700 hPa", "JRA-3Q: 850-500 hPa",
                                  "CLARA-A3: top 440-680 hPa"]),
         "low": ("Low cloud", ["ERA5, MERRA-2: > 700 hPa", "JRA-3Q: 1100-850 hPa",
                               "CLARA-A3: top > 680 hPa"])}
OUT_DATA = config.DATA / "cloud_fraction_monthly" / REGION
OUT_FIG = config.RESULTS / cft.OUT / f"monthly_{REGION}"
rd.CACHE_MAX_CELLS = max(rd.CACHE_MAX_CELLS, 6_000_000)   # an ERA5 year-month: 4 layers x 0.25 deg


def data_file(layer, name, m):
    return OUT_DATA / f"cf_{layer}_{name.replace('-', '')}_{m:02d}.nc"


# ---------------------------------------------------------------- reading (one month)
_PSC = None


def ps_clim():
    global _PSC
    if _PSC is None:
        _PSC = rd.surface_pressure_clim(REGION)
    return _PSC


def era5_month(m):
    """{layer: (year, lat, lon)} ERA5 layer covers of month m, from the 3-D cloud cover."""
    psc = ps_clim()
    d = config.RAW["era5"] / "global" / "fraction_of_cloud_cover"
    files = {}
    for f in sorted(glob.glob(str(d / "global_fraction_of_cloud_cover_*.nc"))):
        mm = re.search(r"(\d{4})\.nc$", f)
        if mm:
            files[(int(mm[1]), m)] = f

    def read(path, y, mon):
        with xr.open_dataset(path) as ds:
            cc = rd.standardise(ds["cc"])
            cc = cc.isel(time=np.flatnonzero(cc.time.dt.month.values == int(mon)))
            cc = rd.cut(cc.where(cc.lev >= cft.TOP, drop=True), REGION).isel(time=0).load()
        p = psc.sel(month=int(mon)).interp(lat=cc.lat, lon=cc.lon, kwargs={"fill_value": None})
        return cft.era5_layers(cc, p).astype("float32")

    fields = rd.cached(f"era5_cloud_layers_{cft.OVERLAP}_{REGION}", files, read)
    years = sorted(k[0] for k in fields)
    cube = xr.concat([fields[(y, m)] for y in years], "year").assign_coords(year=years)
    return {k: cube.sel(layer=k, drop=True) for k in LAYERS}


def load_month(name, m):
    """{layer: (year, lat, lon)} in % for month m."""
    if name == "ERA5":
        return era5_month(m)
    out = {}
    for layer in LAYERS:
        if name == "MERRA-2":
            da = rd.merra2("tavgM_2d_rad_Nx", cft.MERRA2[layer], REGION, scale=100.0, months=[m])
        elif name == "JRA-3Q":
            da = rd.jra3q(cft.JRA3Q[layer], cft.JRA3Q[layer], REGION, months=[m])
        else:
            da = rd.cmsaf("CFC", cft.CLARA[layer], REGION, valid=(-1e-6, 100.001), months=[m])
        out[layer] = da.sel(month=m, drop=True)
    return out


# ---------------------------------------------------------------- compute
def compute(months):
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    for m in months:
        for name in NAMES:
            rd.log(f"--- {name}, {calendar.month_name[m]} ---")
            try:
                fields = load_month(name, m)
            except Exception as e:                       # noqa: BLE001
                rd.log(f"  [WARN] {name} left out: {type(e).__name__}: {e}")
                continue
            for layer, da in fields.items():
                if LAND_ONLY:
                    da = da.where(masks.land_mask(da))
                yrs = config.years(name, m)
                da = stats.select_years(da.rename(f"{name} {layer}"), yrs)
                st = stats.ols_stats(da)
                sig = stats.fdr(st.p)
                ds = xr.Dataset({"trend": st.trend.astype("float32"), "p": st.p.astype("float32"),
                                 "sig": sig.astype("int8")})
                ds.attrs = {"dataset": name, "layer": layer, "month": m,
                            "units": "% points per decade", "years": f"{yrs[0]}-{yrs[1]}",
                            "land_only": int(LAND_ONLY)}
                ds.to_netcdf(data_file(layer, name, m),
                             encoding={v: {"zlib": True} for v in ("trend", "p", "sig")})
                rd.log(f"  {layer}: {stats.pct_area(sig, st.trend.notnull()):.0f} % of the "
                       f"area significant")
                del st, sig, ds
            del fields


# ---------------------------------------------------------------- plot
def read_month(m):
    return {(l, n): xr.load_dataset(data_file(l, n, m)) for l in LAYERS for n in NAMES
            if data_file(l, n, m).exists()}


def summary_csv(months):
    rows = []
    for m in months:
        for (layer, name), ds in read_month(m).items():
            o = rd.cut(ds, BOX)
            rows.append({"month": calendar.month_abbr[m], "month_number": m, "dataset": name,
                         "layer": layer, "years": ds.attrs["years"],
                         "global_mean_trend": float(stats.area_mean(ds.trend)),
                         "pct_global_significant": stats.pct_area(ds.sig.astype(bool),
                                                                  ds.trend.notnull()),
                         "box_mean_trend": float(stats.area_mean(o.trend)),
                         "pct_box_significant": stats.pct_area(o.sig.astype(bool),
                                                               o.trend.notnull())})
    path = OUT_FIG / "cloud_fraction_trends_monthly.csv"
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.4g")
    print(f"Saved: {path}", flush=True)


def image_panel(ax, field, norm, sig, left, bottom):
    """A regular lat-lon field as an image (a cell-by-cell mesh over 360 degrees is very slow
    in cartopy at 0.25 deg), the significance dots and the map frame."""
    import cartopy.crs as ccrs
    f = field.sortby("lat").sortby("lon")
    s = np.asarray(sig.sortby("lat").sortby("lon"), bool)
    dlo = float(abs(f.lon[1] - f.lon[0]))
    dla = float(abs(f.lat[1] - f.lat[0]))
    ax.imshow(f.values, origin="lower", cmap=CMAP, norm=norm, interpolation="nearest",
              extent=(float(f.lon[0]) - dlo / 2, float(f.lon[-1]) + dlo / 2,
                      float(f.lat[0]) - dla / 2, float(f.lat[-1]) + dla / 2),
              transform=ccrs.PlateCarree(), zorder=1, rasterized=True)
    ps.sig_dots(ax, f, s)
    ps.map_frame(ax, REGION, left=left, bottom=bottom, **TICKS)


def plot_month(m, res, norm):
    import cartopy.crs as ccrs
    fs, W = config.FS, config.WIDTH["full"]
    r = config.REGIONS[REGION]
    left, right, gx, gy = 0.62, 0.1, 0.08, 0.12
    pw = (W - left - right - 3 * gx) / 4
    mh = pw * (r["lat"][1] - r["lat"][0]) / (r["lon"][1] - r["lon"][0])
    top = 1.05                                    # title, layer names and their definitions
    cb_top = top + 4 * (mh + gy) + 0.3
    H = cb_top + 0.6
    fig, at = ps.figure("full", H)
    for c, layer in enumerate(LAYERS):
        x = left + c * (pw + gx) + pw / 2
        name, lines = HEADS[layer]
        ps.text(fig, x, top - 0.08 - 0.115 * len(lines), name, ha="center", va="bottom",
                fontsize=fs["small"], fontweight="bold")
        for j, line in enumerate(lines):
            ps.text(fig, x, top - 0.06 - 0.115 * (len(lines) - 1 - j), line, ha="center",
                    va="bottom", fontsize=fs["small"] - 2, color="0.3")
    k = 0
    for i, name in enumerate(NAMES):
        t0 = top + i * (mh + gy)
        d0 = next((res[(l, name)] for l in LAYERS if (l, name) in res), None)
        ps.text(fig, left - 0.45, t0 + mh / 2, ps.label(name) + (f"\n{d0.attrs['years']}"
                if d0 is not None else ""), rotation=90, ha="center", va="center",
                fontsize=fs["small"], fontweight="bold", color=ps.colour(name))
        for c, layer in enumerate(LAYERS):
            ax = at(left + c * (pw + gx), t0, pw, mh, projection=ccrs.PlateCarree())
            last = i == len(NAMES) - 1
            if (layer, name) not in res:
                ps.map_frame(ax, REGION, left=(c == 0), bottom=last, **TICKS)  # no data
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center",
                        fontsize=fs["small"], color="0.4")
            else:
                d = res[(layer, name)]
                image_panel(ax, d.trend, norm, d.sig.astype(bool), c == 0, last)
            ps.letter(ax, k)
            k += 1
    ps.colourbar(fig, at(left + 0.4, cb_top, W - left - right - 0.8, 0.08), norm, CMAP,
                 f"Trend (% points decade$^{{-1}}$); dots: significant (FDR q = "
                 f"{config.ALPHA_FDR:.2f})")
    ps.text(fig, W / 2, 0.06, f"{calendar.month_name[m]} cloud fraction trends"
            + (" over land" if LAND_ONLY else ""), ha="center", va="top",
            fontsize=fs["suptitle"], fontweight="bold")
    ps.save(fig, OUT_FIG / f"cloud_fraction_trends_{m:02d}_{calendar.month_abbr[m]}.png")


def common_norm():
    """One colour scale from all months (every 4th cell of every map: plenty for a percentile)."""
    vals = []
    for m in range(1, 13):
        for d in read_month(m).values():
            vals.append(d.trend.values[::4, ::4])
    return ps.sym_norm(vals) if vals else None


def plot(months):
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    ps.setup()
    summary_csv(months)
    norm_all = common_norm() if SAME_SCALE_ALL_MONTHS else None
    for m in months:
        res = read_month(m)
        if not res:
            rd.log(f"  no data files for {calendar.month_name[m]}")
            continue
        norm = norm_all or ps.sym_norm([d.trend.values for d in res.values()])
        plot_month(m, res, norm)


def months_from(args):
    ms = [int(a) for a in args if a.isdigit()]
    if not ms and os.environ.get("SLURM_ARRAY_TASK_ID", "").isdigit():
        ms = [int(os.environ["SLURM_ARRAY_TASK_ID"])]
    return ms or list(range(1, 13))


if __name__ == "__main__":
    args = sys.argv[1:]
    what = next((a for a in args if not a.isdigit()), "all")
    months = months_from(args)
    todo = [m for m in months if not all(data_file(l, n, m).exists()
                                         for l in LAYERS for n in NAMES)]
    if what == "compute":
        compute(months)
    elif what == "all" and todo:
        compute(todo)
    if what in ("plot", "all"):
        plot(months)