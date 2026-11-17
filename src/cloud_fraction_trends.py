"""
Cloud fraction trends over Southern Africa: total, high, middle and low cloud, from ERA5,
MERRA-2, JRA-3Q and CLARA-A3. One figure per layer.

Run:  python cloud_fraction_trends.py [compute | plot] [total high mid low]

FIGURE  results/cloud_fraction/cloud_fraction_<layer>.png
  a)    monthly boxplots of the per-cell trends over the Southern Africa box (10-40E, 35-10S,
        land), area-weighted
  b-e)  October trend maps, 10-40E, 35S-18N (the box's longitudes, north to the sub-Saharan
        edge; no Madagascar), box dashed; black dots: significant
        (FDR q = 0.10 over the land cells of each map)
  f)    October area-weighted land series over the box, anomalies from 1991-2020

DATA (native grids; % cloud cover; years config.YEARS)
  ERA5      yearly files of 3-D cloud cover cc. Layers from the profile (high p < 400 hPa,
            middle 400-700, low > 700, as MERRA-2), maximum overlap within a layer (OVERLAP),
            random overlap between layers (total = 1 - (1-h)(1-m)(1-l)); levels below the
            MERRA-2 climatological surface are left out.
  MERRA-2   tavgM_2d_rad_Nx CLDTOT, CLDHGH, CLDMID, CLDLOW (fraction -> %)
  JRA-3Q    its own covers: tcdc-tcl-fc-ll125 (total), hcdc-hcl (500-90 hPa),
            mcdc-mcl (850-500), lcdc-lcl (1100-850) -fc-ll125, 1-hourly, monthly files
  CLARA-A3  CFC files: cfc, cfc_high (< 440 hPa cloud top), cfc_middle (440-680), cfc_low
DATA FILES  data/cloud_fraction/cf_<layer>_<dataset>.nc
  trend, p, n_eff, sig (month, lat, lon), % points per decade; October box series (year).
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

import config
import masks
import plotstyle as ps
import readers as rd
import stats

MAP = "africa_10_40"
BOX = "southern_africa"
MONTH = 10
OUT = "cloud_fraction"
NAMES = ["ERA5", "MERRA-2", "JRA-3Q", "CLARA-A3"]
LAYERS = {"total": "Total cloud", "high": "High cloud (< 400 hPa)",
          "mid": "Middle cloud (400-700 hPa)", "low": "Low cloud (> 700 hPa)"}
OVERLAP = "max"                  # ERA5 within-layer overlap: "max" or "random"
TOP, HIGH_MID, MID_LOW = 50.0, 400.0, 700.0
JRA3Q = {"total": "tcdc-tcl-fc-ll125", "high": "hcdc-hcl-fc-ll125",
         "mid": "mcdc-mcl-fc-ll125", "low": "lcdc-lcl-fc-ll125"}
MERRA2 = {"total": "CLDTOT", "high": "CLDHGH", "mid": "CLDMID", "low": "CLDLOW"}
CLARA = {"total": "cfc", "high": "cfc_high", "mid": "cfc_middle", "low": "cfc_low"}
NOTE = ("Layers differ: JRA-3Q high 500-90, middle 850-500, low 1100-850 hPa; CLARA-A3 by "
        "cloud-top pressure (high < 440, middle 440-680, low > 680 hPa).")


def data_file(layer, name):
    return config.DATA / OUT / f"cf_{layer}_{name.replace('-', '')}.nc"


# ---------------------------------------------------------------- reading
def era5_layers(cc, ps_):
    """cc (lev, lat, lon) 0-1 -> layer covers (layer, lat, lon) in %."""
    lev = cc.lev
    if ps_ is not None and config.MASK_BELOW_GROUND:
        cc = cc.where(lev <= ps_)
    out = {}
    for k, sel in (("high", (lev >= TOP) & (lev < HIGH_MID)),
                   ("mid", (lev >= HIGH_MID) & (lev <= MID_LOW)), ("low", lev > MID_LOW)):
        c = cc.where(sel, drop=True).clip(0, 1)
        out[k] = (1 - (1 - c.fillna(0)).prod("lev") if OVERLAP == "random"
                  else c.max("lev").fillna(0))
    out["total"] = 1 - (1 - out["high"]) * (1 - out["mid"]) * (1 - out["low"])
    land_ok = cc.notnull().any("lev")
    return 100 * xr.concat([out[k].where(land_ok) for k in LAYERS], "layer").assign_coords(
        layer=list(LAYERS))


def load_era5():
    psc = rd.surface_pressure_clim(MAP)
    files = {}
    d = config.RAW["era5"] / "global" / "fraction_of_cloud_cover"
    for f in sorted(glob.glob(str(d / "global_fraction_of_cloud_cover_*.nc"))):
        m = re.search(r"(\d{4})\.nc$", f)
        if m:
            files[int(m[1])] = f

    def read(path, y):
        with xr.open_dataset(path) as ds:
            cc = rd.cut(rd.standardise(ds["cc"]), MAP)
            cc = cc.where(cc.lev >= TOP, drop=True).load()
        ps_ = psc.interp(lat=cc.lat, lon=cc.lon, kwargs={"fill_value": None})
        return xr.concat([era5_layers(cc.isel(time=i), ps_.sel(month=int(t.dt.month)))
                          for i, t in enumerate(cc.time)], "time").assign_coords(time=cc.time)
    fields = rd.from_months(rd.per_month(rd.cached(f"era5_cloud_layers_{OVERLAP}_{MAP}", files, read)))
    return {k: fields.sel(layer=k, drop=True) for k in LAYERS}


def load(name):
    """{layer: (year, month, lat, lon) in %}."""
    if name == "ERA5":
        return load_era5()
    if name == "MERRA-2":
        return {k: rd.merra2("tavgM_2d_rad_Nx", v, MAP, scale=100.0) for k, v in MERRA2.items()}
    if name == "JRA-3Q":
        return {k: rd.jra3q(v, v, MAP) for k, v in JRA3Q.items()}
    if name == "CLARA-A3":
        return {k: rd.cmsaf("CFC", v, MAP, valid=(-1e-6, 100.001)) for k, v in CLARA.items()}
    raise KeyError(name)


# ---------------------------------------------------------------- compute
def compute():
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        rd.log(f"--- {name} ---")
        fields = load(name)
        for layer, da in fields.items():
            land = masks.land_mask(da)
            da = stats.select_years(da.where(land).rename(f"{name} {layer}"), config.YEARS[name])
            anom = stats.anomalies(da)
            st = stats.ols_stats(anom)
            sig = stats.fdr_maps(st.p)
            series = stats.regional_series(rd.cut(anom.sel(month=MONTH), BOX))
            sst = stats.ols_stats(series)
            ds = xr.Dataset({"trend": st.trend.astype("float32"), "p": st.p.astype("float32"),
                             "n_eff": st.n_eff.astype("float32"), "sig": sig.astype("int8"),
                             "series": series})
            ds.attrs = {"dataset": name, "layer": layer, "units": "% points per decade",
                        "years": f"{config.YEARS[name][0]}-{config.YEARS[name][1]}",
                        "series_trend": float(sst.trend), "series_p": float(sst.p),
                        "series_n_eff": float(sst.n_eff)}
            ds.to_netcdf(data_file(layer, name),
                         encoding={v: {"zlib": True} for v in ("trend", "p", "n_eff")})
            rd.log(f"  {layer}: October box series trend {float(sst.trend):+.2f} % per decade, "
                   f"{stats.p_text(float(sst.p))}")


# ---------------------------------------------------------------- plot
def summary_csv(layers):
    rows = []
    for layer in layers:
        for n in NAMES:
            if not data_file(layer, n).exists():
                continue
            ds = xr.load_dataset(data_file(layer, n))
            for m in ds.month.values:
                o = rd.cut(ds.sel(month=m), BOX)
                rows.append({"layer": layer, "dataset": n, "years": ds.attrs["years"],
                             "month": int(m), **stats.describe(o.trend),
                             "pct_box_area_sig": stats.pct_area(o.sig.astype(bool),
                                                                o.trend.notnull())})
    path = config.RESULTS / OUT / "cloud_fraction_box_monthly_trends.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved: {path}", flush=True)


def plot_layer(layer):
    import cartopy.crs as ccrs
    res = {n: xr.load_dataset(data_file(layer, n)) for n in NAMES if data_file(layer, n).exists()}
    if not res:
        return
    names = list(res)
    fs, W = config.FS, config.WIDTH["full"]
    years = {n: ps.year_list(res[n].attrs["years"]) for n in names}
    left, right, gap = 0.62, 0.1, 0.12
    r = config.REGIONS[MAP]
    mw = (W - left - right - (len(NAMES) - 1) * gap) / len(NAMES)
    mh = mw * (r["lat"][1] - r["lat"][0]) / (r["lon"][1] - r["lon"][0])
    box_top, box_h = 0.45, 1.25
    head = 0.4                                     # dataset name and years above each map
    map_top = box_top + box_h + 0.55 + head
    cb_top = map_top + mh + 0.3
    ts_top, ts_h = cb_top + 0.8, 1.15
    H = ts_top + ts_h + 0.95 + (0.25 if layer != "total" else 0)
    fig, at = ps.figure("full", H)

    ax = at(left, box_top, W - left - right, box_h)
    ps.monthly_boxplot(ax, {n: {int(m): rd.cut(res[n].trend.sel(month=m), BOX)
                                for m in res[n].month.values} for n in names}, names,
                       "Trend, area-weighted\n(% points decade$^{-1}$)")
    ps.letter(ax, 0)
    ps.dataset_legend(ax, names, years, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                      handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)

    norm = ps.sym_norm([res[n].trend.sel(month=MONTH) for n in names])
    lay = LAYERS[layer][0].lower() + LAYERS[layer][1:]
    ps.text(fig, W / 2, map_top - head - 0.08, f"{calendar.month_name[MONTH]} {lay} trends",
            ha="center", fontsize=fs["title"], fontweight="bold")
    for i, name in enumerate(NAMES):
        x0 = left + i * (mw + gap)
        ax = at(x0, map_top, mw, mh, projection=ccrs.PlateCarree())
        ps.text(fig, x0 + mw / 2, map_top - 0.2, ps.label(name), ha="center",
                fontsize=fs["title"], fontweight="bold", color=ps.colour(name))
        if name in res:
            ps.text(fig, x0 + mw / 2, map_top - 0.03, ps.span(years[name]), ha="center",
                    fontsize=fs["small"], color="0.25")
        if name not in res:
            ps.map_frame(ax, MAP, left=(i == 0), xticks=[15, 30])
            continue
        o = res[name].sel(month=MONTH)
        ps.map_panel(ax, o.trend, norm, sig=o.sig.astype(bool).values, region=MAP,
                     left=(i == 0), box=BOX, xticks=[15, 30])
        ps.letter(ax, i + 1)
    ps.colourbar(fig, at(left + 0.5, cb_top, W - left - right - 1.0, 0.08), norm, "RdBu_r",
                 f"{calendar.month_name[MONTH]} trend (% points decade$^{{-1}}$); dots: "
                 f"significant (FDR q = {config.ALPHA_FDR:.2f})")

    ax = at(left, ts_top, W - left - right, ts_h)
    for name in names:
        s, a = res[name].series, res[name].attrs
        ax.plot(s.year, s, color=ps.colour(name), lw=1.2,
                label=f"{ps.label(name)}: {a['series_trend']:+.2f}, {stats.p_text(a['series_p'])}, "
                      f"n$_{{eff}}$ = {a['series_n_eff']:.0f}")
    ax.axhline(0, color="black", lw=0.6, ls="--", zorder=0)
    ax.set_xlim(min(y[0] for y in years.values()), max(y[-1] for y in years.values()))
    ax.set_ylabel(f"{calendar.month_abbr[MONTH]} anomaly (% points)")
    ax.grid(alpha=0.3, linewidth=0.4)
    ps.letter(ax, len(NAMES) + 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False,
              handlelength=1.2, columnspacing=1.0, title="Box trend (% points decade$^{-1}$)",
              title_fontsize=fs["legend"])
    ps.text(fig, W / 2, ts_top - 0.05, f"{calendar.month_name[MONTH]} area-weighted land anomaly "
            f"over the box (from {config.BASELINE[0]}-{config.BASELINE[1]})", ha="center",
            fontsize=fs["title"], fontweight="bold")
    if layer != "total":
        ps.text(fig, W / 2, H - 0.05, NOTE, ha="center", fontsize=fs["small"], color="0.3",
                va="bottom", wrap=True)
    ps.save(fig, config.RESULTS / OUT / f"cloud_fraction_{layer}.png")


def plot(layers=None):
    layers = layers or list(LAYERS)
    ps.setup()
    summary_csv(layers)
    for layer in layers:
        plot_layer(layer)


if __name__ == "__main__":
    args = sys.argv[1:]
    what = next((a for a in args if a in ("compute", "plot")), "all")
    layers = [a for a in args if a in LAYERS] or list(LAYERS)
    if what == "compute" or (what == "all" and not all(data_file("total", n).exists() for n in NAMES)):
        compute()
    if what in ("plot", "all"):
        plot(layers)
