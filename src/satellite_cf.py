"""
October cloud fraction from satellite records: CLARA-A3, MODIS Terra, MODIS Aqua, Terra+Aqua
and CERES EBAF, as trend maps and as land series in latitude bands (10-38E).

Run:  python cloud_fraction_satellite.py [compute | plot]

FIGURE  results/cloud_fraction_satellite/cloud_fraction_satellite.png
  a)-f)  October trend maps (15W-50E, 35S-20N, land, % points per decade): CLARA-A3 over its
         full record and over 2002-2024, MODIS Terra, Aqua, Terra+Aqua and CERES over
         2002-2024. Black dots: significant (FDR q = 0.10 over the land cells of each map).
         Notches on the left edge: the latitude bands; dashed box: the band region.
  g)...  CLARA-A3 October cloud fraction per band over its full record, with the OLS trend,
         p and n_eff.
  ...    cross validation over 2002-2024: every dataset's band series with dashed OLS trends.
  SERIES_MODE = "anomaly": anomalies (CLARA-A3 full record from 1991-2020 as bars; cross
  validation from 2002-2020, the part of 1991-2020 they all cover); "absolute": the band-mean
  cloud fraction. The band panels of each column share one y axis (SHARED_Y; False: each
  panel fitted to its own data). SERIES_SCALE = 0.01 shows the series (and their trends) as
  cloud fraction, a decimal (0 = clear, 1 = overcast, so -0.05 = 5 % points less cloud);
  1 shows %.
  Anomaly of a band: for each land cell, its October value minus its own mean October over
  the baseline, then the area-weighted (cos latitude) mean over the band's land cells that
  have a value in every year.
TABLE   cf_band_trends.png / .csv: trend, p, n_eff for every band and dataset.
SUPPLEMENTARY  cf_latitude_bands_map.png: the bands on a map.

DATA (native grids, October; % cloud cover)
  CLARA-A3  CFC files, cfc (thematic to 2020, interim after)
  MODIS     MOD08_M3 / MYD08_M3 HDF4, Cloud_Fraction_Mean_Mean, <dir>/<year>/<doy>/
            (Terra+Aqua: their mean where both exist)
  CERES     CERES_EBAF_Ed4.2.1_Subset_*.nc, cldarea_total_daynight_mon
  Note: CERES cloud fraction is retrieved from MODIS on the same satellites.
DATA FILES  data/cloud_fraction_satellite/<dataset>.nc: trend, p, sig (lat, lon);
            band series (band, year) and their trend, p, n_eff.
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

MAP = "cloud_maps"
MONTH = 10
COMMON = (2002, 2024)            # cross-validation years
COMMON_BASE = (2002, 2020)       # their anomaly baseline (the part of 1991-2020 all cover)
SERIES_MODE = "anomaly"          # band series: "anomaly" or "absolute" cloud fraction
SERIES_SCALE = 0.01              # band series in decimals (0-1); 1.0 for %
SHARED_Y = True                  # one y axis for all band panels of a column
OUT = "cloud_fraction_satellite"
BANDS = [("10N-0", 0.0, 10.0), ("0-10S", -10.0, 0.0), ("10S-20S", -20.0, -10.0),
         ("20S-35S", -35.0, -20.0)]
BAND_LON = (10.0, 38.0)
BAND_COLOURS = ["#525252", "#8C510A", "#C51B7D", "#E66101"]
# maps: key, dataset (style), years
MAPS = [("CLARA-A3_full", "CLARA-A3", None), ("CLARA-A3", "CLARA-A3", COMMON),
        ("MODIS Terra", "MODIS Terra", COMMON), ("MODIS Aqua", "MODIS Aqua", COMMON),
        ("MODIS Terra+Aqua", "MODIS Terra+Aqua", COMMON), ("CERES", "CERES", COMMON)]
CROSS = ["CLARA-A3", "MODIS Terra", "MODIS Aqua", "MODIS Terra+Aqua", "CERES"]
CERES_VAR = "cldarea_total_daynight_mon"
MODIS_VAR = "Cloud_Fraction_Mean_Mean"


def data_file(key):
    return config.DATA / OUT / f"{key.replace(' ', '_').replace('+', 'plus').replace('-', '')}.nc"


# ---------------------------------------------------------------- reading (October)
def modis(which):
    """MODIS L3 monthly HDF4 files -> (year, lat, lon) in %."""
    from pyhdf.SD import SD, SDC
    base, prefix = (config.RAW["modis_terra"], "MOD08_M3") if which == "terra" else \
        (config.RAW["modis_aqua"], "MYD08_M3")
    files = {}
    for yd in sorted(glob.glob(str(base / "*"))):
        y = os.path.basename(yd)
        if not y.isdigit():
            continue
        y = int(y)
        doy = (pd.Timestamp(y, MONTH, 1).dayofyear)
        for d in (doy, doy - 1, doy + 1):
            f = glob.glob(os.path.join(yd, f"{d:03d}", f"{prefix}.A*.hdf"))
            if f:
                files[(y, MONTH)] = f[0]
                break

    def read(path, y, m):
        h = SD(path, SDC.READ)
        try:
            s = h.select(MODIS_VAR)
            a = s.attributes()
            v = s[:].astype("float64")
            v = np.where(v == a.get("_FillValue", -9999), np.nan,
                         (v - a.get("add_offset", 0.0)) * a.get("scale_factor", 1.0))
        finally:
            h.end()
        nlat, nlon = v.shape
        lat = np.linspace(90 - 90 / nlat, -90 + 90 / nlat, nlat)
        lon = np.linspace(-180 + 180 / nlon, 180 - 180 / nlon, nlon)
        da = xr.DataArray(100 * v, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
        return rd.cut(da.sortby("lat"), MAP)
    return rd.from_months(rd.cached(f"modis_{which}_cf_{MAP}", files, read)).sel(month=MONTH)


def ceres():
    parts = []
    for f in sorted(glob.glob(str(config.RAW["ceres"] / "CERES_EBAF_Ed4.2.1_Subset_*.nc"))):
        parts.append(rd.single_file(f, CERES_VAR, MAP, months=[MONTH], fill=1e3))
    da = xr.concat(parts, "year")
    _, first = np.unique(da.year.values, return_index=True)
    da = da.isel(year=first)
    return da.where(da >= 0).sel(month=MONTH)


def load_all():
    clara = rd.cmsaf("CFC", "cfc", MAP, valid=(-1e-6, 100.001), months=[MONTH]).sel(month=MONTH)
    terra, aqua = modis("terra"), modis("aqua")
    t, a = xr.align(terra, aqua, join="outer")
    both = xr.concat([t, a], "s").mean("s", skipna=True)
    return {"CLARA-A3": clara, "MODIS Terra": terra, "MODIS Aqua": aqua,
            "MODIS Terra+Aqua": both, "CERES": ceres()}


# ---------------------------------------------------------------- compute
def band_series(da):
    """(band, year) area-weighted means over land cells complete in every year."""
    out = []
    for _, s_, n_ in BANDS:
        sub = da.sel(lat=slice(s_, n_), lon=slice(*BAND_LON))
        out.append(stats.regional_series(sub))
    return xr.concat(out, "band").assign_coords(band=[b[0] for b in BANDS])


def compute():
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    fields = load_all()
    for key, name, years in MAPS:
        rd.log(f"--- {key} ---")
        da = fields[name]
        da = da.where(masks.land_mask(da)).rename(key)
        yrs = years or config.YEARS[name]
        da = stats.select_years(da, yrs)
        st = stats.ols_stats(da)
        sig = stats.fdr(st.p)
        base = config.BASELINE if years is None else COMMON_BASE
        clim = da.sel(year=slice(*base)).mean("year")
        series_abs = band_series(da)
        series = band_series(da - clim)
        sst = stats.ols_stats(series)
        ds = xr.Dataset({"trend": st.trend.astype("float32"), "p": st.p.astype("float32"),
                         "sig": sig.astype("int8"), "series": series, "series_abs": series_abs,
                         "series_trend": sst.trend, "series_p": sst.p, "series_n_eff": sst.n_eff})
        ds.attrs = {"dataset": name, "years": f"{yrs[0]}-{yrs[1]}", "month": MONTH,
                    "baseline": f"{base[0]}-{base[1]}", "units": "% points (per decade)"}
        ds.to_netcdf(data_file(key), encoding={v: {"zlib": True} for v in ("trend", "p")})
        for b in series.band.values:
            s = sst.sel(band=b)
            rd.log(f"  {b}: {float(s.trend):+.2f} % per decade, {stats.p_text(float(s.p))}, "
                   f"n_eff {float(s.n_eff):.1f}")


# ---------------------------------------------------------------- plot
def notches(ax):
    import cartopy.crs as ccrs
    r = config.REGIONS[MAP]
    span = r["lon"][1] - r["lon"][0]
    for k, (_, s_, n_) in enumerate(BANDS):
        for yy in (s_, n_):
            ax.plot([r["lon"][0], r["lon"][0] + 0.05 * span], [yy, yy], color="k", lw=1.0,
                    transform=ccrs.PlateCarree(), zorder=6, solid_capstyle="butt")
        ax.plot([r["lon"][0] + 0.006 * span] * 2, [s_, n_], color=BAND_COLOURS[k], lw=3,
                transform=ccrs.PlateCarree(), zorder=6, solid_capstyle="butt")


def plot():
    import cartopy.crs as ccrs
    from matplotlib.lines import Line2D
    ps.setup()
    res = {k: xr.load_dataset(data_file(k)) for k, _, _ in MAPS if data_file(k).exists()}
    if not res:
        raise SystemExit("no data files: run compute first")
    fs, W = config.FS, config.WIDTH["full"]
    left, right, gap, gy = 0.55, 0.1, 0.15, 0.36
    r = config.REGIONS[MAP]
    mw = (W - left - right - 2 * gap) / 3
    mh = mw * (r["lat"][1] - r["lat"][0]) / (r["lon"][1] - r["lon"][0])
    top = 0.3
    cb_top = top + 2 * mh + gy + 0.28
    band_top, bh, bgap = cb_top + 0.95, 0.85, 0.32
    H = band_top + 4 * bh + 3 * bgap + 0.85
    fig, at = ps.figure("full", H)
    norm = ps.sym_norm([d.trend for d in res.values()])
    box = dict(lon=BAND_LON, lat=(BANDS[-1][1], BANDS[0][2]))
    for i, (key, name, years) in enumerate(MAPS):
        rr, cc = divmod(i, 3)
        x0, t0 = left + cc * (mw + gap), top + rr * (mh + gy)
        ax = at(x0, t0, mw, mh, projection=ccrs.PlateCarree())
        ps.text(fig, x0 + mw / 2, t0 - 0.04, f"{ps.label(name)} " +
                (res[key].attrs["years"] if key in res else ""), ha="center",
                fontsize=fs["title"], fontweight="bold", color=ps.colour(name))
        if key not in res:
            ps.map_frame(ax, MAP, left=(cc == 0), xticks=[0, 20, 40])
            continue
        d = res[key]
        ps.map_panel(ax, d.trend, norm, sig=d.sig.astype(bool).values, region=MAP,
                     left=(cc == 0), box=box, xticks=[0, 20, 40], yticks=[-30, -15, 0, 15])
        notches(ax)
        ps.letter(ax, i)
    ps.colourbar(fig, at(left + 0.5, cb_top, W - left - right - 1.0, 0.08), norm, "RdBu_r",
                 f"{calendar.month_name[MONTH]} cloud fraction trend (% points decade$^{{-1}}$); "
                 f"dots: significant (FDR q = {config.ALPHA_FDR:.2f})")

    # band series
    cw = (W - left - right - 0.55) / 2
    xb, xc = left, left + cw + 0.55
    full = res.get("CLARA-A3_full")
    absolute = SERIES_MODE == "absolute" and all("series_abs" in d for d in res.values())
    if SERIES_MODE == "absolute" and not absolute:
        print("  [WARN] data files have no absolute series (run compute again): anomalies shown")
    sv = "series_abs" if absolute else "series"
    sc = SERIES_SCALE
    dec = sc != 1
    tr_unit = "decade$^{-1}$" if dec else "% points decade$^{-1}$"
    tr_fmt = "+.4f" if dec else "+.2f"

    def fit(ax, vals, low_room=0.0):
        """y axis fitted to this panel's values (+ a little room, and low_room for the text)."""
        v = np.concatenate([np.asarray(x, float).ravel() for x in vals])
        v = v[np.isfinite(v)]
        if not v.size:
            return
        lo, hi = float(v.min()), float(v.max())
        pad = 0.08 * max(hi - lo, 1e-6)
        if not absolute:
            lo, hi = min(lo, 0.0), max(hi, 0.0)
        ax.set_ylim(lo - pad - low_room * (hi - lo), hi + pad)
    def column_range(keys, low_room=0.0):
        """One y range for every band panel of a column (all its series)."""
        allv = [np.asarray(res[k_][sv], float).ravel() * sc for k_ in keys if k_ in res]
        v = np.concatenate(allv) if allv else np.array([])
        v = v[np.isfinite(v)]
        if not (SHARED_Y and v.size):
            return None
        lo, hi = float(v.min()), float(v.max())
        if not absolute:
            lo, hi = min(lo, 0.0), max(hi, 0.0)
        pad = 0.08 * max(hi - lo, 1e-6)
        return lo - pad - low_room * (hi - lo), hi + pad
    shared_b = column_range(["CLARA-A3_full"], low_room=0.35)     # room for the trend text
    shared_c = column_range(CROSS)
    k = len(MAPS)
    for bi, (bname, _, _) in enumerate(BANDS):
        t0 = band_top + bi * (bh + bgap)
        last = bi == len(BANDS) - 1
        axb = at(xb, t0, cw, bh)
        if full is not None:
            s = full[sv].sel(band=bname) * sc
            tr, p, ne = (float(full[v].sel(band=bname)) for v in ("series_trend", "series_p",
                                                                     "series_n_eff"))
            tr *= sc
            yrs = s.year.values.astype(float)
            line = tr / 10 * (yrs - yrs.mean()) + float(s.mean())
            if absolute:
                axb.plot(s.year, s, color=ps.colour("CLARA-A3"), lw=1.3, marker="o", ms=2.2)
                axb.plot(yrs, line, color="k", lw=1.0, ls="--")
            else:
                axb.bar(s.year, s, width=0.75, color=np.where(s > 0, "#B2182B", "#2166AC"))
                axb.plot(yrs, line, color=ps.colour("CLARA-A3"), lw=1.3)
            axb.text(0.01, 0.04, f"{tr:{tr_fmt}} {tr_unit}, {stats.p_text(p)}, "
                     f"n$_{{eff}}$ = {ne:.0f}", transform=axb.transAxes, fontsize=fs["small"],
                     va="bottom", bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))
            if shared_b:
                axb.set_ylim(*shared_b)
            else:
                fit(axb, [s, line], low_room=0.35)
            axb.set_xlim(yrs.min() - 0.5, yrs.max() + 0.5)
        axc = at(xc, t0, cw, bh)
        vals = []
        for key in CROSS:
            if key not in res:
                continue
            s = res[key][sv].sel(band=bname) * sc
            c = ps.colour(res[key].attrs["dataset"])
            axc.plot(s.year, s, color=c, lw=1.3, marker="o", ms=2.2)
            yrs = s.year.values.astype(float)
            tr = float(res[key].series_trend.sel(band=bname)) * sc
            line = tr / 10 * (yrs - yrs.mean()) + float(s.mean())
            axc.plot(yrs, line, color=c, lw=1.0, ls="--")
            vals += [s, line]
        if shared_c:
            axc.set_ylim(*shared_c)
        else:
            fit(axc, vals)
        axc.set_xlim(COMMON[0] - 0.5, COMMON[1] + 0.5)
        kind = "cloud fraction" if absolute else "anomaly"
        for ax, title in ((axb, "CLARA-A3 " + (full.attrs["years"] if full is not None else "")
                           + f", {kind}"),
                          (axc, f"Cross validation {COMMON[0]}-{COMMON[1]}, {kind}")):
            if not absolute:
                ax.axhline(0, color="k", lw=0.5)
            ax.grid(axis="y", alpha=0.3, lw=0.4)
            ax.tick_params(labelbottom=last)
            ax.text(0.0, 1.03, f"{'abcdefghijklmnopqrstuvwxyz'[k]}) {bname}", transform=ax.transAxes,
                    fontsize=fs["label"], fontweight="bold", color=BAND_COLOURS[bi], va="bottom")
            if bi == 0:
                ps.text(fig, (xb if ax is axb else xc) + cw / 2, t0 - 0.24, title, ha="center",
                        fontsize=fs["title"], fontweight="bold")
            k += 1
        unit = "fraction" if dec else ("%" if absolute else "% points")
        axb.set_ylabel(f"{'Cloud fraction' if absolute else 'Anomaly'} ({unit})",
                       fontsize=fs["small"])
    h = [Line2D([], [], color=ps.colour(res[k_].attrs["dataset"]), lw=1.2, marker="o", ms=2.5,
                label=ps.label(res[k_].attrs["dataset"])) for k_ in CROSS if k_ in res]
    h += [Line2D([], [], color="0.3", lw=0.9, ls="--", label="OLS trend")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, handlelength=1.5,
               columnspacing=1.0, bbox_to_anchor=(0.5, 1 - (band_top + 4 * bh + 3 * bgap + 0.22) / H))
    ps.save(fig, config.RESULTS / OUT / "cloud_fraction_satellite.png")
    trend_table(res)
    band_map()


def trend_table(res):
    import matplotlib.pyplot as plt
    rows = []
    for bname, _, _ in BANDS:
        for key, name, _ in MAPS:
            if key in res:
                d = res[key]
                rows.append({"band": bname, "dataset": ps.label(name), "years": d.attrs["years"],
                             "trend_pct_per_decade": float(d.series_trend.sel(band=bname)),
                             "p": float(d.series_p.sel(band=bname)),
                             "n_eff": float(d.series_n_eff.sel(band=bname)), "_key": name})
    path = config.RESULTS / OUT / "cf_band_trends.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop(columns="_key").to_csv(path, index=False)
    print(f"Saved: {path}", flush=True)
    cells = [[r["band"], r["dataset"], r["years"], f"{r['trend_pct_per_decade']:+.2f}",
              f"{r['p']:.3f}", f"{r['n_eff']:.1f}", stats.sig_stars(r["p"])] for r in rows]
    fig, at = ps.figure("two", 0.2 * (len(cells) + 1) + 0.45)
    ax = at(0.05, 0.35, config.WIDTH["two"] - 0.1, 0.2 * (len(cells) + 1))
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=["Band", "Dataset", "Years", "Trend (% dec$^{-1}$)",
                                            "p", "n$_{eff}$", ""],
                 loc="upper center", cellLoc="center",
                 colWidths=[0.12, 0.27, 0.15, 0.18, 0.1, 0.1, 0.08])
    t.auto_set_font_size(False)
    t.set_fontsize(config.FS["small"])
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor("0.75")
        c.set_linewidth(0.4)
        c.set_height(1 / (len(cells) + 1))
        if i == 0:
            c.set_text_props(fontweight="bold")
            c.set_facecolor("#E6E6E6")
        elif rows[i - 1]["p"] < 0.05:
            c.set_facecolor("#FFF3CD")
        if i and j == 1:
            c.set_text_props(color=ps.colour(rows[i - 1]["_key"]), fontweight="bold")
    ps.text(fig, config.WIDTH["two"] / 2, 0.06, f"{calendar.month_name[MONTH]} cloud fraction "
            f"trends by latitude band (land, {BAND_LON[0]:.0f}-{BAND_LON[1]:.0f}E); shaded: p < 0.05",
            ha="center", va="top", fontsize=config.FS["title"], fontweight="bold")
    ps.save(fig, config.RESULTS / OUT / "cf_band_trends.png")


def band_map():
    import cartopy.crs as ccrs
    from matplotlib.patches import Rectangle
    fig, at = ps.figure(3.4, 3.0)
    ax = at(0.5, 0.15, 2.3, 2.5, projection=ccrs.PlateCarree())
    reg = dict(lon=(-20.0, 55.0), lat=(-37.0, 18.0))
    for k, (name, s_, n_) in enumerate(BANDS):
        ax.add_patch(Rectangle((BAND_LON[0], s_), BAND_LON[1] - BAND_LON[0], n_ - s_,
                               facecolor=BAND_COLOURS[k], alpha=0.35, edgecolor=BAND_COLOURS[k],
                               transform=ccrs.PlateCarree(), zorder=2))
        ax.text(BAND_LON[1] + 1.5, (s_ + n_) / 2, name, color=BAND_COLOURS[k],
                fontsize=config.FS["small"], fontweight="bold", va="center",
                transform=ccrs.PlateCarree())
    ps.map_frame(ax, reg, xticks=[-20, 0, 20, 40])
    ps.save(fig, config.RESULTS / OUT / "cf_latitude_bands_map.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "compute" or (what == "all" and not all(data_file(k).exists() for k, _, _ in MAPS)):
        compute()
    if what in ("plot", "all"):
        plot()
