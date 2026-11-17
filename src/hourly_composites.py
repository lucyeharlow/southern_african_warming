"""
October 12-hourly composites and correlations on a Southern Africa 2 m temperature index,
from ERA5 hourly data, 1979-2025.

12-HOUR MEANS  every October day is split into a day half (DAY_HOURS_UTC, 04-15 UTC = 06-17
  South African time) and a night half (the other 12 hours of the same date: 16-23 and 00-03
  UTC), so day and night are ranked and composited separately (31 days x 47 years = 1457
  periods per half).

INDEX  the 2 m temperature anomaly averaged over the land cells of the Southern Africa box
  (10-40E, 35-10S). Anomalies are from a 1991-2020 climatology of each half that follows the
  warming through October (a straight-line fit of the 1991-2020 means against day of month).
  RANKING files: every period, sorted from the warmest to the coolest anomaly, with its date,
  half, UTC hours, anomaly and percentile (one CSV per half and variant).

COMPOSITES  for the periods >= the 90th percentile (high) and <= the 10th (low) of each half,
  the mean anomaly of 2 m temperature, specific humidity and temperature at 850 hPa, omega at
  500 and 300 hPa, the latent heat flux (upward, land) and SW down (day only), and high minus
  low. Significance: the selection is shifted in time by a random number of periods (at least
  MIN_SHIFT_DAYS days; circular over the chronological series of that half) N_RANDOM times,
  which keeps the clustering of warm spells, and p is the share of shifted composites at least
  as large; black dots: significant after FDR (q = config.ALPHA_FDR).
  Everything is done with the trend in and with the linear 1979-2025 trend removed (index and
  fields, per half and cell).

CORRELATIONS  detrended: the index against every variable, as maps (per half) and as box
  means (table); p from stats.corr_and_p (lag-1 n_eff).

Run:  python hourly_composites.py            read (if not cached), compute and plot
      python hourly_composites.py read       only read the hourly files into the cache
      python hourly_composites.py plot       plot from the data files
      add --vars t2m,w500,swdn to work on a subset (t2m, the index, is always included)
  netCDF files are cut with contiguous slices in the file's own order (read_nc_region): reading
  the 37-level hourly files by flipped or reordered indices is extremely slow.
  Reading is the slow part and is done in parallel over years: run it as a SLURM job with
  several CPUs (run_hourly.sh); a variable-year is read once and cached
  (processed_data/cache/era5h12_<var>_composite/<year>.nc).

FIGURES  results/hourly_composites/ (the detrended versions end in _detrended)
  t2m12h_index_october.png                number of high/low periods per year, day and night
  t2m12h_composites_<day|night>_october.png  rows = variables; columns = high, low, high - low
  t2m12h_correlations_october_detrended.png  detrended correlation maps, columns day, night
TABLES  t2m12h_ranking_<day|night>_october[_detrended].csv, t2m12h_box_correlations.csv

DATA  ERA5 hourly, HOURLY/<variable>/{nc,grb}/era5_hourly_<variable>_YYYY10.{nc,grb} (netCDF
  preferred if both exist); accumulated fluxes (J m-2 per hour) / 3600 -> W m-2, H_L = -slhf.
"""
from __future__ import annotations

import calendar
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import config
import masks
import plotstyle as ps
import readers as rd
import stats

HOURLY = Path("/soge-home/data/analysis/era5/0.28125x0.28125/hourly")
MONTH = 10
YEARS = (1979, 2025)
BOX = "southern_africa"
MAP = "composite"                     # area read and mapped
DAY_HOURS_UTC = range(4, 16)          # 06-17 South African time; the rest of the date = night
HALVES = ("day", "night")
PCT = 10
VARIANTS = (False, True)              # with the trend, detrended
N_RANDOM = 1000                       # time-shifted composites for the significance test
MIN_SHIFT_DAYS = 10
OUT = "hourly_composites"
HOUR = 3600.0
SEED = 1
rd.CACHE_MAX_CELLS = max(rd.CACHE_MAX_CELLS, 50_000_000)   # a year of 12-h fields over MAP

# key: (folder, level hPa, scale, row label, units, colour map, land only, day only)
VARS = {
    "t2m": ("2m_temperature", None, 1.0, "2 m temperature", "K", "RdBu_r", False, False),
    "q850": ("specific_humidity", 850.0, 1000.0, "$q$ 850 hPa", "g kg$^{-1}$", "BrBG", False,
             False),
    "t850": ("temperature", 850.0, 1.0, "$T$ 850 hPa", "K", "RdBu_r", False, False),
    "w500": ("vertical_velocity", 500.0, 1.0, "$\\omega$ 500 hPa", "Pa s$^{-1}$", "RdBu_r",
             False, False),
    "w300": ("vertical_velocity", 300.0, 1.0, "$\\omega$ 300 hPa", "Pa s$^{-1}$", "RdBu_r",
             False, False),
    "lh": ("surface_latent_heat_flux", None, -1.0 / HOUR, "$H_L$", "W m$^{-2}$", "BrBG", True,
           False),
    "swdn": ("surface_solar_radiation_downwards", None, 1.0 / HOUR, "SW down", "W m$^{-2}$",
             "RdBu_r", False, True),
}
INDEX = "t2m"
VARIANT = False                       # set by the loops
_SUFFIX = lambda: "_detrended" if VARIANT else ""


# ---------------------------------------------------------------- reading
def month_file(folder, year, month=MONTH):
    for kind, ext in (("nc", "nc"), ("grb", "grb")):
        p = HOURLY / folder / kind / f"era5_hourly_{folder}_{year}{month:02d}.{ext}"
        if p.exists():
            return p
    return None


def open_hourly(path, lev=None):
    """One monthly file -> lazily opened DataArray (time, [lev,] lat, lon) on valid times.
    GRIB files with pressure levels are filtered to the one level wanted while opening."""
    if str(path).endswith(".grb"):
        try:
            import cfgrib  # noqa: F401
        except ImportError:
            raise ImportError("cfgrib is not installed in this Python (run with the myenv "
                              "Python: see run_hourly.sh)") from None
        kw = {"indexpath": ""}
        if lev is not None:
            kw["filter_by_keys"] = {"typeOfLevel": "isobaricInhPa", "level": int(lev)}
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=kw)
    else:
        ds = xr.open_dataset(path)
    da = ds[list(ds.data_vars)[0]]
    return valid_times(da)


def valid_times(da):
    """GRIB accumulations come as (time, step): put every field on its valid time."""
    if "step" in da.dims and "time" in da.dims:
        vt = da["valid_time"].values.ravel()
        da = da.stack(t=("time", "step")).transpose("t", ...)
        da = da.drop_vars([c for c in ("t", "time", "step", "valid_time") if c in da.coords])
        da = da.rename(t="time").assign_coords(time=vt)
    return da                     # analysis fields: readers.standardise sorts out valid_time


def twelve_hourly(da, year, month=MONTH):
    """(time, lat, lon) hourly -> (dom, half, lat, lon) means of the day and night halves of
    every date of the month; dates with fewer than 10 of the 12 hours are NaN."""
    t = pd.DatetimeIndex(da.time.values)
    keep = (t.year == year) & (t.month == month)
    da, t = da.isel(time=np.flatnonzero(keep)), t[keep]
    half = np.where(np.isin(t.hour, list(DAY_HOURS_UTC)), 0, 1)
    ndays = calendar.monthrange(year, month)[1]
    key = (t.day.values - 1) * 2 + half
    vals = da.values.astype("float64")
    out = np.full((ndays * 2,) + vals.shape[1:], np.nan)
    for k in np.unique(key):
        sel = key == k
        if sel.sum() >= 10:
            out[k] = np.nanmean(vals[sel], axis=0)
    out = out.reshape((ndays, 2) + vals.shape[1:])
    return xr.DataArray(out, dims=("dom", "half") + da.dims[1:],
                        coords={"dom": np.arange(1, ndays + 1), "half": list(HALVES),
                                **{d: da[d] for d in da.dims[1:]}})


def _runs(idx):
    """Sorted integer indices -> list of contiguous slices."""
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) != 1) + 1
    return [slice(int(g[0]), int(g[-1]) + 1) for g in np.split(idx, cuts)]


def read_nc_region(path, lev):
    """Fast netCDF read: the region and level are cut in the file's own order with
    contiguous slices only (latitude as stored, longitude in at most two blocks either side of
    0 degrees), so the library reads a few blocks per time step instead of scattered points;
    reordering to ascending latitude and -180..180 longitude is done after loading."""
    r = config.REGIONS[MAP]
    with xr.open_dataset(path) as ds:
        v = ds[list(ds.data_vars)[0]]
        latn = "latitude" if "latitude" in v.dims else "lat"
        lonn = "longitude" if "longitude" in v.dims else "lon"
        levn = next((d for d in ("level", "plev", "lev", "pressure_level") if d in v.dims), None)
        lat = v[latn].values
        lon = ((v[lonn].values + 180.0) % 360.0) - 180.0
        ilat = np.flatnonzero((lat >= r["lat"][0] - 0.5) & (lat <= r["lat"][1] + 0.5))
        ilon = np.flatnonzero((lon >= r["lon"][0] - 0.5) & (lon <= r["lon"][1] + 0.5))
        sel = {latn: slice(int(ilat[0]), int(ilat[-1]) + 1)}
        if levn is not None:
            levs = v[levn].values.astype(float)
            k = int(np.argmin(np.abs(levs - lev)))
            if abs(levs[k] - lev) > 1:
                raise ValueError(f"no {lev:g} hPa level in {path}")
            sel[levn] = k
        parts = [v.isel(**sel, **{lonn: s}).load() for s in _runs(ilon)]
    da = xr.concat(parts, lonn) if len(parts) > 1 else parts[0]
    da = da.assign_coords({lonn: ((da[lonn] + 180.0) % 360.0) - 180.0}).sortby(lonn).sortby(latn)
    return rd.standardise(da)


def reader(key):
    folder, lev, scale, *_ = VARS[key]

    def read(path, year):
        if str(path).endswith(".nc"):
            da = read_nc_region(path, lev)
        else:
            da = rd.cut(rd.standardise(open_hourly(path, lev)), MAP)
            if lev is not None and "lev" in da.dims:
                da = rd.pick_levels(da, lev)
            da = da.load()
        da = rd.cut(da, MAP)
        da = da.drop_vars([c for c in ("lev",) if c in da.coords])
        return twelve_hourly(rd.clean(da) * scale, int(year))
    return read


def load(key):
    """(year, dom, half, lat, lon) 12-hour means of one variable."""
    folder = VARS[key][0]
    files = {y: month_file(folder, y) for y in range(YEARS[0], YEARS[1] + 1)}
    missing = [y for y, p in files.items() if p is None]
    if missing:
        rd.log(f"  [NOTE] {key}: no October file for {missing}")
    files = {y: p for y, p in files.items() if p is not None}
    fields = rd.cached(f"era5h12_{key}_{MAP}", files, reader(key))
    years = sorted(fields)
    if not years:
        rd.log(f"  [WARN] {key}: no file could be read - left out")
        return None
    return xr.concat([fields[y] for y in years], "year").assign_coords(year=years)


# ---------------------------------------------------------------- statistics
def anomalies(da):
    """Remove the 1991-2020 climatology of each half and cell, a straight line in day of
    month; detrended (linear in year, per half and cell) when VARIANT."""
    base = da.sel(year=slice(*config.BASELINE)).mean("year")
    d = base.dom - base.dom.mean()
    bm = base.mean("dom")
    b = (d * (base - bm)).sum("dom") / (d * d).sum()
    a = da - (bm + b * d)
    if VARIANT:
        y = a.year - a.year.mean()
        ya = a.mean("dom")                                  # per year: same slope, less work
        ya = ya - ya.mean("year")
        slope = (y * ya).sum("year") / (y * y).sum()
        a = a - slope * y
    return a


def flatten(a, half):
    """(year, dom, [lat, lon]) of one half -> (t, ...) chronological, without missing dates."""
    s = a.sel(half=half, drop=True).stack(t=("year", "dom")).transpose("t", ...)
    return s


def index_series(t2m):
    a = anomalies(t2m).where(masks.land_mask(t2m.isel(year=0, dom=0, half=0, drop=True)))
    return stats.area_mean(rd.cut(a, BOX))                    # (year, dom, half)


def shifted_p(x, idx_a, idx_b, obs, rng):
    """Circular time-shift test: x (t, cells); idx_a (and idx_b for a difference) selected
    periods; p = share of shifted composites with |value| >= |obs|."""
    n = x.shape[0]
    shifts = rng.integers(MIN_SHIFT_DAYS, n - MIN_SHIFT_DAYS, N_RANDOM)
    o = np.abs(obs)
    count = np.zeros_like(o)
    for k in shifts:
        c = x[(idx_a + k) % n].mean(axis=0)
        if idx_b is not None:
            c = c - x[(idx_b + k) % n].mean(axis=0)
        count += np.abs(c) >= o - 1e-12
    return (count + 1) / (N_RANDOM + 1)


def composites_half(a, sel_hi, sel_lo, land, rng):
    """Composites of one variable and one half: a (t, lat, lon) anomalies."""
    if land:
        a = a.where(masks.land_mask(a.isel(t=0, drop=True)))
    x = a.values.reshape(a.sizes["t"], -1)
    ok = np.isfinite(x).all(axis=0)
    xo = np.where(ok, x, 0.0)
    ih, il = np.flatnonzero(sel_hi), np.flatnonzero(sel_lo)
    out = {}
    for name, ia, ib in (("high", ih, None), ("low", il, None), ("diff", ih, il)):
        c = xo[ia].mean(axis=0) - (xo[ib].mean(axis=0) if ib is not None else 0)
        p = shifted_p(xo, ia, ib, c, rng)
        c[~ok], p[~ok] = np.nan, np.nan
        shape = a.shape[1:]
        out[name] = (("lat", "lon"), c.reshape(shape).astype("float32"))
        out[f"p_{name}"] = (("lat", "lon"), p.reshape(shape).astype("float32"))
        out[f"sig_{name}"] = (("lat", "lon"), stats.fdr(p.reshape(shape)).astype("int8"))
    return xr.Dataset(out, coords={"lat": a.lat, "lon": a.lon})


def plain(da):
    """Drop the (year, dom) index of a flattened series: a plain t axis for stats.corr_and_p."""
    return da.drop_vars([c for c in ("t", "year", "dom") if c in da.coords])


def data_file(name):
    return config.DATA / OUT / f"{name}{_SUFFIX()}.nc"


# ---------------------------------------------------------------- compute
def read_all():
    rd.log("--- reading ERA5 hourly October files (12-hour means, cached) ---")
    out = {k: load(k) for k in VARS}
    out = {k: v for k, v in out.items() if v is not None}
    if INDEX not in out:
        raise SystemExit(f"the index variable {INDEX} could not be read")
    return out


def compute():
    global VARIANT
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    (config.RESULTS / OUT).mkdir(parents=True, exist_ok=True)
    fields = read_all()
    years = sorted(set.intersection(*[set(v.year.values) for v in fields.values()]))
    fields = {k: v.sel(year=years) for k, v in fields.items()}
    rows = []
    for VARIANT in VARIANTS:
        rd.log(f"=== {'detrended' if VARIANT else 'with the trend'} ===")
        index = index_series(fields[INDEX])
        index.rename("index").to_netcdf(data_file("index"))
        anoms = {k: anomalies(v) for k, v in fields.items()}
        for half in HALVES:
            s = flatten(index, half)
            ok = np.isfinite(s.values)
            v = s.values
            t_hi, t_lo = np.nanpercentile(v, 100 - PCT), np.nanpercentile(v, PCT)
            hi, lo = ok & (v >= t_hi), ok & (v <= t_lo)
            ranking(s, t_hi, t_lo, half)
            rd.log(f"  {half}: {ok.sum()} periods; P{100 - PCT} = {t_hi:+.2f} K "
                   f"({hi.sum()} periods), P{PCT} = {t_lo:+.2f} K ({lo.sum()})")
            rng = np.random.default_rng(SEED)
            for key, a in anoms.items():
                if VARS[key][7] and half == "night":
                    continue
                fa = flatten(a, half).isel(t=np.flatnonzero(ok))
                ds = composites_half(fa, hi[ok], lo[ok], VARS[key][6], rng)
                ds.attrs = {"n_high": int(hi.sum()), "n_low": int(lo.sum()),
                            "p_high": float(t_hi), "p_low": float(t_lo)}
                ds.to_netcdf(data_file(f"{key}_{half}"))
                if VARIANT:                                     # detrended correlations
                    x = plain(s.isel(t=np.flatnonzero(ok)))
                    f_ = plain(fa)
                    if VARS[key][6]:
                        f_ = f_.where(masks.land_mask(f_.isel(t=0, drop=True)))
                    r, p = stats.corr_and_p(x, f_, dim="t")
                    xr.Dataset({"r": r.astype("float32"), "p": p.astype("float32"),
                                "sig": stats.fdr(p).astype("int8")}).to_netcdf(
                        data_file(f"corr_{key}_{half}"))
                    bm = stats.area_mean(rd.cut(f_, BOX))
                    rb, pb = stats.corr_and_p(x, bm, dim="t")
                    rows.append({"variable": key, "half": half, "r_box_mean": float(rb),
                                 "p": float(pb), "n": int(ok.sum())})
                rd.log(f"    {key}: composites done")
    VARIANT = False
    if rows:
        path = config.RESULTS / OUT / "t2m12h_box_correlations.csv"
        pd.DataFrame(rows).to_csv(path, index=False, float_format="%.4g")
        rd.log(f"Saved: {path}")
        print(pd.DataFrame(rows).round(3).to_string(index=False))


def ranking(s, t_hi, t_lo, half):
    """CSV of every period of one half, from the warmest to the coolest anomaly."""
    df = pd.DataFrame({"year": s.year.values, "month": MONTH, "day": s.dom.values,
                       "half": half, "anomaly_K": s.values})
    df["hours_utc"] = ("04-15" if half == "day" else "16-23, 00-03")
    df = df.dropna(subset=["anomaly_K"]).sort_values("anomaly_K", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    df["percentile"] = 100.0 * (1 - (df["rank"] - 0.5) / len(df))
    df["group"] = np.where(df.anomaly_K >= t_hi, "high", np.where(df.anomaly_K <= t_lo, "low", ""))
    cols = ["rank", "year", "month", "day", "half", "hours_utc", "anomaly_K", "percentile", "group"]
    path = config.RESULTS / OUT / f"t2m12h_ranking_{half}_october{_SUFFIX()}.csv"
    df[cols].to_csv(path, index=False, float_format="%.4g")
    rd.log(f"  Saved: {path}")


# ---------------------------------------------------------------- plot
def plot_index():
    import matplotlib.pyplot as plt
    fs, W = config.FS, config.WIDTH["full"]
    idx = xr.load_dataarray(data_file("index"))
    fig, at = ps.figure("full", 3.6)
    for i, half in enumerate(HALVES):
        s = flatten(idx, half)
        v = s.values
        t_hi, t_lo = np.nanpercentile(v, 100 - PCT), np.nanpercentile(v, PCT)
        yrs = s.year.values
        years = np.unique(yrs)
        n_hi = np.array([np.sum((yrs == y) & (v >= t_hi)) for y in years])
        n_lo = np.array([np.sum((yrs == y) & (v <= t_lo)) for y in years])
        ax = at(0.7, 0.55 + i * 1.5, W - 0.85, 1.15)
        ax.bar(years, n_hi, color="#C63F3E", width=0.8, label=f"≥ P{100 - PCT} (warm)")
        ax.bar(years, -n_lo, color="#245E55", width=0.8, label=f"≤ P{PCT} (cool)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(f"{half.capitalize()} periods\nper October")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{abs(int(x))}"))
        ax.grid(axis="y", alpha=0.3, lw=0.4)
        ax.set_title(f"{half.capitalize()} ({'04-15' if half == 'day' else '16-03'} UTC)",
                     loc="left", fontsize=fs["small"] + 0.5, fontweight="bold", pad=2)
        ax.tick_params(labelbottom=(i == 1))
        if i == 0:
            ax.legend(loc="upper right", fontsize=fs["small"] - 1, frameon=False, ncol=2)
        ps.letter(ax, i)
    ps.text(fig, W / 2, 0.08, "October 12-hour 2 m temperature extremes over Southern Africa "
            "(land), ERA5" + (" (detrended)" if VARIANT else ""), ha="center", va="top",
            fontsize=fs["title"], fontweight="bold")
    ps.save(fig, config.RESULTS / OUT / f"t2m12h_index_october{_SUFFIX()}.png")


def plot_composites(half):
    import cartopy.crs as ccrs
    fs, W = config.FS, config.WIDTH["full"]
    keys = [k for k in VARS if data_file(f"{k}_{half}").exists()]
    if not keys:
        return
    res = {k: xr.load_dataset(data_file(f"{k}_{half}")) for k in keys}
    r = config.REGIONS[MAP]
    aspect = (r["lat"][1] - r["lat"][0]) / (r["lon"][1] - r["lon"][0])
    left, right, gx, gy, cbw = 0.55, 0.12, 0.08, 0.12, 0.72
    pw = (W - left - right - cbw - 2 * gx) / 3
    mh = pw * aspect
    top = 0.95
    H = top + len(keys) * (mh + gy) + 0.25
    fig, at = ps.figure("full", H)
    a = res[keys[0]].attrs
    heads = [f"Warm (≥ P{100 - PCT}, n = {a['n_high']})",
             f"Cool (≤ P{PCT}, n = {a['n_low']})", "Warm minus cool"]
    for c, h in enumerate(heads):
        ps.text(fig, left + c * (pw + gx) + pw / 2, top - 0.2, h, ha="center",
                fontsize=fs["title"], fontweight="bold")
    k = 0
    for i, key in enumerate(keys):
        _, _, _, lab, unit, cmap, _, _ = VARS[key]
        ds = res[key]
        t0 = top + i * (mh + gy)
        norm = ps.sym_norm([ds["high"], ds["low"], ds["diff"]], cmap=cmap)
        ps.text(fig, left - 0.42, t0 + mh / 2, lab, rotation=90, ha="center", va="center",
                fontsize=fs["label"], fontweight="bold")
        for c, which in enumerate(("high", "low", "diff")):
            ax = at(left + c * (pw + gx), t0, pw, mh, projection=ccrs.PlateCarree())
            ps.map_panel(ax, ds[which], norm, cmap=cmap,
                         sig=ds[f"sig_{which}"].astype(bool).values, region=MAP,
                         left=(c == 0), bottom=(i == len(keys) - 1), box=BOX,
                         xticks=[0, 30], yticks=[-30, 0])
            ps.letter(ax, k)
            k += 1
        ps.colourbar(fig, at(left + 3 * pw + 2 * gx + 0.12, t0 + 0.08, 0.08, mh - 0.16), norm,
                     cmap, unit, orientation="vertical", nticks=5)
    ps.text(fig, W / 2, 0.06, f"October {half} composites on the Southern Africa 12-hour "
            "2 m temperature index, ERA5" + (" (detrended)" if VARIANT else ""), ha="center",
            va="top", fontsize=fs["suptitle"] - 1, fontweight="bold")
    ps.text(fig, W / 2, H - 0.12, f"anomalies from {config.BASELINE[0]}-{config.BASELINE[1]}; "
            f"dots: significant against {N_RANDOM} time-shifted composites (FDR q = "
            f"{config.ALPHA_FDR:.2f}); dashed: Southern Africa box", ha="center", va="top",
            fontsize=fs["small"] - 1, color="0.3")
    ps.save(fig, config.RESULTS / OUT / f"t2m12h_composites_{half}_october{_SUFFIX()}.png")


def plot_correlations():
    import cartopy.crs as ccrs
    global VARIANT
    VARIANT = True
    fs, W = config.FS, config.WIDTH["full"]
    keys = [k for k in VARS if k != INDEX and data_file(f"corr_{k}_day").exists()]
    if not keys:
        VARIANT = False
        return
    r = config.REGIONS[MAP]
    aspect = (r["lat"][1] - r["lat"][0]) / (r["lon"][1] - r["lon"][0])
    left, gx, gy = 0.6, 0.15, 0.12
    pw = 2.2
    mh = pw * aspect
    top = 0.8
    H = top + len(keys) * (mh + gy) + 0.75
    fig, at = ps.figure("full", H)
    x0 = (W - 2 * pw - gx) / 2 + 0.2
    from matplotlib.colors import BoundaryNorm
    norm = BoundaryNorm(np.linspace(-0.8, 0.8, 17), 256, extend="both")
    for c, half in enumerate(HALVES):
        ps.text(fig, x0 + c * (pw + gx) + pw / 2, top - 0.15, half.capitalize(), ha="center",
                fontsize=fs["title"], fontweight="bold")
    k = 0
    for i, key in enumerate(keys):
        t0 = top + i * (mh + gy)
        ps.text(fig, x0 - 0.45, t0 + mh / 2, VARS[key][3], rotation=90, ha="center",
                va="center", fontsize=fs["label"], fontweight="bold")
        for c, half in enumerate(HALVES):
            ax = at(x0 + c * (pw + gx), t0, pw, mh, projection=ccrs.PlateCarree())
            f = data_file(f"corr_{key}_{half}")
            if f.exists():
                ds = xr.load_dataset(f)
                ps.map_panel(ax, ds.r, norm, cmap="RdBu_r", sig=ds.sig.astype(bool).values,
                             region=MAP, left=(c == 0), bottom=(i == len(keys) - 1), box=BOX,
                             xticks=[0, 30], yticks=[-30, 0])
            else:
                ps.map_frame(ax, MAP, left=(c == 0), bottom=(i == len(keys) - 1), box=BOX,
                             xticks=[0, 30], yticks=[-30, 0])
                ax.text(0.5, 0.5, "day only", transform=ax.transAxes, ha="center",
                        va="center", fontsize=fs["small"], color="0.4")
            ps.letter(ax, k)
            k += 1
    ps.colourbar(fig, at(x0, H - 0.5, 2 * pw + gx, 0.1), norm, "RdBu_r",
                 "r with the 12-hour 2 m temperature index (detrended); dots: significant (FDR)",
                 nticks=9)
    ps.text(fig, W / 2, 0.06, "October detrended correlations with the Southern Africa "
            "2 m temperature index, ERA5 12-hourly", ha="center", va="top",
            fontsize=fs["title"], fontweight="bold")
    ps.save(fig, config.RESULTS / OUT / "t2m12h_correlations_october_detrended.png")
    VARIANT = False


def plot():
    global VARIANT
    ps.setup()
    for VARIANT in VARIANTS:
        if not data_file("index").exists():
            continue
        plot_index()
        for half in HALVES:
            plot_composites(half)
    VARIANT = False
    plot_correlations()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="October 12-hourly t2m-index composites (ERA5 hourly).")
    ap.add_argument("what", nargs="?", default="all", choices=["all", "read", "compute", "plot"])
    ap.add_argument("--vars", help="comma-separated subset of " + ",".join(VARS) +
                    " (t2m, the index, is always included), e.g. --vars t2m,w500,swdn")
    args = ap.parse_args()
    if args.vars:
        want = {INDEX, *[v.strip() for v in args.vars.split(",") if v.strip()]}
        bad = want - set(VARS)
        if bad:
            raise SystemExit(f"unknown --vars {sorted(bad)}; choose from {list(VARS)}")
        for k in list(VARS):
            if k not in want:
                del VARS[k]
    if args.what == "read":
        read_all()
    if args.what in ("all", "compute"):
        compute()
    if args.what in ("all", "plot"):
        plot()