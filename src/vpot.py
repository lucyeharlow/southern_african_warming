"""
October 200 hPa velocity potential (chi) and divergence: ERA5, MERRA-2, JRA-3Q, CMIP6 AMIP and
CMIP6 historical (multi-model means).

Run:  python velocity_potential.py [compute | plot]

Run the figures on their own with velocity_potential_reanalyses.py and
velocity_potential_cmip6.py (each computes first if its data files are missing).

FIGURES  results/velocity_potential/chi200_reanalyses_october.png  velocity potential, ERA5,
         results/velocity_potential/div200_reanalyses_october.png  divergence      MERRA-2, JRA-3Q
         results/velocity_potential/chi200_cmip6_october.png       velocity potential, CMIP6
         results/velocity_potential/div200_cmip6_october.png       divergence      AMIP, historical
  Datasets as rows (name and period on the left); columns climatology and trend, black dots
  where the trend is significant; a colour bar under each column. Reanalyses: FDR q = 0.10 over the global
  map. CMIP6: one-sample t test of the models' trends against zero, then the same FDR. One
  colour scale per column, set from that figure's datasets.

DATA (native grids, global; reanalysis years from config.YEARS)
  ERA5     RAW['era5_chi'] chi_october_pl.nc (chi on pressure levels, computed from u, v)
  JRA-3Q   vpot-pres-an-ll125 (analysed velocity potential), October means of the 6-hourly fields
  MERRA-2  instM_3d_asm_Np U, V at 200 hPa -> chi with skyborn's VectorWind (latitudes passed
           north-to-south, as the solver requires)
  CMIP6    RAW['cmip6_vp']/{amip, amip_200hpa, historical}/<MODEL>.nc: per-model October trend,
           pvalue and climatology mean_clt (a model in amip and amip_200hpa is used once)
  Divergence = Laplacian of chi (spectral, truncated at DIV_TRUNC so all datasets show the same
  scales); for CMIP6 the Laplacian of each model's climatology and trend.
DATA FILES  data/velocity_potential/<chi|div>200_<dataset>.nc: clim, trend, p, sig (lat, lon)
"""
from __future__ import annotations

import calendar
import glob
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats as sps

import config
import plotstyle as ps
import readers as rd
import stats

MONTH, LEVEL = 10, 200.0
OUT = "velocity_potential"
REGION = "global"
REANALYSES = ["ERA5", "MERRA-2", "JRA-3Q"]
CMIP = {"CMIP6 AMIP": ["amip", "amip_200hpa"], "CMIP6 historical": ["historical"]}
COLUMNS = REANALYSES + list(CMIP)
DIV_TRUNC = 63
EARTH_RADIUS = 6.371229e6
QUANT = {"chi": dict(scale=1e-6, unit="10$^6$ m$^2$ s$^{-1}$", title="velocity potential"),
         "div": dict(scale=1e6, unit="10$^{-6}$ s$^{-1}$", title="divergence")}
TICKS = dict(xticks=[-120, 0, 120], yticks=[-60, 0, 60])
STYLE = {"CMIP6 AMIP": "#555555", "CMIP6 historical": "#555555"}


def data_file(q, name):
    return config.DATA / OUT / f"{q}{LEVEL:.0f}_{name.replace(' ', '_').replace('-', '')}.nc"


# ---------------------------------------------------------------- spherical harmonics
def _pole_to_pole(lat):
    d = np.diff(lat)
    return np.allclose(d, d[0], atol=1e-4) and np.isclose(np.abs(lat).max(), 90, atol=1e-3)


def velocity_potential(u, v):
    """chi (lat, lon) from u, v DataArrays on a regular pole-to-pole grid (any lat order)."""
    from skyborn.windspharm.standard import VectorWind
    lat = u.lat.values
    if not _pole_to_pole(lat):
        raise ValueError("needs an evenly spaced pole-to-pole grid")
    n2s = lambda a: np.nan_to_num(np.asarray(a, float))[::-1] if lat[0] < lat[-1] else np.nan_to_num(np.asarray(a, float))
    _, chi = VectorWind(n2s(u), n2s(v), gridtype="regular").sfvp()
    chi = chi[::-1] if lat[0] < lat[-1] else chi
    return u.copy(data=chi.astype("float32"))


def laplacian(da):
    """Spectral Laplacian of (..., lat, lon) on a regular pole-to-pole grid, truncated."""
    from skyborn.spharm import Spharmt, getspecindx
    lat = da.lat.values
    if not _pole_to_pole(lat):
        raise ValueError("needs an evenly spaced pole-to-pole grid")
    da = da.transpose(..., "lat", "lon")
    flip = lat[0] < lat[-1]
    nlat, nlon = da.sizes["lat"], da.sizes["lon"]
    sh = Spharmt(nlon, nlat, rsphere=EARTH_RADIUS, gridtype="regular")
    nt = min(DIV_TRUNC or nlat - 1, nlat - 1)
    _, n = getspecindx(nt)
    vals = da.values.reshape(-1, nlat, nlon)
    out = []
    for f in vals:
        f = np.nan_to_num(f[::-1] if flip else f).astype(np.float64)
        g = sh.spectogrd(sh.grdtospec(f, nt) * (-n * (n + 1) / EARTH_RADIUS ** 2))
        out.append(g[::-1] if flip else g)
    return da.copy(data=np.stack(out).reshape(da.shape).astype("float32"))


# ---------------------------------------------------------------- reading
def load_reanalysis(name):
    """(year, lat, lon) October chi at LEVEL."""
    if name == "ERA5":
        with xr.open_dataset(config.RAW["era5_chi"]) as ds:
            da = rd.standardise(next(iter(ds.data_vars.values())))
            da = rd.pick_levels(da, LEVEL).load()
        return da
    if name == "JRA-3Q":
        return rd.jra3q("vpot-pres-an-ll125", "vpot-pres-an-ll125", REGION, lev=LEVEL,
                        months=[MONTH]).sel(month=MONTH)
    u = rd.merra2("instM_3d_asm_Np", "U", REGION, lev=LEVEL, months=[MONTH]).sel(month=MONTH)
    v = rd.merra2("instM_3d_asm_Np", "V", REGION, lev=LEVEL, months=[MONTH]).sel(month=MONTH)
    return xr.concat([velocity_potential(u.sel(year=y), v.sel(year=y)) for y in u.year.values],
                     "year")


def load_cmip(folders):
    files = {}
    for f in folders:
        for p in sorted(glob.glob(str(Path(config.RAW["cmip6_vp"]) / f / "*.nc"))):
            files[Path(p).stem] = p                          # later folder wins
    clim, trend, used, periods = [], [], [], set()
    for model, p in sorted(files.items()):
        with xr.open_dataset(p) as ds:
            if f"{LEVEL:.0f}" not in str(ds.attrs.get("level", f"{LEVEL:.0f}hPa")):
                rd.log(f"  SKIPPED {model}: level {ds.attrs.get('level')}")
                continue
            t, c = rd.standardise(ds["trend"]).load(), rd.standardise(ds["mean_clt"]).load()
            periods.add(str(ds.attrs.get("period", "?")))
        if trend and (t.shape != trend[0].shape):
            rd.log(f"  SKIPPED {model}: different grid")
            continue
        trend.append(t)
        clim.append(c)
        used.append(model)
    if not used:
        return None
    rd.log(f"  {len(used)} models: {', '.join(used)}")
    return (xr.concat(clim, "model"), xr.concat(trend, "model"), ", ".join(sorted(periods)))


def model_ttest(trend):
    """p of a one-sample t test of the models' trends against zero."""
    n = trend.sizes["model"]
    t = trend.mean("model") / (trend.std("model", ddof=1) / np.sqrt(n))
    return xr.apply_ufunc(lambda x: 2 * sps.t.sf(np.abs(x), n - 1), t)


# ---------------------------------------------------------------- compute
def save(q, name, clim, trend, p, sub):
    sig = stats.fdr(p)
    ds = xr.Dataset({"clim": clim.astype("float32"), "trend": trend.astype("float32"),
                     "p": p.astype("float32"), "sig": sig.astype("int8")})
    ds.attrs = {"dataset": name, "years": sub, "month": MONTH, "level_hPa": LEVEL}
    ds.to_netcdf(data_file(q, name), encoding={v: {"zlib": True} for v in ("clim", "trend", "p")})
    rd.log(f"  {q}: {stats.pct_area(sig, trend.notnull()):.0f} % of area significant")


def compute():
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    for name in REANALYSES:
        rd.log(f"--- {name} ---")
        chi = stats.select_years(load_reanalysis(name).rename(name), config.YEARS[name])
        sub = f"{config.YEARS[name][0]}-{config.YEARS[name][1]}"
        st = stats.ols_stats(chi)
        save("chi", name, chi.mean("year"), st.trend, st.p, sub)
        try:
            div = laplacian(chi)
            sd = stats.ols_stats(div)
            save("div", name, div.mean("year"), sd.trend, sd.p, sub)
        except Exception as e:
            rd.log(f"  divergence SKIPPED: {type(e).__name__}: {e}")
    for name, folders in CMIP.items():
        rd.log(f"--- {name} ---")
        got = load_cmip(folders)
        if got is None:
            continue
        clim, trend, period = got
        sub = f"{period}, mean of {trend.sizes['model']} models"
        save("chi", name, clim.mean("model"), trend.mean("model"), model_ttest(trend), sub)
        try:
            dt = laplacian(trend)
            save("div", name, laplacian(clim).mean("model"), dt.mean("model"), model_ttest(dt), sub)
        except Exception as e:
            rd.log(f"  divergence SKIPPED: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- plot
def row_label(n, years):
    """Dataset name and period for the left of a row (CMIP6: name and period on short lines)."""
    if n.startswith("CMIP6 "):
        n = n.replace("CMIP6 ", "CMIP6\n")
        years = years.replace(", mean of ", "\n")
    return n, years


def plot_quantity(names, q, stem, title):
    """One figure for one quantity (chi or div) and a group of datasets: datasets as rows (name
    and period on the left), columns climatology and trend (black dots: significant). One
    colour scale per column, from this figure's datasets, as a colour bar under the column."""
    import cartopy.crs as ccrs
    ps.setup()
    fs, W = config.FS, config.WIDTH["full"]
    res = {n: xr.load_dataset(data_file(q, n)) for n in names if data_file(q, n).exists()}
    names = [n for n in names if n in res]
    if not names:
        print(f"no {q} data files for {stem}: run compute first")
        return
    meta, sc = QUANT[q], QUANT[q]["scale"]
    cols = [("clim", "Climatology"), ("trend", f"Trend (dots: significant, FDR q = {config.ALPHA_FDR:.2f})")]
    lab_w, tick_w, right, gx, gy = 0.85, 0.33, 0.05, 0.08, 0.08
    x_first = lab_w + tick_w
    pw = (W - x_first - right - gx) / 2
    ph = pw / 2
    top = 0.5
    xlab, cb_gap, cb_h, cb_lab = 0.17, 0.06, 0.07, 0.62
    tops = [top + i * (ph + gy) for i in range(len(names))]
    H = tops[-1] + ph + xlab + cb_gap + cb_h + cb_lab
    fig, at = ps.figure("full", H)
    norms = {var: ps.sym_norm([d[var] * sc for d in res.values()]) for var, _ in cols}
    for ci, (var, head) in enumerate(cols):
        ps.text(fig, x_first + ci * (pw + gx) + pw / 2, top - 0.04, head, ha="center",
                fontsize=fs["title"], fontweight="bold")
    k = 0
    for ri, n in enumerate(names):
        d, t0 = res[n], tops[ri]
        last = ri == len(names) - 1
        name, years = row_label(n, d.attrs["years"])
        colour = STYLE.get(n, ps.colour(n) if n in config.DATASETS else "k")
        ps.text(fig, 0.03, t0 + ph / 2 - 0.02, name, ha="left", va="bottom",
                fontsize=fs["title"], fontweight="bold", color=colour)
        ps.text(fig, 0.03, t0 + ph / 2 + 0.02, years, ha="left", va="top",
                fontsize=fs["small"], color="0.3")
        for ci, (var, _) in enumerate(cols):
            ax = at(x_first + ci * (pw + gx), t0, pw, ph, projection=ccrs.PlateCarree())
            dots = var == "trend"
            ps.map_panel(ax, d[var] * sc, norms[var], sig=d.sig.astype(bool).values if dots else None,
                         region=REGION, left=(ci == 0), bottom=last, **TICKS)
            ps.letter(ax, k)
            k += 1
    cb_top = tops[-1] + ph + xlab + cb_gap
    for ci, (var, _) in enumerate(cols):
        unit = meta["unit"] + (" decade$^{-1}$" if var == "trend" else "")
        what = f"{meta['title'].capitalize()} {'climatology' if var == 'clim' else 'trend'}"
        x0 = x_first + ci * (pw + gx)
        ps.colourbar(fig, at(x0 + 0.08 * pw, cb_top, 0.84 * pw, cb_h), norms[var], "RdBu_r",
                     f"{what}\n({unit})", nticks=5)
    ps.text(fig, W / 2, 0.06, title, ha="center", va="top", fontsize=fs["suptitle"],
            fontweight="bold")
    ps.save(fig, config.RESULTS / OUT / f"{stem}.png")


def plot_group(names, group, label):
    """The velocity potential figure and the divergence figure for a group of datasets."""
    for q in QUANT:
        plot_quantity(names, q, f"{q}{LEVEL:.0f}_{group}_october",
                      f"{calendar.month_name[MONTH]} {LEVEL:.0f} hPa {QUANT[q]['title']}, {label}")


def plot():
    plot_group(REANALYSES, "reanalyses", "reanalyses")
    plot_group(list(CMIP), "cmip6", "CMIP6 multi-model means")


def missing(names):
    return not all(data_file(q, n).exists() for q in QUANT for n in names)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "compute" or (what == "all" and missing(REANALYSES)):
        compute()
    if what in ("plot", "all"):
        plot()
