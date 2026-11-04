"""
October 200 hPa velocity potential (chi) and divergence: ERA5, MERRA-2, JRA-3Q, CMIP6 AMIP and
CMIP6 historical (multi-model means).

Run:  python velocity_potential.py [compute | plot]

Run the figures on their own with velocity_potential_reanalyses.py and
velocity_potential_cmip6.py (each computes first if its data files are missing).

FIGURES  results/velocity_potential/vp200_reanalyses_october.png   ERA5, MERRA-2, JRA-3Q
         results/velocity_potential/vp200_cmip6_october.png        CMIP6 AMIP, historical
  Datasets as columns; rows: chi climatology, chi trend, divergence climatology, divergence
  trend, black dots where the trend is significant. Reanalyses: FDR q = 0.10 over the global
  map. CMIP6: one-sample t test of the models' trends against zero, then the same FDR. One
  colour scale per row, set from that figure's datasets.

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
def plot_group(names, stem, title):
    """One figure for a group of datasets: datasets as columns (titles on top); rows chi
    climatology, chi trend, divergence climatology, divergence trend (black dots: significant).
    One colour scale per row, from this figure's datasets."""
    import cartopy.crs as ccrs
    ps.setup()
    fs, W = config.FS, config.WIDTH["full"]
    res = {q: {n: xr.load_dataset(data_file(q, n)) for n in names if data_file(q, n).exists()}
           for q in QUANT}
    names = [n for n in names if any(n in res[q] for q in QUANT)]
    if not names:
        print(f"no data files for {', '.join(names) or stem}: run compute first")
        return
    rows = [(q, var) for q in QUANT for var in ("clim", "trend") if res[q]]
    left, right, gx, cbw = 0.42, 0.1, 0.1, 0.75
    nc = len(names)
    pw = (W - left - right - cbw - (nc - 1) * gx) / nc
    ph = pw / 2
    head, gy = 0.2, 0.32
    top = 0.92
    tops = [top + i * (ph + gy + head) for i in range(len(rows))]
    H = tops[-1] + ph + 0.4
    fig, at = ps.figure("full", H)
    for ci, n in enumerate(names):
        any_ds = next(res[q][n] for q in QUANT if n in res[q])
        x0 = left + ci * (pw + gx)
        ps.text(fig, x0 + pw / 2, 0.47, n, ha="center", fontsize=fs["title"], fontweight="bold",
                color=STYLE.get(n, ps.colour(n) if n in config.DATASETS else "k"))
        ps.text(fig, x0 + pw / 2, 0.5, any_ds.attrs["years"], ha="center", va="top",
                fontsize=fs["small"], color="0.3")
    k = 0
    for ri, (q, var) in enumerate(rows):
        meta, sc = QUANT[q], QUANT[q]["scale"]
        sym = "χ" if q == "chi" else "div"
        norm = ps.sym_norm([d[var] * sc for d in res[q].values()])
        t0 = tops[ri]
        ps.text(fig, left, t0 - 0.05,
                f"{sym} {'climatology' if var == 'clim' else 'trend'}"
                + (" (dots: significant)" if var == "trend" else ""),
                fontsize=fs["label"], fontweight="bold")
        for ci, n in enumerate(names):
            x0 = left + ci * (pw + gx)
            ax = at(x0, t0, pw, ph, projection=ccrs.PlateCarree())
            if n not in res[q]:
                ps.map_frame(ax, REGION, left=(ci == 0), bottom=True, **TICKS)
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center",
                        fontsize=fs["small"], color="0.4")
                continue
            d = res[q][n]
            dots = var == "trend"
            ps.map_panel(ax, d[var] * sc, norm, sig=d.sig.astype(bool).values if dots else None,
                         region=REGION, left=(ci == 0), bottom=True, **TICKS)
            if dots:
                ax.text(0.98, 0.04, f"{stats.pct_area(d.sig.astype(bool), d.trend.notnull()):.0f}% "
                        "of area sig.", transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=fs["small"], zorder=10,
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
            ps.letter(ax, k)
            k += 1
        unit = meta["unit"] + (" decade$^{-1}$" if var == "trend" else "")
        ps.colourbar(fig, at(W - right - cbw + 0.12, t0, 0.08, ph), norm, "RdBu_r", unit,
                     orientation="vertical", nticks=5)
    ps.text(fig, W / 2, 0.06, title, ha="center", va="top", fontsize=fs["suptitle"],
            fontweight="bold")
    ps.save(fig, config.RESULTS / OUT / f"{stem}.png")


def figure_title(group):
    return (f"{calendar.month_name[MONTH]} {LEVEL:.0f} hPa velocity potential and divergence, "
            f"{group}")


def plot():
    plot_group(REANALYSES, f"vp{LEVEL:.0f}_reanalyses_october", figure_title("reanalyses"))
    plot_group(list(CMIP), f"vp{LEVEL:.0f}_cmip6_october", figure_title("CMIP6 multi-model means"))


def missing(names):
    return not all(data_file(q, n).exists() for q in QUANT for n in names)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "compute" or (what == "all" and missing(REANALYSES)):
        compute()
    if what in ("plot", "all"):
        plot()
