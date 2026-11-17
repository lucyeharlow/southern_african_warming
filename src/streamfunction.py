"""
Quick look: October meridional mass streamfunction over Southern Africa (10-40E): early and
late decades, their difference and the linear trend. ERA5 only.

Run:  python psi_section.py [compute | plot]

STREAMFUNCTION  For a longitude band the full meridional wind does not conserve mass (air also
  leaves and enters through the band's sides), so the "local" streamfunction of Schwendike et
  al. (2014, JGR 119, 1322-1339) is used, from the divergent meridional wind:
      psi(phi, p) = (2 pi a cos(phi) / g) * integral from the top to p of v_chi dp
  with v_chi = (1/a) d(chi)/d(phi) the 10-40E mean (vchi_section.py: every October of
  config.YEARS['ERA5'], its per-year files reused, so nothing is read twice). It is scaled to
  a full latitude circle, as the zonal-mean streamfunction is, so the numbers compare with it.
  Integrated from the top level (100 hPa; psi = 0 there) with the trapezium rule over LEVELS;
  levels below the ground add nothing and are left out of the plot.
  Sign: psi > 0 is clockwise in this latitude-pressure view (northward flow aloft, southward
  below); a Southern Hemisphere Hadley cell is negative.

FIGURE  results/psi/psi_section_october.png
  a) early decade (EARLY, 1980-1989) and b) late decade (LATE, 2016-2025) October means:
     shading and contours, one colour scale for both
  c) late minus early (shading), d) linear trend over config.YEARS['ERA5'] per decade
     (shading; stats.ols_stats, lag-1 n_eff; dots where significant, FDR q = 0.10); in c-d the
     black contours are the climatology of all the years (dashed negative). c and d, the
     changes, are drawn larger.
  Brown line: MERRA-2 climatological October surface pressure.
DATA FILE  data/psi/psi_yearly_october.nc (psi: year, lev, lat, kg s-1; ps: lat)
"""
from __future__ import annotations

import calendar
import sys

import numpy as np
import xarray as xr

import config
import plotstyle as ps
import readers as rd
import stats
import div_wind as vs

OUT = "psi"
REGION = (10.0, 40.0)                     # longitudes of the Southern Africa box
EARLY, LATE = (1980, 1989), (2016, 2025)
G = 9.80665
PSI_SCALE = 1e-10                         # kg s-1 -> 10^10 kg s-1 on the figure


def data_file():
    return config.DATA / OUT / f"psi_yearly_{calendar.month_name[vs.MONTH].lower()}.nc"


def streamfunction(v, ps_):
    """psi (..., lev, lat) in kg s-1 from band-mean v_chi (..., lev, lat; NaN below ground):
    (2 pi a cos(phi) / g) times the trapezium integral of v from the top level down to p."""
    v = v.sortby("lev")
    p = v.lev.values * 100.0                                   # Pa, top first
    x = np.nan_to_num(v.transpose(..., "lev", "lat").values)
    seg = 0.5 * (x[..., 1:, :] + x[..., :-1, :]) * np.diff(p)[:, None]
    integ = np.concatenate([np.zeros_like(x[..., :1, :]), np.cumsum(seg, axis=-2)], axis=-2)
    coslat = np.cos(np.deg2rad(v.lat.values))
    psi = v.transpose(..., "lev", "lat").copy(data=2 * np.pi * vs.A * coslat / G * integ)
    return psi.where(psi.lev <= ps_)                           # leave out below the ground


# ---------------------------------------------------------------- compute
def compute():
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    psc = rd.surface_pressure_clim(vs.SECTION, months=[vs.MONTH]).sel(month=vs.MONTH)
    lat_out = np.arange(vs.SECTION["lat"][0], vs.SECTION["lat"][1] + 1e-6, vs.GRID)
    got = vs.make_year_files()
    v = vs.band_means(vs.load_year_files(got), psc, [REGION]).isel(band=0, drop=True)
    v = v.interp(lat=lat_out)                                             # (year, lev, lat)
    pg = psc.interp(lat=lat_out, lon=np.arange(vs.SECTION["lon"][0], vs.SECTION["lon"][1] + 1e-6,
                                               vs.GRID), kwargs={"fill_value": None})
    ps_ = pg.sel(lon=slice(REGION[0], REGION[1] - 1e-6)).mean("lon")
    psi = streamfunction(v, ps_).transpose("year", "lev", "lat")
    ds = xr.Dataset({"psi": psi.astype("float32"), "ps": ps_.astype("float32")})
    ds.attrs = {"month": vs.MONTH, "lon": f"{REGION[0]:.0f}-{REGION[1]:.0f}E",
                "units": "psi kg s-1 (full latitude circle), ps hPa",
                "definition": "psi = 2 pi a cos(phi) / g * integral_top^p v_chi dp"}
    ds.to_netcdf(data_file())
    clim = psi.mean("year")
    rd.log(f"saved {data_file()} ({got[0]}-{got[-1]}, {len(got)} Octobers)")
    rd.log(f"  climatological psi {float(clim.min()) * PSI_SCALE:+.1f} to "
           f"{float(clim.max()) * PSI_SCALE:+.1f} x 10^10 kg s-1")


# ---------------------------------------------------------------- plot
def contour_levels(da, n=6):
    """Symmetric, rounded contour interval with about n lines on each side."""
    m = float(np.nanpercentile(np.abs(da), 98)) if np.isfinite(da).any() else 1.0
    step = float(f"{m / n:.1g}") or 1.0
    top = step * np.ceil(m / step)
    lv = np.arange(-top, top + step / 2, step)
    return lv[np.abs(lv) > step / 2]                           # no zero line


def plot():
    import matplotlib.pyplot as plt
    if not data_file().exists():
        raise SystemExit("no data file: run 'python psi_section.py compute' first")
    ds = xr.load_dataset(data_file())
    if "band" in ds.dims:
        raise SystemExit("data file from the band version: run 'python psi_section.py compute'")
    psi = ds.psi * PSI_SCALE
    y0, y1 = int(ds.year.min()), int(ds.year.max())
    span = lambda p: f"{max(p[0], y0)}-{min(p[1], y1)}"
    clim = psi.mean("year")
    early = psi.sel(year=slice(*EARLY)).mean("year")
    late = psi.sel(year=slice(*LATE)).mean("year")
    diff = late - early
    st = stats.ols_stats(psi.rename("psi"))
    sig = stats.fdr(st.p)
    ps.setup()
    fs, W = config.FS, config.WIDTH["full"]
    lat, lev = ds.lat.values, ds.lev.values
    mnorm = ps.sym_norm([early, late])
    dnorm = ps.sym_norm([diff])
    tnorm = ps.sym_norm([st.trend])
    clev = contour_levels(clim)
    left, right, gx = 0.62, 0.08, 0.2
    pw = (W - left - right - gx) / 2
    ph_ab, ph_cd = 1.6, 2.1
    top = 0.5
    xlab, cb_h, cb_lab = 0.22, 0.07, 0.45
    top2 = top + ph_ab + xlab + cb_h + cb_lab + 0.35
    H = top2 + ph_cd + xlab + cb_h + cb_lab + 0.45
    fig, at = ps.figure("full", H)
    panels = [(early, f"Early decade, {span(EARLY)}", mnorm, early, 0, top, ph_ab, False),
              (late, f"Late decade, {span(LATE)}", mnorm, late, 1, top, ph_ab, False),
              (diff, "Late minus early", dnorm, clim, 0, top2, ph_cd, True),
              (st.trend, f"Linear trend, {y0}-{y1}", tnorm, clim, 1, top2, ph_cd, True)]
    for k, (d, title, norm, cont, col, t0, ph, emph) in enumerate(panels):
        ax = at(left + col * (pw + gx), t0, pw, ph)
        ax.pcolormesh(lat, lev, d, cmap="RdBu_r", norm=norm, shading="nearest", rasterized=True)
        if np.isfinite(cont).any():
            ax.contour(lat, lev, cont, levels=clev, colors="k", linewidths=0.6,
                       negative_linestyles="dashed", zorder=3)
        if k == 3:
            ps.section_dots(ax, lat, lev, sig.values.astype(bool), np.isfinite(d.values))
        ax.plot(lat, ds.ps, color=vs.C_SP_LINE, lw=0.9, zorder=4)
        ax.set_ylim(1000, 100)
        ax.set_xlim(lat[0], lat[-1])
        ax.set_xticks(np.arange(np.ceil((lat[0] + 1) / 15) * 15, lat[-1] - 1, 15))
        ax.set_yticks([200, 400, 600, 800, 1000])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda v, _: f"{abs(v):.0f}°{'S' if v < 0 else 'N' if v > 0 else ''}"))
        ax.tick_params(labelleft=(col == 0))
        if col == 0:
            ax.set_ylabel("Pressure (hPa)")
        if emph:
            for sp in ax.spines.values():
                sp.set_linewidth(1.3)
        ax.set_title(title, fontsize=fs["title"] + (1 if emph else 0), fontweight="bold",
                     pad=3, loc="left")
        ps.letter(ax, k)
    cb1 = top + ph_ab + xlab + 0.04
    ps.colourbar(fig, at(left + 0.25 * (2 * pw + gx), cb1, 0.5 * (2 * pw + gx), cb_h), mnorm,
                 "RdBu_r", "ψ (10$^{10}$ kg s$^{-1}$), a-b", nticks=7)
    cb2 = top2 + ph_cd + xlab + 0.04
    ps.colourbar(fig, at(left + 0.1 * pw, cb2, 0.8 * pw, cb_h), dnorm, "RdBu_r",
                 "Δψ (10$^{10}$ kg s$^{-1}$)", nticks=5)
    ps.colourbar(fig, at(left + pw + gx + 0.1 * pw, cb2, 0.8 * pw, cb_h), tnorm, "RdBu_r",
                 "ψ trend (10$^{10}$ kg s$^{-1}$ decade$^{-1}$)", nticks=5)
    step = clev[1] - clev[0] if len(clev) > 1 else np.nan
    ps.text(fig, W / 2, 0.05, f"ERA5 {calendar.month_name[vs.MONTH]} meridional mass "
            f"streamfunction, Southern Africa ({REGION[0]:.0f}-{REGION[1]:.0f}°E)",
            ha="center", va="top", fontsize=fs["suptitle"], fontweight="bold")
    ps.text(fig, W / 2, H - 0.06, f"ψ = (2πa cos φ / g) ∫ v$_χ$ dp (local streamfunction, "
            "Schwendike et al. 2014; ψ < 0: anticlockwise, e.g. the southern Hadley cell)\n"
            f"Contours every {step:g} × 10$^{{10}}$ kg s$^{{-1}}$ (dashed < 0): a-b each decade, "
            f"c-d the {y0}-{y1} mean; d) dots: significant (FDR q = {config.ALPHA_FDR:.2f})",
            ha="center", va="bottom", fontsize=fs["small"] - 0.5, color="0.3")
    ps.save(fig, config.RESULTS / OUT / f"psi_section_{calendar.month_name[vs.MONTH].lower()}.png")


def up_to_date():
    if not data_file().exists():
        return False
    with xr.open_dataset(data_file()) as ds:
        return "band" not in ds.dims


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "compute" or (what == "all" and not up_to_date()):
        compute()
    if what in ("plot", "all"):
        plot()