"""
October: detrended correlation between omega at 300 hPa and cloud fraction at every level of
the column, in 5-degree longitude bands from 10 to 40E (latitude-pressure cross sections, land
cells), and the long-term changes where they co-vary. One figure per dataset: ERA5, MERRA-2,
JRA-3Q.

Run:  python omega_cf_correlation.py [compute | plot] [ERA5 MERRA-2 JRA-3Q]
      omega_cf_sensitivity.py repeats this with omega at other levels.

FIGURE  results/omega_cf/omega300_cf_<dataset>.png
  Row 1  detrended r(omega300, CF at each level); black dots where significant (FDR q = 0.10
         per panel)
  Row 2  CF decreasing; row 3 CF increasing. Colour = omega300 regime at that latitude
         (stats.regimes); colour strength = |r|; light grey = CF changes the other way.
  Each band's panels end at its first and last latitude with land (no white sea).
  Brown line: MERRA-2 climatological October surface pressure (levels below it are left out).

METHOD  omega and CF averaged over the land cells of each band (cells with a value in every
  year); r and p from the detrended series (stats.corr_and_p: regression slope t test with
  n_eff - 3 df, n_eff from the lag-1 autocorrelation of the residuals, capped at n).
DATA    omega: ERA5 global_vertical_velocity_YYYY.nc (w); MERRA-2 instM_3d_asm_Np OMEGA;
        JRA-3Q vvel-pres-an-ll125 (as omega_trends.py).
        CF: ERA5 global_fraction_of_cloud_cover_YYYY.nc (cc, pressure levels); MERRA-2
        tavgM_3d_rad_Np CLOUD (pressure levels); JRA-3Q has no pressure-level cloud cover in
        the data here, so its column is its three layer covers (JRA3Q_CF = "layers": high
        hcdc 90-500, middle mcdc 500-850, low lcdc 850-1100 hPa), each drawn over its pressure
        range; the statistics (and FDR) are done on the three layers. If a pressure-level
        JRA-3Q cloud cover folder exists, set JRA3Q_CF to its name.
        Native grids; config.YEARS.
DATA FILES  data/omega_cf/omega<level>_cf_<dataset>.nc: r, p, sig, cat, cf_trend (band, lev,
        lat), ps (band, lat)
"""
from __future__ import annotations

import calendar
import sys

import numpy as np
import xarray as xr

import config
import masks
import omega_trends as ot
import plotstyle as ps
import readers as rd
import stats

REGION = "omega_sfc"
CF_READ = "africa_10_40"          # JRA-3Q layer covers read as in cloud_fraction_trends.py
MONTH, OMEGA_LEV, PLEV_TOP = 10, 300.0, 100.0
OUT = "omega_cf"
NAMES = ["ERA5", "MERRA-2", "JRA-3Q"]
BANDS = [(10.0, 15.0), (15.0, 20.0), (20.0, 25.0), (25.0, 30.0), (30.0, 35.0), (35.0, 40.0)]
JRA3Q_CF = "layers"               # or the folder of a pressure-level JRA-3Q cloud cover
JRA3Q_LAYERS = [("hcdc-hcl-fc-ll125", 90.0, 500.0), ("mcdc-mcl-fc-ll125", 500.0, 850.0),
                ("lcdc-lcl-fc-ll125", 850.0, 1100.0)]
FINE = np.arange(100.0, 1000.1, 25.0)   # pressure axis the JRA-3Q layers are drawn on
C_SP_LINE = "#7B3F00"


def data_file(name, lev=OMEGA_LEV):
    return config.DATA / OUT / f"omega{lev:.0f}_cf_{name.replace('-', '')}.nc"


# ---------------------------------------------------------------- reading
def load_omega(name, lev):
    """(year, lat, lon) October omega at one level."""
    if name == "ERA5":
        w = rd.era5_yearly(config.RAW["era5"] / "global" / "vertical_velocity",
                           "global_vertical_velocity_*.nc", "w", REGION, lev=lev, months=[MONTH])
    elif name == "MERRA-2":
        w = rd.merra2("instM_3d_asm_Np", "OMEGA", REGION, lev=lev, months=[MONTH])
    elif ot.JRA3Q_SOURCE == "grib":
        w = rd.jra3q_grib(str(config.RAW["jra3q_grib"] / "anl_p125_vvel"), "anl_p125_vvel",
                          REGION, lev=lev, months=[MONTH], years=config.YEARS["JRA-3Q"])
    else:
        w = rd.jra3q(ot.JRA3Q_FOLDER, "vvel-pres-an-ll125", REGION, lev=lev, months=[MONTH])
    return w.sel(month=MONTH).drop_vars("lev", errors="ignore").squeeze(drop=True)


def load_cf(name):
    """CF (year, lev, lat, lon) in %, October; JRA-3Q layers: lev = layer index 0-2."""
    if name == "ERA5":
        d = config.RAW["era5"] / "global"
        cf = rd.era5_yearly(d / "fraction_of_cloud_cover", "global_fraction_of_cloud_cover_*.nc",
                            "cc", REGION, scale=100.0, months=[MONTH])
    elif name == "MERRA-2":
        cf = rd.merra2("tavgM_3d_rad_Np", "CLOUD", REGION, scale=100.0, months=[MONTH])
    elif JRA3Q_CF != "layers":
        cf = rd.jra3q(JRA3Q_CF, JRA3Q_CF, REGION, months=[MONTH])
    else:
        lay = [rd.cut(rd.jra3q(v, v, CF_READ, months=[MONTH]), REGION) for v, _, _ in JRA3Q_LAYERS]
        return xr.concat(lay, "lev").assign_coords(lev=np.arange(len(JRA3Q_LAYERS))).sel(
            month=MONTH)
    cf = cf.sel(month=MONTH)
    return cf.where(cf.lev >= PLEV_TOP, drop=True).sortby("lev")


def layered(name):
    return name == "JRA-3Q" and JRA3Q_CF == "layers"


def to_fine(ds):
    """Layer statistics (band, lev=layer, lat) -> drawn on FINE pressure levels."""
    idx = np.full(FINE.size, -1)
    for k, (_, top, bot) in enumerate(JRA3Q_LAYERS):
        idx[(FINE >= top) & (FINE < bot)] = k
    idx[FINE >= JRA3Q_LAYERS[-1][1]] = len(JRA3Q_LAYERS) - 1
    out = ds.isel(lev=xr.DataArray(idx, dims="lev")).assign_coords(lev=FINE)
    return out


# ---------------------------------------------------------------- compute
def compute_one(name, lev, psc):
    w = load_omega(name, lev)
    cf = load_cf(name)
    w = stats.select_years(w.where(masks.land_mask(w)).rename(name), config.YEARS[name])
    cf = cf.where(masks.land_mask(cf))
    if not layered(name):
        cf = cf.where(rd.above_ground_on(cf, psc))
    cf = stats.select_years(cf.rename(name), config.YEARS[name])
    ps_ = psc.interp(lat=cf.lat, lon=cf.lon, kwargs={"fill_value": None})
    out = []
    for lo, hi in BANDS:
        wb = stats.complete(w.sel(lon=slice(lo, hi - 1e-6))).mean("lon")       # (year, lat)
        cfb = stats.complete(cf.sel(lon=slice(lo, hi - 1e-6))).mean("lon")     # (year, lev, lat)
        cfb = cfb.interp(lat=wb.lat) if not np.array_equal(cfb.lat, wb.lat) else cfb
        r, p = stats.corr_and_p(stats.detrend(wb), stats.detrend(cfb))
        sig = stats.fdr(p)
        reg = stats.regimes(wb, stats.ols_stats(wb).trend)
        cft = stats.ols_stats(cfb).trend
        cat = xr.where((reg >= 0) & cft.notnull(), 2 * reg + xr.where(cft > 0, 0, 1), -1)
        out.append(xr.Dataset({"r": r, "p": p, "sig": sig.astype("int8"),
                               "cat": cat.astype("int8"), "cf_trend": cft,
                               "ps": ps_.sel(lon=slice(lo, hi - 1e-6)).mean("lon").interp(
                                   lat=wb.lat)}))
        rd.log(f"  omega{lev:.0f}, {lo:.0f}-{hi:.0f}E: {stats.pct_area(sig, r.notnull()):.0f} % "
               "of the section significant")
    ds = xr.concat(out, "band").assign_coords(band=[f"{a:.0f}-{b:.0f}E" for a, b in BANDS])
    ds = ds.transpose("band", "lev", "lat")
    if layered(name):
        ds = to_fine(ds)
    ds.attrs = {"dataset": name, "month": MONTH, "omega_level_hPa": float(lev),
                "cf": "three layer covers" if layered(name) else "pressure levels",
                "years": f"{config.YEARS[name][0]}-{config.YEARS[name][1]}"}
    for v in ("r", "p", "cf_trend", "ps"):
        ds[v] = ds[v].astype("float32")
    ds.to_netcdf(data_file(name, lev))
    return ds


def compute(names=NAMES, levels=(OMEGA_LEV,)):
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    psc = rd.surface_pressure_clim(REGION, months=[MONTH]).sel(month=MONTH)
    for name in names:
        rd.log(f"--- {name} ---")
        for lev in levels:
            compute_one(name, lev, psc)


# ---------------------------------------------------------------- plot
def lat_range(o):
    """First and last latitude with data in this band (any level, any row)."""
    ok = np.isfinite(o.r.values).any(0)
    lat = o.lat.values
    if not ok.any():
        return lat.min(), lat.max()
    d = float(np.median(np.abs(np.diff(lat)))) / 2
    return lat[ok].min() - d, lat[ok].max() + d


def lat_ticks(lo, hi):
    step = 20 if hi - lo > 25 else 10 if hi - lo > 12 else 5
    return np.arange(np.ceil(lo / step) * step, hi + 1e-6, step)


def section(ax, o, var, norm=None, strength=None, cols=None, show=None, valid=None,
            omega_lev=OMEGA_LEV, cmap="RdBu_r"):
    """Draw one latitude-pressure panel; returns nothing."""
    import matplotlib.pyplot as plt
    lat, lev = o.lat.values, o.lev.values
    if var == "r":
        ax.pcolormesh(lat, lev, o.r, cmap=cmap, norm=norm, shading="nearest", rasterized=True)
    else:
        code, other = var
        ps.categories(ax, lat, lev, code, strength, cols, other=other)
    if show is not None:
        ps.section_dots(ax, lat, lev, show, valid)
    ax.plot(lat, o.ps, color=C_SP_LINE, lw=0.8, zorder=4)
    ax.plot([0, 0.06], [omega_lev] * 2, color="k", lw=2, transform=ax.get_yaxis_transform(),
            solid_capstyle="butt", zorder=5)
    lo, hi = lat_range(o)
    ax.set_xlim(lo, hi)
    ax.set_ylim(1000, PLEV_TOP)
    ax.set_yticks([200, 400, 600, 800, 1000])
    ax.set_xticks(lat_ticks(lo, hi))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{abs(v):.0f}°{'S' if v < 0 else 'N' if v > 0 else ''}"))


def plot(names=NAMES, lev=OMEGA_LEV):
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt
    ps.setup()
    fs, W = config.FS, config.WIDTH["full"]
    cols = [config.REGIME_COLOURS[k] for k in config.REGIMES]
    wl = f"ω$_{{{lev:.0f}}}$"
    for name in names:
        if not data_file(name, lev).exists():
            continue
        ds = xr.load_dataset(data_file(name, lev))
        nb = ds.sizes["band"]
        left, right, gx, gy = 0.75, 0.08, 0.1, 0.42
        pw = (W - left - right - (nb - 1) * gx) / nb
        ph = 1.2
        top = 0.62
        tops = [top + i * (ph + gy) for i in range(3)]
        cb_top = tops[2] + ph + 0.45
        H = cb_top + 1.35 + (0.2 if layered(name) else 0)
        fig, at = ps.figure("full", H)
        rnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        rows = ["Detrended r", "CF decreasing", "CF increasing"]
        k = 0
        for ri, rlab in enumerate(rows):
            ps.text(fig, 0.1, tops[ri] + ph / 2, rlab, rotation=90, ha="center", va="center",
                    fontsize=fs["label"], fontweight="bold")
            for bi in range(nb):
                o = ds.isel(band=bi)
                ax = at(left + bi * (pw + gx), tops[ri], pw, ph)
                sig = o.sig.astype(bool).values
                if ri == 0:
                    section(ax, o, "r", norm=rnorm, show=sig, valid=np.isfinite(o.r.values),
                            omega_lev=lev)
                    ax.set_title(str(o.band.values), fontsize=fs["title"], fontweight="bold",
                                 pad=3)
                else:
                    cat = o.cat.values
                    want = (cat >= 0) & (cat % 2 == (1 if ri == 1 else 0))
                    section(ax, o, (np.where(want, cat // 2, -1), (cat >= 0) & ~want),
                            strength=np.clip(np.abs(np.nan_to_num(o.r.values)), 0, 1), cols=cols,
                            show=sig & want, valid=want, omega_lev=lev)
                ax.tick_params(labelleft=(bi == 0), labelbottom=True, length=2, pad=1.5)
                if bi == 0:
                    ax.set_ylabel("Pressure (hPa)", fontsize=fs["small"])
                ps.letter(ax, k)
                k += 1
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=rnorm, cmap="RdBu_r"),
                          cax=at(left, cb_top, 2.0, 0.08), orientation="horizontal",
                          ticks=[-1, -0.5, 0, 0.5, 1])
        cb.set_label(f"Detrended r({wl}, CF)", fontsize=fs["label"])
        cb.ax.tick_params(labelsize=fs["tick"], length=2)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1),
                                                cmap=ps.shade_cmap("0.2")),
                          cax=at(left + 2.6, cb_top, 2.0, 0.08), orientation="horizontal")
        cb.set_label("Colour strength = |r| (rows 2-3)", fontsize=fs["label"])
        cb.ax.tick_params(labelsize=fs["tick"], length=2)
        h = ps.regime_handles() + [
            Patch(facecolor=config.C_OTHER, label="CF changes the other way"),
            Line2D([], [], color=C_SP_LINE, lw=1, label="Surface pressure"),
            ps.dot_handle(f"r significant (FDR q = {config.ALPHA_FDR:.2f})")]
        fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, handlelength=1.2,
                   columnspacing=1.0, bbox_to_anchor=(0.5, 1 - (cb_top + 0.5) / H),
                   title=f"{wl} long-term change (rows 2-3; black mark = {lev:.0f} hPa)",
                   title_fontsize=fs["legend"])
        if layered(name):
            ps.text(fig, W / 2, H - 0.08, "JRA-3Q cloud: its high (90-500 hPa), middle (500-850) "
                    "and low (850-1100) cloud covers, each drawn over its pressure range",
                    ha="center", fontsize=fs["small"], color="0.3")
        ps.text(fig, W / 2, 0.06, f"{ps.label(name)} {calendar.month_name[MONTH]} "
                f"{ds.attrs['years']}: detrended r({wl}, cloud fraction through the column), land",
                ha="center", va="top", fontsize=fs["suptitle"], fontweight="bold")
        ps.save(fig, config.RESULTS / OUT / f"omega{lev:.0f}_cf_{name.replace('-', '')}.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    what = next((a for a in args if a in ("compute", "plot")), "all")
    names = [a for a in args if a in NAMES] or NAMES
    if what == "compute" or (what == "all" and not all(data_file(n).exists() for n in names)):
        compute(names)
    if what in ("plot", "all"):
        plot(names)
