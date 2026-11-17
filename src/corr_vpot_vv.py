"""
October: detrended correlation between the 200 hPa velocity potential (chi200) and omega at
every level of the column, in 5-degree longitude bands from 10 to 40E (latitude-pressure cross
sections, land cells), and how omega has changed where chi200 rose or fell. One figure per
dataset: ERA5, MERRA-2, JRA-3Q. Laid out like omega_cf_correlation.py. Also maps of the same
correlation at every cell, one omega level at a time (laid out like omega_surface_correlation.py).

Run:  python chi_omega_correlation.py [compute | plot] [sections | maps] [ERA5 MERRA-2 JRA-3Q]
                                       [--lev 300,500]
      sections and maps both by default; maps for every omega level unless --lev.

FIGURE 1 (sections)  results/chi_omega/chi200_omega_<dataset>.png
  Row 1  detrended r(chi200, omega at each level); black dots where significant (FDR q = 0.10
         per panel)
  Row 2  latitudes where chi200 increased over the period; row 3 where it decreased. Colour =
         the omega regime at that level (stats.regimes: strengthening, weakening or new
         subsidence or ascent; the colours of the omega-CF figures); colour strength = |r|, so
         strong colours are regimes where omega also co-varies with chi200 from year to year;
         light grey = chi200 changed the other way at that latitude.
  Each band's panels end at its first and last latitude with land.
  Brown line: MERRA-2 climatological October surface pressure (levels below it are left out).

FIGURE 2 (maps)  results/chi_omega/maps/chi200_omega<level>.png, one per omega level, over
  10-40E, 35S-10N (land; the Southern Africa box dashed). Rows = datasets; columns:
  1 chi200 trend (one colour scale for all datasets; dots: significant, FDR q = 0.10)
  2 detrended r(chi200, omega at that level) at every cell (dots: significant)
  3 chi200 decreasing, 4 chi200 increasing: colour = the omega regime at that level and cell
    (stats.regimes), strength = |r|; light grey = chi200 changed the other way.

SIGN  With the divergent wind V = grad(chi) (the windspharm and JRA-3Q convention), upper-level
  divergence sits at chi200 minima: a falling chi200 means stronger divergence aloft, which
  goes with more ascent below (negative omega), so r > 0 is the expected sign.

METHOD  chi200 and omega averaged over the land cells of each band (cells with a value in
  every year; chi200 is first interpolated to the omega grid); r and p from the detrended
  series (stats.corr_and_p: regression slope t test with n_eff - 3 df, n_eff from the lag-1
  autocorrelation of the residuals, capped at n); trends stats.ols_stats (per decade).
DATA    chi200: velocity_potential.load_reanalysis (ERA5 chi file, JRA-3Q vpot, MERRA-2 from
        U and V); omega: omega_trends.load (ERA5 w, MERRA-2 OMEGA, JRA-3Q vvel), the same
        caches. Native grids; config.YEARS.
DATA FILES  data/chi_omega/chi200_omega_<dataset>.nc: r, p, sig, omega_trend, regime
        (band, lev, lat; regime -1 where there is no value),
        chi_trend, chi_p, ps (band, lat)
        data/chi_omega/maps/chi200_omega<level>_<dataset>.nc: r, p, sig, cat, chi_trend,
        chi_sig (lat, lon); cat = 2 x regime + (0 chi200 rising, 1 falling), -1 none
"""
from __future__ import annotations

import calendar
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import config
import masks
import omega_cf_corr as occ
import plotstyle as ps
import readers as rd
import stats

REGION = "omega_sfc"
MONTH, CHI_LEV = 10, 200.0
OUT = "chi_omega"
NAMES = ["ERA5", "MERRA-2", "JRA-3Q"]
BANDS = occ.BANDS
CHI_SCALE = 1e-6                          # m2 s-1 -> 10^6 m2 s-1 (logs, map colour bar)
BOX = "southern_africa"                   # dashed on the maps


def data_file(name):
    return config.DATA / OUT / f"chi{CHI_LEV:.0f}_omega_{name.replace('-', '')}.nc"


def map_file(name, lev):
    return config.DATA / OUT / "maps" / f"chi{CHI_LEV:.0f}_omega{lev:.0f}_{name.replace('-', '')}.nc"


def omega_levels():
    import omega_trends as ot
    return [float(l) for l in ot.LEVELS]


NEEDED = ("r", "sig", "regime", "chi_trend", "ps")     # variables the current plot uses


def up_to_date(name):
    """True if the data file exists and has every variable the plot needs (files written by an
    older version of this script are recomputed)."""
    f = data_file(name)
    if not f.exists():
        return False
    with xr.open_dataset(f) as ds:
        return all(v in ds for v in NEEDED)


# ---------------------------------------------------------------- reading
def load_chi(name):
    """(year, lat, lon) October chi200 in m2 s-1 over REGION (padded)."""
    import vpot as vp
    if vp.LEVEL != CHI_LEV:
        raise ValueError(f"velocity_potential.LEVEL is {vp.LEVEL}, not {CHI_LEV}")
    return rd.cut(vp.load_reanalysis(name), REGION, pad=2.0)


def load_omega(name):
    """(year, lev, lat, lon) October omega in Pa s-1 over REGION."""
    import omega_trends as ot
    return rd.cut(ot.load(name), REGION).sortby("lev")


# ---------------------------------------------------------------- compute
def prepare(name, psc):
    """omega (year, lev, lat, lon) and chi200 (year, lat, lon) on omega's grid: land, above the
    ground, the dataset's years."""
    w = load_omega(name)
    chi = load_chi(name)
    chi = chi.interp(lat=w.lat, lon=w.lon)                     # chi is smooth: to omega's grid
    land = masks.land_mask(w.isel(lev=0, drop=True))
    yrs = config.YEARS[name]
    w = w.where(land).where(rd.above_ground_on(w, psc))
    w = stats.select_years(w.rename(name), yrs)
    chi = stats.select_years(chi.where(land).rename(f"{name} chi"), yrs)
    return w, chi, yrs


def compute_sections(name, w, chi, yrs, psc):
    ps_ = psc.interp(lat=w.lat, lon=w.lon, kwargs={"fill_value": None})
    out = []
    for lo, hi in BANDS:
        band = dict(lon=slice(lo, hi - 1e-6))
        cb = stats.complete(chi.sel(**band)).mean("lon")                  # (year, lat)
        wb = stats.complete(w.sel(**band)).mean("lon")                    # (year, lev, lat)
        r, p = stats.corr_and_p(stats.detrend(cb), stats.detrend(wb))
        sig = stats.fdr(p)
        cst, wst = stats.ols_stats(cb), stats.ols_stats(wb)
        reg = stats.regimes(wb, wst.trend).where(r.notnull(), -1)
        out.append(xr.Dataset({"r": r, "p": p, "sig": sig.astype("int8"),
                               "omega_trend": wst.trend.where(r.notnull()),
                               "regime": reg.astype("int8"),
                               "chi_trend": cst.trend, "chi_p": cst.p,
                               "ps": ps_.sel(**band).mean("lon")}))
        up = float((cst.trend > 0).sum()), float(cst.trend.notnull().sum())
        rd.log(f"  sections {lo:.0f}-{hi:.0f}E: {stats.pct_area(sig, r.notnull()):.0f} % "
               f"significant; median r {float(r.median()):+.2f}; chi200 rising at "
               f"{up[0]:.0f}/{up[1]:.0f} latitudes")
    ds = xr.concat(out, "band").assign_coords(band=[f"{a:.0f}-{b:.0f}E" for a, b in BANDS])
    ds = ds.transpose("band", "lev", "lat")
    ds.attrs = {"dataset": name, "month": MONTH, "chi_level_hPa": CHI_LEV,
                "years": f"{yrs[0]}-{yrs[1]}",
                "units": "omega_trend Pa s-1 per decade; chi_trend m2 s-1 per decade"}
    for v in ("r", "p", "omega_trend", "chi_trend", "chi_p", "ps"):
        ds[v] = ds[v].astype("float32")
    ds.to_netcdf(data_file(name))
    rd.log(f"  saved {data_file(name)}")


def compute_maps(name, w, chi, yrs, levels):
    """Every cell: detrended r(chi200, omega at one level), and the omega regime split by the
    sign of the chi200 trend."""
    (config.DATA / OUT / "maps").mkdir(parents=True, exist_ok=True)
    chi = stats.complete(chi)
    cst = stats.ols_stats(chi)
    chi_sig = stats.fdr(cst.p)
    cd = stats.detrend(chi)
    for lev in levels:
        if lev not in w.lev.values:
            rd.log(f"  [WARN] maps: no omega at {lev:.0f} hPa")
            continue
        wl = stats.complete(w.sel(lev=lev, drop=True))
        r, p = stats.corr_and_p(cd, stats.detrend(wl))
        sig = stats.fdr(p)
        reg = stats.regimes(wl, stats.ols_stats(wl).trend)
        cat = xr.where((reg >= 0) & r.notnull() & cst.trend.notnull(),
                       2 * reg + xr.where(cst.trend > 0, 0, 1), -1)
        ds = xr.Dataset({"r": r.astype("float32"), "p": p.astype("float32"),
                         "sig": sig.astype("int8"), "cat": cat.astype("int8"),
                         "chi_trend": cst.trend.astype("float32"),
                         "chi_sig": chi_sig.astype("int8")})
        ds.attrs = {"dataset": name, "month": MONTH, "chi_level_hPa": CHI_LEV,
                    "omega_level_hPa": lev, "years": f"{yrs[0]}-{yrs[1]}",
                    "units": "chi_trend m2 s-1 per decade"}
        ds.to_netcdf(map_file(name, lev))
        rd.log(f"  maps omega{lev:.0f}: {stats.pct_area(sig, r.notnull()):.0f} % of land with r "
               f"significant; area-weighted median r {stats.wmedian(r):+.2f}")


def compute(names=NAMES, kinds=("sections", "maps"), levels=None):
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    psc = rd.surface_pressure_clim(REGION, months=[MONTH]).sel(month=MONTH)
    levels = levels or omega_levels()
    for name in names:
        rd.log(f"--- {name} ---")
        try:
            w, chi, yrs = prepare(name, psc)
            if "sections" in kinds:
                compute_sections(name, w, chi, yrs, psc)
            if "maps" in kinds:
                compute_maps(name, w, chi, yrs, levels)
        except Exception as e:
            rd.log(f"  [WARN] {name} left out: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- plot
def plot(names=NAMES):
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt
    ps.setup()
    fs, W = config.FS, config.WIDTH["full"]
    chi_l = f"χ$_{{{CHI_LEV:.0f}}}$"
    for name in names:
        if not up_to_date(name):
            print(f"{name}: data file missing or from an older version of this script: run "
                  f"'python {Path(__file__).name} compute {name}' first")
            continue
        ds = xr.load_dataset(data_file(name))
        nb = ds.sizes["band"]
        left, right, gx, gy = 0.75, 0.08, 0.1, 0.42
        pw = (W - left - right - (nb - 1) * gx) / nb
        ph = 1.2
        top = 0.62
        tops = [top + i * (ph + gy) for i in range(3)]
        cb_top = tops[2] + ph + 0.45
        H = cb_top + 1.35
        fig, at = ps.figure("full", H)
        rnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        cols = [config.REGIME_COLOURS[k] for k in config.REGIMES]
        rows = ["Detrended r", f"{chi_l} increasing", f"{chi_l} decreasing"]
        k = 0
        for ri, rlab in enumerate(rows):
            ps.text(fig, 0.1, tops[ri] + ph / 2, rlab, rotation=90, ha="center", va="center",
                    fontsize=fs["label"], fontweight="bold")
            for bi in range(nb):
                o = ds.isel(band=bi)
                ax = at(left + bi * (pw + gx), tops[ri], pw, ph)
                sig = o.sig.astype(bool).values
                valid = np.isfinite(o.r.values)
                if ri == 0:
                    occ.section(ax, o, "r", norm=rnorm, show=sig, valid=valid, omega_lev=CHI_LEV)
                    ax.set_title(str(o.band.values), fontsize=fs["title"], fontweight="bold",
                                 pad=3)
                else:
                    ct = o.chi_trend.values
                    rising = np.broadcast_to((ct > 0)[None, :], valid.shape)
                    falling = np.broadcast_to((ct < 0)[None, :], valid.shape)
                    want = valid & (rising if ri == 1 else falling)
                    other = valid & (falling if ri == 1 else rising)
                    reg = o.regime.values
                    occ.section(ax, o, (np.where(want & (reg >= 0), reg, -1), other),
                                strength=np.clip(np.abs(np.nan_to_num(o.r.values)), 0, 1),
                                cols=cols, show=sig & want, valid=want, omega_lev=CHI_LEV)
                ax.tick_params(labelleft=(bi == 0), labelbottom=True, length=2, pad=1.5)
                if bi == 0:
                    ax.set_ylabel("Pressure (hPa)", fontsize=fs["small"])
                ps.letter(ax, k)
                k += 1
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=rnorm, cmap="RdBu_r"),
                          cax=at(left, cb_top, 2.0, 0.08), orientation="horizontal",
                          ticks=[-1, -0.5, 0, 0.5, 1])
        cb.set_label(f"Detrended r({chi_l}, ω) (row 1)", fontsize=fs["label"])
        cb.ax.tick_params(labelsize=fs["tick"], length=2)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1),
                                                cmap=ps.shade_cmap("0.2")),
                          cax=at(left + 2.6, cb_top, 2.0, 0.08), orientation="horizontal")
        cb.set_label("Colour strength = |r| (rows 2-3)", fontsize=fs["label"])
        cb.ax.tick_params(labelsize=fs["tick"], length=2)
        h = ps.regime_handles() + [
             Patch(facecolor=config.C_OTHER, label=f"{chi_l} changed the other way"),
             Line2D([], [], color=occ.C_SP_LINE, lw=1, label="Surface pressure"),
             ps.dot_handle(f"r significant (FDR q = {config.ALPHA_FDR:.2f})")]
        fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, handlelength=1.2,
                   columnspacing=1.0, bbox_to_anchor=(0.5, 1 - (cb_top + 0.5) / H),
                   title=f"ω long-term change at each level (rows 2-3); black mark on the axis: "
                         f"{CHI_LEV:.0f} hPa",
                   title_fontsize=fs["legend"])
        ps.text(fig, W / 2, 0.06, f"{ps.label(name)} {calendar.month_name[MONTH]} "
                f"{ds.attrs['years']}: detrended r({chi_l}, ω through the column), land",
                ha="center", va="top", fontsize=fs["suptitle"], fontweight="bold")
        ps.save(fig, config.RESULTS / OUT / f"chi{CHI_LEV:.0f}_omega_{name.replace('-', '')}.png")


def plot_maps(names=NAMES, levels=None):
    """One figure per omega level: rows = datasets; columns chi200 trend, r, chi200 decreasing,
    chi200 increasing (as omega_surface_correlation.py)."""
    import cartopy.crs as ccrs
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt
    ps.setup()
    fs, W = config.FS, config.WIDTH["full"]
    cols = [config.REGIME_COLOURS[k] for k in config.REGIMES]
    reg = config.REGIONS[REGION]
    left, right, gx, gy = 0.85, 0.08, 0.08, 0.22
    pw = (W - left - right - 3 * gx) / 4
    ph = pw * (reg["lat"][1] - reg["lat"][0]) / (reg["lon"][1] - reg["lon"][0])
    top = 0.62
    chi_l = f"χ$_{{{CHI_LEV:.0f}}}$"
    for lev in levels or omega_levels():
        res = {n: xr.load_dataset(map_file(n, lev)) for n in names if map_file(n, lev).exists()}
        if not res:
            continue
        wl = f"ω$_{{{lev:.0f}}}$"
        names_ = list(res)
        tops = [top + i * (ph + gy) for i in range(len(names_))]
        cb_top = tops[-1] + ph + 0.4
        H = cb_top + 1.45
        fig, at = ps.figure("full", H)
        tnorm = ps.sym_norm([d.chi_trend * CHI_SCALE for d in res.values()])
        rnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        heads = [f"{chi_l} trend", f"Detrended r({chi_l}, {wl})", f"{chi_l} decreasing",
                 f"{chi_l} increasing"]
        k = 0
        for ri, name in enumerate(names_):
            d = res[name]
            ps.text(fig, 0.12, tops[ri] + ph / 2, f"{ps.label(name)}\n{d.attrs['years']}",
                    rotation=90, ha="center", va="center", fontsize=fs["label"],
                    fontweight="bold", color=ps.colour(name))
            cat = d.cat.values
            for ci in range(4):
                x0 = left + ci * (pw + gx)
                if ri == 0:
                    ps.text(fig, x0 + pw / 2, top - 0.05, heads[ci], ha="center",
                            fontsize=fs["small"], fontweight="bold")
                ax = at(x0, tops[ri], pw, ph, projection=ccrs.PlateCarree())
                kw = dict(region=REGION, left=(ci == 0), bottom=(ri == len(names_) - 1),
                          xticks=[15, 30], box=BOX)
                if ci == 0:
                    ps.map_panel(ax, d.chi_trend * CHI_SCALE, tnorm,
                                 sig=d.chi_sig.astype(bool).values, **kw)
                elif ci == 1:
                    ps.map_panel(ax, d.r, rnorm, sig=d.sig.astype(bool).values, **kw)
                else:
                    want = (cat >= 0) & (cat % 2 == (1 if ci == 2 else 0))
                    ps.categories(ax, d.lon, d.lat, np.where(want, cat // 2, -1),
                                  np.clip(np.abs(np.nan_to_num(d.r.values)), 0, 1), cols,
                                  other=(cat >= 0) & ~want, transform=ccrs.PlateCarree())
                    ps.sig_dots(ax, d.r, d.sig.astype(bool).values & want)
                    ps.map_frame(ax, **kw)
                ps.letter(ax, k)
                k += 1
        ps.colourbar(fig, at(left, cb_top, 1.75, 0.08), tnorm, "RdBu_r",
                     f"{chi_l} trend (10$^6$ m$^2$ s$^{{-1}}$ decade$^{{-1}}$)", nticks=5)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=rnorm, cmap="RdBu_r"),
                          cax=at(left + 2.2, cb_top, 1.75, 0.08), orientation="horizontal",
                          ticks=[-1, -0.5, 0, 0.5, 1])
        cb.set_label("r (colour strength in columns 3-4 = |r|)", fontsize=fs["label"])
        cb.ax.tick_params(labelsize=fs["tick"], length=2)
        h = ps.regime_handles() + [
            Patch(facecolor=config.C_OTHER, label=f"{chi_l} changed the other way"),
            ps.dot_handle(f"Significant (FDR q = {config.ALPHA_FDR:.2f})")]
        fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, handlelength=1.2,
                   columnspacing=1.0, bbox_to_anchor=(0.5, 1 - (cb_top + 0.55) / H),
                   title=f"{wl} long-term change (columns 3-4); dashed: Southern Africa "
                         "(10-40°E, 35-10°S)", title_fontsize=fs["legend"])
        ps.text(fig, W / 2, 0.06, f"{calendar.month_name[MONTH]}: detrended correlation of "
                f"{chi_l} with ω at {lev:.0f} hPa, land", ha="center", va="top",
                fontsize=fs["suptitle"], fontweight="bold")
        ps.save(fig, config.RESULTS / OUT / "maps" / f"chi{CHI_LEV:.0f}_omega{lev:.0f}.png")


def parse_levels(args):
    for i, a in enumerate(args):
        if a == "--lev" and i + 1 < len(args):
            return [float(x) for x in args[i + 1].split(",")]
        if a.startswith("--lev="):
            return [float(x) for x in a.split("=", 1)[1].split(",")]
    return None


if __name__ == "__main__":
    args = sys.argv[1:]
    what = next((a for a in args if a in ("compute", "plot")), "all")
    kinds = [a for a in args if a in ("sections", "maps")] or ["sections", "maps"]
    names = [a for a in args if a in NAMES] or NAMES
    levels = parse_levels(args) or omega_levels()
    if what == "compute":
        compute(names, kinds, levels)
    else:
        need = [n for n in names
                if ("sections" in kinds and not up_to_date(n))
                or ("maps" in kinds and not all(map_file(n, l).exists() for l in levels))]
        if need:
            rd.log(f"computing (missing or older data files): {', '.join(need)}")
            compute(need, kinds, levels)
    if what in ("plot", "all"):
        if "sections" in kinds:
            plot(names)
        if "maps" in kinds:
            plot_maps(names, levels)