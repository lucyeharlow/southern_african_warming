"""
Quick look: the October divergent circulation over sub-Saharan Africa (10-40E mean, 35S-18N),
climatology and linear trend, drawn two ways. ERA5 only.

Run:  python vchi_section.py [compute | plot]
      compute reads every October of config.YEARS['ERA5'] once and keeps one small file per
      year (data/vchi/years/), so an interrupted run carries on where it stopped.

FIGURE  results/vchi/vchi_section_october.png (columns: climatology, linear trend per decade)
  Row 1, latitude-pressure, vertical and horizontal motion kept apart (no arrows):
    a, b  shading omega (red: descent); contours v_chi (solid northward, dashed southward)
  Row 2, latitude-height in physical units:
    c, d  shading w = -omega / (rho g) (red: ascent); arrows (v_chi, w), both in m s-1, drawn
          in a frame of distance from the equator (y = a phi, km) and height (km), so an
          arrow points along the air's actual path in the drawn frame; the vertical axis is
          stretched (the factor is given), which is why the paths look steep.
  b, d dots: omega trend significant (FDR q = 0.10; stats.ols_stats, lag-1 n_eff); w's trend
  is omega's divided by the climatological rho g, so the same cells.
  Brown: ground (MERRA-2 climatological October surface pressure; row 2 as height).

METHOD
  1 Each October: u and v (ERA5 monthly GRIB, LEVELS) on a 1-degree pole-to-pole grid; chi per
    level (velocity_potential.velocity_potential: skyborn windspharm, divergent wind =
    grad(chi)); v_chi = (1/a) d(chi)/d(phi), checked against the solver's own irrotational
    wind (logged).
  2 v_chi and omega averaged over 10-40E (land and sea; LAND_ONLY to change), leaving out
    levels below the ground.
  3 rho = p / (R_d T) and heights (hypsometric equation, from 1000 hPa up) from the October
    climatological temperature of the same 10-40E mean (GRIB archive 'temperature' folder if
    present; otherwise the standard atmosphere, logged). w = -omega / (rho g) for each year.
  4 Climatology = mean of all the years; trend = OLS per decade.
DATA  u, v (and T, omega if present): GRIB_DIR/<variable>/grb/era5_monthly_<variable>_YYYYMM.grb
      omega otherwise from the repository's ERA5 global_vertical_velocity_YYYY.nc.
DATA FILES  data/vchi/years/vchi_<year>.nc (v_chi on the grid, 10-40E)
            data/vchi/vchi_region_october.nc (vchi, omega, w: year, lev, lat; t, rho, z: lev,
            lat; ps, zs: lat)
"""
from __future__ import annotations

import calendar
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import config
import masks
import plotstyle as ps
import readers as rd
import stats

GRIB_DIR = Path("/soge-home/data/analysis/era5/0.28125x0.28125/monthly")
MONTH = 10
NAME = "ERA5"
LEVELS = [100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 850.0, 925.0, 1000.0]
SECTION = {"lon": (10.0, 40.0), "lat": (-40.0, 20.0)}      # read (the year files)
PLOT_LAT = (-35.0, 18.0)                                    # sub-Saharan Africa, as the maps
LAND_ONLY = False
GRID = 1.0                                # degrees, pole to pole, for chi
A = 6.371229e6                            # Earth radius (m), as velocity_potential.py
G = 9.80665
RD = 287.05                               # dry-air gas constant (J kg-1 K-1)
T_SAMPLE = 46                             # Octobers of temperature read for the climatology
ARROW_STEP = 4.0                          # degrees of latitude between arrows (row 2)
ARROW_LEVELS = [200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 850.0]
REF_IN = 0.2                              # length (inches) of a reference-size arrow
OUT = "vchi"
C_SP_LINE = "#7B3F00"


def years():
    return config.YEARS[NAME]


def year_file(y):
    return config.DATA / OUT / "years" / f"vchi_{y}.nc"


def region_file():
    return config.DATA / OUT / f"vchi_region_{calendar.month_name[MONTH].lower()}.nc"


# ---------------------------------------------------------------- reading
def grib_path(var, year):
    return GRIB_DIR / var / "grb" / f"era5_monthly_{var}_{year}{MONTH:02d}.grb"


def read_grib(var, year, region=None):
    """(lev, lat, lon) one October field from the ERA5 monthly GRIB archive."""
    with xr.open_dataset(grib_path(var, year), engine="cfgrib", backend_kwargs={
            "indexpath": "", "filter_by_keys": {"typeOfLevel": "isobaricInhPa"}}) as ds:
        da = rd.standardise(ds[list(ds.data_vars)[0]])
        da = rd.pick_levels(da, LEVELS)
        if region is not None:
            da = rd.cut(da, region, pad=1.0)
        da = da.load()
    return da.drop_vars([c for c in da.coords if c not in ("lat", "lon", "lev")])


def target_grid():
    return (np.arange(-90.0, 90.0 + 1e-6, GRID), np.arange(-180.0, 180.0 - 1e-6, GRID))


def read_uv(year):
    """(lev, lat, lon) October u and v on the 1-degree global grid."""
    lat, lon = target_grid()
    uv = xr.Dataset({k: read_grib(f"{k}_component_of_wind", year) for k in ("u", "v")})
    return uv.interp(lat=lat, lon=lon, kwargs={"fill_value": None})


def yearly_omega():
    """(year, lev, lat, lon) October omega (Pa s-1) over the section's region."""
    y0, y1 = years()
    if (GRIB_DIR / "vertical_velocity" / "grb").exists():
        got = [y for y in range(y0, y1 + 1) if grib_path("vertical_velocity", y).exists()]
        if got:
            rd.log(f"omega from the GRIB archive ({len(got)} Octobers)")
            return xr.concat([read_grib("vertical_velocity", y, SECTION) for y in got],
                             "year").assign_coords(year=got)
    w = rd.era5_yearly(config.RAW["era5"] / "global" / "vertical_velocity",
                       "global_vertical_velocity_*.nc", "w", SECTION, pad=1.0, lev=LEVELS,
                       months=[MONTH]).sel(month=MONTH)
    rd.log("omega from global_vertical_velocity_YYYY.nc")
    return w.sel(year=slice(y0, y1))


# ---------------------------------------------------------------- chi and v_chi
def vchi_from_uv(uv, check=False):
    """(lev, lat, lon) v_chi = (1/a) d(chi)/d(phi); with check, compare with the solver's own
    irrotational wind."""
    import velocity_potential as vp
    phi = np.deg2rad(uv.lat.values)
    out, ref = [], []
    for lev in uv.lev.values:
        u, v = uv.u.sel(lev=lev), uv.v.sel(lev=lev)
        chi = vp.velocity_potential(u, v)                          # lat ascending, like u
        out.append(chi.copy(data=np.gradient(chi.values.astype(float), phi, axis=0) / A))
        if check:
            from skyborn.windspharm.standard import VectorWind
            n2s = lambda a: np.asarray(a, float)[::-1]
            ref.append(VectorWind(n2s(u), n2s(v), gridtype="regular").irrotationalcomponent()[1][::-1])
    vchi = xr.concat(out, "lev").assign_coords(lev=uv.lev.values)
    if check:
        inner = np.abs(uv.lat.values) < 80
        d = vchi.values[:, inner] - np.stack(ref)[:, inner]
        rd.log(f"  check: v_chi from d(chi)/d(phi) against the solver's irrotational wind "
               f"(|lat| < 80): RMS difference {np.sqrt(np.mean(d ** 2)):.3g} m s-1, RMS v_chi "
               f"{np.sqrt(np.mean(np.stack(ref)[:, inner] ** 2)):.3g} m s-1")
    return vchi


def year_ok(y):
    """True if this year's file holds v_chi on the grid (files of the earlier version held a
    10-40E mean and are recomputed)."""
    if not year_file(y).exists():
        return False
    with xr.open_dataarray(year_file(y)) as da:
        return "lon" in da.dims


def region_mean(da, psc):
    """(..., lev, lat, lon) -> (..., lev, lat): mean over 10-40E of the cells above the ground
    (and on land if LAND_ONLY)."""
    da = rd.cut(da, SECTION)
    da = da.where(rd.above_ground_on(da, psc))
    if LAND_ONLY:
        da = da.where(masks.land_mask(da.isel({d: 0 for d in da.dims if d not in ("lat", "lon")},
                                               drop=True)))
    return da.sel(lon=slice(*SECTION["lon"])).mean("lon")


def standard_t(p):
    """Standard-atmosphere temperature (K) at pressure p (hPa), troposphere and tropopause."""
    return np.maximum(288.15 * (np.asarray(p, float) / 1013.25) ** 0.190263, 216.65)


def climatological_t(psc, lat_out):
    """(lev, lat) October mean temperature over 10-40E, from the GRIB archive if it has
    temperature, otherwise the standard atmosphere."""
    y0, y1 = years()
    got = [y for y in range(y0, y1 + 1) if grib_path("temperature", y).exists()]
    if got:
        got = got[:: max(1, len(got) // T_SAMPLE)]
        t = xr.concat([read_grib("temperature", y, SECTION) for y in got], "year").mean("year")
        rd.log(f"temperature from the GRIB archive ({len(got)} Octobers)")
        return region_mean(t, psc).interp(lat=lat_out)
    rd.log("[WARN] no temperature in the GRIB archive: rho and heights from the standard "
           "atmosphere")
    lev = np.array(LEVELS)
    return xr.DataArray(np.repeat(standard_t(lev)[:, None], lat_out.size, 1), dims=("lev", "lat"),
                        coords={"lev": lev, "lat": lat_out})


def heights(t):
    """(lev, lat) height (m) above the 1000 hPa level, hypsometric equation from 1000 hPa up
    (temperatures below the ground filled from the level above)."""
    t = t.sortby("lev", ascending=False)                                # 1000 hPa first
    v = t.values.astype(float).copy()
    for j in range(v.shape[1]):                                        # fill below the ground
        col = v[:, j]
        ok = np.isfinite(col)
        if ok.any():
            first = np.argmax(ok)
            col[:first] = col[first]
            col[~np.isfinite(col)] = np.interp(np.flatnonzero(~np.isfinite(col)),
                                               np.flatnonzero(np.isfinite(col)),
                                               col[np.isfinite(col)])
    t = t.copy(data=v)
    p = t.lev.values
    tm = 0.5 * (t.values[1:] + t.values[:-1])
    dz = RD * tm / G * np.log(p[:-1] / p[1:])[:, None]
    z = np.concatenate([np.zeros((1, t.sizes["lat"])), np.cumsum(dz, axis=0)])
    return t.copy(data=z).sortby("lev")


# ---------------------------------------------------------------- compute
def make_year_files():
    """Write the missing per-year v_chi files; returns the years that have one."""
    (config.DATA / OUT / "years").mkdir(parents=True, exist_ok=True)
    y0, y1 = years()
    first = True
    for y in range(y0, y1 + 1):
        if year_ok(y):
            continue
        if not (grib_path("u_component_of_wind", y).exists()
                and grib_path("v_component_of_wind", y).exists()):
            rd.log(f"  [WARN] no u/v GRIB for October {y}")
            continue
        v = rd.cut(vchi_from_uv(read_uv(y), check=first), SECTION)    # (lev, lat, lon)
        first = False
        v.astype("float32").rename("vchi").to_netcdf(year_file(y))
        rd.log(f"  October {y}: v_chi done")
    got = [y for y in range(y0, y1 + 1) if year_ok(y)]
    if not got:
        raise FileNotFoundError(f"no u/v GRIB files in {GRIB_DIR}")
    return got


def load_year_files(got):
    """(year, lev, lat, lon) v_chi from the per-year files."""
    return xr.concat([xr.load_dataarray(year_file(y)) for y in got], "year").assign_coords(year=got)


def compute():
    psc = rd.surface_pressure_clim(SECTION, months=[MONTH]).sel(month=MONTH)
    lat_out = np.arange(PLOT_LAT[0], PLOT_LAT[1] + 1e-6, GRID)
    got = make_year_files()
    vchi = region_mean(xr.concat([xr.load_dataarray(year_file(y)) for y in got], "year")
                       .assign_coords(year=got), psc).interp(lat=lat_out)
    omega = region_mean(yearly_omega(), psc).interp(lat=lat_out)
    common = sorted(set(got) & set(int(y) for y in omega.year.values))
    vchi, omega = vchi.sel(year=common), omega.sel(year=common)
    t = climatological_t(psc, lat_out)
    z = heights(t)
    rho = (t.lev * 100.0) / (RD * t)
    w = -omega / (rho * G)
    pg = psc.interp(lat=lat_out, lon=np.arange(SECTION["lon"][0], SECTION["lon"][1] + 1e-6, GRID),
                    kwargs={"fill_value": None})
    ps_ = pg.mean("lon")
    zs = xr.DataArray([np.interp(np.log(p), np.log(z.lev.values), z.sel(lat=la).values)
                       for la, p in zip(lat_out, ps_.values)], dims="lat", coords={"lat": lat_out})
    ds = xr.Dataset({"vchi": vchi, "omega": omega, "w": w.transpose("year", "lev", "lat"),
                     "t": t, "rho": rho, "z": z, "ps": ps_, "zs": zs})
    ds.attrs = {"month": MONTH, "lon": f"{SECTION['lon'][0]:.0f}-{SECTION['lon'][1]:.0f}E",
                "land_only": int(LAND_ONLY),
                "units": "vchi, w m s-1; omega Pa s-1; t K; rho kg m-3; z, zs m; ps hPa"}
    region_file().parent.mkdir(parents=True, exist_ok=True)
    ds.astype("float32").to_netcdf(region_file())
    rd.log(f"saved {region_file()} ({common[0]}-{common[-1]}, {len(common)} Octobers); "
           f"climatological w {float(w.mean('year').min()) * 100:+.2f} to "
           f"{float(w.mean('year').max()) * 100:+.2f} cm s-1")


# ---------------------------------------------------------------- plot
def contour_levels(da, n=5):
    """Symmetric, rounded contour levels with about n lines on each side, no zero line."""
    a = np.abs(np.asarray(da, float))
    m = float(np.nanpercentile(a, 98)) if np.isfinite(a).any() else 1.0
    step = float(f"{m / n:.1g}") or 1.0
    top = step * np.ceil(m / step)
    lv = np.arange(-top, top + step / 2, step)
    return lv[np.abs(lv) > step / 2], step


def lat_formatter():
    import matplotlib.pyplot as plt
    return plt.FuncFormatter(lambda v, _: f"{abs(v):.0f}°{'S' if v < 0 else 'N' if v > 0 else ''}")


def plot():
    import matplotlib.pyplot as plt
    if not region_file().exists():
        raise SystemExit("no data file: run 'python vchi_section.py compute' first")
    ds = xr.load_dataset(region_file())
    y0, y1 = int(ds.year.min()), int(ds.year.max())
    clim = ds[["vchi", "omega", "w"]].mean("year")
    tr = {v: stats.ols_stats(ds[v].rename(v)) for v in ("vchi", "omega", "w")}
    sig = stats.fdr(tr["omega"].p).values.astype(bool)
    trend = xr.Dataset({v: tr[v].trend for v in tr})
    ps.setup()
    fs, W = config.FS, config.WIDTH["full"]
    lat, lev = ds.lat.values, ds.lev.values
    yk = A * np.deg2rad(lat) / 1e3                               # distance from the equator, km
    zk = ds.z.values / 1e3                                       # (lev, lat) km
    ykk = np.broadcast_to(yk, zk.shape)

    left, right, gx = 0.62, 0.08, 0.25
    pw = (W - left - right - gx) / 2
    ph1, ph2 = 1.75, 1.75
    top = 0.55
    under = 0.25 + 0.07 + 0.62                                   # x labels, colour bar, label
    top2 = top + ph1 + under + 0.2 + 0.35
    H = top2 + ph2 + 0.3 + under + 0.55
    fig, at = ps.figure("full", H)
    S = {"omega": (1e2, 1e3), "w": (1e2, 1e3)}                   # clim, trend display scales
    for ci, (d, kind) in enumerate(((clim, "clim"), (trend, "trend"))):
        so, sw = S["omega"][ci], S["w"][ci]
        x0 = left + ci * (pw + gx)
        per = " decade$^{-1}$" if ci else ""
        # ---- row 1: omega shading, v_chi contours
        ax = at(x0, top, pw, ph1)
        norm = ps.sym_norm([d.omega * so])
        ax.pcolormesh(lat, lev, d.omega * so, cmap="RdBu_r", norm=norm, shading="nearest",
                      rasterized=True)
        lv, step = contour_levels(d.vchi)
        cs = ax.contour(lat, lev, d.vchi, levels=lv, colors="k", linewidths=0.7,
                        negative_linestyles="dashed", zorder=3)
        ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=fs["small"] - 2, inline_spacing=1)
        if ci:
            ps.section_dots(ax, lat, lev, sig, np.isfinite(d.omega.values))
        ax.plot(lat, ds.ps, color=C_SP_LINE, lw=0.9, zorder=4)
        ax.set_ylim(1000, 100)
        ax.set_xlim(lat[0], lat[-1])
        ax.set_yticks([200, 400, 600, 800, 1000])
        ax.xaxis.set_major_formatter(lat_formatter())
        ax.tick_params(labelleft=(ci == 0))
        if ci == 0:
            ax.set_ylabel("Pressure (hPa)")
        ax.set_title(f"{'Climatology' if ci == 0 else 'Linear trend'}, {y0}-{y1}",
                     fontsize=fs["title"], fontweight="bold", pad=3, loc="left")
        ps.letter(ax, ci)                             # a, b on the top row
        ps.colourbar(fig, at(x0 + 0.08 * pw, top + ph1 + 0.27, 0.84 * pw, 0.07), norm, "RdBu_r",
                     f"ω (10$^{{-{2 if ci == 0 else 3}}}$ Pa s$^{{-1}}${per}; red: descent)\n"
                     f"contours: v$_χ$ every {step:g} m s$^{{-1}}${per}", nticks=5)
        # ---- row 2: w shading, (v_chi, w) arrows in distance-height space
        ax = at(x0, top2, pw, ph2)
        normw = ps.sym_norm([d.w * sw])
        ax.pcolormesh(ykk, zk, d.w.values * sw, cmap="RdBu", norm=normw, shading="nearest",
                      rasterized=True)                          # RdBu: red = descent (w < 0)
        if ci:                                    # same thinning as row 1 (mean heights)
            ps.section_dots(ax, yk, zk.mean(axis=1), sig, np.isfinite(d.w.values))
        sel = np.isin(lev, ARROW_LEVELS)
        step_i = max(1, int(round(ARROW_STEP / GRID)))
        li = np.arange(int(round((np.ceil(lat[0] / ARROW_STEP) * ARROW_STEP - lat[0]) / GRID)),
                       lat.size, step_i)
        U, V = d.vchi.values[np.ix_(sel, li)], d.w.values[np.ix_(sel, li)]
        mag = np.hypot(U, V)
        ref = float(f"{np.nanpercentile(mag, 95):.1g}") if np.isfinite(mag).any() else 1.0
        q = ax.quiver(ykk[np.ix_(sel, li)], zk[np.ix_(sel, li)], U, V, angles="xy",
                      scale=ref / REF_IN, scale_units="inches", width=0.004, headwidth=3.5,
                      headlength=4, headaxislength=3.5, color="k", zorder=4)
        ax.quiverkey(q, 0.08, -0.3, ref, f"{ref:g} m s$^{{-1}}${per} (arrow length ≈ speed, "
                     "mostly v$_χ$)", labelpos="E", coordinates="axes",
                     fontproperties={"size": fs["small"] - 1})
        ax.fill_between(yk, -1, ds.zs.values / 1e3, color=C_SP_LINE, alpha=0.85, lw=0, zorder=6)
        ax.set_xlim(yk[0], yk[-1])
        ax.set_ylim(0, 16)
        ax.tick_params(labelleft=(ci == 0))
        if ci == 0:
            ax.set_ylabel("Height (km)")
        ax.set_xlabel("Distance from the equator (10$^3$ km)", fontsize=fs["small"])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e3:g}"))
        top_ax = ax.secondary_xaxis("top", functions=(lambda y: np.rad2deg(y * 1e3 / A),
                                                      lambda la: A * np.deg2rad(la) / 1e3))
        top_ax.xaxis.set_major_formatter(lat_formatter())
        top_ax.tick_params(labelsize=fs["tick"] - 1, length=2, pad=1)
        # vertical stretch of the drawn frame: (km per inch across) / (km per inch up)
        stretch = ((yk[-1] - yk[0]) / pw) / (16.0 / ph2)
        ax.text(1.0, 1.13, f"vertical stretch ×{stretch:.0f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=fs["small"] - 1)
        ax.set_title(f"{'Climatology' if ci == 0 else 'Linear trend'}: v$_χ$, w",
                     fontsize=fs["title"], fontweight="bold", pad=14, loc="left")
        ps.letter(ax, 2 + ci)                         # c, d on the bottom row
        ps.colourbar(fig, at(x0 + 0.08 * pw, top2 + ph2 + 0.3 + 0.32, 0.84 * pw, 0.07), normw,
                     "RdBu", f"w (10$^{{-{2 if ci == 0 else 3}}}$ m s$^{{-1}}${per}; red: "
                     "descent)", nticks=5)
    lon = ds.attrs.get("lon", "10-40E")
    ps.text(fig, W / 2, 0.05, f"ERA5 {calendar.month_name[MONTH]} divergent circulation, "
            f"sub-Saharan Africa ({lon.replace('E', '°E')} mean)", ha="center", va="top",
            fontsize=fs["suptitle"], fontweight="bold")
    ps.text(fig, W / 2, H - 0.06, "a-b: v$_χ$ = (1/a) ∂χ/∂φ contours, solid northward, dashed "
            "southward. c-d: w = -ω/(ρg), ρ = p/(R$_d$T);\narrows follow the air's path in the "
            f"drawn frame. Dots: significant ω (= w) trend (FDR q = {config.ALPHA_FDR:.2f}); "
            "brown: ground", ha="center", va="bottom", fontsize=fs["small"] - 1, color="0.3")
    ps.save(fig, config.RESULTS / OUT / f"vchi_section_{calendar.month_name[MONTH].lower()}.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "compute" or (what == "all" and not region_file().exists()):
        compute()
    if what in ("plot", "all"):
        plot()