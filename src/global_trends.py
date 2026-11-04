"""
October surface flux trends, 40S-40N and 120W-120E, land: ERA5, MERRA-2, JRA-3Q, CLARA-A3.

Run:  python global_flux_trends.py [compute | plot]

FIGURE  results/global_flux/october_flux_trends_tropics.png
  Rows = quantities, columns = datasets:
    a) net radiation Q_rad = SW_net + LW_net     ERA5, MERRA-2, JRA-3Q, CLARA-A3
    b) downwelling SW                            all four
    c) downwelling LW                            all four
    d) latent heat flux H_L (upward positive)    ERA5, MERRA-2, JRA-3Q
    e) sensible heat flux H_S (upward positive)  ERA5, MERRA-2, JRA-3Q
    f) surface radiation budget residual Q_rad - H_L - H_S (own colour bar)
  White dots: significant (FDR q = 0.10 over the land cells of each map).

DATA (native grids; years config.YEARS)
  ERA5      global_surface_*.nc (J m-2 per day -> W m-2): Q_rad = ssr + str, SW down = ssrd,
            LW down = strd, H_L = -slhf, H_S = -sshf
  MERRA-2   tavgM_2d_rad_Nx: SWGNT + LWGNT, SWGDN, LWGAB; tavgM_2d_flx_Nx: EFLUX, HFLUX
  CLARA-A3  Q_rad = SNS + SNL (surface net SW + net LW), SW down = SIS, LW down = SDL
  JRA-3Q    {dswrf,uswrf,dlwrf,ulwrf,lhtfl,shtfl}1have-sfc-fc-ll125; Q_rad = dswrf - uswrf
            + dlwrf - ulwrf; the turbulent fluxes' sign is set so the October latent heat
            over the land region is upward (logged)

DATA FILES  data/global_flux/<quantity>_<dataset>.nc: trend, p, sig (lat, lon), W m-2 per decade
"""
from __future__ import annotations

import calendar
import sys

import numpy as np
import xarray as xr

import config
import masks
import plotstyle as ps
import readers as rd
import stats

REGION = "tropics"
MONTH = 10
OUT = "global_flux"
NAMES = ["ERA5", "MERRA-2", "JRA-3Q", "CLARA-A3"]
QUANTITIES = [
    ("Qrad", "Q$_{rad}$"),
    ("SWdn", r"SW$\downarrow$"),
    ("LWdn", r"LW$\downarrow$"),
    ("LH", "H$_L$"),
    ("SH", "H$_S$"),
    ("RES", "Q$_{rad}$-\nH$_L$-H$_S$"),
]
JRA3Q_TURB_UPWARD = None
DAY = 86400.0
TICKS = dict(xticks=[-90, 0, 90], yticks=[-30, 0, 30])


def data_file(q, name):
    return config.DATA / OUT / f"{q}_{name.replace('-', '')}.nc"


# ---------------------------------------------------------------- reading
def load(name):
    """{quantity: (year, month, lat, lon)} October fields in W m-2 (H_L, H_S upward)."""
    mo = [MONTH]
    if name == "ERA5":
        f = lambda v, n: rd.single_file(config.RAW["era5"] / f"global_{n}.nc", v, REGION,
                                        scale=1 / DAY, months=mo)
        out = {"Qrad": f("ssr", "surface_net_solar_radiation")
                       + f("str", "surface_net_thermal_radiation"),
               "SWdn": f("ssrd", "surface_solar_radiation_downwards"),
               "LWdn": f("strd", "surface_thermal_radiation_downwards"),
               "LH": -f("slhf", "surface_latent_heat_flux"),
               "SH": -f("sshf", "surface_sensible_heat_flux")}
    elif name == "MERRA-2":
        m = lambda c, v: rd.merra2(c, v, REGION, months=mo)
        out = {"Qrad": m("tavgM_2d_rad_Nx", "SWGNT") + m("tavgM_2d_rad_Nx", "LWGNT"),
               "SWdn": m("tavgM_2d_rad_Nx", "SWGDN"), "LWdn": m("tavgM_2d_rad_Nx", "LWGAB"),
               "LH": m("tavgM_2d_flx_Nx", "EFLUX"), "SH": m("tavgM_2d_flx_Nx", "HFLUX")}
    elif name == "CLARA-A3":
        c = lambda v: rd.cmsaf(v, v, REGION, months=mo)
        sw, lw = c("SIS"), c("SDL")                    # downwelling SW and LW
        out = {"Qrad": c("SNS") + c("SNL"),            # net SW + net LW (downward positive)
               "SWdn": sw, "LWdn": lw}
    elif name == "JRA-3Q":
        j = lambda v: rd.jra3q(f"{v}1have-sfc-fc-ll125", f"{v}1have-sfc-fc-ll125", REGION, months=mo)
        sw, lw = j("dswrf"), j("dlwrf")
        out = {"Qrad": sw - j("uswrf") + lw - j("ulwrf"), "SWdn": sw, "LWdn": lw}
        lh, sh = j("lhtfl"), j("shtfl")
        if JRA3Q_TURB_UPWARD is None:
            land = masks.land_mask(lh)
            m = float(stats.area_mean(lh.sel(month=MONTH).mean("year").where(land)))
            sign = 1.0 if m > 0 else -1.0
            rd.log(f"  JRA-3Q October latent heat over land {m:+.1f} W m-2: files are "
                   f"{'upward' if m > 0 else 'downward (flipped to upward)'} positive")
        else:
            sign = 1.0 if JRA3Q_TURB_UPWARD else -1.0
        out["LH"], out["SH"] = sign * lh, sign * sh
    else:
        raise KeyError(name)
    if "LH" in out:
        out["RES"] = out["Qrad"] - out["LH"] - out["SH"]
    return {k: v.sel(month=MONTH) for k, v in out.items()}


# ---------------------------------------------------------------- compute
def compute():
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        rd.log(f"--- {name} ---")
        fields = load(name)
        land = masks.land_mask(next(iter(fields.values())))
        for q, da in fields.items():
            da = stats.select_years(da.where(land).rename(f"{name} {q}"), config.YEARS[name])
            st = stats.ols_stats(da)
            sig = stats.fdr(st.p)
            ds = xr.Dataset({"trend": st.trend.astype("float32"), "p": st.p.astype("float32"),
                             "sig": sig.astype("int8")})
            ds.attrs = {"dataset": name, "quantity": q, "month": MONTH,
                        "units": "W m-2 per decade",
                        "years": f"{config.YEARS[name][0]}-{config.YEARS[name][1]}"}
            ds.to_netcdf(data_file(q, name), encoding={v: {"zlib": True} for v in ("trend", "p")})
            rd.log(f"  {q}: area-weighted mean {float(stats.area_mean(st.trend)):+.2f} W m-2 per "
                   f"decade, {stats.pct_area(sig, st.trend.notnull()):.0f} % of land significant")


# ---------------------------------------------------------------- plot
def bottom_labels(res, ri, n):
    """Longitude labels on the last two rows and wherever the panel below is empty."""
    below = [(QUANTITIES[j][0], n) in res for j in range(ri + 1, len(QUANTITIES) - 1)]
    return ri >= len(QUANTITIES) - 2 or not any(below)


def plot():
    import cartopy.crs as ccrs
    ps.setup()
    res = {(q, n): xr.load_dataset(data_file(q, n)) for q, _ in QUANTITIES for n in NAMES
           if data_file(q, n).exists()}
    if not res:
        raise SystemExit("no data files: run 'python global_flux_trends.py compute' first")
    fs, W = config.FS, config.WIDTH["full"]
    names = [n for n in NAMES if any(k[1] == n for k in res)]
    left, right, gx, gy = 1.0, 0.08, 0.1, 0.16
    nc = len(names)
    r = config.REGIONS[REGION]
    pw = (W - left - right - (nc - 1) * gx) / nc
    ph = pw * (r["lat"][1] - r["lat"][0]) / (r["lon"][1] - r["lon"][0])
    top = 0.42
    tops = [top + i * (ph + gy) for i in range(len(QUANTITIES))]
    tops[-1] += 0.62                                       # room for the shared colour bar
    H = tops[-1] + ph + 0.75
    fig, at = ps.figure("full", H)
    shared = ps.sym_norm([d.trend for (q, n), d in res.items() if q != "RES"])
    res_norm = ps.sym_norm([d.trend for (q, n), d in res.items() if q == "RES"] or [[1.0]])
    k = 0
    for ri, (q, qlab) in enumerate(QUANTITIES):
        norm = res_norm if q == "RES" else shared
        ps.text(fig, 0.05, tops[ri] + ph / 2, qlab, ha="left", va="center",
                fontsize=fs["label"], fontweight="bold")
        for ci, n in enumerate(names):
            x0 = left + ci * (pw + gx)
            if ri == 0:
                ps.text(fig, x0 + pw / 2, top - 0.05, ps.label(n), ha="center",
                        fontsize=fs["title"], fontweight="bold", color=ps.colour(n))
            d = res.get((q, n))
            if d is None:
                continue
            ax = at(x0, tops[ri], pw, ph, projection=ccrs.PlateCarree())
            ps.map_panel(ax, d.trend, norm, sig=d.sig.astype(bool).values, region=REGION,
                         left=(ci == 0), bottom=bottom_labels(res, ri, n), **TICKS)
            ax.text(0.01, 0.95, f"{'abcdefghijklmnopqrstuvwxyz'[k]})", transform=ax.transAxes,
                    fontsize=fs["small"], fontweight="bold", va="top", zorder=10,
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.8))
            k += 1
    cb_top = tops[-2] + ph + 0.2
    ps.colourbar(fig, at(left + 0.3, cb_top, W - left - right - 0.6, 0.08), shared, "RdBu_r",
                 "Trend, rows 1-5 (W m$^{-2}$ decade$^{-1}$)")
    ps.colourbar(fig, at(left + 0.3, tops[-1] + ph + 0.22, W - left - right - 0.6, 0.08),
                 res_norm, "RdBu_r", "Residual trend (W m$^{-2}$ decade$^{-1}$); "
                 f"dots: significant (FDR q = {config.ALPHA_FDR:.2f}); H$_L$, H$_S$ upward")
    yrs = sorted({d.attrs["years"] for d in res.values()})
    ps.text(fig, W / 2, 0.02, f"{calendar.month_name[MONTH]} trends over land "
            f"({', '.join(yrs)})", ha="center", va="top", fontsize=fs["suptitle"],
            fontweight="bold")
    ps.save(fig, config.RESULTS / OUT / "october_flux_trends_tropics.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    need = [(q, n) for q, _ in QUANTITIES for n in NAMES
            if not (n == "CLARA-A3" and q in ("LH", "SH", "RES"))]
    if what == "compute" or (what == "all" and not all(data_file(q, n).exists() for q, n in need)):
        compute()
    if what in ("plot", "all"):
        plot()
