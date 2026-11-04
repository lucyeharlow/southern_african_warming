"""
Surface energy / radiation budget trends over Southern Africa (10-40E, 35-10S, land).

Run:  python radiation_budget_trends.py             compute (if data files missing) and plot
      python radiation_budget_trends.py compute     re-read the raw data
      python radiation_budget_trends.py plot        plot from data/ only

FIGURES (results/radiation_budget/)
  qred_full_budget.png  Q_red = Q_rad - LHF - SHF = SW_dn - SW_up + LW_dn - LW_up - LHF - SHF
                        (net surface energy flux: what goes into the ground and storage)
                        ERA5, MERRA-2, JRA-3Q
  qrad_budget.png       Q_rad = SW_dn - SW_up + LW_dn - LW_up (net radiation only)
                        ERA5, MERRA-2, CLARA-A3, JRA-3Q
  a) monthly boxplots of the per-cell trends (area-weighted, land)
  b)... October maps, one column per dataset:
     row 1 trend; row 2 dominant flux where it is strengthening; row 3 where it is weakening.
     Dominant flux = the component with the largest |contribution| to the trend (signed as it
     enters the budget). Colour strength = the size of that contribution in W m-2 per
     decade, compared with every cell of the October maps of the figure: full colour at the
     98th percentile of |dominant contribution| over all of them (CMAP_PERCENTILE), pale =
     small change. So cells can be compared across the maps and with the trend colour bar.
     Black dots: trend significant (FDR q = 0.10 over the land cells of the map).

DATA (each dataset on its own grid; LHF, SHF upward positive)
  ERA5     global_surface_*.nc monthly means of daily accumulations (J m-2 -> W m-2):
           SW_dn = ssrd, SW_up = ssrd - ssr, LW_dn = strd, LW_up = strd - str,
           LHF = -slhf, SHF = -sshf
  MERRA-2  tavgM_2d_rad_Nx SWGDN, SWGNT, LWGAB, LWGNT; tavgM_2d_flx_Nx EFLUX, HFLUX
  CLARA-A3 SIS, SNS, SDL, SNL (SW_up = SIS - SNS, LW_up = SDL - SNL)
  JRA-3Q   {dswrf,uswrf,dlwrf,ulwrf,lhtfl,shtfl}1have-sfc-fc-ll125 (1-hourly, monthly files);
           the turbulent fluxes' sign is set so that the October LHF + SHF over the box is
           upward (JRA3Q_TURB_UPWARD = None), and logged.
Years: config.YEARS (each dataset its own). Conventions: stats.py.

DATA FILES  data/radiation_budget/<budget>_<dataset>.nc
  trend, p, n_eff, sig (month, lat, lon); comp_trend (comp, month, lat, lon); W m-2 per decade
"""
from __future__ import annotations

import calendar
import colorsys
import sys

import numpy as np
import pandas as pd
import xarray as xr

import config
import masks
import plotstyle as ps
import readers as rd
import stats

REGION = "southern_africa"
MONTH = 10
OUT = "radiation_budget"
BUDGETS = {"full": ["ERA5", "MERRA-2", "JRA-3Q"],
           "rad": ["ERA5", "MERRA-2", "JRA-3Q", "CLARA-A3"]}
JRA3Q_TURB_UPWARD = None          # True / False if known; None = decide from the data
COMPS = ["SW_dn", "SW_up", "LW_dn", "LW_up", "LHF", "SHF"]
SIGN = {"SW_dn": 1, "SW_up": -1, "LW_dn": 1, "LW_up": -1, "LHF": -1, "SHF": -1}
LABELS = {"SW_dn": r"SW$\downarrow$", "SW_up": r"SW$\uparrow$", "LW_dn": r"LW$\downarrow$",
          "LW_up": r"LW$\uparrow$", "LHF": "LHF", "SHF": "SHF"}
HUES = {"SW_dn": 48, "SW_up": 280, "LW_dn": 130, "LW_up": 25, "LHF": 210, "SHF": 330}
CMAP_PERCENTILE = config.COLOUR_PERCENTILE   # full colour: this percentile of |dominant change|
DAY = 86400.0


def data_file(budget, name):
    return config.DATA / OUT / f"{budget}_{name.replace('-', '')}.nc"


def deep(c):
    return np.array(colorsys.hls_to_rgb(HUES[c] / 360, 0.42, 0.75))


def share_cmap(c):
    col = deep(c)
    return ps.shade_cmap(col, start=0.85 * np.ones(3) + 0.15 * col)


# ---------------------------------------------------------------- reading
def components(name, full):
    """{comp: (year, month, lat, lon) W m-2}."""
    keys = COMPS if full else COMPS[:4]
    era = config.RAW["era5"]
    if name == "ERA5":
        f = lambda v, n: rd.single_file(era / f"global_{n}.nc", v, REGION, scale=1 / DAY)
        ssrd, ssr = f("ssrd", "surface_solar_radiation_downwards"), f("ssr", "surface_net_solar_radiation")
        strd = f("strd", "surface_thermal_radiation_downwards")
        str_ = f("str", "surface_net_thermal_radiation")
        out = {"SW_dn": ssrd, "SW_up": ssrd - ssr, "LW_dn": strd, "LW_up": strd - str_}
        if full:
            out["LHF"] = -f("slhf", "surface_latent_heat_flux")
            out["SHF"] = -f("sshf", "surface_sensible_heat_flux")
    elif name == "MERRA-2":
        m = lambda v: rd.merra2("tavgM_2d_rad_Nx", v, REGION)
        dn, sw, lw_ab, lw = m("SWGDN"), m("SWGNT"), m("LWGAB"), m("LWGNT")
        out = {"SW_dn": dn, "SW_up": dn - sw, "LW_dn": lw_ab, "LW_up": lw_ab - lw}
        if full:
            out["LHF"] = rd.merra2("tavgM_2d_flx_Nx", "EFLUX", REGION)
            out["SHF"] = rd.merra2("tavgM_2d_flx_Nx", "HFLUX", REGION)
    elif name == "CLARA-A3":
        c = lambda v: rd.cmsaf(v, v, REGION)
        sis, sdl = c("SIS"), c("SDL")
        out = {"SW_dn": sis, "SW_up": sis - c("SNS"), "LW_dn": sdl, "LW_up": sdl - c("SNL")}
    elif name == "JRA-3Q":
        j = lambda v: rd.jra3q(f"{v}1have-sfc-fc-ll125", f"{v}1have-sfc-fc-ll125", REGION)
        out = {"SW_dn": j("dswrf"), "SW_up": j("uswrf"), "LW_dn": j("dlwrf"), "LW_up": j("ulwrf")}
        if full:
            lh, sh = j("lhtfl"), j("shtfl")
            if JRA3Q_TURB_UPWARD is None:
                land = masks.land_mask(lh)
                m = float(stats.area_mean((lh + sh).sel(month=MONTH).mean("year").where(land)))
                sign = 1.0 if m > 0 else -1.0
                rd.log(f"  JRA-3Q October LHF + SHF over the box {m:+.1f} W m-2: files are "
                       f"{'upward' if m > 0 else 'downward (flipped to upward)'} positive")
            else:
                sign = 1.0 if JRA3Q_TURB_UPWARD else -1.0
            out["LHF"], out["SHF"] = sign * lh, sign * sh
    else:
        raise KeyError(name)
    return {k: out[k] for k in keys}


# ---------------------------------------------------------------- compute
def compute():
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    cache = {}
    for budget, names in BUDGETS.items():
        full = budget == "full"
        for name in names:
            rd.log(f"--- {name}, {budget} budget ---")
            if (name, full) not in cache:
                cache[(name, full)] = components(name, full)
            comp = cache[(name, full)]
            land = masks.land_mask(comp["SW_dn"])
            comp = {k: stats.select_years(v.where(land).rename(f"{name} {k}"), config.YEARS[name])
                    for k, v in comp.items()}
            q = sum(SIGN[k] * v for k, v in comp.items())
            st = stats.ols_stats(q)
            sig = stats.fdr_maps(st.p)
            ct = xr.concat([stats.ols_stats(comp[k]).trend for k in comp], "comp").assign_coords(
                comp=list(comp))
            ds = xr.Dataset({"trend": st.trend.astype("float32"), "p": st.p.astype("float32"),
                             "n_eff": st.n_eff.astype("float32"), "sig": sig.astype("int8"),
                             "comp_trend": ct.astype("float32")})
            ds.attrs = {"dataset": name, "budget": budget, "units": "W m-2 per decade",
                        "years": f"{config.YEARS[name][0]}-{config.YEARS[name][1]}"}
            enc = {v: {"zlib": True, "complevel": 4} for v in ("trend", "p", "n_eff", "comp_trend")}
            ds.to_netcdf(data_file(budget, name), encoding=enc)
            o = ds.sel(month=MONTH)
            rd.log(f"  October: area-weighted mean trend {stats.describe(o.trend)['mean']:+.2f} "
                   f"W m-2 per decade, {stats.pct_area(o.sig.astype(bool), o.trend.notnull()):.0f}"
                   f" % of area significant; saved {data_file(budget, name).name}")


# ---------------------------------------------------------------- plot
def dominant(o):
    """Index into COMPS of the largest |signed contribution|, whether that flux increases,
    and the size of that contribution (W m-2 per decade)."""
    contrib = xr.concat([SIGN[str(c)] * o.comp_trend.sel(comp=c) for c in o.comp.values], "comp")
    a = np.abs(contrib.fillna(0)).values
    raw = o.comp_trend.values
    idx = a.argmax(0)
    ok = np.isfinite(o.trend.values)
    rising = np.take_along_axis(raw, idx[None], 0)[0] > 0
    size = np.take_along_axis(a, idx[None], 0)[0]
    code = np.array([COMPS.index(str(c)) for c in o.comp.values])[idx]
    return np.where(ok, code, -1), rising & ok, np.where(ok, size, np.nan)


def plot_budget(budget):
    import cartopy.crs as ccrs
    from matplotlib.colors import Normalize
    names = [n for n in BUDGETS[budget] if data_file(budget, n).exists()]
    if not names:
        print(f"no data files for the {budget} budget: run compute first")
        return
    res = {n: xr.load_dataset(data_file(budget, n)) for n in names}
    qlab = "Q$_{red}$" if budget == "full" else "Q$_{rad}$"
    fs, W = config.FS, config.WIDTH["full"]
    left, right, gap = 0.95, 0.12, 0.12
    nd = len(names)
    r = config.REGIONS[REGION]
    mw = (W - left - right - (nd - 1) * gap) / nd
    mh = mw * (r["lat"][1] - r["lat"][0]) / (r["lon"][1] - r["lon"][0])
    box_top, box_h = 0.6, 1.2
    map_top = box_top + box_h + 0.5
    gy = 0.24
    cb1 = map_top + mh + 0.32
    row2 = cb1 + 0.62
    leg_top = row2 + 2 * mh + gy + 0.3
    H = leg_top + 0.95
    fig, at = ps.figure("full", H)
    years = {n: ps.year_list(res[n].attrs["years"]) for n in names}

    # a) monthly boxplots
    ax = at(left, box_top, W - left - right, box_h)
    ps.monthly_boxplot(ax, {n: {int(m): res[n].trend.sel(month=m) for m in res[n].month.values}
                            for n in names}, names,
                       f"{qlab} trend, area-weighted\n(W m$^{{-2}}$ decade$^{{-1}}$)")
    ps.letter(ax, 0)
    ps.dataset_legend(ax, names, years, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                      ncol=2, handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)

    # October maps
    norm = ps.sym_norm([res[n].trend.sel(month=MONTH) for n in names])
    rows = [f"{calendar.month_abbr[MONTH]} {qlab}\ntrend", "Dominant flux,\nstrengthening",
            "Dominant flux,\nweakening"]
    tops = [map_top, row2, row2 + mh + gy]
    deep_cmaps = [share_cmap(c) for c in COMPS]
    sizes = np.concatenate([dominant(res[n].sel(month=MONTH))[2].ravel() for n in names])
    sizes = sizes[np.isfinite(sizes)]
    smax = float(np.percentile(sizes, CMAP_PERCENTILE)) if sizes.size else 1.0
    smax = smax if smax > 0 else 1.0
    letter_i = 1
    for ri, t0 in enumerate(tops):
        ps.text(fig, 0.12, t0 + mh / 2, rows[ri], rotation=90, ha="center", va="center",
                fontsize=fs["label"], fontweight="bold")
        for ci, name in enumerate(names):
            o = res[name].sel(month=MONTH)
            x0 = left + ci * (mw + gap)
            ax = at(x0, t0, mw, mh, projection=ccrs.PlateCarree())
            sig = o.sig.astype(bool).values
            if ri == 0:
                ps.map_panel(ax, o.trend, norm, sig=sig, region=REGION, left=(ci == 0))
                ps.text(fig, x0 + mw / 2, t0 - 0.05, ps.label(name), ha="center",
                        fontsize=fs["title"], fontweight="bold", color=ps.colour(name))
            else:
                code, rising, size = dominant(o)
                want = rising if ri == 1 else ~rising
                strength = size / smax           # colour maps start pale, not white
                ps.categories(ax, o.lon, o.lat, np.where(want, code, -1), np.clip(strength, 0, 1),
                              [deep(c) for c in COMPS], other=(code >= 0) & ~want,
                              transform=ccrs.PlateCarree(), cmaps=deep_cmaps)
                ps.sig_dots(ax, o.trend, sig)
                ps.map_frame(ax, REGION, left=(ci == 0), bottom=(ri == 2))
            ps.letter(ax, letter_i)
            letter_i += 1
    ps.colourbar(fig, at(left + 0.5, cb1, W - left - right - 1.0, 0.09), norm, "RdBu_r",
                 f"{qlab} trend (W m$^{{-2}}$ decade$^{{-1}}$)")

    # legend: flux colours, share scale, other, dots
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    used = sorted({str(c) for n in names for c in res[n].comp.values}, key=COMPS.index)
    h = [Patch(facecolor=deep(c), edgecolor="0.3", linewidth=0.4, label=f"{LABELS[c]} dominant")
         for c in used]
    h += [Patch(facecolor=config.C_OTHER, edgecolor="0.4", linewidth=0.4,
                label="Changes the other way"),
          ps.dot_handle(f"Trend significant (FDR q = {config.ALPHA_FDR:.2f})")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(left / W, 1 - leg_top / H),
               ncol=3, frameon=False, handlelength=1.2, columnspacing=1.0)
    cax = at(left + 0.4, leg_top + 0.62, 2.3, 0.08)
    cb = fig.colorbar(plt_mappable(Normalize(0, smax)), cax=cax, orientation="horizontal",
                      extend="max")
    cb.ax.tick_params(labelsize=fs["tick"], length=2)
    cb.outline.set_linewidth(0.4)
    ps.text(fig, left + 2.85, leg_top + 0.72, "Colour strength (any flux): size of the\n"
            "dominant flux's change, W m$^{-2}$ decade$^{-1}$\n"
            f"(full colour: largest {100 - CMAP_PERCENTILE:.0f}% of cells in these maps)",
            fontsize=fs["small"], va="center")
    ps.save(fig, config.RESULTS / OUT / ("qred_full_budget.png" if budget == "full"
                                         else "qrad_budget.png"))


def plt_mappable(norm):
    import matplotlib.pyplot as plt
    return plt.cm.ScalarMappable(norm=norm, cmap=ps.shade_cmap("0.15", start="0.85"))


def summary_csv():
    rows = []
    for budget, names in BUDGETS.items():
        for n in names:
            if not data_file(budget, n).exists():
                continue
            ds = xr.load_dataset(data_file(budget, n))
            for m in ds.month.values:
                o = ds.sel(month=m)
                rows.append({"budget": budget, "dataset": n, "years": ds.attrs["years"],
                             "month": int(m), **stats.describe(o.trend),
                             "pct_area_sig_fdr": stats.pct_area(o.sig.astype(bool),
                                                                o.trend.notnull())})
    path = config.RESULTS / OUT / "qred_qrad_budget_monthly_trends.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved: {path}", flush=True)


def plot():
    ps.setup()
    summary_csv()
    for budget in BUDGETS:
        plot_budget(budget)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    missing = not all(data_file(b, n).exists() for b, ns in BUDGETS.items() for n in ns)
    if what == "compute" or (what == "all" and missing):
        compute()
    if what in ("plot", "all"):
        plot()
