"""
Downwelling shortwave (SW down) trends over Southern Africa, in the style of the temperature
figure: ERA5, MERRA-2, JRA-3Q, CLARA-A3, SARAH-3, plus CERES EBAF in the series.

Run:  python sw_down_trends.py            compute (if the data files are missing) and plot
      python sw_down_trends.py compute    re-read the raw data and rewrite the data files
      python sw_down_trends.py plot       plot from the data files only

FIGURE  results/sw_down/sw_down_trends_southern_africa.png
  a) monthly boxplots of the cell trends over Southern Africa (10-40E, 35-10S), land,
     area-weighted (box = weighted IQR, line = weighted median, whiskers = most extreme cell
     within 1.5 IQR)
  b)-f) October trend maps, 10-40E, 35S-18N (the box's longitudes, north to the sub-Saharan
     edge; no Madagascar), land, the Southern Africa box dashed; black dots: significant (FDR q = 0.10 over the land cells
     of each map)
  g) October area-weighted SW down anomaly over the Southern Africa land (from each
     dataset's 1991-2020 October mean; CERES EBAF from its 2000-2020 mean), with the OLS
     trend and its p in the legend

DATA (native grids; years config.years(dataset, month), so each dataset keeps its own period)
  ERA5      global_surface_solar_radiation_downwards.nc  ssrd (J m-2 per day -> W m-2)
  MERRA-2   tavgM_2d_rad_Nx                              SWGDN
  JRA-3Q    dswrf1have-sfc-fc-ll125 (1-hourly averages, monthly files; read over the
            40S-40N region so the October months cached by global_flux_trends.py are reused)
  CLARA-A3  SIS (February from 1986: February 1985 is empty in CLARA-A3)
  SARAH-3   SIS
  CERES     CERES_EBAF_Ed4.2.1_Subset_*.nc  sfc_sw_down_all_mon, 2000-2024, series only

METHOD (stats.py, masks.py): land cells (Natural Earth 50 m land fraction >= 0.5) with a value
  in every year; OLS trend per decade, lag-1 n_eff <= n; Benjamini-Hochberg FDR q = 0.10 per
  map; boxplots, series and % of area significant area-weighted (cos latitude).

DATA FILES  data/sw_down/sw_down_<dataset>.nc
  trend, p, n_eff, sig, first_year, last_year (month, lat, lon / month); October series.
OUTPUTS     results/sw_down/sw_down_trends_southern_africa.png, .csv (summary per dataset and
  month), sw_down_october_series.csv
"""
from __future__ import annotations

import calendar
import glob
import sys

import numpy as np
import pandas as pd
import xarray as xr

import config
import masks
import plotstyle as ps
import readers as rd
import stats

BOX = "southern_africa"           # boxplots and series
MAP = "africa_10_40"              # maps
MONTH = 10
ORDER = ["ERA5", "MERRA-2", "JRA-3Q", "CLARA-A3", "SARAH-3"]
SERIES = ORDER + ["CERES"]
CERES_YEARS = (2000, 2024)        # CERES EBAF's own October record (it starts in March 2000)
OUT = "sw_down"
UNITS = "W m$^{-2}$"


def data_file(name):
    return config.DATA / OUT / f"sw_down_{name.replace('-', '')}.nc"


# ---------------------------------------------------------------- reading
def load(name):
    """(year, month, lat, lon) SW down in W m-2 over MAP."""
    if name == "ERA5":
        return rd.single_file(config.RAW["era5"] / "global_surface_solar_radiation_downwards.nc",
                              "ssrd", MAP, scale=1 / 86400.0)
    if name == "MERRA-2":
        return rd.merra2("tavgM_2d_rad_Nx", "SWGDN", MAP)
    if name == "JRA-3Q":
        v = "dswrf1have-sfc-fc-ll125"
        return rd.cut(rd.jra3q(v, v, "tropics"), MAP)
    if name == "CLARA-A3":
        return rd.cmsaf("SIS", "SIS", MAP)
    if name == "SARAH-3":
        return rd.cmsaf("SIS", "SIS", MAP, dirs=[config.RAW["sarah"]])
    raise KeyError(name)


def load_ceres():
    """(year, lat, lon) CERES EBAF October all-sky SW down over BOX."""
    parts = [rd.single_file(f, "sfc_sw_down_all_mon", BOX, months=[MONTH], fill=1e3)
             for f in sorted(glob.glob(str(config.RAW["ceres"] / "CERES_EBAF_Ed4.2.1_Subset_*.nc")))]
    if not parts:
        raise FileNotFoundError(f"no CERES EBAF files in {config.RAW['ceres']}")
    da = xr.concat(parts, "year")
    _, first = np.unique(da.year.values, return_index=True)
    da = da.isel(year=first)
    return da.where(da >= 0).sel(month=MONTH)


def october_series(da, years):
    """Area-weighted October anomaly over the BOX land cells complete in every year."""
    s = stats.regional_series(rd.cut(da, BOX))
    b0, b1 = max(config.BASELINE[0], years[0]), min(config.BASELINE[1], years[1])
    return s, s - s.sel(year=slice(b0, b1)).mean(), (b0, b1)


def series_attrs(s, base):
    st = stats.ols_stats(s)
    return {"series_trend": float(st.trend), "series_p": float(st.p),
            "series_n_eff": float(st.n_eff), "series_n": float(st.n),
            "series_base": f"{base[0]}-{base[1]}"}


# ---------------------------------------------------------------- compute
def compute():
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    for name in ORDER:
        rd.log(f"--- {name} ---")
        da = load(name)
        da = da.where(masks.land_mask(da))
        parts = []
        for m in range(1, 13):
            yrs = config.years(name, m)
            x = stats.select_years(da.sel(month=m).rename(f"{name} {calendar.month_abbr[m]}"), yrs)
            st = stats.ols_stats(x)
            parts.append(xr.Dataset({"trend": st.trend, "p": st.p, "n_eff": st.n_eff,
                                     "sig": stats.fdr(st.p),
                                     "first_year": yrs[0], "last_year": yrs[1]}))
        ds = xr.concat(parts, xr.DataArray(np.arange(1, 13), dims="month", name="month"))
        yrs = config.years(name, MONTH)
        oct_ = stats.select_years(da.sel(month=MONTH).rename(name), yrs)
        raw, anom, base = october_series(oct_, yrs)
        out = xr.Dataset({"trend": ds.trend.astype("float32"), "p": ds.p.astype("float32"),
                          "n_eff": ds.n_eff.astype("float32"), "sig": ds.sig.astype("int8"),
                          "first_year": ds.first_year.astype("int16"),
                          "last_year": ds.last_year.astype("int16"),
                          "series": anom, "series_abs": raw})
        out.attrs = {"dataset": name, "years": f"{yrs[0]}-{yrs[1]}", "units": "W m-2 per decade",
                     "series_month": MONTH, **series_attrs(raw, base)}
        out.to_netcdf(data_file(name),
                      encoding={v: {"zlib": True, "complevel": 4} for v in ("trend", "p", "n_eff")})
        o = rd.cut(out.sel(month=MONTH), BOX)
        rd.log(f"  {out.attrs['years']}: October Southern Africa median trend "
               f"{stats.wmedian(o.trend):+.2f} W m-2/decade, "
               f"{stats.pct_area(o.sig.astype(bool), o.trend.notnull()):.0f} % of area significant; "
               f"regional {out.attrs['series_trend']:+.2f}, {stats.p_text(out.attrs['series_p'])}")
    rd.log("--- CERES ---")
    try:
        da = load_ceres()
    except Exception as e:
        rd.log(f"  [WARN] CERES left out: {type(e).__name__}: {e}")
        return
    da = stats.select_years(da.where(masks.land_mask(da)).rename("CERES"), CERES_YEARS)
    raw, anom, base = october_series(da, CERES_YEARS)
    out = xr.Dataset({"series": anom, "series_abs": raw})
    out.attrs = {"dataset": "CERES", "years": f"{CERES_YEARS[0]}-{CERES_YEARS[1]}",
                 "series_month": MONTH, **series_attrs(raw, base)}
    out.to_netcdf(data_file("CERES"))
    rd.log(f"  regional {out.attrs['series_trend']:+.2f}, {stats.p_text(out.attrs['series_p'])}")


# ---------------------------------------------------------------- plot
def read_all():
    out = {}
    for name in SERIES:
        if data_file(name).exists():
            with xr.open_dataset(data_file(name)) as ds:
                out[name] = ds.load()
    return out


def tables(res):
    rows, srows = [], []
    for name in ORDER:
        if name not in res:
            continue
        ds = res[name]
        for m in range(1, 13):
            o = rd.cut(ds.sel(month=m), BOX)
            rows.append({"dataset": name, "month": m,
                         "years": f"{int(o.first_year)}-{int(o.last_year)}",
                         **{f"{k}_W_m2_per_decade" if k != "n_cells" else k: v
                            for k, v in stats.describe(o.trend).items()},
                         "median_n_eff": stats.wmedian(o.n_eff),
                         "pct_area_sig_fdr": stats.pct_area(o.sig.astype(bool), o.trend.notnull())})
    for name in SERIES:
        if name in res:
            s = res[name]
            srows += [{"dataset": name, "year": int(y), "sw_down_W_m2": float(a),
                       "anomaly_W_m2": float(b)}
                      for y, a, b in zip(s.year.values, s.series_abs.values, s.series.values)]
    d = config.RESULTS / OUT
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / "sw_down_trends_southern_africa.csv", index=False)
    pd.DataFrame(srows).to_csv(d / "sw_down_october_series.csv", index=False)
    print(f"Saved: {d / 'sw_down_trends_southern_africa.csv'}, sw_down_october_series.csv",
          flush=True)


def plot():
    import cartopy.crs as ccrs
    res = read_all()
    if not any(n in res for n in ORDER):
        raise SystemExit("no data files: run 'python sw_down_trends.py compute' first")
    tables(res)
    ps.setup()
    fs = config.FS
    years = {n: ps.year_list(res[n].attrs["years"]) for n in res}
    W = config.WIDTH["full"]
    left, right = 0.62, 0.15
    box_top, box_h = 0.55, 1.2
    gap, ncol = 0.1, len(ORDER)                           # one row of maps
    mw = (W - left - right - (ncol - 1) * gap) / ncol
    r = config.REGIONS[MAP]
    mh = mw * (r["lat"][1] - r["lat"][0]) / (r["lon"][1] - r["lon"][0])
    head = 0.5
    map_top = box_top + box_h + 0.55 + head
    cb_top = map_top + mh + 0.3
    ts_top, ts_h = cb_top + 0.85, 1.05
    H = ts_top + ts_h + 1.15
    fig, at = ps.figure("full", H)

    # a) monthly boxplots over Southern Africa
    names = [n for n in ORDER if n in res]
    ax = at(left, box_top, W - left - right, box_h)
    ps.monthly_boxplot(ax, {n: {m: rd.cut(res[n].trend.sel(month=m), BOX) for m in range(1, 13)}
                            for n in names}, names,
                       f"Trend, area-weighted\n({UNITS} decade$^{{-1}}$)")
    ps.letter(ax, 0)
    ps.dataset_legend(ax, names, {n: years[n] for n in names}, loc="lower center",
                      bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.2, columnspacing=1.2,
                      borderaxespad=0.2)
    feb = [n for n in names if int(res[n].first_year.sel(month=2)) != years[n][0]]
    if feb:
        ax.text(1.0, -0.2, "; ".join(f"{ps.label(n)} Feb {int(res[n].first_year.sel(month=2))}-"
                                        f"{int(res[n].last_year.sel(month=2))}" for n in feb),
                transform=ax.transAxes, ha="right", va="top", fontsize=fs["small"],
                color="0.3")

    # b)-f) October maps
    norm = ps.sym_norm([res[n].trend.sel(month=MONTH) for n in names])
    ps.text(fig, W / 2, map_top - head - 0.08, f"{calendar.month_name[MONTH]} trends",
            ha="center", fontsize=fs["title"], fontweight="bold")
    for i, name in enumerate(names):
        x0, t0 = left + i * (mw + gap), map_top
        o = res[name].sel(month=MONTH)
        axm = at(x0, t0, mw, mh, projection=ccrs.PlateCarree())
        ps.map_panel(axm, o.trend, norm, sig=o.sig.astype(bool).values, region=MAP, box=BOX,
                     left=(i == 0), bottom=True, xticks=[15, 30])
        ps.letter(axm, i + 1)
        ps.text(fig, x0 + mw / 2, t0 - 0.3, ps.label(name), ha="center",
                fontsize=fs["title"], fontweight="bold", color=ps.colour(name))
        ps.text(fig, x0 + mw / 2, t0 - 0.03,
                f"{ps.span(years[name])}\n"
                f"n$_{{eff}}$ ≈ {stats.wmedian(o.n_eff):.0f}/{len(years[name])}, "
                f"{stats.pct_area(o.sig.astype(bool), o.trend.notnull()):.0f}% sig.",
                ha="center", fontsize=fs["small"], color="0.25", linespacing=1.1)
    ps.colourbar(fig, at(left + 0.6, cb_top, W - left - right - 1.2, 0.1), norm, "RdBu_r",
                 f"{calendar.month_name[MONTH]} SW down trend ({UNITS} decade$^{{-1}}$); "
                 f"dots: significant (FDR q = {config.ALPHA_FDR:.2f})")

    # g) October regional anomaly series
    ax = at(left, ts_top, W - left - right, ts_h)
    snames = [n for n in SERIES if n in res]
    for name in snames:
        s, a = res[name].series, res[name].attrs
        ax.plot(s.year, s, color=ps.colour(name), lw=1.2,
                label=f"{ps.label(name)}: {a['series_trend']:+.2f}, {stats.p_text(a['series_p'])}, "
                      f"n$_{{eff}}$ = {a['series_n_eff']:.0f}")
    ax.axhline(0, color="black", lw=0.6, ls="--", zorder=0)
    ax.set_xlim(min(years[n][0] for n in snames), max(years[n][-1] for n in snames))
    ax.set_ylabel(f"{calendar.month_abbr[MONTH]} anomaly ({UNITS})")
    ax.grid(alpha=0.3, linewidth=0.4)
    ps.letter(ax, len(names) + 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False,
              handlelength=1.2, columnspacing=1.0,
              title=f"Regional trend ({UNITS} decade$^{{-1}}$)", title_fontsize=fs["legend"])
    note = (f"; CERES EBAF {res['CERES'].attrs['series_base']}" if "CERES" in res else "")
    ps.text(fig, W / 2, ts_top - 0.05,
            f"{calendar.month_name[MONTH]} land anomaly, Southern Africa box "
            f"(from {config.BASELINE[0]}-{config.BASELINE[1]}{note})",
            ha="center", fontsize=fs["label"], fontweight="bold")
    ps.save(fig, config.RESULTS / OUT / "sw_down_trends_southern_africa.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "compute" or (what == "all" and not all(data_file(n).exists() for n in SERIES)):
        compute()
    if what in ("plot", "all"):
        plot()
