"""
October composites on a southern African circulation index, ERA5, 1979-2025. Two indices:

  python chi_composites.py                   200 hPa velocity potential (chi200) index
  python chi_composites.py --index omega     500 hPa vertical velocity (omega) index
  python chi_composites.py --index t2m       2 m temperature index (land cells)
  python chi_composites.py --index swdn      downwelling SW index (land cells)

INDEX  October anomaly (from config.BASELINE, 1991-2020) averaged over the Southern Africa box
  (10-40E, 35-10S; cos-latitude weights). High years: index >= its 90th percentile; low years:
  index <= its 10th percentile (5 of 47 years each).
  chi200: positive anomalies usually mean anomalous upper-level convergence over the box (more
  descent), negative upper-level divergence (more ascent).
  omega (OMEGA_LEVEL hPa): positive = descent, so the high years are the most subsiding
  Octobers. t2m and swdn: the warmest and the sunniest Octobers (box mean over land cells,
  as the regional series of the paper). With every index but chi200, chi200 is added to the
  composites as the first row.

COMPOSITES  for the high and low years, the mean October anomaly of
    omega at OMEGA_LEVEL hPa (Pa s-1, positive = descent), total cloud cover (%), 2 m
    temperature (K), downwelling SW at the surface (W m-2), the surface radiation budget
    residual Q_rad - H_L - H_S (W m-2, land), latent heat flux H_L (W m-2, upward, land) and
    SST (K)
  and high minus low. Significance: the composite is compared with N_RANDOM composites of the
  same number of randomly drawn years (for high minus low: two disjoint random sets), p = share
  of random composites at least as large; black dots: significant after Benjamini-Hochberg FDR
  (q = config.ALPHA_FDR) over each map.
  Everything is done twice (VARIANTS): with the trend, and with the linear 1979-2025 trend
  removed from the index and every field first, so that the detrended composites show
  year-to-year variability only (with the trend in, the high and low years can cluster at the
  two ends of the record and the composites then partly show the long-term change).

CLIMATE INDICES  from the same ERA5 SST: Nino 3.4 (5S-5N, 170-120W) and the Dipole Mode Index
  (50-70E, 10S-10N minus 90-110E, 10S-0), October anomalies from config.BASELINE. Their
  correlations with the index are computed on detrended series (stats.corr_and_p:
  lag-1 n_eff).

Run:  python chi_composites.py [--index omega]            compute (if needed) and plot
      python chi_composites.py compute [--index omega]    re-read the raw data
      python chi_composites.py plot [--index omega]       plot from the data files only

FIGURES  results/<index>_composites/ (omega: omega<OMEGA_LEVEL>_composites, e.g. omega300_,
  so the levels do not overwrite each other); names start with the index name; each twice,
  the detrended versions end in _detrended
  <index>_index_october.png         the index, the 10th/90th percentiles and the chosen years
  <index>_composites_october.png  rows = [chi200 for the omega index,] omega, cloud, T2m, SW down, residual, H_L; columns =
                                high, low, high minus low (Africa, config.REGIONS['composite'])
  <index>_composites_sst_october.png  SST (shading) and chi200 (contours) composites, 60S-60N
  <index>_enso_iod_october.png  the detrended index with the other box index (chi200 or omega),
                                Nino 3.4 and the DMI: standardised series, and scatter plots
                                with r and p
TABLE  <index>_composites_october[_detrended].csv: box means of the composites and the share of
  the box that is significant

DATA (ERA5 monthly means, config.RAW)
  chi200    era5_chi: chi_october_pl.nc (as velocity_potential.py)
  omega     era5/global/vertical_velocity/global_vertical_velocity_YYYY.nc: w
  cloud     era5/global_total_cloud_cover.nc: tcc if present, otherwise the total cloud from the
            3-D cloud fraction (cloud_fraction_trends.py, its cache; 10-40E, 35S-18N only)
  T2m, SW down, fluxes   era5/global_<name>.nc as temperature_trends.py, sw_down_trends.py and
            global_flux_trends.py (J m-2 per day -> W m-2; H_L = -slhf, H_S = -sshf)
  SST       era5_sst: global_era5_ssts.nc: sst (averaged to 1 degree)
"""
from __future__ import annotations

import calendar
import sys

import numpy as np
import pandas as pd
import xarray as xr

import config
import masks
import plotstyle as ps
import readers as rd
import stats

MONTH = 10
YEARS = (1979, 2025)
BOX = "southern_africa"
MAP = "composite"                 # Africa maps
SST_MAP = "sst_band"              # SST maps
OMEGA_LEVEL = 300.0               # hPa
PCT = 10                          # high: >= (100 - PCT)th percentile, low: <= PCT-th
VARIANTS = (False, True)          # composites without and with the linear trend removed
DETREND = False                   # the variant being computed/plotted (set by the loop)
N_RANDOM = 5000                   # random composites for the significance test
SST_COARSEN = 4                   # 0.25 -> 1 degree for the SST maps
INDEX = "chi"                     # index variable: "chi" or "omega" (python ... --index omega)
OUT = "chi_composites"            # set from INDEX in set_index()
DAY = 86400.0
SEED = 1

# key: (row label, units, colour map, land only)
VARS = {
    "omega": (f"$\\omega$ {OMEGA_LEVEL:g} hPa", "Pa s$^{-1}$", "RdBu_r", False),
    "cloud": ("Total cloud", "%", "BrBG", False),
    "t2m": ("2 m temperature", "K", "RdBu_r", False),
    "swdn": ("SW down", "W m$^{-2}$", "RdBu_r", False),
    "res": ("$Q_{rad} - H_L - H_S$", "W m$^{-2}$", "PuOr_r", True),
    "lh": ("$H_L$", "W m$^{-2}$", "BrBG", True),
}
# climate indices from the ERA5 SST (October anomalies from config.BASELINE, area means)
NINO34 = dict(lon=(-170.0, -120.0), lat=(-5.0, 5.0))
DMI_WEST = dict(lon=(50.0, 70.0), lat=(-10.0, 10.0))
DMI_EAST = dict(lon=(90.0, 110.0), lat=(-10.0, 0.0))

# index definitions: TeX symbol, long name, units, what a positive value means
# land: the box mean uses land cells only (as the regional series of the paper)
INDICES = {
    "chi": dict(tex="\\chi_{200}", name="200 hPa velocity potential",
                unit="10$^6$ m$^2$ s$^{-1}$", plain="10^6 m2 s-1", land=False,
                sign="positive: upper-level convergence"),
    "omega": dict(tex=f"\\omega_{{{OMEGA_LEVEL:g}}}", name=f"{OMEGA_LEVEL:g} hPa vertical velocity",
                  unit="Pa s$^{-1}$", plain="Pa s-1", land=False,
                  sign="positive: descent (most subsiding years)"),
    "t2m": dict(tex="T_{2m}", name="2 m temperature (land)", unit="K", plain="K", land=True,
                sign="high: the warmest Octobers"),
    "swdn": dict(tex="SW_{\\downarrow}", name="downwelling SW at the surface (land)",
                 unit="W m$^{-2}$", plain="W m-2", land=True,
                 sign="high: the sunniest Octobers"),
}
SERIES = {"chi": "$\\chi_{200}$ index", "omega": f"$\\omega_{{{OMEGA_LEVEL:g}}}$ index",
          "t2m": "$T_{2m}$ index", "swdn": "SW$_{\\downarrow}$ index",
          "nino34": "Ni\u00f1o 3.4", "dmi": "Dipole Mode Index"}
SERIES_UNIT = {"chi": "10$^6$ m$^2$ s$^{-1}$", "omega": "Pa s$^{-1}$", "t2m": "K",
               "swdn": "W m$^{-2}$", "nino34": "K", "dmi": "K"}
SERIES_COL = {"chi": "#808BC5", "omega": "#245E55", "t2m": "#C63F3E", "swdn": "#EAC119",
              "nino34": "#ED773C", "dmi": "#9ED6DF"}


def set_index(name):
    """Select the index variable; for every index but chi200, chi200 is added as the first
    composite row."""
    global INDEX, OUT
    if name not in INDICES:
        raise SystemExit(f"--index must be one of {list(INDICES)}")
    INDEX = name
    OUT = f"omega{OMEGA_LEVEL:g}_composites" if name == "omega" else f"{name}_composites"
    if name != "chi" and "chi_map" not in VARS:                     # chi200 as the first row
        rows = dict(VARS)
        VARS.clear()
        VARS.update({"chi_map": ("$\\chi_{200}$", "10$^6$ m$^2$ s$^{-1}$", "RdBu_r", False)}, **rows)


def ix():
    return INDICES[INDEX]["tex"]


def suffix():
    return "_detrended" if DETREND else ""


def data_file(key):
    return config.DATA / OUT / f"{key}{suffix()}.nc"


def index_file():
    return config.DATA / OUT / "climate_indices.nc"


# ---------------------------------------------------------------- reading (October, year, lat, lon)
def october(da):
    da = da.sel(month=MONTH, drop=True) if "month" in da.dims else da
    da = da.drop_vars([c for c in ("lev",) if c in da.coords and c not in da.dims])
    return da.sel(year=slice(*YEARS))


def era5_file(name, var, region, scale=1.0):
    return october(rd.single_file(config.RAW["era5"] / f"global_{name}.nc", var, region,
                                  scale=scale, months=[MONTH]))


def load_chi():
    """ERA5 October chi at 200 hPa (as velocity_potential.py), in 10^6 m2 s-1."""
    with xr.open_dataset(config.RAW["era5_chi"]) as ds:
        da = rd.standardise(next(iter(ds.data_vars.values())))
        da = rd.pick_levels(da, 200.0).load()
    return october(da) * 1e-6


def load_omega():
    return october(rd.era5_yearly(config.RAW["era5"] / "global" / "vertical_velocity",
                                  "global_vertical_velocity_*.nc", "w", MAP, lev=OMEGA_LEVEL,
                                  months=[MONTH]))


def load_cloud():
    f = config.RAW["era5"] / "global_total_cloud_cover.nc"
    if f.exists():
        return era5_file("total_cloud_cover", "tcc", MAP, scale=100.0)
    rd.log("  no global_total_cloud_cover.nc: total cloud from the 3-D cloud fraction "
           "(cloud_fraction_trends.py cache, 10-40E only)")
    import cloud_fraction_trends as cft
    return october(cft.load_era5()["total"])


def load_fluxes():
    f = lambda n, v: era5_file(n, v, MAP, scale=1 / DAY)
    lh = -f("surface_latent_heat_flux", "slhf")
    sh = -f("surface_sensible_heat_flux", "sshf")
    qrad = f("surface_net_solar_radiation", "ssr") + f("surface_net_thermal_radiation", "str")
    return {"lh": lh, "res": qrad - lh - sh}


def load_sst():
    sst = october(rd.single_file(config.RAW["era5_sst"], "sst", SST_MAP, months=[MONTH]))
    if SST_COARSEN > 1:
        sst = sst.coarsen(lat=SST_COARSEN, lon=SST_COARSEN, boundary="trim").mean()
    return sst


def load_all():
    out = {"omega": load_omega(), "cloud": load_cloud(),
           "t2m": era5_file("2m_temperature", "t2m", MAP),
           "swdn": era5_file("surface_solar_radiation_downwards", "ssrd", MAP, scale=1 / DAY)}
    out.update(load_fluxes())
    out["sst"] = load_sst()
    return out


# ---------------------------------------------------------------- statistics
def anomaly(da):
    """October anomaly from the baseline (detrended if DETREND)."""
    b0, b1 = config.BASELINE
    a = da - da.sel(year=slice(b0, b1)).mean("year")
    return stats.detrend(a) if DETREND else a


def random_weights(n, k, rng):
    """(N_RANDOM, n) weights: mean of k random years; and the difference of two disjoint
    random sets of k years."""
    one = np.zeros((N_RANDOM, n))
    two = np.zeros((N_RANDOM, n))
    for b in range(N_RANDOM):
        idx = rng.permutation(n)
        one[b, idx[:k]] = 1.0 / k
        two[b, idx[:k]] = 1.0 / k
        two[b, idx[k:2 * k]] = -1.0 / k
    return one, two


def random_p(anom, obs, w, chunk=20000):
    """Share of random composites (weights w) with |value| >= |obs|, per cell."""
    x = anom.values.reshape(anom.sizes["year"], -1)
    o = np.abs(obs.values.ravel())
    ok = np.isfinite(x).all(axis=0)
    p = np.full(x.shape[1], np.nan)
    cols = np.flatnonzero(ok)
    for i in range(0, cols.size, chunk):
        c = cols[i:i + chunk]
        r = np.abs(w @ x[:, c])
        p[c] = ((r >= o[c][None, :] - 1e-12).sum(axis=0) + 1) / (w.shape[0] + 1)
    return obs.copy(data=p.reshape(obs.shape))


def composite(da, high, low, w1, w2, land=False):
    """Dataset with high, low, diff composites and their p and FDR significance."""
    a = anomaly(da)
    if land:
        a = a.where(masks.land_mask(a))
    a = a.where(np.isfinite(a).all("year"))                       # complete cells only
    hi = a.sel(year=high).mean("year")
    lo = a.sel(year=low).mean("year")
    out = {"high": hi, "low": lo, "diff": hi - lo}
    ds = xr.Dataset({k: v.astype("float32") for k, v in out.items()})
    for k, w in (("high", w1), ("low", w1), ("diff", w2)):
        p = random_p(a, out[k], w)
        ds[f"p_{k}"] = p.astype("float32")
        ds[f"sig_{k}"] = stats.fdr(p).astype("int8")
    return ds


def box_mean(field, land=False):
    f = rd.cut(field, BOX)
    if land:
        f = f.where(masks.land_mask(f))
    return float(stats.area_mean(f))


# ---------------------------------------------------------------- compute
def region_mean(da, region):
    return stats.area_mean(rd.cut(da, region)).reset_coords(drop=True)


def anom(da):
    return da - da.sel(year=slice(*config.BASELINE)).mean("year")


def box_index(key, chi, fields):
    """October box-mean anomaly of one index variable (land cells only where INDICES says)."""
    a = anom(chi if key == "chi" else fields[key])
    if INDICES[key]["land"]:
        a = a.where(masks.land_mask(a))
    return region_mean(a, BOX)


def climate_indices(chi, fields):
    """October box indices of chi200, omega and the selected index, Nino 3.4 and the Dipole
    Mode Index (anomalies, not detrended) -> data file."""
    a = anom(fields["sst"])
    series = {k: box_index(k, chi, fields) for k in dict.fromkeys(["chi", "omega", INDEX])}
    ds = xr.Dataset({**series, "nino34": region_mean(a, NINO34),
                     "dmi": region_mean(a, DMI_WEST) - region_mean(a, DMI_EAST)})
    ds = ds.dropna("year", how="any")
    ds.to_netcdf(index_file())
    for k in [k for k in ds.data_vars if k != INDEX]:
        r, p = stats.corr_and_p(stats.detrend(ds[k]), stats.detrend(ds[INDEX]))
        rd.log(f"  detrended r({INDEX} index, {k}) = {float(r):+.2f}, {stats.p_text(float(p))} "
               f"({ds.sizes['year']} Octobers)")
    return ds


def compute_variant(chi, fields):
    """Index, high/low years and composites for the current DETREND setting."""
    rd.log(f"=== {'detrended' if DETREND else 'with the trend'} ===")
    index = box_index(INDEX, chi, fields)
    if DETREND:
        index = stats.detrend(index)
    yrs = index.year.values
    if yrs[0] > YEARS[0] or yrs[-1] < YEARS[1]:
        rd.log(f"  [NOTE] the {INDEX} index covers {yrs[0]}-{yrs[-1]}, not all of {YEARS[0]}-{YEARS[1]}")
    v = index.values
    t_hi, t_lo = np.percentile(v, 100 - PCT), np.percentile(v, PCT)
    high = yrs[v >= t_hi]
    low = yrs[v <= t_lo]
    r_year = float(np.corrcoef(yrs, v)[0, 1])
    rd.log(f"  {len(yrs)} years; P{100 - PCT} = {t_hi:+.3g}, P{PCT} = {t_lo:+.3g} {INDICES[INDEX]['plain']}")
    rd.log(f"  high years: {', '.join(map(str, high))}")
    rd.log(f"  low years:  {', '.join(map(str, low))}")
    rd.log(f"  correlation of the index with time: r = {r_year:+.2f}")
    xr.Dataset({"index": index, "high": ("year", np.isin(yrs, high)),
                "low": ("year", np.isin(yrs, low))},
               attrs={"p_high": t_hi, "p_low": t_lo, "r_with_time": r_year,
                      "detrended": int(DETREND)}).to_netcdf(data_file("index"))

    rng = np.random.default_rng(SEED)
    w1, w2 = random_weights(len(yrs), len(high), rng)
    for key, da in fields.items():
        da = da.sel(year=yrs)
        land = VARS.get(key, (0, 0, 0, False))[3]
        rd.log(f"--- composite: {key} ---")
        ds = composite(da, high, low, w1, w2, land=land)
        ds.attrs = {"high_years": ",".join(map(str, high)), "low_years": ",".join(map(str, low))}
        ds.to_netcdf(data_file(key))


def compute():
    global DETREND
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    rd.log("--- reading ERA5 (October) ---")
    chi = load_chi()
    fields = load_all()
    c = rd.cut(chi, SST_MAP)                                       # contours on the SST maps,
    n = max(1, int(round(1.0 / float(abs(c.lat[1] - c.lat[0])))))   # at about 1 degree
    fields["chi"] = c.coarsen(lat=n, lon=n, boundary="trim").mean() if n > 1 else c
    if "chi_map" in VARS:
        fields["chi_map"] = rd.cut(chi, MAP)
    rd.log("--- climate indices (from the ERA5 SST) ---")
    climate_indices(chi, fields)
    for DETREND in VARIANTS:
        compute_variant(chi, fields)
    DETREND = False


# ---------------------------------------------------------------- plot
def read_all():
    return {k: xr.load_dataset(data_file(k)) for k in list(VARS) + ["sst", "chi", "index"]
            if data_file(k).exists()}


def years_text(ds, which):
    return ds.attrs[f"{which}_years"].replace(",", ", ")


def summary_csv(res):
    rows = []
    for k, (lab, unit, _, land) in VARS.items():
        if k not in res:
            continue
        ds = res[k]
        rows.append({"variable": k, "units": unit.replace("$", "").replace("^{-1}", "-1")
                     .replace("^{-2}", "-2"), "land_only": land,
                     **{f"box_{c}": box_mean(ds[c], land) for c in ("high", "low", "diff")},
                     **{f"pct_area_sig_{c}": stats.pct_area(rd.cut(ds[f"sig_{c}"], BOX).astype(bool),
                                                            rd.cut(ds[c], BOX).notnull())
                        for c in ("high", "low", "diff")}})
    path = config.RESULTS / OUT / f"{INDEX}_composites_october{suffix()}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, float_format="%.4g")
    print(f"Saved: {path}", flush=True)
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(df.round(3).to_string(index=False))


def plot_index(res):
    import matplotlib.pyplot as plt
    fs, W = config.FS, config.WIDTH["full"]
    d = res["index"]
    fig, at = ps.figure("full", 2.6)
    ax = at(0.7, 0.55, W - 1.05, 1.55)
    yrs, v = d.year.values, d["index"].values
    col = np.where(d.high, "#C63F3E", np.where(d.low, "#245E55", "0.75"))
    ax.bar(yrs, v, color=col, width=0.8, edgecolor="none")
    for t, c, lab in ((d.attrs["p_high"], "#C63F3E", f"P{100 - PCT}"),
                      (d.attrs["p_low"], "#245E55", f"P{PCT}")):
        ax.axhline(t, color=c, lw=0.8, ls="--")
        ax.text(1.005, t, lab, color=c, va="center", ha="left", fontsize=fs["small"],
                transform=ax.get_yaxis_transform())
    ax.axhline(0, color="k", lw=0.5)
    for y, val, h in zip(yrs, v, d.high.values | d.low.values):
        if h:
            ax.text(y, val + (0.04 if val > 0 else -0.04) * np.ptp(v), str(y), rotation=90,
                    ha="center", va="bottom" if val > 0 else "top", fontsize=fs["small"] - 1.5)
    ax.set_xlim(yrs[0] - 1, yrs[-1] + 1)
    lo, hi = np.nanmin(v), np.nanmax(v)
    ax.set_ylim(lo - 0.3 * np.ptp(v), hi + 0.3 * np.ptp(v))       # room for the year labels
    ax.set_ylabel(f"${ix()}$ anomaly\n({INDICES[INDEX]['unit']})")
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    ps.text(fig, W / 2, 0.08, f"{calendar.month_name[MONTH]} {INDICES[INDEX]['name']} over "
            f"Southern Africa (10-40°E, 35-10°S), ERA5" + (" (detrended)" if DETREND else ""),
            ha="center", va="top", fontsize=fs["title"], fontweight="bold")
    ps.text(fig, W / 2, 0.33, f"anomaly from {config.BASELINE[0]}-{config.BASELINE[1]}; "
            f"{INDICES[INDEX]['sign']}; r with time = {d.attrs['r_with_time']:+.2f}",
            ha="center", va="top",
            fontsize=fs["small"], color="0.3")
    ps.save(fig, config.RESULTS / OUT / f"{INDEX}_index_october{suffix()}.png")


def plot_composites(res):
    import cartopy.crs as ccrs
    fs, W = config.FS, config.WIDTH["full"]
    keys = [k for k in VARS if k in res]
    r = config.REGIONS[MAP]
    aspect = (r["lat"][1] - r["lat"][0]) / (r["lon"][1] - r["lon"][0])
    left, right, gx, gy, cbw = 0.55, 0.12, 0.08, 0.12, 0.72
    pw = (W - left - right - cbw - 2 * gx) / 3
    mh = pw * aspect
    top = 0.95
    H = top + len(keys) * (mh + gy) + 0.25
    fig, at = ps.figure("full", H)
    ref = res[keys[0]]
    heads = [(f"High ${ix()}$ (≥ P{100 - PCT})", years_text(ref, "high")),
             (f"Low ${ix()}$ (≤ P{PCT})", years_text(ref, "low")),
             ("High minus low", "")]
    for c, (h, y) in enumerate(heads):
        x0 = left + c * (pw + gx) + pw / 2
        ps.text(fig, x0, top - 0.3, h, ha="center", fontsize=fs["title"], fontweight="bold")
        if y:
            ps.text(fig, x0, top - 0.1, y, ha="center", fontsize=fs["small"] - 1, color="0.3")
    k = 0
    for i, key in enumerate(keys):
        lab, unit, cmap, _ = VARS[key]
        ds = res[key]
        t0 = top + i * (mh + gy)
        norm = ps.sym_norm([ds["high"], ds["low"], ds["diff"]], cmap=cmap)
        ps.text(fig, left - 0.42, t0 + mh / 2, lab, rotation=90, ha="center", va="center",
                fontsize=fs["label"], fontweight="bold")
        for c, which in enumerate(("high", "low", "diff")):
            ax = at(left + c * (pw + gx), t0, pw, mh, projection=ccrs.PlateCarree())
            ps.map_panel(ax, ds[which], norm, cmap=cmap, sig=ds[f"sig_{which}"].astype(bool).values,
                         region=MAP, left=(c == 0), bottom=(i == len(keys) - 1), box=BOX,
                         xticks=[0, 30], yticks=[-30, 0])
            ps.letter(ax, k)
            k += 1
        ps.colourbar(fig, at(left + 3 * pw + 2 * gx + 0.12, t0 + 0.08, 0.08, mh - 0.16), norm,
                     cmap, unit, orientation="vertical", nticks=5)
    ps.text(fig, W / 2, 0.06, f"{calendar.month_name[MONTH]} composites on the Southern "
            f"Africa ${ix()}$ index, ERA5" + (" (detrended)" if DETREND else ""),
            ha="center", va="top", fontsize=fs["suptitle"], fontweight="bold")
    ps.text(fig, W / 2, H - 0.12, f"anomalies from {config.BASELINE[0]}-{config.BASELINE[1]}; "
            f"dots: significant against random {len(years_text(ref, 'high').split(','))}-year "
            f"composites (FDR q = {config.ALPHA_FDR:.2f}); dashed: Southern Africa box",
            ha="center", va="top", fontsize=fs["small"] - 1, color="0.3")
    ps.save(fig, config.RESULTS / OUT / f"{INDEX}_composites_october{suffix()}.png")


def plot_sst(res):
    import cartopy.crs as ccrs
    if "sst" not in res:
        return
    fs, W = config.FS, config.WIDTH["full"]
    sst, chi = res["sst"], res.get("chi")
    r = config.REGIONS[SST_MAP]
    aspect = (r["lat"][1] - r["lat"][0]) / (r["lon"][1] - r["lon"][0])
    left, right, cbw, gy = 0.5, 0.1, 0.75, 0.32
    pw = W - left - right - cbw
    mh = pw * aspect
    top = 0.55
    H = top + 3 * (mh + gy) + 0.35
    fig, at = ps.figure("full", H)
    norm = ps.sym_norm([sst["high"], sst["low"], sst["diff"]], cmap="RdBu_r")
    titles = {"high": f"High ${ix()}$ years ({years_text(sst, 'high')})",
              "low": f"Low ${ix()}$ years ({years_text(sst, 'low')})",
              "diff": "High minus low"}
    for i, which in enumerate(("high", "low", "diff")):
        t0 = top + i * (mh + gy)
        ax = at(left, t0, pw, mh, projection=ccrs.PlateCarree())
        f = sst[which]                                 # an image: pcolormesh over 360 degrees
        dlo = float(abs(f.lon[1] - f.lon[0]))          # is very slow in cartopy
        dla = float(abs(f.lat[1] - f.lat[0]))
        ax.imshow(f.values, origin="lower", cmap="RdBu_r", norm=norm, interpolation="nearest",
                  extent=(float(f.lon[0]) - dlo / 2, float(f.lon[-1]) + dlo / 2,
                          float(f.lat[0]) - dla / 2, float(f.lat[-1]) + dla / 2),
                  transform=ccrs.PlateCarree(), zorder=1)
        ps.sig_dots(ax, f, sst[f"sig_{which}"].astype(bool).values)
        ps.map_frame(ax, SST_MAP, True, i == 2, BOX, xticks=[-120, -60, 0, 60, 120],
                     yticks=[-40, 0, 40])
        if chi is not None:
            c = chi[which]
            lim = float(np.nanpercentile(np.abs(np.concatenate(
                [chi[w].values.ravel() for w in ("high", "low", "diff")])), 98)) or 1.0
            lev = np.linspace(-lim, lim, 11)
            lev = lev[np.abs(lev) > 1e-9]
            ax.contour(c.lon, c.lat, c, levels=lev, colors="k", linewidths=0.5,
                       negative_linestyles="dashed", transform=ccrs.PlateCarree(), zorder=3)
        ax.set_title(titles[which], loc="left", fontsize=fs["small"] + 0.5, fontweight="bold",
                     pad=2)
        ps.letter(ax, i)
    ps.colourbar(fig, at(left + pw + 0.15, top + 0.2, 0.09, 3 * mh + 2 * gy - 0.4), norm,
                 "RdBu_r", "SST anomaly (K)", orientation="vertical", nticks=7)
    ps.text(fig, W / 2, 0.06, f"{calendar.month_name[MONTH]} SST and $\\chi_{{200}}$ composites "
            f"on the ${ix()}$ index, ERA5" + (" (detrended)" if DETREND else ""), ha="center", va="top",
            fontsize=fs["suptitle"], fontweight="bold")
    ps.text(fig, W / 2, 0.3, "shading: SST (dots: significant, FDR); contours: $\\chi_{200}$ "
            "(solid positive, dashed negative); dashed box: Southern Africa", ha="center",
            va="top", fontsize=fs["small"] - 1, color="0.3")
    ps.save(fig, config.RESULTS / OUT / f"{INDEX}_composites_sst_october{suffix()}.png")


def plot_indices():
    """Detrended index against the other box index, Nino 3.4 and the DMI: series and scatter
    plots."""
    if not index_file().exists():
        return
    global DETREND
    fs, W = config.FS, config.WIDTH["full"]
    ds = xr.load_dataset(index_file())
    keys = [INDEX] + [k for k in ("chi", "omega", "nino34", "dmi") if k != INDEX and k in ds]
    keys = keys[:5]
    det = {k: stats.detrend(ds[k]) for k in keys}
    z = {k: (v - v.mean()) / v.std() for k, v in det.items()}
    sel = None
    DETREND = True
    if data_file("index").exists():
        sel = xr.load_dataset(data_file("index"))
    DETREND = False
    others = keys[1:]
    fig, at = ps.figure("full", 5.5)
    ax = at(0.7, 0.62, W - 0.85, 1.5)
    for k in keys:
        ax.plot(ds.year, z[k], color="#1D1D1B" if k == INDEX else SERIES_COL[k],
                lw=1.6 if k == INDEX else 1.0, label=SERIES[k], marker="o" if k == INDEX else None, ms=2.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Standardised,\ndetrended")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=len(keys), frameon=False,
              fontsize=fs["legend"] - 0.5, columnspacing=1.0, handlelength=1.5)
    ps.letter(ax, 0)
    gx = 0.4
    sw = (W - 0.85 - gx * (len(others) - 1)) / len(others)
    for i, k in enumerate(others):
        ax = at(0.7 + i * (sw + gx), 2.75, sw, 1.75)
        x, y = det[k], det[INDEX]
        c = ["0.6"] * x.size
        if sel is not None:
            hi = set(sel.year.values[sel.high.values]); lo = set(sel.year.values[sel.low.values])
            c = ["#C63F3E" if yy in hi else "#245E55" if yy in lo else "0.6" for yy in ds.year.values]
        ax.scatter(x, y, c=c, s=10, edgecolors="none", zorder=3)
        b = float((x * y).sum() / (x * x).sum())
        xs = np.array([float(x.min()), float(x.max())])
        ax.plot(xs, b * xs, color="k", lw=0.8)
        r, p = stats.corr_and_p(x, y)
        ax.text(0.97, 0.04, f"r = {float(r):+.2f}, {stats.p_text(float(p))}",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=fs["small"] - 0.5,
                fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
        ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
        ax.set_xlabel(f"{SERIES[k]}\n({SERIES_UNIT[k]}, detrended)", fontsize=fs["small"])
        if i == 0:
            ax.set_ylabel(f"{SERIES[INDEX]}\n({SERIES_UNIT[INDEX]}, detrended)")
        ax.tick_params(labelsize=fs["tick"] - 1)
        ax.grid(alpha=0.3, lw=0.4)
        ps.letter(ax, i + 1)
    ps.text(fig, W / 2, 0.06, f"{calendar.month_name[MONTH]} ${ix()}$ index against other "
            f"indices, ERA5 {int(ds.year[0])}-{int(ds.year[-1])} (detrended)", ha="center",
            va="top", fontsize=fs["title"], fontweight="bold")
    ps.text(fig, W / 2, H_NOTE, "indices: Southern Africa box means; red/green dots: the "
            "high/low years of the detrended index;\n"
            "Ni\u00f1o 3.4: 5\u00b0S-5\u00b0N, 170-120\u00b0W; DMI: 50-70\u00b0E, 10\u00b0S-10\u00b0N "
            "minus 90-110\u00b0E, 10\u00b0S-0\u00b0", ha="center", va="top",
            fontsize=fs["small"] - 1, color="0.3")
    ps.save(fig, config.RESULTS / OUT / f"{INDEX}_enso_iod_october.png")


H_NOTE = 5.12


def plot():
    global DETREND
    ps.setup()
    done = False
    for DETREND in VARIANTS:
        res = read_all()
        if "index" not in res:
            continue
        done = True
        rd.log(f"=== {'detrended' if DETREND else 'with the trend'} ===")
        summary_csv(res)
        plot_index(res)
        plot_composites(res)
        plot_sst(res)
    DETREND = False
    if not done:
        raise SystemExit(f"no data files: run 'python chi_composites.py compute --index {INDEX}' first")
    plot_indices()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="October composites on a Southern Africa "
                                 "chi200 or omega index (ERA5).")
    ap.add_argument("what", nargs="?", default="all", choices=["all", "compute", "plot"],
                    help="all (default): compute if needed, then plot")
    ap.add_argument("--index", default="chi", choices=list(INDICES))
    ap.add_argument("--compute", action="store_true", help="same as 'compute' and then plot")
    args = ap.parse_args()
    set_index(args.index)
    what = "all_force" if args.compute else args.what
    missing = not index_file().exists() or not all(
        data_file(k).exists() for k in list(VARS) + ["sst", "chi", "index"])
    if what in ("compute", "all_force") or (what == "all" and missing):
        compute()
    if what in ("plot", "all", "all_force"):
        plot()