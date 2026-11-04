"""
Checks of the assumptions behind the trend test used in every figure (stats.ols_stats):
an OLS linear trend, its residuals treated as AR(1), n_eff = n (1 - r1) / (1 + r1) (capped at
n) from the lag-1 autocorrelation r1, and a t test with n_eff - 2 degrees of freedom.

Run:  python stats_assumptions.py [compute | plot] [variable ...] [--month M]
      variables: t2m sw_down cf_total cf_high cf_mid cf_low qred qrad omega300 rh850
      (default all; October unless --month)

For every variable and dataset, October, over the Southern Africa box (10-40E, 35-10S, land
cells with a value in every year of the dataset's own years, config.years):
  PER CELL  each check at every cell; the % of area (cos latitude weights) where it fails at
            p < 0.05. If the assumption holds about 5 % fails by chance.
  BOX       the same checks on the area-weighted box-mean series, with all the numbers.

CHECKS (on the OLS residuals e)
  ar1_not_enough  Ljung-Box (lags 1..K, K = 5) on e_t - r1 e_(t-1): what AR(1) leaves.
                  Also r2 (lag-2 autocorrelation of e) next to r1^2, its AR(1) value.
  r1_bias         r1 is biased low for short series; Marriott-Pope r1c = (n r1 + 1) / (n - 3).
                  % of cells whose significance (p < 0.05) changes with r1c in n_eff.
  non_normal      D'Agostino-Pearson K^2 per cell; Shapiro-Wilk for the box series.
  heteroscedastic Breusch-Pagan: n R^2 of e^2 regressed on time (chi^2, 1 df).
  curvature       quadratic term of y ~ a + b t + c t^2 (t test with n_eff - 3 df).
  breakpoint      a step at an unknown year on top of the trend: sup-F (Andrews 1993) of
                  y ~ a + b t + d [year >= k] over k in the middle 70 % of the record, F scaled
                  by n_eff / n; break where sup F > 8.85 (Andrews' 5 % value, 15 % trimming).
                  (Pettitt on the raw series would count the trend itself as a change point.)
  small_n_eff     n_eff < 10.
  test_disagrees  significance differs from a Mann-Kendall test whose variance is inflated by
                  n / n_eff (the same AR(1) correction).
  box only        Theil-Sen slope, moving-block bootstrap p (blocks of BLOCK years, NBOOT
                  resamples), the largest Cook's distance and its year.

OUTPUTS
  data/stats_assumptions/cells_<month>.csv, box_<month>.csv   (the numbers)
  results/stats_assumptions/stats_assumptions_<MM>.png         a) % of area failing each check
      per variable and dataset; b) the box-series p of each check (bold: p < 0.05)
  results/stats_assumptions/stats_assumptions_box_<MM>.csv     the box table, for the paper
"""
from __future__ import annotations

import calendar
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats as sps

import config
import masks
import plotstyle as ps
import readers as rd
import stats

OUT = "stats_assumptions"
BOX = "southern_africa"
ALPHA = 0.05
LB_LAGS = 5
BLOCK = 4
NBOOT = 2000
SMALL_NEFF = 10
TRIM = 0.15
SUPF_CRIT = 8.85                 # Andrews (1993), 1 restriction, 15 % trimming, 5 %
CHECKS = [("ar1_not_enough", "AR(1) not\nenough"), ("r1_bias", "Sig. changes\nwith r1 bias"),
          ("non_normal", "Non-normal\nresiduals"), ("heteroscedastic", "Hetero-\nscedastic"),
          ("curvature", "Curvature"), ("breakpoint", "Step beyond\ntrend"),
          ("small_n_eff", "n$_{eff}$ < 10"), ("test_disagrees", "Sig. differs\nfrom MK")]


# ---------------------------------------------------------------- variables
def _t2m(name, m):
    import temperature_trends as tt
    return tt.load(name).sel(month=m)


def _sw(name, m):
    import sw_down_trends as sd
    return sd.load(name).sel(month=m)


def _cf(layer):
    def f(name, m):
        import cloud_fraction_trends as cf
        return cf.load(name)[layer].sel(month=m)
    return f


def _budget(full):
    def f(name, m):
        import radiation_budget_trends as rb
        comp = rb.components(name, full)
        return sum(rb.SIGN[k] * v for k, v in comp.items()).sel(month=m)
    return f


def _omega(name, m):
    import omega_trends as ot
    if m != ot.MONTH:
        raise ValueError("omega300 is only read for October")
    return ot.load(name).sel(lev=300.0).drop_vars("lev", errors="ignore")


def _rh(name, m):
    try:
        import rh_decomposition as rhd
    except ImportError:
        import rh as rhd
    return rhd.load(name, 850)["rh"].sel(month=m)


T2M = ["ERA5", "MERRA-2", "JRA-3Q", "NOAA", "CHIRTS", "CRUTEM5"]
REAN = ["ERA5", "MERRA-2", "JRA-3Q"]
VARIABLES = {
    "t2m": ("T2m", _t2m, T2M),
    "sw_down": ("SW$\\downarrow$", _sw, REAN + ["CLARA-A3", "SARAH-3"]),
    "cf_total": ("Total cloud", _cf("total"), REAN + ["CLARA-A3"]),
    "cf_high": ("High cloud", _cf("high"), REAN + ["CLARA-A3"]),
    "cf_mid": ("Middle cloud", _cf("mid"), REAN + ["CLARA-A3"]),
    "cf_low": ("Low cloud", _cf("low"), REAN + ["CLARA-A3"]),
    "qred": ("Q$_{red}$", _budget(True), REAN),
    "qrad": ("Q$_{rad}$", _budget(False), REAN + ["CLARA-A3"]),
    "omega300": ("ω$_{300}$", _omega, REAN),
    "rh850": ("RH$_{850}$", _rh, REAN),
}


# ---------------------------------------------------------------- the checks (vectorised)
def residual_stats(Y, years):
    """Y (n, m) complete columns. OLS pieces shared by the checks."""
    n = Y.shape[0]
    t = years - years.mean()
    sxx = (t ** 2).sum()
    b = (t[:, None] * (Y - Y.mean(0))).sum(0) / sxx
    e = Y - Y.mean(0) - b * t[:, None]
    ssr = (e ** 2).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r1 = (e[1:] * e[:-1]).sum(0) / ssr
        r2 = (e[2:] * e[:-2]).sum(0) / ssr
    return dict(n=n, t=t, sxx=sxx, b=b, e=e, ssr=ssr, r1=r1, r2=r2)


def santer_p(b, ssr, sxx, n, r1):
    with np.errstate(invalid="ignore", divide="ignore"):
        n_eff = np.minimum(n * (1 - r1) / (1 + r1), n)
        se = np.sqrt(ssr / (n_eff - 2) / sxx)
        p = 2 * sps.t.sf(np.abs(b / se), n_eff - 2)
    return np.where(n_eff > 2, p, np.nan), n_eff


def ljung_box(e, r1):
    u = e[1:] - r1 * e[:-1]
    u = u - u.mean(0)
    n = u.shape[0]
    den = (u ** 2).sum(0)
    q = np.zeros(u.shape[1])
    for k in range(1, LB_LAGS + 1):
        with np.errstate(invalid="ignore", divide="ignore"):
            rk = (u[k:] * u[:-k]).sum(0) / den
        q += rk ** 2 / (n - k)
    q *= n * (n + 2)
    return sps.chi2.sf(q, LB_LAGS - 1)


def breusch_pagan(e, t):
    e2 = e ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        r = ((e2 - e2.mean(0)) * (t - t.mean())[:, None]).sum(0) / np.sqrt(
            ((e2 - e2.mean(0)) ** 2).sum(0) * ((t - t.mean()) ** 2).sum())
    return sps.chi2.sf(e.shape[0] * r ** 2, 1)


def curvature_p(Y, s, n_eff):
    t = s["t"]
    t2 = t ** 2 - (t ** 2).mean()
    t2 = t2 - (t2 * t).sum() / s["sxx"] * t                     # orthogonal to t
    c = (t2[:, None] * s["e"]).sum(0) / (t2 ** 2).sum()
    eq = s["e"] - c * t2[:, None]
    n = s["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt((eq ** 2).sum(0) / (n_eff - 3) / (t2 ** 2).sum())
        p = 2 * sps.t.sf(np.abs(c / se), n_eff - 3)
    return np.where(n_eff > 3, p, np.nan)


def mann_kendall_p(Y, n_eff, chunk=4000):
    n = Y.shape[0]
    iu = np.triu_indices(n, 1)
    S = np.concatenate([np.sign(Y[iu[1], i:i + chunk] - Y[iu[0], i:i + chunk]).sum(0)
                        for i in range(0, Y.shape[1], chunk)])
    var = n * (n - 1) * (2 * n + 5) / 18 * (n / np.maximum(n_eff, 1))
    z = (S - np.sign(S)) / np.sqrt(var)
    return 2 * sps.norm.sf(np.abs(z))


def step_test(s, n_eff):
    """Sup-F test for a step at an unknown year on top of the trend (Andrews 1993): for every
    year k in the middle (1 - 2 TRIM) of the record, F of adding a step at k to y ~ a + b t;
    F scaled by n_eff / n for the autocorrelation; a break where max F > SUPF_CRIT.
    Returns (break flag, largest F, index of the step year)."""
    e, t, n = s["e"], s["t"], s["n"]
    lo = max(int(np.floor(TRIM * n)), 2)
    best, arg = np.zeros(e.shape[1]), np.zeros(e.shape[1], int)
    for k in range(lo, n - lo + 1):
        d = (np.arange(n) >= k).astype(float)
        d = d - d.mean()
        d = d - (d * t).sum() / s["sxx"] * t                       # orthogonal to 1 and t
        dd = (d ** 2).sum()
        red = (d[:, None] * e).sum(0) ** 2 / dd
        with np.errstate(invalid="ignore", divide="ignore"):
            F = red / ((s["ssr"] - red) / (n - 3)) * np.minimum(n_eff / n, 1)
        F = np.nan_to_num(F)
        better = F > best
        best, arg = np.where(better, F, best), np.where(better, k, arg)
    return best > SUPF_CRIT, best, arg


def all_checks(Y, years):
    """Every check for every column of Y (n, m); returns a dict of (m,) arrays."""
    s = residual_stats(Y, years)
    n = s["n"]
    p, n_eff = santer_p(s["b"], s["ssr"], s["sxx"], n, s["r1"])
    r1c = np.clip((n * s["r1"] + 1) / (n - 3), -0.99, 0.99)
    pc, _ = santer_p(s["b"], s["ssr"], s["sxx"], n, r1c)
    p_mk = mann_kendall_p(Y, n_eff)
    return dict(
        trend=10 * s["b"], p=p, r1=s["r1"], r1c=r1c, r2=s["r2"], n_eff=n_eff,
        p_ar1=ljung_box(s["e"], s["r1"]),
        p_normal=sps.normaltest(s["e"], axis=0).pvalue,
        p_bp=breusch_pagan(s["e"], s["t"]),
        p_curv=curvature_p(Y, s, n_eff),
        **dict(zip(("breakpoint", "sup_f", "step_index"), step_test(s, n_eff))),
        p_r1c=pc, p_mk=p_mk)


def flags(c):
    """Failure (True) of each check per column."""
    sig, sig_c, sig_mk = c["p"] < ALPHA, c["p_r1c"] < ALPHA, c["p_mk"] < ALPHA
    return {"ar1_not_enough": c["p_ar1"] < ALPHA, "r1_bias": sig != sig_c,
            "non_normal": c["p_normal"] < ALPHA, "heteroscedastic": c["p_bp"] < ALPHA,
            "curvature": c["p_curv"] < ALPHA, "breakpoint": c["breakpoint"],
            "small_n_eff": c["n_eff"] < SMALL_NEFF, "test_disagrees": sig != sig_mk}


def box_extras(y, years, c):
    """Box series only: Shapiro, Theil-Sen, block bootstrap, Cook's distance, step year."""
    n = y.size
    t = years - years.mean()
    b = (t * (y - y.mean())).sum() / (t ** 2).sum()
    e = y - y.mean() - b * t
    out = {"p_shapiro": float(sps.shapiro(e).pvalue),
           "theil_sen": 10 * float(sps.theilslopes(y, years)[0])}
    rng = np.random.default_rng(0)
    nb = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, size=(NBOOT, nb))
    idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(NBOOT, -1)[:, :n]
    eb = e[idx]                                                  # null: no trend
    bb = (eb * t).sum(1) / (t ** 2).sum()
    out["p_bootstrap"] = float((np.abs(bb) >= abs(b)).mean())
    h = 1 / n + t ** 2 / (t ** 2).sum()
    s2 = (e ** 2).sum() / (n - 2)
    d = e ** 2 / (2 * s2) * h / (1 - h) ** 2
    out["cooks_max"], out["cooks_year"] = float(d.max()), int(years[d.argmax()])
    out["step_year"] = int(years[int(c["step_index"])])
    return out


# ---------------------------------------------------------------- compute
def compute(variables, month):
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    cell_rows, box_rows = [], []
    for var in variables:
        label, loader, names = VARIABLES[var]
        for name in names:
            rd.log(f"--- {var}, {name} ---")
            try:
                da = loader(name, month)
            except Exception as e:
                rd.log(f"  [WARN] left out: {type(e).__name__}: {e}")
                continue
            da = rd.cut(da, BOX)
            da = da.where(masks.land_mask(da))
            yrs = config.years(name, month)
            da = stats.select_years(da.rename(f"{var} {name}"), yrs)
            da = stats.complete(da).transpose("year", "lat", "lon")
            years = da.year.values.astype(float)
            Y = da.values.reshape(da.sizes["year"], -1)
            ok = np.isfinite(Y).all(0)
            if not ok.any():
                rd.log("  [WARN] no complete cells")
                continue
            w = np.broadcast_to(np.cos(np.deg2rad(da.lat.values))[:, None],
                                (da.sizes["lat"], da.sizes["lon"])).ravel()[ok]
            c = all_checks(Y[:, ok], years)
            f = flags(c)
            row = {"variable": var, "dataset": name, "years": f"{yrs[0]}-{yrs[1]}",
                   "n_cells": int(ok.sum()), "median_r1": float(np.median(c["r1"])),
                   "median_r1c": float(np.median(c["r1c"])),
                   "median_n_eff": float(np.median(c["n_eff"])),
                   "pct_sig_trend": 100 * float((w * (c["p"] < ALPHA)).sum() / w.sum())}
            for k, v in f.items():
                row[f"pct_{k}"] = 100 * float((w * v).sum() / w.sum())
            cell_rows.append(row)
            y = stats.regional_series(da).values.astype(float)
            cb = {k: float(np.asarray(v)[0]) for k, v in all_checks(y[:, None], years).items()}
            fb = {k: bool(np.asarray(v)[0]) for k, v in flags(
                {k: np.asarray([v]) for k, v in cb.items()}).items()}
            ex = box_extras(y, years, cb)
            fb["non_normal"] = ex["p_shapiro"] < ALPHA        # box: Shapiro-Wilk (shown)
            box_rows.append({"variable": var, "dataset": name, "years": row["years"],
                             **cb, **ex,
                             **{f"fails_{k}": v for k, v in fb.items()}})
            rd.log(f"  cells: {row['n_cells']}, median r1 {row['median_r1']:+.2f} "
                   f"(bias-corrected {row['median_r1c']:+.2f}); % of area failing: " +
                   ", ".join(f"{k} {row[f'pct_{k}']:.0f}" for k, _ in CHECKS))
            rd.log(f"  box: trend {cb['trend']:+.3f}/decade, p {cb['p']:.3f} (r1 corrected "
                   f"{cb['p_r1c']:.3f}, Mann-Kendall {cb['p_mk']:.3f}, bootstrap "
                   f"{box_rows[-1]['p_bootstrap']:.3f}); r1 {cb['r1']:+.2f}, r2 {cb['r2']:+.2f} "
                   f"(AR(1): {cb['r1'] ** 2:.2f})")
    pd.DataFrame(cell_rows).to_csv(config.DATA / OUT / f"cells_{month:02d}.csv", index=False,
                                   float_format="%.4g")
    pd.DataFrame(box_rows).to_csv(config.DATA / OUT / f"box_{month:02d}.csv", index=False,
                                  float_format="%.4g")
    rd.log(f"saved {config.DATA / OUT}/cells_{month:02d}.csv, box_{month:02d}.csv")


# ---------------------------------------------------------------- plot
def plot(month):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    fc, fb = (config.DATA / OUT / f"{k}_{month:02d}.csv" for k in ("cells", "box"))
    if not (fc.exists() and fb.exists()):
        raise SystemExit("no results: run 'python stats_assumptions.py compute' first")
    cells, box = pd.read_csv(fc), pd.read_csv(fb)
    ps.setup()
    fs, W = config.FS, config.WIDTH["full"]
    rows = [(r.variable, r.dataset) for r in cells.itertuples()]
    label = lambda v, d: f"{VARIABLES[v][0]}, {ps.label(d)}"
    nr, nc = len(rows), len(CHECKS)
    left, right, rh = 1.55, 0.1, 0.165
    cw = (W - left - right) / nc
    top = 0.95
    th = nr * rh
    top2 = top + th + 0.85
    H = top2 + th + 0.5
    fig, at = ps.figure("full", H)

    # a) % of area failing
    ax = at(left, top, W - left - right, th)
    vals = np.array([[cells.loc[i, f"pct_{k}"] for k, _ in CHECKS] for i in range(nr)])
    bounds = [0, 5, 10, 20, 30, 50, 100]
    cmap = ListedColormap(["#FFFFFF", "#FEE5D9", "#FCAE91", "#FB6A4A", "#DE2D26", "#A50F15"])
    norm = BoundaryNorm(bounds, cmap.N)
    ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    for i in range(nr):
        for j in range(nc):
            v = vals[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=fs["small"] - 1,
                    color="white" if v >= 30 else "black")
    fail_frame(ax, rows, label, fs)
    ax.set_xticks(range(nc))
    ax.set_xticklabels([l for _, l in CHECKS], fontsize=fs["small"] - 1)
    ax.xaxis.tick_top()
    ps.text(fig, 0.05, top - 0.42, "a) % of the box area where each check fails (p < 0.05; "
            "about 5 % by chance)", fontsize=fs["title"], fontweight="bold")

    # b) box-series p values
    pcols = [("p_ar1", "ar1_not_enough"), ("p_r1c", "r1_bias"), ("p_shapiro", "non_normal"),
             ("p_bp", "heteroscedastic"), ("p_curv", "curvature"), (None, "breakpoint"),
             ("n_eff", "small_n_eff"), ("p_mk", "test_disagrees")]
    ax = at(left, top2, W - left - right, th)
    ax.imshow(np.zeros((nr, nc)), cmap=ListedColormap(["#FFFFFF"]), aspect="auto")
    for i, (v, d) in enumerate(rows):
        b = box[(box.variable == v) & (box.dataset == d)]
        if b.empty:
            continue
        b = b.iloc[0]
        for j, (col, key) in enumerate(pcols):
            fail = bool(b[f"fails_{key}"])
            if col is None:
                txt = str(int(b["step_year"])) if fail else "-"
            elif col == "n_eff":
                txt = f"{b['n_eff']:.0f}"
            elif key in ("r1_bias", "test_disagrees"):
                txt = f"{b['p']:.2f}/" + (f"{b[col]:.2f}" if np.isfinite(b[col]) else "n/a")
            else:
                txt = f"{b[col]:.2f}" if np.isfinite(b[col]) else "n/a"
            ax.text(j, i, txt, ha="center", va="center", fontsize=fs["small"] - 1,
                    fontweight="bold" if fail else "normal",
                    color="#A50F15" if fail else "black")
    fail_frame(ax, rows, label, fs)
    ax.set_xticks(range(nc))
    ax.set_xticklabels([l for _, l in CHECKS], fontsize=fs["small"] - 1)
    ax.xaxis.tick_top()
    ps.text(fig, 0.05, top2 - 0.42, "b) Box-mean series: p of each check (bold red: fails)",
            fontsize=fs["title"], fontweight="bold")
    ps.text(fig, 0.05, top2 + th + 0.1, "Columns 2 and 8: p of the trend / p with the r1 bias "
            "correction or Mann-Kendall; step: year of the step; n/a: n$_{eff}$ too small "
            "for the test.", fontsize=fs["small"] - 1, va="top")
    ps.text(fig, W / 2, 0.06, f"{calendar.month_name[month]}: checks of the AR(1) trend test "
            "(OLS, lag-1 n$_{eff}$), Southern Africa box, land", ha="center", va="top",
            fontsize=fs["suptitle"], fontweight="bold")
    ps.save(fig, config.RESULTS / OUT / f"stats_assumptions_{month:02d}.png")
    box.to_csv(config.RESULTS / OUT / f"stats_assumptions_box_{month:02d}.csv", index=False,
               float_format="%.4g")


def fail_frame(ax, rows, label, fs):
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label(v, d) for v, d in rows], fontsize=fs["small"] - 1)
    ax.set_xticks(np.arange(-0.5, len(CHECKS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="0.8", lw=0.4)
    ax.tick_params(which="both", length=0)
    prev = None
    for i, (v, _) in enumerate(rows):                 # line between variables
        if prev is not None and v != prev:
            ax.axhline(i - 0.5, color="0.3", lw=0.8)
        prev = v


if __name__ == "__main__":
    args = sys.argv[1:]
    what = next((a for a in args if a in ("compute", "plot")), "all")
    month = 10
    if "--month" in args:
        month = int(args[args.index("--month") + 1])
    variables = [a for a in args if a in VARIABLES] or list(VARIABLES)
    if what in ("compute", "all"):
        compute(variables, month)
    if what in ("plot", "all"):
        plot(month)
