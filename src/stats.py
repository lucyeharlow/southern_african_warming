"""
Statistics shared by every figure.

Conventions (the same everywhere)
  Trends   OLS slope per decade along 'year'. Only cells with a value in EVERY year are used
           (complete()); the year axis must be gap-free, so lag 1 is always the previous year.
  p        lag-1 autocorrelation r1 of the residuals (sum e_t e_t-1 / sum e_t^2), effective
           sample size n_eff = n (1 - r1) / (1 + r1) capped at n; SE^2 = SSR / (n_eff - 2) / Sxx;
           two-sided t test with n_eff - 2 degrees of freedom (Santer et al. 2000, JGR 105,
           7337). Where n_eff <= 2 the trend is kept and p is NaN (never significant).
  Maps     Benjamini-Hochberg FDR at config.ALPHA_FDR over the cells of the map that is drawn
           (pass the field already cut to the plotted domain and land-masked).
  Averages area-weighted by cos(latitude) everywhere (box statistics, regional series, % area).
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import stats as sps

import config


# ---------------------------------------------------------------- years and anomalies
def select_years(da, years, dim="year"):
    """Exactly the years first-last on a gap-free axis; a year missing from the data entirely
    is an error (config.STRICT_YEARS) or left as NaN (so every cell drops out)."""
    want = np.arange(years[0], years[1] + 1)
    have = set(np.asarray(da[dim].values).tolist())
    missing = [int(y) for y in want if y not in have]
    if missing:
        msg = f"{da.name or 'field'}: no data for {missing}"
        if config.STRICT_YEARS:
            raise ValueError(msg + " (set config.YEARS or STRICT_YEARS)")
        print("  [WARN] " + msg, flush=True)
    da = da.reindex({dim: want})
    report_gaps(da, dim)
    return da


def report_gaps(da, dim="year"):
    """Log the years in which cells that have data in other years are missing (these cells
    are left out of every trend), so a gap in a download or a bad file is easy to find."""
    other = [d for d in da.dims if d != dim]
    if not other:
        return
    some = da.notnull().any(dim)
    n = int(some.sum())
    if n == 0:
        print(f"  [WARN] {da.name or 'field'}: no data at all", flush=True)
        return
    miss = (da.isnull() & some).sum(other) / n
    bad = [(int(y), float(f)) for y, f in zip(miss[dim].values, miss.values) if f > 0.01]
    if bad:
        txt = ", ".join(f"{y} ({100 * f:.0f}%)" for y, f in bad[:12])
        print(f"  [WARN] {da.name or 'field'}: cells missing in {txt}"
              + (" ..." if len(bad) > 12 else "")
              + "; those cells are left out of the trends", flush=True)


def complete(da, dim="year"):
    """Cells with a value at every step of dim; everything else NaN."""
    return da.where(da.notnull().all(dim))


def to_year_month(da):
    """(time, ...) monthly -> (year, month, ...)."""
    t = da.time
    da = da.assign_coords(year=("time", t.dt.year.values), month=("time", t.dt.month.values))
    return da.set_index(time=["year", "month"]).unstack("time")


def anomalies(da, base=None):
    """(year, month, ...) minus each month's base-period mean (config.BASELINE)."""
    b0, b1 = base or config.BASELINE
    clim = da.sel(year=slice(b0, b1)).mean("year")
    return da - clim


# ---------------------------------------------------------------- trends
def ols_stats(da, dim="year"):
    """Trend per decade, p, n, r1 and n_eff along dim for every other index.

    Cells without a value in every year are NaN in all outputs."""
    da = complete(da, dim)
    yrs = da[dim].values.astype(float)
    if np.any(np.diff(yrs) != 1):
        raise ValueError(f"{dim} axis has gaps; select it with select_years() first")
    n = da.notnull().sum(dim)
    x = xr.DataArray(yrs - yrs.mean(), dims=dim, coords={dim: da[dim]})
    ya = da - da.mean(dim)
    sxx = float((x ** 2).sum())
    slope = (x * ya).sum(dim, skipna=False) / sxx
    resid = ya - slope * x
    ssr = (resid ** 2).sum(dim, skipna=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        r1 = (resid * resid.shift({dim: 1})).sum(dim) / ssr
        n_eff = np.minimum(n * (1 - r1) / (1 + r1), n)
        se = np.sqrt((ssr / (n_eff - 2) / sxx).where(n_eff > 2))
        p = xr.apply_ufunc(lambda t, df: 2 * sps.t.sf(np.abs(t), df), slope / se, n_eff - 2)
    ok = n == da.sizes[dim]
    return xr.Dataset({"trend": (10 * slope).where(ok), "p": p.where(ok), "n": n.where(ok),
                       "r1": r1.where(ok), "n_eff": n_eff.where(ok)})


def fdr(p, alpha=None):
    """Benjamini-Hochberg over all finite p of one map: True where adjusted p <= alpha."""
    alpha = config.ALPHA_FDR if alpha is None else alpha
    v = np.asarray(p, float)
    out = np.zeros(v.shape, bool)
    ok = np.isfinite(v)
    if ok.any():
        out[ok] = sps.false_discovery_control(v[ok]) <= alpha
    return xr.DataArray(out, dims=p.dims, coords=p.coords) if isinstance(p, xr.DataArray) else out


def sig_stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def p_text(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ---------------------------------------------------------------- correlation (detrended)
def detrend(da, dim="year"):
    yrs = da[dim].values.astype(float)
    x = xr.DataArray(yrs - yrs.mean(), dims=dim, coords={dim: da[dim]})
    ya = da - da.mean(dim)
    return ya - (x * ya).sum(dim) / float((x ** 2).sum()) * x


def corr_and_p(x, y, dim="year"):
    """Pearson r of two (detrended) series and the p of the regression slope of y on x, with
    n_eff from the residuals' lag-1 autocorrelation (capped at n) and n_eff - 3 df."""
    x, y = xr.broadcast(x, y)
    ok = (x.notnull() & y.notnull()).all(dim)
    x, y = x.where(ok), y.where(ok)
    n = x.sizes[dim]
    sxx, syy, sxy = (x * x).sum(dim), (y * y).sum(dim), (x * y).sum(dim)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = sxy / np.sqrt(sxx * syy)
        b = sxy / sxx
        e = y - b * x
        sse = (e * e).sum(dim)
        r1 = (e * e.shift({dim: 1})).sum(dim) / sse
        n_eff = np.minimum(n * (1 - r1) / (1 + r1), n)
        se = np.sqrt((sse / (n_eff - 3) / sxx).where(n_eff > 3))
        p = xr.apply_ufunc(lambda t, df: 2 * sps.t.sf(np.abs(t), df), b / se, n_eff - 3)
    return r.where(ok), p.where(ok)


# ---------------------------------------------------------------- area weighting
def weights(da):
    """cos(latitude), broadcast to da and NaN where da is NaN."""
    return np.cos(np.deg2rad(da.lat)).broadcast_like(da).where(da.notnull())


def area_mean(da, dims=("lat", "lon")):
    """cos(latitude)-weighted mean over dims, skipping NaN cells."""
    w = weights(da)
    return (da * w).sum(dims) / w.sum(dims)


def regional_series(da, dim="year"):
    """Area-weighted mean over the cells that have a value in every step of dim."""
    return area_mean(complete(da, dim))


def weighted_values(da):
    v = np.asarray(da.values, float).ravel()
    w = np.cos(np.deg2rad(da.lat)).broadcast_like(da).values.ravel()
    ok = np.isfinite(v)
    return v[ok], w[ok]


def wquantile(v, w, q):
    """Area-weighted quantiles (Hazen positions; equal weights give the usual quantiles)."""
    i = np.argsort(v)
    v, w = v[i], w[i]
    return np.interp(q, (np.cumsum(w) - 0.5 * w) / w.sum(), v)


def wmedian(da):
    v, w = weighted_values(da)
    return float(wquantile(v, w, 0.5)) if v.size else np.nan


def box_stats(da):
    """Area-weighted box for matplotlib's bxp: weighted quartiles, whiskers at the most
    extreme cells within 1.5 IQR, no fliers."""
    v, w = weighted_values(da)
    if not v.size:
        return None
    q1, med, q3 = wquantile(v, w, [0.25, 0.5, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return {"q1": q1, "med": med, "q3": q3, "fliers": [], "mean": np.average(v, weights=w),
            "whislo": v[v >= lo].min(), "whishi": v[v <= hi].max()}


def describe(da):
    """Area-weighted summary of a field (n cells, mean, quartiles, min, max)."""
    v, w = weighted_values(da)
    if not v.size:
        return {"n_cells": 0}
    q = wquantile(v, w, [0.25, 0.5, 0.75])
    return {"n_cells": v.size, "mean": np.average(v, weights=w), "min": v.min(), "q25": q[0],
            "median": q[1], "q75": q[2], "max": v.max()}


def pct_area(mask, valid):
    """Percentage of the valid area (cos-lat weighted) where mask is True."""
    w = np.cos(np.deg2rad(valid.lat)).broadcast_like(valid).where(valid)
    tot = float(w.sum())
    return 100 * float((w * mask).sum()) / tot if tot > 0 else np.nan


def fdr_maps(p, over=("lat", "lon"), alpha=None):
    """FDR separately on every map (the dims in over) of a multi-dimensional p."""
    return xr.apply_ufunc(lambda a: fdr(a, alpha), p, input_core_dims=[list(over)],
                          output_core_dims=[list(over)], vectorize=True)


# ---------------------------------------------------------------- omega regimes
CHUNK = 40_000                   # cells per block in regimes() and pettitt() (memory)


def _regime_np(x, slope):
    """x (year, N), slope (N) -> index into config.REGIMES (-1 none). First persistent sign
    change (config.MIN_PERSIST years of one sign, pre- and post-switch means of opposite sign)
    -> new subsidence / new ascent; else mean sign x trend sign."""
    n, P = x.shape[0], config.MIN_PERSIST
    fin = np.isfinite(x)
    cum = lambda a: np.concatenate([np.zeros_like(a[:1], float), np.cumsum(a, 0)])
    c_sum, c_fin = cum(np.where(fin, x, 0.0)), cum(fin)
    c_pos, c_neg = cum(fin & (x > 0)), cum(fin & (x < 0))
    ks = np.arange(1, max(n - P, 1))
    win = lambda c: c[ks + P] - c[ks]
    with np.errstate(invalid="ignore", divide="ignore"):
        pre_n, post_n = c_fin[ks], c_fin[n] - c_fin[ks]
        pre, post = c_sum[ks] / pre_n, (c_sum[n] - c_sum[ks]) / post_n
        ok = ((win(c_fin) == P) & ((win(c_pos) == P) | (win(c_neg) == P)) & (pre_n >= 2)
              & (post_n >= 2) & (pre != 0) & (post != 0) & (np.sign(pre) != np.sign(post)))
        found = ok.any(0)
        post_first = np.take_along_axis(post, ok.argmax(0)[None], 0)[0]
        mean = c_sum[n] / np.maximum(c_fin[n], 1)
    code = np.where(mean > 0, np.where(slope > 0, 0, 1), np.where(slope < 0, 4, 3))
    code = np.where(found, np.where(post_first > 0, 2, 5), code)
    return np.where(np.isfinite(slope) & fin.all(0), code, -1)


def regimes(da, slope, dim="year"):
    """Regime code for every cell of da (dim, ...) given its trend slope (...)."""
    da = da.transpose(dim, ...)
    shape = da.shape[1:]
    x = da.values.reshape(da.shape[0], -1)
    sl = np.asarray(slope.transpose(*da.dims[1:]).values).ravel()
    code = np.concatenate([_regime_np(x[:, i:i + CHUNK], sl[i:i + CHUNK])
                           for i in range(0, max(x.shape[1], 1), CHUNK)])
    return xr.DataArray(code.reshape(shape), dims=da.dims[1:],
                        coords={d: da[d] for d in da.dims[1:]})


def pettitt(da, dim="year", alpha=0.05):
    """Pettitt change-point test per cell: True where p < alpha."""
    da = da.transpose(dim, ...)
    x_all = da.values.reshape(da.shape[0], -1)
    tri = np.triu(np.ones((x_all.shape[0],) * 2), 1)
    out = []
    for i in range(0, max(x_all.shape[1], 1), CHUNK // 4):      # (year, year, cells) arrays
        x = x_all[:, i:i + CHUNK // 4]
        n = np.isfinite(x).sum(0)
        S = np.nan_to_num(np.sign(x[None] - x[:, None]))
        U = np.einsum("tj...,tj->t...", np.cumsum(S, 0), tri)
        K = np.abs(U).max(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.clip(2 * np.exp(-6 * K ** 2 / (n ** 3 + n ** 2)), 0, 1) < alpha)
    return xr.DataArray(np.concatenate(out).reshape(da.shape[1:]), dims=da.dims[1:],
                        coords={d: da[d] for d in da.dims[1:]})
