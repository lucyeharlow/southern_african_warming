"""
Reading raw files: standard names and orientation, domain cut, monthly means, file finders and
a per-month cache. Everything stays on the dataset's own grid (no regridding).

Monthly fields are returned as (year, month, lat, lon) or (year, month, lev, lat, lon).
Every reader caches what it reads in processed_data/cache/<tag>/ (one small file per month or
per source file), so a second run reads only new files.
"""
from __future__ import annotations

import calendar
import glob
import os
import re
import time

import numpy as np
import pandas as pd
import xarray as xr

import config

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")      # file locking fails on NFS homes

CACHE_MAX_CELLS = 5_000_000      # fields bigger than this (e.g. global SARAH-3 at 0.05 deg) are
                                 # read again each run instead of being cached
NAMES = {"latitude": "lat", "longitude": "lon", "Latitude": "lat", "Longitude": "lon",
         "valid_time": "time", "pressure_level": "lev", "isobaricInhPa": "lev",
         "level": "lev", "plev": "lev"}


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- fields
def standardise(da):
    """lat/lon/time/lev names, longitude -180..180, lat and lon ascending, size-1 extras dropped."""
    if "valid_time" in da.coords and "time" in da.coords:      # GRIB (cfgrib): analysis time
        if "valid_time" in da.dims or "time" not in da.dims:   # and valid time both present
            da = da.drop_vars("time")
        else:
            da = da.drop_vars("valid_time")
    if "expver" in da.dims:                    # ERA5 + ERA5T (expver 1 and 5): one value each
        da = da.max("expver", skipna=True)     # time step, whichever version has it
    da = da.rename({k: v for k, v in NAMES.items() if k in da.dims or k in da.coords})
    da = da.drop_vars(["number", "expver", "step"], errors="ignore")
    if float(da.lon.max()) > 180:
        da = da.assign_coords(lon=(da.lon + 180) % 360 - 180)
    if "lev" in da.coords and float(da.lev.max()) > 2000:            # Pa -> hPa
        da = da.assign_coords(lev=da.lev / 100.0)
    extra = [d for d in da.dims if d not in ("time", "lev", "lat", "lon") and da.sizes[d] == 1]
    return da.squeeze(extra, drop=True).sortby("lat").sortby("lon")


def region_of(region):
    return config.REGIONS[region] if isinstance(region, str) else region


def cut(da, region, pad=0.0):
    """Cells whose centres lie inside the region (+ pad degrees)."""
    r = region_of(region)
    return da.sel(lat=slice(r["lat"][0] - pad, r["lat"][1] + pad),
                  lon=slice(r["lon"][0] - pad, r["lon"][1] + pad))


def clean(da, fill=1e10):
    """Fill values (1e15, 9.999e20, -999 ...) to NaN."""
    return da.where(np.abs(da) < fill)


def pick_levels(da, lev):
    """One level or a list of levels (hPa), exactly."""
    if lev is None:
        return da
    return da.sel(lev=lev, method="nearest", tolerance=0.5)


def steps_needed(times, year, month, frac=0.9):
    """frac of the time steps a full month has, from the spacing of the given times."""
    tt = pd.to_datetime(np.asarray(times))
    hours = (float(np.median(np.diff(tt).astype("timedelta64[m]").astype(float))) / 60
             if tt.size > 1 else 24.0)
    return frac * calendar.monthrange(year, month)[1] * 24.0 / max(hours, 1.0)


def to_year_month(da):
    """(time, ...) of monthly means -> (year, month, ...), one value per year and month."""
    t = pd.to_datetime(da.time.values)
    da = da.assign_coords(year=("time", t.year.values), month=("time", t.month.values))
    return da.set_index(time=["year", "month"]).unstack("time").transpose("year", "month", ...)


def from_months(fields):
    """{(year, month): field} -> (year, month, ...); missing year-months are NaN."""
    if not fields:
        raise RuntimeError("no readable files")
    keys = sorted(fields)
    years = sorted({y for y, _ in keys})
    like = fields[keys[0]]
    out = []
    for y in years:
        out.append(xr.concat([fields.get((y, m), xr.full_like(like, np.nan)) for m in range(1, 13)],
                             "month", coords="minimal", compat="override")
                   .assign_coords(month=np.arange(1, 13)))
    return xr.concat(out, "year", coords="minimal", compat="override").assign_coords(
        year=years).transpose("year", "month", ...)


def per_month(fields_by_file):
    """{key: (time, ...) field} -> {(year, month): field} (for files holding several months)."""
    out = {}
    for da in fields_by_file.values():
        for i, t in enumerate(pd.to_datetime(da.time.values)):
            out[(t.year, t.month)] = da.isel(time=i, drop=True)
    return out


def _tag(region):
    if isinstance(region, str):
        return region
    return "r{}_{}_{}_{}".format(*region["lon"], *region["lat"]).replace("-", "m")


def _levtag(lev):
    if lev is None:
        return ""
    return "_" + "-".join(f"{v:g}" for v in np.atleast_1d(lev))


# ---------------------------------------------------------------- cache
def _cache_name(key):
    return "_".join(str(int(k)) if np.issubdtype(type(k), np.integer) or isinstance(k, int)
                    else str(k) for k in np.atleast_1d(key))


def _source(path):
    return os.path.basename(str(path if not isinstance(path, (tuple, list)) else path[0]))


_JOB = None                      # the reader, for the worker processes (inherited by fork)


def _work(item):
    key, path = item
    try:
        return key, _JOB(path, *np.atleast_1d(key)).astype("float32"), None
    except Exception as e:
        return key, None, f"{_source(path)}: {type(e).__name__}: {e}"


def workers():
    """Parallel file reading: SAW_WORKERS if set (e.g. export SAW_WORKERS=2 for very large
    files), else one worker per CPU given by SLURM (--cpus-per-task), else 1."""
    n = os.environ.get("SAW_WORKERS") or os.environ.get("SLURM_CPUS_PER_TASK") or "1"
    return max(1, int(n))


def cached(tag, files, reader):
    """{key: field} for every entry of files ({key: path}), reading each file once: fields are
    cached as processed_data/cache/<tag>/<key>.nc and re-read only if the source name changes.
    Files not yet cached are read in parallel (workers()). Unreadable files are reported and
    left out."""
    global _JOB
    d = config.CACHE / tag
    d.mkdir(parents=True, exist_ok=True)
    out, todo, bad = {}, [], []
    for key, path in files.items():
        cpath = d / f"{_cache_name(key)}.nc"
        if cpath.exists():
            try:
                with xr.open_dataarray(cpath) as c:
                    if c.attrs.get("source") == _source(path):
                        out[key] = c.load()
                        continue
            except Exception:
                pass
        todo.append((key, path))
    n_cached, n = len(out), workers()
    if todo:
        log(f"  {tag}: {n_cached} cached, reading {len(todo)} file(s) with {n} worker(s) ...")
    _JOB = reader
    t0 = last = time.time()
    done = 0

    def keep(key, da, err):
        nonlocal done, last
        done += 1
        if err:
            bad.append(err)
        else:
            da.attrs = {"source": _source(files[key])}
            if da.size <= CACHE_MAX_CELLS:             # very large fields are not cached
                cpath = d / f"{_cache_name(key)}.nc"
                tmp = cpath.with_suffix(f".{os.getpid()}.tmp")
                da.rename("v").to_netcdf(tmp)
                os.replace(tmp, cpath)
            out[key] = da
        now = time.time()
        if done % 25 == 0 or done == len(todo) or now - last > 60:   # slow files: each one
            log(f"    {done}/{len(todo)} read ({(now - t0) / 60:.1f} min)")
            last = now

    if n > 1 and len(todo) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool
        left = dict(todo)
        try:
            with ProcessPoolExecutor(min(n, len(todo)),
                                     mp_context=multiprocessing.get_context("fork")) as ex:
                futs = [ex.submit(_work, item) for item in todo]
                for f in as_completed(futs):
                    key, da, err = f.result()
                    left.pop(key, None)
                    keep(key, da, err)
        except BrokenProcessPool:
            log(f"  [WARN] {tag}: a reading process was killed, most likely out of memory "
                f"(ask for more with --mem, or export SAW_WORKERS=2). Reading the "
                f"{len(left)} remaining file(s) one at a time ...")
            for item in left.items():
                keep(*_work(item))
    else:
        for item in todo:
            keep(*_work(item))
    log(f"  {tag}: {len(out)} files ({len(out) - n_cached} read now, {n_cached} from the cache)")
    for b in bad:
        log(f"  SKIPPED {b}")
    return out


def seed_from_old_caches(tag, var, files, region):
    """Copy months already averaged by the old scripts (config.OLD_JRA3Q_CACHES, files
    jra3q_<var>_*.nc) into the new cache, so they are not read again. Only entries whose
    source file name matches the current file, and only caches covering the region, are used."""
    r = region_of(region)
    d = config.CACHE / tag
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for folder in getattr(config, "OLD_JRA3Q_CACHES", []):
        for old in sorted(glob.glob(os.path.join(str(folder), f"jra3q_{var}_*.nc"))):
            try:
                with xr.open_dataset(old) as ds:
                    ds = ds.load()
            except Exception:
                continue
            v = standardise(ds["v"]) if "v" in ds else None
            if v is None or not (float(v.lat.min()) <= r["lat"][0] + 1.3 and
                                 float(v.lat.max()) >= r["lat"][1] - 1.3 and
                                 float(v.lon.min()) <= r["lon"][0] + 1.3 and
                                 float(v.lon.max()) >= r["lon"][1] - 1.3):
                continue
            entry = "entry" if "entry" in v.dims else "year"
            month = re.search(r"_(\d{2})\.nc$", old)
            for i in range(v.sizes[entry]):
                e = v.isel({entry: i})
                y = int(ds["yr"].values[i] if "yr" in ds else ds["year"].values[i])
                m = int(ds["mo"].values[i]) if "mo" in ds else int(month[1]) if month else None
                key = (y, m)
                if m is None or key not in files or str(ds["source"].values[i]) != _source(files[key]):
                    continue
                cpath = d / f"{_cache_name(key)}.nc"
                if cpath.exists():
                    continue
                da = cut(e.drop_vars([c for c in e.coords if c not in ("lat", "lon")]), r)
                da = da.astype("float32")
                da.attrs = {"source": _source(files[key])}
                da.rename("v").to_netcdf(cpath)
                n += 1
    if n:
        log(f"  {tag}: {n} month(s) taken from the old scripts' caches")


# ---------------------------------------------------------------- file finders
def merra2_files(collection):
    """{(year, month): path} for MERRA2_*.<collection>.YYYYMM.nc4 (searched recursively);
    where a month exists in two versions (e.g. 400 and the reprocessed 401) the highest wins."""
    found = {}
    pat = os.path.join(str(config.RAW["merra2"]), "**", f"MERRA2_*.{collection}.*.nc4")
    for f in glob.glob(pat, recursive=True):
        m = re.match(rf"MERRA2_(\d{{3}})\.{re.escape(collection)}\.(\d{{4}})(\d{{2}})\.nc4$",
                     os.path.basename(f))
        if m:
            key, ver = (int(m[2]), int(m[3])), int(m[1])
            if key not in found or ver > found[key][0]:
                found[key] = (ver, f)
    return {k: v[1] for k, v in sorted(found.items())}


def jra3q_files(folder, name=None):
    """{(year, month): path} for JRA-3Q monthly netCDF files (*.YYYYMM0100_YYYYMMDDHH.nc) in
    config.RAW['jra3q']/<folder> (folder '' = the top folder)."""
    d = os.path.join(str(config.RAW["jra3q"]), folder)
    out = {}
    for f in glob.glob(os.path.join(d, f"*{name or folder}*.nc")):
        m = re.search(r"\.(\d{4})(\d{2})0100_\d{10}\.nc$", os.path.basename(f))
        if m:
            out[(int(m[1]), int(m[2]))] = f
    return dict(sorted(out.items()))


def cmsaf_files(code, dirs=None):
    """{(year, month): path} for CM SAF monthly files *<code>mmYYYYMM01*.nc (CLARA-A3,
    SARAH-3; searched recursively); the thematic record is preferred over the interim one."""
    dirs = dirs if dirs is not None else config.RAW["clara"]
    found = {}
    for d in np.atleast_1d(dirs):
        for f in sorted(glob.glob(os.path.join(str(d), "**", f"*{code}mm*.nc"), recursive=True)):
            m = re.search(rf"{code}mm(\d{{4}})(\d{{2}})01", os.path.basename(f))
            if not m:
                continue
            key = (int(m[1]), int(m[2]))
            interim = "AVPOSI" in os.path.basename(f) or "interim" in os.path.basename(f)
            if key not in found or (found[key][0] and not interim):
                found[key] = (interim, f)
    return {k: v[1] for k, v in sorted(found.items())}


# ---------------------------------------------------------------- readers
def single_file(path, var, region, scale=1.0, offset=0.0, pad=0.0, fill=1e10, lev=None,
                months=None):
    """One multi-year file of monthly means (ERA5, NOAAGlobalTemp, CRUTEM5) ->
    (year, month, [lev,] lat, lon); months: only these calendar months are read."""
    with xr.open_dataset(path) as ds:
        da = pick_levels(cut(standardise(ds[var]), region, pad), lev)
        if months is not None:
            da = da.isel(time=np.flatnonzero(np.isin(da.time.dt.month.values, months)))
        da = (clean(da, fill) * scale + offset).load()
    _, first = np.unique(da.time.values, return_index=True)      # one value per month
    return to_year_month(da.isel(time=first))


def era5_yearly(directory, pattern, var, region, scale=1.0, offset=0.0, pad=0.0, lev=None,
                months=None):
    """ERA5 one-file-per-year monthly means (..._YYYY.nc) -> (year, month, [lev,] lat, lon)."""
    files = {}
    for f in sorted(glob.glob(os.path.join(str(directory), pattern))):
        m = re.search(r"(\d{4})\.nc$", f)
        if m:
            files[int(m[1])] = f

    def read(path, y):
        with xr.open_dataset(path) as ds:
            da = pick_levels(cut(standardise(ds[var]), region, pad), lev)
            if months is not None:
                da = da.isel(time=np.flatnonzero(np.isin(da.time.dt.month.values, months)))
            return (clean(da) * scale + offset).load()

    tag = f"era5_{var}_{_tag(region)}{_levtag(lev)}" + (f"_m{'-'.join(map(str, months))}"
                                                        if months is not None else "")
    return from_months(per_month(cached(tag, files, read)))


def only(files, months):
    return files if months is None else {k: v for k, v in files.items() if k[1] in months}


def merra2(collection, var, region, scale=1.0, offset=0.0, pad=0.0, lev=None, months=None):
    """MERRA-2 monthly files -> (year, month, [lev,] lat, lon)."""
    def read(path, y, m):
        with xr.open_dataset(path) as ds:
            da = pick_levels(cut(standardise(ds[var]), region, pad), lev)
            if "time" in da.dims:
                da = da.isel(time=0, drop=True)
            return (clean(da) * scale + offset).load()
    tag = f"merra2_{collection}_{var}_{_tag(region)}{_levtag(lev)}"
    return from_months(cached(tag, only(merra2_files(collection), months), read))


def open_any(path):
    """open_dataset; if the time units cannot be decoded, open without decoding times."""
    try:
        return xr.open_dataset(path)
    except ValueError as e:
        if "time" not in str(e).lower() and "calendar" not in str(e).lower():
            raise
        return xr.open_dataset(path, decode_times=False)


def time_dim(da):
    """The time dimension of a sub-monthly field, whatever it is called (time, valid_time,
    initial_time0_hours, t, ...): a dimension with 'time' in its name, else the first one that is
    not lat, lon or lev."""
    named = [d for d in da.dims if "time" in d.lower()]
    other = [d for d in da.dims if d not in ("lat", "lon", "lev")]
    if not (named or other):
        raise ValueError(f"no time dimension in {da.dims}")
    return (named or other)[0]


def steps_needed_any(da, t, year, month, frac=0.9):
    """frac of a full month's time steps: from the time spacing if the times are dates, else
    assuming 1-hourly data if there are more than 8 steps per day, 6-hourly otherwise."""
    vals = da[t].values if t in da.coords else None
    if vals is not None and np.issubdtype(np.asarray(vals).dtype, np.datetime64):
        return steps_needed(vals, year, month, frac)
    days = calendar.monthrange(year, month)[1]
    per_day = 24 if da.sizes[t] > 8 * days else 4
    return frac * per_day * days


def time_mean(da, block=8):
    """Mean over time, reading a few time steps at a time (low memory for large files)."""
    total = count = None
    for i in range(0, da.sizes["time"], block):
        x = clean(da.isel(time=slice(i, i + block))).load()
        s, c = x.sum("time", skipna=True), x.notnull().sum("time")
        total = s if total is None else total + s
        count = c if count is None else count + c
    return total.where(count > 0) / count.where(count > 0)


def jra3q(folder, var, region, scale=1.0, offset=0.0, pad=0.0, lev=None, min_frac=0.9,
          months=None):
    """JRA-3Q sub-monthly netCDF (one file per month) -> monthly means (year, month, [lev,]
    lat, lon); a month is kept only with at least min_frac of its time steps. The year and
    month come from the file name, so the time variable is only used to count the steps."""
    def read(path, y, m):
        with open_any(path) as ds:
            da = ds[var] if var in ds else next(v for v in ds.data_vars.values() if v.ndim >= 3)
            t = time_dim(da)
            if t != "time":
                da = da.rename({t: "time"})
            if "time" in da.coords and not np.issubdtype(da.time.dtype, np.datetime64):
                da = da.drop_vars("time")                 # undecodable times: count steps only
            da = pick_levels(cut(standardise(da), region, pad), lev)
            need = steps_needed_any(da, "time", y, m, min_frac)
            if da.sizes["time"] < need:
                raise ValueError(f"only {da.sizes['time']} time steps: incomplete month")
            da = da.drop_vars([c for c in da.coords if c not in ("lat", "lon", "lev")])
            return (time_mean(da) * scale + offset)
    tag = f"jra3q_{var}_{_tag(region)}{_levtag(lev)}"
    files = only(jra3q_files(folder, var), months)
    if lev is None and scale == 1.0 and offset == 0.0:
        seed_from_old_caches(tag, var, files, region)
    return from_months(cached(tag, files, read))


def jra3q_grib(directory, prefix, region, lev=None, months=None, hours=(0, 6, 12, 18),
               min_frac=0.9, years=None):
    """JRA-3Q 6-hourly analysis GRIB files <prefix>.YYYYMMDDHH (cfgrib) -> monthly means."""
    index = {}
    for name in os.listdir(directory) if os.path.isdir(directory) else []:
        m = re.fullmatch(re.escape(prefix) + r"\.(\d{4})(\d{2})(\d{2})(\d{2})", name)
        if (m and int(m[4]) in hours and (months is None or int(m[2]) in months)
                and (years is None or years[0] <= int(m[1]) <= years[1])):
            index.setdefault((int(m[1]), int(m[2])), []).append(os.path.join(directory, name))

    def read(paths, y, m):
        need = min_frac * len(hours) * calendar.monthrange(y, m)[1]
        if len(paths) < need:
            raise ValueError(f"only {len(paths)} analysis times: incomplete month")
        total = None
        for p in sorted(paths):
            with xr.open_dataset(p, engine="cfgrib", backend_kwargs={
                    "indexpath": "", "filter_by_keys": {"typeOfLevel": "isobaricInhPa"}}) as ds:
                da = ds[list(ds.data_vars)[0]]
                da = pick_levels(cut(standardise(da), region, pad=0), lev).load()
                da = da.drop_vars([c for c in da.coords if c not in ("lat", "lon", "lev")])
            da = clean(da).astype("float64")
            total = da if total is None else total + da
        return total / len(paths)
    tag = f"jra3q_grib_{prefix}_{_tag(region)}{_levtag(lev)}"
    files = {k: tuple(v) for k, v in sorted(index.items())}
    return from_months(cached(tag, files, read))


def cmsaf(code, var, region, dirs=None, scale=1.0, valid=(-900, 1e10), months=None):
    """CLARA-A3 / SARAH-3 monthly files -> (year, month, lat, lon); records whose
    record_status is not 0 are left out."""
    def read(path, y, m):
        with xr.open_dataset(path) as ds:
            if "record_status" in ds and int(np.asarray(ds["record_status"].values).ravel()[0]) != 0:
                raise ValueError("record_status not ok")
            name = var if var in ds else var.lower()
            da = cut(standardise(ds[name]), region)
            if "time" in da.dims:
                da = da.isel(time=0, drop=True)
            return (da.where((da > valid[0]) & (da < valid[1])) * scale).load()
    src = "" if dirs is None else "_" + re.sub(r"\W", "", os.path.basename(
        str(np.atleast_1d(dirs)[0]).rstrip("/")))         # e.g. _sarah_3: own cache folder
    tag = f"cmsaf{src}_{code}_{var}_{_tag(region)}"
    return from_months(cached(tag, only(cmsaf_files(code, dirs), months), read))


def surface_pressure_clim(region, pad=1.0, months=None):
    """MERRA-2 monthly climatological surface pressure (month, lat, lon) in hPa, used to mask
    levels below the ground for every dataset (interpolated to their grids: a mask only)."""
    ps = merra2("instM_3d_asm_Np", "PS", region, scale=0.01, pad=pad, months=months)
    return ps.mean("year")


def above_ground_on(field, ps_clim, month=None):
    """True where each level of field (lev, lat, lon) is above the MERRA-2 climatological
    surface (ps_clim (month, lat, lon) interpolated to field's grid)."""
    if not config.MASK_BELOW_GROUND or ps_clim is None:
        return xr.ones_like(field, dtype=bool)
    ps = ps_clim if month is None else ps_clim.sel(month=month)
    ps = ps.interp(lat=field.lat, lon=field.lon, kwargs={"fill_value": None})
    return field.lev <= ps
