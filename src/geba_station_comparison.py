"""
GEBA stations against the gridded datasets: October downwelling shortwave (SW down) at each
station that measures global radiation, next to the value of ERA5, MERRA-2, JRA-3Q, CLARA-A3
and SARAH-3 at the same place.

Run:  python geba_station_sw.py            compute (if the data file is missing) and plot
      python geba_station_sw.py compute    re-read GEBA and the gridded data
      python geba_station_sw.py plot       plot from the data file only
      python geba_station_sw.py check      report on the GEBA files and the gridded
                                           datasets' variables (see CHECK)

STATIONS  GEBA global radiation (ebcode GLOBAL), 10-40E, south of 3S, with more than
          MIN_ENTRIES October values after MIN_YEAR (newest *_metadata.csv / *_monthly_data.csv
          in RAW['geba']; converted_flux_avg, W m-2).
GRIDDED   October SW down from the readers, as in the SW figure (sw_trends.py /
          sw_down_trends.py: ERA5 ssrd, MERRA-2 SWGDN, JRA-3Q dswrf1have (1-hourly averages,
          monthly files), CLARA-A3 SIS, SARAH-3 SIS; the same caches), from FIRST_GRIDDED
          (1982); each dataset's nearest cell that has data.
VALUES    raw monthly means in W m-2 (no anomalies, no trends), so the biases are visible.
          A missing year leaves a gap in the line; a value with no neighbours is a dot.
QC        GEBA months whose clearness index (SW down / top-of-atmosphere October mean) is
          outside K_RANGE are left out and logged (see CHECK).
BIAS      above each panel: mean of (dataset - GEBA) over that station's GEBA years (the years
          with a station value, from FIRST_GRIDDED), W m-2, ordered from the smallest |bias|
          to the largest, each with its dataset tag (TAG). Long station names are drawn
          smaller (down to TITLE_MIN_FS) to fit the panel, then shortened if still too long.
CHECK     results/geba/geba_check.txt: the columns of both files; the energy balance codes;
          every metadata column of the selected stations; series at the same place (possible
          duplicates); per station the October values used (years, gaps, range, repeated
          station-years); converted_flux_avg against flux_avg (the conversion factor); every
          flag-like column; and the clearness index K = SW down / top-of-atmosphere October
          mean, which must lie between about 0.3 and 0.85 for a monthly mean of global
          radiation (values outside suggest a wrong variable, units or a bad month).
          Gridded datasets: for the file each one is read from (an October file), the variable
          used, its attributes (long and standard name, units, cell methods), the file's
          global attributes, the conversion this script applies and the converted October
          value at each station, with its clearness index; a variable whose names do not
          describe downwelling shortwave at the surface, or a clearness index outside K_RANGE,
          is marked CHECK.

FIGURE  results/geba/geba_station_sw_october.png
  map of the stations (colours from STATION_COLOURS, colour-blind friendly), then one panel
  per station (same year and flux ranges in every panel); title in the station's colour.
DATA FILE  data/geba/geba_station_sw_october.csv: tskey, station, lat, lon, source, year,
  sw_down_W_m2 (source GEBA or a dataset); the cell used for each dataset is in
  results/geba/geba_station_cells.csv with the mean bias and correlation against the station.
"""
from __future__ import annotations

import calendar
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import config
import plotstyle as ps
import readers as rd

OUT = "geba"
MONTH = 10
MAP = "africa_10_40"
LON_RANGE, LAT_MAX = (10.0, 40.0), -3.0
TARGET_TYPE = "GLOBAL"
MIN_YEAR, MIN_ENTRIES = 1980, 15          # more than 15 October values after 1980
DATASETS = ["ERA5", "MERRA-2", "JRA-3Q", "CLARA-A3", "SARAH-3"]     # order of the SW figure
FIRST_GRIDDED = 1982                      # gridded series from this year
FLUX = "converted_flux_avg"               # GEBA monthly mean flux (W m-2)
K_RANGE = (0.3, 0.85)                     # plausible clearness index of a monthly mean
STATION_COLOURS = ["#245E55", "#ED773C", "#808BC5", "#C63F3E", "#EAC119", "#EAA7C7", "#9ED6DF"]
STATION_MARKERS = ["o", "s", "^", "D"]    # next shape after every len(STATION_COLOURS) stations
STATION_LINE = "#1D1D1B"                  # GEBA series
LAND, SEA = "#EAE4DA", "#FFFFFF"
TAG = {"ERA5": "E5", "MERRA-2": "M2", "CLARA-A3": "CL", "SARAH-3": "SA", "JRA-3Q": "JR"}
TITLE_MIN_FS = 6.0                        # long station names shrink down to this size


def data_file():
    return config.DATA / OUT / f"geba_station_sw_{calendar.month_name[MONTH].lower()}.csv"


# ---------------------------------------------------------------- GEBA
def newest(pattern):
    files = sorted(glob.glob(str(Path(config.RAW["geba"]) / pattern)))
    if not files:
        raise FileNotFoundError(f"no {pattern} in {config.RAW['geba']}")
    return files[-1]


def load_geba():
    """(stations DataFrame, long DataFrame of the stations' October values)."""
    meta = pd.read_csv(newest("*_metadata.csv"), sep=";", low_memory=False)
    mon = pd.read_csv(newest("*_monthly_data.csv"), sep=";", low_memory=False)
    for c in ("tskey", "sgxlon", "sgylat"):
        meta[c] = pd.to_numeric(meta[c], errors="coerce")
    for c in ("tskey", "year", "month", FLUX):
        mon[c] = pd.to_numeric(mon[c], errors="coerce")
    st = meta[(meta.ebcode == TARGET_TYPE) & meta.sgylat.notna() & meta.sgxlon.notna()
              & (meta.sgylat < LAT_MAX) & meta.sgxlon.between(*LON_RANGE)]
    octo = mon[mon.tskey.isin(st.tskey) & mon[FLUX].notna()
               & (mon.month == MONTH) & (mon.year > MIN_YEAR)]
    # drop months outside the plausible clearness index (wrong units or a bad month)
    toa = st.set_index("tskey").sgylat.map(toa_month_mean)
    k = octo[FLUX] / octo.tskey.map(toa)
    bad = (k < K_RANGE[0]) | (k > K_RANGE[1])
    for r in octo[bad].itertuples():
        rd.log(f"  [WARN] GEBA {int(r.tskey)} {int(r.year)}: {getattr(r, FLUX):.1f} W m-2 is a "
               f"clearness index of {k[r.Index]:.2f} (outside {K_RANGE[0]}-{K_RANGE[1]}): left out")
    octo = octo[~bad]
    count = octo.groupby("tskey")[FLUX].count()
    st = st[st.tskey.isin(count[count > MIN_ENTRIES].index)].sort_values("sgname")
    rd.log(f"GEBA: {len(st)} stations ({TARGET_TYPE}, > {MIN_ENTRIES} {calendar.month_abbr[MONTH]} "
           f"values after {MIN_YEAR})")
    vals = (octo[octo.tskey.isin(st.tskey)].groupby(["tskey", "year"])[FLUX].mean()
            .rename("sw_down_W_m2").reset_index())
    return st.reset_index(drop=True), vals


# ---------------------------------------------------------------- gridded data
def sw_module():
    """The SW figure's script (sw_trends.py on the server, sw_down_trends.py in the repository):
    same readers and caches."""
    try:
        import sw_trends as sd
    except ImportError:
        import sw_down_trends as sd
    return sd


def load_gridded(name):
    """(year, lat, lon) October SW down in W m-2."""
    return sw_module().load(name).sel(month=MONTH)


def nearest_valid(da, lat, lon):
    """The series of the nearest cell (great-circle) that has at least one value."""
    has = da.notnull().any("year").values
    la, lo = np.meshgrid(np.deg2rad(da.lat.values), np.deg2rad(da.lon.values), indexing="ij")
    p, q = np.deg2rad(lat), np.deg2rad(lon)
    d = np.arccos(np.clip(np.sin(p) * np.sin(la) + np.cos(p) * np.cos(la) * np.cos(lo - q), -1, 1))
    d[~has] = np.inf
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return da.isel(lat=i, lon=j), float(da.lat[i]), float(da.lon[j]), float(6371 * d[i, j])


# ---------------------------------------------------------------- compute
def compute():
    (config.DATA / OUT).mkdir(parents=True, exist_ok=True)
    st, vals = load_geba()
    if st.empty:
        raise SystemExit("no GEBA station passes the filter")
    info = st.set_index("tskey")[["sgname", "sgylat", "sgxlon"]]
    rows = [{"tskey": int(r.tskey), "source": "GEBA", "year": int(r.year),
             "sw_down_W_m2": float(r.sw_down_W_m2), "cell_lat": np.nan, "cell_lon": np.nan,
             "cell_km": np.nan} for r in vals.itertuples()]
    for name in DATASETS:
        rd.log(f"--- {name} ---")
        try:
            da = load_gridded(name)
        except Exception as e:
            rd.log(f"  [WARN] left out: {type(e).__name__}: {e}")
            continue
        for k, s in info.iterrows():
            ser, la, lo, km = nearest_valid(da, s.sgylat, s.sgxlon)
            rows += [{"tskey": int(k), "source": name, "year": int(y), "sw_down_W_m2": float(v),
                      "cell_lat": la, "cell_lon": lo, "cell_km": km}
                     for y, v in zip(ser.year.values, ser.values)
                     if np.isfinite(v) and y >= FIRST_GRIDDED]
    df = pd.DataFrame(rows)
    df.insert(1, "station", df.tskey.map(info.sgname))
    df.insert(2, "lat", df.tskey.map(info.sgylat))
    df.insert(3, "lon", df.tskey.map(info.sgxlon))
    df.to_csv(data_file(), index=False, float_format="%.3f")
    rd.log(f"saved {data_file()}")


# ---------------------------------------------------------------- plot
def gap_series(ax, s, colour, years, lw, marker=None, ms=2.5, zorder=3):
    """s: Series indexed by year. Missing years break the line; isolated values show as dots."""
    y = s.reindex(years)
    ax.plot(years, y.values, color=colour, lw=lw, marker=marker, ms=ms, zorder=zorder,
            solid_capstyle="butt")
    if marker is None:
        v = y.values
        prev = np.concatenate([[np.nan], v[:-1]])
        nxt = np.concatenate([v[1:], [np.nan]])
        alone = np.isfinite(v) & ~np.isfinite(prev) & ~np.isfinite(nxt)
        ax.plot(np.asarray(years)[alone], v[alone], ls="none", marker="o", ms=ms, color=colour,
                zorder=zorder)


def fit_title(ax, text, colour, size, renderer):
    """Station title above the bias row, shrunk (down to TITLE_MIN_FS) to fit the panel width,
    then shortened if it still does not fit."""
    width = ax.get_window_extent(renderer).width
    t = ax.text(0, 1.2, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=size,
                fontweight="bold", color=colour)
    while t.get_window_extent(renderer).width > width and size > TITLE_MIN_FS:
        size -= 0.5
        t.set_fontsize(size)
    name, key = text.rsplit(" (", 1)
    while t.get_window_extent(renderer).width > width and len(name) > 8:
        name = name[:-1]
        t.set_text(f"{name.rstrip()}... ({key}")
    return t


def short_name(name, tskey, n=26):
    t = str(name).strip().title()
    return (t if len(t) <= n else t[:n - 1].rstrip() + "...") + f" ({tskey})"


def station_style(i):
    n = len(STATION_COLOURS)
    return STATION_COLOURS[i % n], STATION_MARKERS[(i // n) % len(STATION_MARKERS)]


def station_bias(g, name):
    """Mean (dataset - GEBA), correlation and the years used: the station's GEBA years (from
    FIRST_GRIDDED) for which the dataset has a value."""
    obs = g[(g.source == "GEBA") & (g.year >= FIRST_GRIDDED)].set_index("year").sw_down_W_m2
    x = g[g.source == name].set_index("year").sw_down_W_m2
    com = obs.index.intersection(x.index)
    bias = float((x[com] - obs[com]).mean()) if len(com) else np.nan
    r = float(np.corrcoef(x[com], obs[com])[0, 1]) if len(com) > 2 else np.nan
    return bias, r, com, len(obs)


def summary(df):
    """Per station and dataset: cell used, mean bias and correlation over the station's years."""
    rows = []
    for k, g in df.groupby("tskey", sort=False):
        for name in DATASETS:
            d = g[g.source == name]
            if d.empty:
                continue
            bias, r, com, n_obs = station_bias(g, name)
            rows.append({"tskey": k, "station": g.station.iloc[0], "dataset": name,
                         "cell_lat": d.cell_lat.iloc[0], "cell_lon": d.cell_lon.iloc[0],
                         "cell_km": d.cell_km.iloc[0], "geba_years": n_obs,
                         "years_used": len(com),
                         "first_last": f"{com.min()}-{com.max()}" if len(com) else "",
                         "mean_bias_W_m2": bias, "correlation": r})
    path = config.RESULTS / OUT / "geba_station_cells.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.3f")
    print(f"Saved: {path}", flush=True)


def plot():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.lines import Line2D
    if not data_file().exists():
        raise SystemExit("no data file: run 'python geba_station_sw.py compute' first")
    df = pd.read_csv(data_file())
    df = df[(df.source == "GEBA") | (df.source.isin(DATASETS) & (df.year >= FIRST_GRIDDED))]
    summary(df)
    ps.setup()
    fs, W = config.FS, config.WIDTH["full"]
    stations = df.drop_duplicates("tskey")[["tskey", "station", "lat", "lon"]].reset_index(drop=True)
    names = [n for n in DATASETS if n in set(df.source)]
    years = np.arange(df.year.min(), df.year.max() + 1)
    lo_, hi_ = df.sw_down_W_m2.min(), df.sw_down_W_m2.max()
    ylim = (np.floor(lo_ / 25) * 25, np.ceil(hi_ / 25) * 25)

    # layout (inches from the top-left)
    left, right, gx, gy = 0.72, 0.08, 0.12, 0.55
    pw = (W - left - right - gx) / 2
    ph = 1.1
    map_w, map_h = 2.2, 2.2
    top = 0.42
    p_top = top + map_h + 0.55
    nrow = int(np.ceil(len(stations) / 2))
    H = p_top + nrow * ph + (nrow - 1) * gy + 0.4
    fig, at = ps.figure("full", H)

    # map and legends
    ax = at(left, top, map_w, map_h, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale(config.COASTLINE), facecolor=LAND, zorder=0)
    ax.set_facecolor(SEA)
    ps.map_frame(ax, MAP, xticks=[15, 25, 35], yticks=[-30, -20, -10])
    ax.set_extent([LON_RANGE[0] - 1, LON_RANGE[1] + 1, -36, LAT_MAX + 1], ccrs.PlateCarree())
    ps.letter(ax, 0)
    st_handles = []
    for i, s in stations.iterrows():
        c, m = station_style(i)
        ax.scatter(s.lon, s.lat, s=28, color=c, marker=m, edgecolors="black", linewidths=0.5,
                   transform=ccrs.PlateCarree(), zorder=5)
        st_handles.append(Line2D([], [], ls="none", marker=m, ms=5.5, color=c, mec="black",
                                 mew=0.5, label=short_name(s.station, int(s.tskey))))
    x_leg = left + map_w + 0.25
    lax = at(x_leg, top, W - x_leg - right, map_h)
    lax.axis("off")
    leg = lax.legend(handles=st_handles, loc="upper left", frameon=False, fontsize=fs["legend"],
                     title="GEBA stations (global radiation)", title_fontsize=fs["legend"],
                     ncol=1 if len(st_handles) <= 10 else 2, handletextpad=0.3,
                     borderaxespad=0.0, alignment="left")
    lax.add_artist(leg)
    yrs = df.groupby("source").year.agg(["min", "max"])
    ds_handles = [Line2D([], [], color=STATION_LINE, lw=1.6, marker="o", ms=3,
                         label="GEBA station")]
    ds_handles += [Line2D([], [], color=ps.colour(n), lw=1.2,
                          label=f"{ps.label(n)} [{TAG[n]}]")
                   for n in names]
    lax.legend(handles=ds_handles, loc="lower left", frameon=False, fontsize=fs["legend"],
               ncol=2, handlelength=1.4, columnspacing=1.0, borderaxespad=0.0,
               alignment="left")

    # one panel per station
    renderer = fig.canvas.get_renderer()
    for i, s in stations.iterrows():
        r, c = divmod(i, 2)
        x0, t0 = left + c * (pw + gx), p_top + r * (ph + gy)
        ax = at(x0, t0, pw, ph)
        g = df[df.tskey == s.tskey]
        for n in names:
            d = g[g.source == n].set_index("year").sw_down_W_m2
            if not d.empty:
                gap_series(ax, d, ps.colour(n), years, lw=1.0)
        obs = g[g.source == "GEBA"].set_index("year").sw_down_W_m2
        gap_series(ax, obs, STATION_LINE, years, lw=1.6, marker="o", ms=2.5, zorder=5)
        ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.3, lw=0.4)
        last = i + 2 >= len(stations)
        if not last:
            ax.tick_params(labelbottom=False)
        if c == 0:
            ax.set_ylabel(f"{calendar.month_abbr[MONTH]} SW down\n(W m$^{{-2}}$)")
        else:
            ax.tick_params(labelleft=False)
        col, _ = station_style(i)
        fit_title(ax, f"{s.station.title()} ({int(s.tskey)})", col, fs["title"], renderer)
        # mean bias over this station's GEBA years: smallest |bias| first, each with its tag
        bias = [(n, station_bias(g, n)[0]) for n in names]
        bias.sort(key=lambda t: (not np.isfinite(t[1]), abs(t[1]) if np.isfinite(t[1]) else 0))
        prev = ax.text(0, 1.03, "Bias:", transform=ax.transAxes, ha="left", va="bottom",
                       fontsize=fs["small"], color="0.3")
        for n, b in bias:
            prev = ax.annotate(f" {TAG[n]} " + ("-" if not np.isfinite(b) else f"{round(b):+d}") + " ",
                               xy=(1, 0), xycoords=prev, va="bottom", fontsize=fs["small"],
                               fontweight="bold", color=ps.colour(n))
        ps.letter(ax, i + 1)
    ps.text(fig, W / 2, 0.06, f"{calendar.month_name[MONTH]} GEBA Stations", ha="center",
            va="top", fontsize=fs["suptitle"], fontweight="bold")
    ps.save(fig, config.RESULTS / OUT / f"geba_station_sw_{calendar.month_name[MONTH].lower()}.png")


# ---------------------------------------------------------------- check
def toa_month_mean(lat, month=MONTH, year=2001):
    """Top-of-atmosphere daily insolation (W m-2) averaged over the month's days."""
    S0 = 1361.0
    days = pd.date_range(f"{year}-{month:02d}-01", periods=calendar.monthrange(year, month)[1])
    g = 2 * np.pi * (days.dayofyear.values - 1) / 365.0
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    e0 = (1.000110 + 0.034221 * np.cos(g) + 0.001280 * np.sin(g) + 0.000719 * np.cos(2 * g)
          + 0.000077 * np.sin(2 * g))
    phi = np.deg2rad(lat)
    h0 = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    q = S0 / np.pi * e0 * (h0 * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(h0))
    return float(q.mean())


def check():
    """Report on the GEBA files and the values this script uses (see CHECK above)."""
    lines = []
    out = lambda t="": (print(t, flush=True), lines.append(t))
    fm, fd = newest("*_metadata.csv"), newest("*_monthly_data.csv")
    meta = pd.read_csv(fm, sep=";", low_memory=False)
    mon = pd.read_csv(fd, sep=";", low_memory=False)
    out(f"GEBA check, {calendar.month_name[MONTH]}\nmetadata: {fm} ({len(meta)} rows)\n"
        f"monthly:  {fd} ({len(mon)} rows)")
    for name, df in (("metadata", meta), ("monthly", mon)):
        out(f"\n{name} columns:")
        for c in df.columns:
            ex = df[c].dropna()
            out(f"  {c:<28} {str(df[c].dtype):<8} e.g. {', '.join(map(str, ex.unique()[:4]))}")
    out("\nenergy balance codes (ebcode), number of series:")
    desc = [c for c in meta.columns if c != "ebcode" and c.startswith("eb")]
    tab = meta.groupby("ebcode").size().sort_values(ascending=False)
    for code, n in tab.items():
        extra = "; ".join(str(meta.loc[meta.ebcode == code, c].iloc[0]) for c in desc)
        out(f"  {code:<14} {n:6d}  {extra}")

    st, vals = load_geba()
    out(f"\nSelected: {TARGET_TYPE}, {LON_RANGE[0]:.0f}-{LON_RANGE[1]:.0f}E, south of "
        f"{-LAT_MAX:.0f}S, > {MIN_ENTRIES} {calendar.month_abbr[MONTH]} values after {MIN_YEAR}: "
        f"{len(st)} series")
    for _, r in st.iterrows():
        out(f"\n  tskey {int(r.tskey)}:")
        for c in meta.columns:
            out(f"    {c:<26} {r[c]}")

    out("\nSeries at the same place (within 5 km) or with the same name (possible duplicates):")
    dup = False
    for i, a in st.iterrows():
        for j, b in st.iterrows():
            if j <= i:
                continue
            km = 111.2 * np.hypot(a.sgylat - b.sgylat,
                                  (a.sgxlon - b.sgxlon) * np.cos(np.deg2rad(a.sgylat)))
            if km < 5 or str(a.sgname).strip().lower() == str(b.sgname).strip().lower():
                dup = True
                out(f"  {int(a.tskey)} {a.sgname} and {int(b.tskey)} {b.sgname}: {km:.1f} km")
    if not dup:
        out("  none")

    for c in ("tskey", "year", "month", FLUX):
        mon[c] = pd.to_numeric(mon[c], errors="coerce")
    sel = mon[mon.tskey.isin(st.tskey) & (mon.month == MONTH) & (mon.year > MIN_YEAR)]
    flags = [c for c in mon.columns if any(k in c.lower() for k in ("flag", "qc", "qual"))]
    raw = [c for c in mon.columns if c != FLUX and "flux" in c.lower()]
    out(f"\nOctober values used ({FLUX}); flag-like columns: {', '.join(flags) or 'none'}; "
        f"other flux columns: {', '.join(raw) or 'none'}")
    for _, r in st.iterrows():
        d = sel[sel.tskey == r.tskey]
        v = d.dropna(subset=[FLUX])
        yrs = sorted(v.year.astype(int).unique())
        gaps = [y for y in range(yrs[0], yrs[-1] + 1) if y not in yrs]
        rep_ = v.groupby("year").size()
        toa = toa_month_mean(r.sgylat)
        k = v[FLUX] / toa
        out(f"\n  {int(r.tskey)} {r.sgname} ({r.sgylat:.2f}, {r.sgxlon:.2f}): {len(yrs)} years "
            f"{yrs[0]}-{yrs[-1]}; missing {gaps or 'none'}")
        out(f"    {FLUX}: min {v[FLUX].min():.1f}, median {v[FLUX].median():.1f}, "
            f"max {v[FLUX].max():.1f} W m-2; rows without a value: {int(d[FLUX].isna().sum())}")
        if (rep_ > 1).any():
            out(f"    more than one row in {list(rep_[rep_ > 1].index.astype(int))} "
                "(the script averages them)")
        out(f"    top of atmosphere {calendar.month_abbr[MONTH]} mean {toa:.0f} W m-2; clearness "
            f"index K {k.min():.2f}-{k.max():.2f}" + (
                f"  OUTSIDE {K_RANGE[0]}-{K_RANGE[1]} in "
                f"{list(v.year[(k < K_RANGE[0]) | (k > K_RANGE[1])].astype(int))}"
                if ((k < K_RANGE[0]) | (k > K_RANGE[1])).any() else "  (plausible)"))
        for c in raw:
            x = pd.to_numeric(v[c], errors="coerce")
            ok = x.notna() & (x != 0)
            if ok.any():
                f = (v[FLUX][ok] / x[ok])
                out(f"    {FLUX} / {c}: median {f.median():.4g} (range {f.min():.4g}-"
                    f"{f.max():.4g})")
        for c in flags:
            out(f"    {c}: " + ", ".join(f"{a} x{b}" for a, b in
                                         d[c].astype(str).value_counts().items()))
    check_gridded(out, st)
    path = config.RESULTS / OUT / "geba_check.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {path}")


# what each dataset is read from: (variable, conversion to W m-2, file finder)
def gridded_sources():
    f_era5 = config.RAW["era5"] / "global_surface_solar_radiation_downwards.nc"
    return {
        "ERA5": ("ssrd", 1 / 86400.0, "J m-2 (monthly mean of daily accumulations) / 86400",
                 lambda: {(0, MONTH): str(f_era5)} if f_era5.exists() else {}),
        "MERRA-2": ("SWGDN", 1.0, "none (W m-2)",
                    lambda: rd.merra2_files("tavgM_2d_rad_Nx")),
        "JRA-3Q": ("dswrf1have-sfc-fc-ll125", 1.0,
                   "none (W m-2; monthly mean of the file's 1-hourly averages)",
                   lambda: rd.jra3q_files("dswrf1have-sfc-fc-ll125")),
        "CLARA-A3": ("SIS", 1.0, "none (W m-2)", lambda: rd.cmsaf_files("SIS")),
        "SARAH-3": ("SIS", 1.0, "none (W m-2)",
                    lambda: rd.cmsaf_files("SIS", dirs=[config.RAW["sarah"]])),
    }


GLOBAL_KEYS = ("title", "product", "product_version", "source", "institution", "summary",
               "references", "history", "Conventions", "id", "keywords")


def check_gridded(out, st):
    """For each gridded dataset: the variable read, its attributes, the conversion applied and
    the converted October value at each GEBA station (with its clearness index)."""
    out("\n" + "=" * 90 + "\nGRIDDED DATASETS: the variable each one is read from")
    for name, (var, scale, conv, finder) in gridded_sources().items():
        out(f"\n--- {name}: {var}")
        try:
            files = finder()
        except Exception as e:
            out(f"  could not list the files: {type(e).__name__}: {e}")
            continue
        octo = [f for (y, m), f in files.items() if m == MONTH]
        if not octo:
            out("  no file found")
            continue
        path = octo[len(octo) // 2]                      # a mid-record October
        out(f"  {len(files)} file(s); inspected: {path}")
        try:
            ds = xr.open_dataset(path)
        except Exception as e:
            out(f"  could not open: {type(e).__name__}: {e}")
            continue
        with ds:
            v = var if var in ds else var.lower()
            if v not in ds:                              # JRA-3Q: the file's only 3-D field
                v = next((n for n, x in ds.data_vars.items() if x.ndim >= 3), v)
            if v not in ds:
                out(f"  VARIABLE {var} NOT IN THE FILE; data variables: {', '.join(ds.data_vars)}")
                continue
            da = ds[v]
            out(f"  dims: {dict(da.sizes)}")
            for k, a in da.attrs.items():
                out(f"    {k}: {a}")
            text = " ".join(str(da.attrs.get(k, "")) for k in
                            ("long_name", "standard_name", "description")).lower() + " " + v.lower()
            ok_name = (any(w in text for w in ("downw", "incoming", "incident", "ssrd", "sis"))
                       and any(w in text for w in ("shortwave", "short-wave", "solar", "ssrd",
                                                   "sis", "sw")))
            out(f"  names describe surface downwelling shortwave: "
                f"{'yes' if ok_name else 'CHECK (names do not say so)'}")
            out(f"  units in the file: {da.attrs.get('units', '?')}; conversion here: {conv}")
            gl = {k: ds.attrs[k] for k in GLOBAL_KEYS if k in ds.attrs}
            if gl:
                out("  global attributes:")
                for k, a in gl.items():
                    out(f"    {k}: {str(a)[:300]}")
            # converted October value at each station (nearest cell with data)
            tdim = next((d for d in ("time", "valid_time") if d in da.dims), None)
            if tdim:
                t = da[tdim]
                one_month = (np.issubdtype(t.dtype, np.datetime64) and t.size > 1
                             and np.unique(t.dt.strftime("%Y%m").values).size == 1)
                if one_month:                            # sub-monthly file: its monthly mean
                    out(f"  value checked: mean of {t.size} steps, {str(t.values[0])[:7]}")
                    da = da.mean(tdim)
                else:                                    # the first October in the file
                    i = 0
                    if np.issubdtype(t.dtype, np.datetime64):
                        hit = np.flatnonzero(t.dt.month.values == MONTH)
                        i = int(hit[0]) if hit.size else 0
                    out(f"  value checked: {str(t.values[i])[:10]}")
                    da = da.isel({tdim: i}, drop=True)
            da = rd.standardise(da).squeeze(drop=True)
            out(f"  converted October value at each station (nearest cell with data):")
            for _, r in st.iterrows():
                try:
                    sub = rd.cut(da, {"lon": (r.sgxlon - 1.5, r.sgxlon + 1.5),
                                      "lat": (r.sgylat - 1.5, r.sgylat + 1.5)}).load()
                    sub = sub.where(np.abs(sub) < 1e9)
                    if "year" in sub.dims:
                        sub = sub.isel(year=0)
                    sub = sub.expand_dims(year=[0])
                    ser, la, lo, km = nearest_valid(sub, r.sgylat, r.sgxlon)
                    val = float(ser.values[0]) * scale
                    k = val / toa_month_mean(r.sgylat)
                    flag = "" if K_RANGE[0] <= k <= K_RANGE[1] else "  CHECK"
                    out(f"    {int(r.tskey)} {str(r.sgname)[:24]:<24} {val:7.1f} W m-2 "
                        f"(cell {la:.2f}, {lo:.2f}, {km:.0f} km), K = {k:.2f}{flag}")
                except Exception as e:
                    out(f"    {int(r.tskey)} {r.sgname}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "check":
        check()
        sys.exit()
    if what == "compute" or (what == "all" and not data_file().exists()):
        compute()
    if what in ("plot", "all"):
        plot()
