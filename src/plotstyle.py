"""
One plotting style for every figure: AMS widths, one font, sizes at the printed size, a-style
panel letters, hyphens only (no en dashes, hyphen-minus for negative numbers), pcolormesh maps
with 50 m coastlines, cartopy gridlines with tick-based lat/lon labels, white significance dots
thinned to about config.DOT_TARGET per map, colour limits at the 98th percentile, and the same
area-weighted monthly boxplots everywhere.

Layout is in inches from the top-left corner: fig, at = figure("full", height);
ax = at(x, top, w, h[, projection=...]). bbox_inches="tight" is never used (it can crop
cartopy axes), so the saved figure is exactly config.WIDTH[...] wide.
"""
from __future__ import annotations

import calendar
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch

import config
import stats

MONTHS = np.arange(1, 13)
_FONT = None


def setup():
    """Apply the style (call once per script, before making figures)."""
    global _FONT
    have = {f.name for f in font_manager.fontManager.ttflist}
    _FONT = next((f for f in config.FONT if f in have), "DejaVu Sans")
    if _FONT not in ("Arial", "Helvetica"):
        warnings.warn(f"Arial/Helvetica not installed: using {_FONT} (metric-compatible with "
                      "Arial if Liberation Sans). Install Arial for the final figures.")
    fs = config.FS
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [_FONT],
        "mathtext.fontset": "custom", "mathtext.rm": _FONT, "mathtext.it": f"{_FONT}:italic",
        "mathtext.bf": f"{_FONT}:bold", "mathtext.cal": _FONT, "mathtext.sf": _FONT,
        "mathtext.tt": _FONT, "mathtext.default": "regular",          # (no 'cursive' lookups)
        "font.size": fs["label"], "axes.labelsize": fs["label"], "axes.titlesize": fs["title"],
        "xtick.labelsize": fs["tick"], "ytick.labelsize": fs["tick"],
        "legend.fontsize": fs["legend"], "axes.unicode_minus": False,      # hyphen-minus
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.2,
        "savefig.dpi": config.DPI, "savefig.facecolor": "white", "figure.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def figure(width, height):
    """(fig, at): a figure config.WIDTH[width] inches wide; at(x, top, w, h, **kw) adds axes
    placed in inches from the top-left corner."""
    W = config.WIDTH[width] if isinstance(width, str) else float(width)
    fig = plt.figure(figsize=(W, height))
    at = lambda x, t, w, h, **kw: fig.add_axes([x / W, 1 - (t + h) / height, w / W, h / height], **kw)
    fig._inches = (W, height)
    return fig, at


def text(fig, x, top, s, **kw):
    """Figure text at (x, top) inches from the top-left corner."""
    W, H = fig._inches
    kw.setdefault("va", "bottom")
    return fig.text(x / W, 1 - top / H, s, **kw)


def letter_text(i):
    """Panel label for index i: a ... z, then aa, ab, ... (spreadsheet style)."""
    abc, out = "abcdefghijklmnopqrstuvwxyz", ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        out = abc[r] + out
    return out


def letter(ax, i, inside=True, **kw):
    """Panel letter 'a)', 'b)', ... in the top-left corner of ax (inside on a white box)."""
    s = f"{letter_text(i)})"
    kw = dict(fontsize=config.FS["letter"], fontweight="bold", transform=ax.transAxes, **kw)
    if inside:
        return ax.text(0.02, 0.97, s, va="top", ha="left", zorder=10,
                       bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8), **kw)
    return ax.text(0.0, 1.02, s, va="bottom", ha="left", **kw)


def span(years):
    """'1980-2025' (hyphen); a list with gaps gets its count."""
    y = sorted(int(v) for v in years)
    return f"{y[0]}-{y[-1]}" + ("" if len(y) == y[-1] - y[0] + 1 else f", {len(y)} yrs")


def year_list(attr):
    """'1980-2025' -> [1980, ..., 2025]."""
    y0, y1 = (int(v) for v in str(attr).split("-"))
    return list(range(y0, y1 + 1))


def label(name):
    return config.DATASETS[name]["label"]


def colour(name):
    return config.DATASETS[name]["colour"]


# ---------------------------------------------------------------- colours
def sym_norm(fields, cmap="RdBu_r", pct=None, nbins=12):
    """Discrete symmetric levels up to the pct-th percentile of |values| over all fields."""
    pct = config.COLOUR_PERCENTILE if pct is None else pct
    v = np.concatenate([np.abs(np.asarray(f, float)).ravel() for f in fields])
    v = v[np.isfinite(v)]
    lim = float(np.percentile(v, pct)) if v.size else 1.0
    lim = lim if lim > 0 else 1.0
    bounds = mticker.MaxNLocator(nbins=nbins, symmetric=True).tick_values(-lim, lim)
    return BoundaryNorm(bounds, plt.get_cmap(cmap).N, extend="both")


def colourbar(fig, cax, norm, cmap, label_, orientation="horizontal", nticks=7):
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation=orientation, extend="both")
    b = norm.boundaries
    step = max(1, int(np.ceil((len(b) - 1) / (nticks - 1))))
    mid = len(b) // 2
    cb.set_ticks(np.concatenate([b[mid::-step][::-1], b[mid + step::step]]))
    cb.set_label(label_, fontsize=config.FS["label"])
    cb.ax.tick_params(labelsize=config.FS["tick"], length=2)
    cb.outline.set_linewidth(0.5)
    return cb


# ---------------------------------------------------------------- maps
def _ticks(lo, hi, lat=False):
    span_ = hi - lo
    if lat and span_ > 120:
        step = 30
    else:
        step = 5 if span_ <= 20 else 10 if span_ <= 80 else 30 if span_ <= 160 else 60
    return np.arange(np.ceil(lo / step) * step, hi + 1e-6, step)


def map_frame(ax, region, left=True, bottom=True, box=None, xticks=None, yticks=None):
    """Coastlines and borders (config.COASTLINE), extent, cartopy gridlines and tick labels."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    r = config.REGIONS[region] if isinstance(region, str) else region
    pc = ccrs.PlateCarree()
    ax.add_feature(cfeature.COASTLINE.with_scale(config.COASTLINE), linewidth=0.5, zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale(config.COASTLINE), linewidth=0.3,
                   edgecolor="0.3", zorder=4)
    if box is not None:
        b = config.REGIONS[box] if isinstance(box, str) else box
        ax.plot([b["lon"][0], b["lon"][1], b["lon"][1], b["lon"][0], b["lon"][0]],
                [b["lat"][0], b["lat"][0], b["lat"][1], b["lat"][1], b["lat"][0]],
                color="k", lw=0.8, ls="--", transform=pc, zorder=5)
    xt = np.asarray(xticks, float) if xticks is not None else _ticks(*r["lon"])
    yt = np.asarray(yticks, float) if yticks is not None else _ticks(*r["lat"], lat=True)
    ax.gridlines(crs=pc, xlocs=xt, ylocs=yt, draw_labels=False, linewidth=0.3, color="0.5",
                 alpha=0.6, linestyle=":")
    xt = xt[xt < r["lon"][1] - 1e-6] if xt.size > 2 else xt     # no label on the right edge,
    ax.set_xticks(xt, crs=pc)                                  # where it meets the next map
    ax.set_yticks(yt, crs=pc)
    ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol="°"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol="°"))
    ax.tick_params(labelsize=config.FS["tick"], length=2, pad=1.5,
                   labelleft=left, labelbottom=bottom)
    if r["lon"][1] - r["lon"][0] >= 359 and r["lat"][1] - r["lat"][0] >= 179:
        ax.set_global()
    else:
        ax.set_extent([r["lon"][0], r["lon"][1], r["lat"][0], r["lat"][1]], crs=pc)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def thin_dots(sig, valid, sy, sx):
    """Fractional (row, column) positions for thinned significance dots: one dot per block of
    sy x sx cells where at least half of the block's valid cells are significant, placed at
    the mean position of its significant cells (so small significant patches still get a
    dot, and a dot never sits on a non-significant area)."""
    sig = np.asarray(sig, bool)
    valid = np.asarray(valid, bool) | sig
    rows, cols = [], []
    for i0 in range(0, sig.shape[0], sy):
        for j0 in range(0, sig.shape[1], sx):
            s, v = sig[i0:i0 + sy, j0:j0 + sx], valid[i0:i0 + sy, j0:j0 + sx]
            if s.any() and s.sum() >= 0.5 * v.sum():
                ii, jj = np.nonzero(s)
                rows.append(i0 + ii.mean())
                cols.append(j0 + jj.mean())
    return np.array(rows), np.array(cols)


def _at(coord, idx):
    c = np.asarray(coord, float)
    return np.interp(idx, np.arange(c.size), c)


def sig_dots(ax, field, sig):
    """Dots (config.DOT_COLOUR) on significant areas, about config.DOT_TARGET per map at
    most (thin_dots)."""
    import cartopy.crs as ccrs
    sig = np.asarray(sig, bool)
    if not sig.any():
        return
    valid = np.isfinite(np.asarray(field, float))
    step = max(1, int(np.ceil(np.sqrt(valid.sum() / config.DOT_TARGET))))
    i, j = thin_dots(sig, valid, step, step)
    ax.scatter(_at(field.lon.values, j), _at(field.lat.values, i), s=config.DOT_SIZE,
               color=config.DOT_COLOUR, linewidths=0, transform=ccrs.PlateCarree(), zorder=3)


def section_dots(ax, x, y, sig, valid=None):
    """Dots on significant areas of a (y, x) cross section, at most about config.SECTION_DOTS
    (columns, rows) per panel (thin_dots)."""
    sig = np.asarray(sig, bool)
    if not sig.any():
        return
    nx, ny = config.SECTION_DOTS
    sx = max(1, int(np.ceil(sig.shape[1] / nx)))
    sy = max(1, int(np.ceil(sig.shape[0] / ny)))
    i, j = thin_dots(sig, sig if valid is None else valid, sy, sx)
    ax.scatter(_at(x, j), _at(y, i), s=config.DOT_SIZE * 1.5, color=config.DOT_COLOUR,
               linewidths=0, zorder=3)


def dot_handle(label_):
    """Legend entry for the significance dots."""
    from matplotlib.lines import Line2D
    return Line2D([], [], marker="o", ls="none", color=config.DOT_COLOUR, markersize=2.5,
                  label=label_)


def map_panel(ax, field, norm, cmap="RdBu_r", sig=None, region="southern_africa",
              left=True, bottom=True, box=None, **ticks):
    """A (lat, lon) field drawn cell by cell, dots where sig, and the map frame."""
    import cartopy.crs as ccrs
    m = ax.pcolormesh(field.lon, field.lat, field, cmap=cmap, norm=norm, shading="nearest",
                      transform=ccrs.PlateCarree(), rasterized=True, zorder=1)
    if sig is not None:
        sig_dots(ax, field, sig)
    map_frame(ax, region, left, bottom, box, **ticks)
    return m


# ---------------------------------------------------------------- monthly boxplots
def monthly_boxplot(ax, fields, names, ylabel):
    """fields: {name: {month: (lat, lon) DataArray}}. One area-weighted box per dataset and
    month (box = weighted IQR, line = weighted median, whiskers = most extreme cell within
    1.5 IQR), datasets side by side in config colours."""
    monthly_boxplot_stats(ax, {n: {m: stats.box_stats(f) for m, f in fields[n].items()}
                               for n in names if fields.get(n)}, names, ylabel)


def monthly_boxplot_stats(ax, boxes_by_name, names, ylabel):
    """As monthly_boxplot, from box statistics already computed (stats.box_stats dicts):
    boxes_by_name = {name: {month: dict or None}}."""
    line = dict(color="black", linewidth=0.6)
    names = [n for n in names if boxes_by_name.get(n)]
    width = 0.8 / (len(names) + 0.5)
    offs = (np.arange(len(names)) - (len(names) - 1) / 2) * width
    for name, off in zip(names, offs):
        boxes = boxes_by_name[name]
        have = [m for m in MONTHS if boxes.get(m)]
        if have:
            ax.bxp([boxes[m] for m in have], positions=np.array(have) + off, widths=width * 0.85,
                   patch_artist=True, showfliers=False, manage_ticks=False, whiskerprops=line,
                   capprops=line, medianprops=line, capwidths=width * 0.5,
                   boxprops=dict(facecolor=colour(name), edgecolor="black", linewidth=0.6))
    ax.axhline(0, linestyle="--", linewidth=0.6, color="black", zorder=0)
    ax.set(xlim=(0.5, 12.5), xticks=MONTHS, xticklabels=[m[0] for m in calendar.month_abbr[1:]])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)


def dataset_legend(ax_or_fig, names, years=None, **kw):
    """Coloured patches with 'NAME (first-last)'."""
    h = [Patch(facecolor=colour(n), edgecolor="black", linewidth=0.5,
               label=label(n) + (f" ({span(years[n])})" if years and n in years else ""))
         for n in names]
    kw.setdefault("frameon", False)
    kw.setdefault("fontsize", config.FS["legend"])
    return ax_or_fig.legend(handles=h, **kw)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}", flush=True)


# ---------------------------------------------------------------- category maps
def shade_cmap(colour_, start="white"):
    """From start (no signal) to colour_ (full signal)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("", [start, colour_])


def categories(ax, x, y, code, strength, colours, other=None, transform=None, cmaps=None):
    """Cells coloured by category (code = index into colours, -1 none), shaded from white by
    strength (0-1); 'other' cells light grey (config.C_OTHER). Works on maps (transform) and
    plain axes (cross sections)."""
    from matplotlib.colors import ListedColormap, Normalize
    kw = dict(shading="nearest", rasterized=True, zorder=1)
    if transform is not None:
        kw["transform"] = transform
    if other is not None and np.any(other):
        ax.pcolormesh(x, y, np.where(other, 1.0, np.nan), cmap=ListedColormap([config.C_OTHER]),
                      vmin=0, vmax=1, **kw)
    for k, c in enumerate(colours):
        f = np.where(code == k, strength, np.nan)
        if np.isfinite(f).any():
            ax.pcolormesh(x, y, f, cmap=(cmaps[k] if cmaps else shade_cmap(c)),
                          norm=Normalize(0, 1), **kw)


def shade_bars(fig, at, x, top, w, h, colours, vmax, label_, cmaps=None, vmin=0.0):
    """One thin vertical colour bar per category (white -> colour), ticks on the last."""
    from matplotlib.colors import Normalize
    k = len(colours)
    bw = w / (k + 0.4 * (k - 1))
    for i, c in enumerate(colours):
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin, vmax),
                                                cmap=cmaps[i] if cmaps else shade_cmap(c)),
                          cax=at(x + i * 1.4 * bw, top, bw, h), orientation="vertical")
        cb.outline.set_linewidth(0.4)
        if i < k - 1:
            cb.set_ticks([])
        else:
            cb.ax.tick_params(labelsize=config.FS["tick"], length=2)
            cb.set_label(label_, fontsize=config.FS["label"])


def regime_handles():
    return [Patch(facecolor=config.REGIME_COLOURS[k], label=config.REGIME_LABELS[k])
            for k in config.REGIMES]
