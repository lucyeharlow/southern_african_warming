"""
Method schematic for the single column model (SCM) experiments, drawn with matplotlib
(vector shapes only, so it stays sharp at any size).

  a) the SCM: one atmospheric column over the land surface, driven by large-scale forcing,
     with T and q nudged towards the reference profiles on a 24 h timescale
  b) control runs: the SCM driven by the early (1980-1989) and late (2016-2025) October
     decadal-mean forcing -> surface SW down
  c) swap experiments: early-decade forcing with one variable taken from the late decade
  d) evaluation: the change from the swap as a fraction of the late-minus-early SW gap
(panel_forcing, the October forcing files averaged into the decadal means, is kept in the
script but not drawn)

Run:  python scm_schematic.py
FIGURE  results/schematic/scm_schematic.{png,pdf}

Edit the settings below (decades, variables, the highlighted swap variable, colours).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

# ---------------------------------------------------------------- settings
OUT = Path(__file__).resolve().parent.parent / "results" / "schematic"
YEARS = (1979, 2025)
EARLY = (1980, 1989)
LATE = (2016, 2025)
NUDGE_HOURS = 24
SWAP_VARS = ["Vertical velocity", "Temperature", "Soil moisture", "Specific humidity",
             "Surface roughness", "U", "V", "Skin temperature"]
HIGHLIGHT = "Vertical velocity"              # the swap shown as the example in c)

C_EARLY = "#E8DCAF"                          # early decade (beige)
C_LATE = "#F4A3B4"                           # late decade (pink)
C_MID = "#14A8E0"                            # other years (blue)
C_CARD = "#B9E2F7"                           # variable list card
C_LINE = "#1D1D1B"                           # flow arrows, fraction line, minus signs
C_T, C_Q = "#C63F3E", "#245E55"              # temperature and humidity profiles
EDGE = "#1D1D1B"
W, H = 6.5, 8.6                              # inches (AMS full width)
FS = 7.5

plt.rcParams.update({"font.size": FS, "font.family": "sans-serif",
                     "mathtext.default": "regular"})


# ---------------------------------------------------------------- drawing helpers
def box(ax, x, y, w, h, fc, text="", fs=FS, ec=EDGE, lw=0.8, r=0.06, weight="normal",
        z=2, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight=weight, zorder=z + 1, color=tc)


def gradient_box(ax, x, y, w, h, c1, c2, text="", fs=FS, r=0.06, z=2):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                           fc="none", ec=EDGE, lw=0.8, zorder=z + 1)
    ax.add_patch(patch)
    cmap = LinearSegmentedColormap.from_list("g", [c1, c1, c2])
    im = ax.imshow(np.linspace(0, 1, 256)[None, :], extent=(x, x + w, y, y + h),
                   cmap=cmap, aspect="auto", zorder=z, interpolation="bicubic")
    im.set_clip_path(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                zorder=z + 2)


def arrow(ax, x1, y1, x2, y2, c=C_LINE, lw=1.1, text="", tx=0, ty=0.06, fs=FS - 0.5,
          ha="center", style="-|>", ms=9, z=3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                                 color=c, lw=lw, zorder=z, shrinkA=0, shrinkB=0))
    if text:
        ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, text, ha=ha, va="bottom",
                fontsize=fs, zorder=z)


def sun(ax, cx, cy, r, body, rays, z=4):
    """Sun icon: body colour (or (c1, c2) for a left-to-right blend) and a list of ray
    colours (cycled over 8 rays)."""
    n = 8
    for k in range(n):
        a = np.pi / 2 + 2 * np.pi * k / n
        tip = (cx + 1.75 * r * np.cos(a), cy + 1.75 * r * np.sin(a))
        b1 = (cx + 1.15 * r * np.cos(a - 0.28), cy + 1.15 * r * np.sin(a - 0.28))
        b2 = (cx + 1.15 * r * np.cos(a + 0.28), cy + 1.15 * r * np.sin(a + 0.28))
        ax.add_patch(Polygon([tip, b1, b2], fc=rays[k % len(rays)], ec=EDGE, lw=0.6,
                             zorder=z))
    circ = plt.Circle((cx, cy), r, fc=body if isinstance(body, str) else "none",
                      ec=EDGE, lw=0.7, zorder=z + 1)
    ax.add_patch(circ)
    if not isinstance(body, str):
        cmap = LinearSegmentedColormap.from_list("s", [body[0], body[0], body[1]])
        im = ax.imshow(np.linspace(0, 1, 128)[None, :], extent=(cx - r, cx + r, cy - r, cy + r),
                       cmap=cmap, aspect="auto", zorder=z + 0.5, interpolation="bicubic")
        im.set_clip_path(circ)


def console(ax, x, y, w, h, n=9, label="SCM run"):
    """Model integration: a dark box with hourly timestep lines (as the SCM log)."""
    box(ax, x, y, w, h, "#111111", ec="#111111", r=0.03)
    for k in range(n):
        ax.text(x + 0.05, y + h - 0.06 - k * (h - 0.1) / n,
                f"Timestep {k:3d}  Day 0  Time {k:02d}:00:00", color="0.92",
                family="monospace", fontsize=3.0, va="top", zorder=4)
    ax.text(x + w / 2, y - 0.05, label, ha="center", va="top", fontsize=FS - 1,
            color="0.25")


def panel_title(ax, x, y, letter, text):
    ax.text(x, y, f"{letter})", fontsize=FS + 2, fontweight="bold", va="top")
    ax.text(x + 0.22, y - 0.005, text, fontsize=FS + 1, fontweight="bold", va="top")


def year_colour(yr):
    if EARLY[0] <= yr <= EARLY[1]:
        return C_EARLY
    if LATE[0] <= yr <= LATE[1]:
        return C_LATE
    return C_MID


# ---------------------------------------------------------------- panels
def panel_scm(ax, x0, y0):
    """a) the SCM column with nudged T and q (y0 = bottom of the panel, 2.75 in tall)."""
    panel_title(ax, 0.1, y0 + 2.75, "a", "Single column model")
    cx, cw = x0 + 1.15, 0.7                      # column
    zb, zt = y0 + 0.55, y0 + 1.85
    # land surface
    ax.add_patch(Rectangle((x0 + 0.35, zb - 0.18), 2.2, 0.18, fc="#9C7B55", ec=EDGE,
                           lw=0.6, zorder=2))
    
    # column with model levels
    ax.add_patch(Rectangle((cx, zb), cw, zt - zb, fc="#EAF6FB", ec=EDGE, lw=0.8, zorder=2))
    for z in np.linspace(zb, zt, 11)[1:-1]:
        ax.plot([cx, cx + cw], [z, z], color="0.7", lw=0.4, zorder=2.5)
    # large-scale forcing inside the column
    arrow(ax, cx + 0.2, zb + 0.2, cx + 0.2, zt - 0.2, c="0.25", lw=1.0, style="<|-|>", ms=7)
    ax.text(cx + 0.27, (zb + zt) / 2, r"$\omega$", fontsize=FS + 1, va="center")
    for z in (zb + 0.45, zt - 0.4):
        arrow(ax, cx + 0.38, z, cx + 0.64, z, c="0.25", lw=0.8, ms=6)
    ax.text(cx + cw / 2 + 0.17, zt - 0.3, "u, v", fontsize=FS - 1.5, ha="center")

    # T and q profiles beside the column: model (solid) nudged towards reference (dashed)
    z = np.linspace(zb + 0.05, zt - 0.05, 40)
    s = (z - zb) / (zt - zb)
    def profile(xl, xr, shape, offset, colour, name):
        ref = xl + (xr - xl) * shape(s)
        mod = ref + offset * np.sin(np.pi * s)
        ax.plot(ref, z, color=colour, lw=1.0, ls="--", zorder=3)
        ax.plot(mod, z, color=colour, lw=1.2, zorder=3)
        for k in (12, 26):
            arrow(ax, mod[k], z[k], ref[k] + np.sign(mod[k] - ref[k]) * 0.015, z[k],
                  c=colour, lw=0.8, ms=5)
        ax.text((xl + xr) / 2, zt + 0.04, name, ha="center", va="bottom", fontsize=FS,
                fontweight="bold", color=colour)
    profile(x0 + 0.95, x0 + 0.3, lambda s: s, 0.12, C_T, "T")
    profile(x0 + 2.65, x0 + 2.05, lambda s: 1 - np.exp(-3.5 * s), -0.12, C_Q, "q")
    ax.text(x0 + 1.5, y0 + 2.5, f"T and q nudged to the reference profiles,"
            f" $\\tau$ = {NUDGE_HOURS} h", ha="center", fontsize=FS - 0.5)
    ax.text(x0 + 1.5, y0 + 2.32, r"$\partial X/\partial t$ = physics + forcing"
            r" $-$ $(X - X_{ref})/\tau$", ha="center", fontsize=FS - 0.5)
    ax.plot([], [], color="0.3", lw=1.2, label="model")
    ax.plot([], [], color="0.3", lw=1.0, ls="--", label="reference")


def panel_forcing(ax, x0, y0):
    """b) the forcing files and the two decadal means."""
    panel_title(ax, x0, y0 + 2.75, "b", "October forcing")
    years = np.arange(YEARS[0], YEARS[1] + 1)
    n = years.size
    bx, by = x0 + 0.1, y0 + 2.2                  # top-left bar
    dx, dy, bw, bh = 0.029, -0.034, 0.05, 0.42
    # base
    base = [(bx - 0.06, by - bh - 0.02), (bx + (n - 1) * dx + bw + 0.06,
            by + (n - 1) * dy - bh - 0.02), (bx + (n - 1) * dx + bw + 0.1,
            by + (n - 1) * dy - bh - 0.14), (bx - 0.02, by - bh - 0.14)]
    ax.add_patch(Polygon(base, fc="#555555", ec="none", zorder=1))
    for i, yr in enumerate(years):
        ax.add_patch(Rectangle((bx + i * dx, by + i * dy - bh), bw, bh, fc=year_colour(yr),
                               ec=EDGE, lw=0.4, zorder=2 + i * 0.01))
    ax.text(x0 + 0.05, y0 + 0.75, f"Every October\nforcing file,\n{YEARS[0]}-{YEARS[1]}",
            ha="left", va="top", fontsize=FS - 0.5)
    # early and late groups -> decadal means
    ie = [i for i, y in enumerate(years) if EARLY[0] <= y <= EARLY[1]]
    il = [i for i, y in enumerate(years) if LATE[0] <= y <= LATE[1]]
    ex, ey = bx + ie[-1] * dx + bw + 0.05, by + ie[len(ie) // 2] * dy - bh / 2 + 0.12
    lx, ly = bx + il[-1] * dx + bw + 0.05, by + il[len(il) // 2] * dy - bh / 2
    bxx, bww = x0 + 1.95, 1.3
    arrow(ax, ex, ey, bxx - 0.03, ey)
    box(ax, bxx, ey - 0.17, bww, 0.34, C_EARLY, f"{EARLY[0]}-{EARLY[1]}\nOctober decadal mean",
        fs=FS - 0.5)
    arrow(ax, lx, ly, bxx - 0.03, ly)
    ax.text(bxx + bww / 2, (ey + ly) / 2, "averaged at\neach timestep", ha="center",
            va="center", fontsize=FS - 0.5, style="italic", color="0.25")
    box(ax, bxx, ly - 0.17, bww, 0.34, C_LATE, f"{LATE[0]}-{LATE[1]}\nOctober decadal mean",
        fs=FS - 0.5)
    return bxx + bww / 2, ly - 0.17


def panel_controls(ax, x0, y0, w):
    """c) control runs."""
    panel_title(ax, x0, y0 + 2.2, "b", "Control runs")
    for k, (dec, c, name) in enumerate([(EARLY, C_EARLY, "early"), (LATE, C_LATE, "late")]):
        cx = x0 + w * (0.27 + 0.46 * k)
        box(ax, cx - 0.75, y0 + 1.55, 1.5, 0.3, c, f"{dec[0]}-{dec[1]} forcing", fs=FS - 0.5)
        arrow(ax, cx, y0 + 1.55, cx, y0 + 1.33)
        console(ax, cx - 0.55, y0 + 0.88, 1.1, 0.45)
        arrow(ax, cx, y0 + 0.73, cx, y0 + 0.56)
        sun(ax, cx, y0 + 0.3, 0.13, c, [c])
        ax.text(cx + 0.33, y0 + 0.3, f"SW$\\downarrow_{{\\mathrm{{{name}}}}}$",
                fontsize=FS + 0.5, va="center")


def panel_swap(ax, x0, y0):
    """d) swap experiments."""
    panel_title(ax, x0, y0 + 2.55, "c", "Swap experiments")
    cw, ch = 1.62, 1.72
    cx, cy = x0 + 0.05, y0 + 0.25
    gradient_box(ax, cx, cy + ch + 0.08, cw, 0.3, C_EARLY, C_LATE,
                 f"{EARLY[0]}-{EARLY[1]} forcing,\none variable from {LATE[0]}-{LATE[1]}",
                 fs=FS - 1.5)
    # card with a folded corner
    ax.add_patch(Polygon([(cx, cy), (cx + cw - 0.16, cy), (cx + cw, cy + 0.16),
                          (cx + cw, cy + ch), (cx, cy + ch)], fc=C_CARD, ec=C_LATE,
                         lw=0.9, zorder=2))
    ax.add_patch(Polygon([(cx + cw - 0.16, cy), (cx + cw - 0.16, cy + 0.16),
                          (cx + cw, cy + 0.16)], fc="#7CC3E8", ec="none", zorder=3))
    step = (ch - 0.16) / len(SWAP_VARS)
    for k, v in enumerate(SWAP_VARS):
        yy = cy + ch - 0.12 - k * step
        if v == HIGHLIGHT:
            box(ax, cx + 0.07, yy - step * 0.45, cw - 0.2, step * 0.9, C_LATE, ec="#B57FB3",
                lw=0.6, r=0.03, z=3)
        ax.text(cx + 0.1, yy, f"• {v}", fontsize=FS - 1, va="center", zorder=4,
                color="#6B3060" if v == HIGHLIGHT else "black")
    # run -> mixed sun
    arrow(ax, cx + cw + 0.05, cy + ch / 2 + 0.2, cx + cw + 0.32, cy + ch / 2 + 0.2)
    console(ax, cx + cw + 0.36, cy + ch / 2 - 0.02, 1.0, 0.45)
    arrow(ax, cx + cw + 0.86, cy + ch / 2 - 0.17, cx + cw + 0.86, cy + ch / 2 - 0.36)
    sun(ax, cx + cw + 0.86, cy + 0.3, 0.13, (C_EARLY, C_LATE), [C_EARLY, C_LATE])
    ax.text(cx + cw + 1.17, cy + 0.3, r"SW$\downarrow_{\mathrm{swap}}$", fontsize=FS + 0.5,
            va="center")


def panel_eval(ax, x0, y0, w):
    """d) evaluation against the SW gap, as a formula."""
    panel_title(ax, x0, y0 + 2.55, "d", "Evaluation")
    mid = x0 + w / 2 + 0.12
    yc = y0 + 1.33                               # level with the arrow from c)
    sw = r"SW\!\downarrow"
    ax.text(mid, yc, r"$f_X = \dfrac{%s_{\mathrm{swap},X} - %s_{\mathrm{early}}}"
            r"{%s_{\mathrm{late}} - %s_{\mathrm{early}}}$" % (sw, sw, sw, sw),
            ha="center", va="center", fontsize=FS + 2)
    ax.text(mid, yc - 0.62, "Fraction of the late-minus-early SW gap\nproduced by swapping "
            "variable $X$", ha="center", va="top", fontsize=FS - 0.5, style="italic")
   

# ---------------------------------------------------------------- figure
def main():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    row1, row2, row3 = 5.7, 3.25, 0.1            # bottoms of the three rows
    panel_scm(ax, (W - 2.9) / 2, row1)
    panel_controls(ax, 0.1, row2, W - 0.2)
    panel_swap(ax, 0.1, row3)
    panel_eval(ax, 3.95, row3, 2.45)

  
    # separators
    for y in (row2 + 2.35, row3 + 2.85):
        ax.plot([0.1, W - 0.1], [y, y], color="0.85", lw=0.6, zorder=0)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"scm_schematic.{ext}", dpi=600, facecolor="white")
        print(f"Saved: {OUT / f'scm_schematic.{ext}'}")
    plt.close(fig)


if __name__ == "__main__":
    main()