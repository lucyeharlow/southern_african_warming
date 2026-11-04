"""
Land and below-ground masks, on any dataset's own grid.

land_mask: a cell is land if at least config.LAND_MIN_FRACTION of it is land in the Natural
Earth 50 m land polygons (estimated from a k x k sub-grid of points, ~0.1 deg apart, per cell).
above_ground: True where a pressure level is above the (monthly climatological) surface.
"""
from __future__ import annotations

from functools import cache

import numpy as np
import xarray as xr

import config


@cache
def _land(lon0, lon1, lat0, lat1):
    from cartopy.io import shapereader
    from shapely import box
    from shapely.ops import unary_union
    path = config.LAND_SHAPEFILE or shapereader.natural_earth("50m", "physical", "land")
    land = unary_union(list(shapereader.Reader(str(path)).geometries()))
    return land.intersection(box(lon0, lat0, lon1, lat1))


def land_fraction(lat, lon):
    """(lat, lon) DataArray of the land fraction of each cell (0-1)."""
    from shapely import contains_xy
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    step = lambda x: float(np.median(np.abs(np.diff(x)))) if x.size > 1 else 1.0
    dlat, dlon = step(lat), step(lon)
    k = int(np.clip(np.ceil(max(dlat, dlon) / 0.1), 1, 10))
    off = (np.arange(k) + 0.5) / k - 0.5
    la = lat[:, None, None, None] + off[None, None, :, None] * dlat
    lo = lon[None, :, None, None] + off[None, None, None, :] * dlon
    la, lo = np.broadcast_arrays(la, lo)
    geom = _land(round(lon.min() - 2), round(lon.max() + 2), round(lat.min() - 2),
                 round(lat.max() + 2))
    frac = contains_xy(geom, lo, la).mean(axis=(2, 3))
    return xr.DataArray(frac, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


def land_mask(da_or_lat, lon=None):
    """Boolean (lat, lon) land mask for a DataArray's grid (or for lat, lon arrays)."""
    if lon is None:
        lat, lon = da_or_lat.lat.values, da_or_lat.lon.values
    else:
        lat = da_or_lat
    return land_fraction(lat, lon) >= config.LAND_MIN_FRACTION


def above_ground(lev, ps):
    """lev: pressure levels (hPa, DataArray with dim 'lev'); ps: surface pressure (hPa) on the
    field's grid. True where the level is above the ground (p <= ps)."""
    if not config.MASK_BELOW_GROUND or ps is None:
        return xr.ones_like(lev * 1.0, dtype=bool)
    return lev <= ps
