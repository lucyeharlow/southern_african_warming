"""
Settings shared by every figure script in southern_african_warming/src.

Change a setting here and every figure follows. Figure scripts read these values when they
run (config.X inside functions), so a test or a one-off run can override them first.

Layout of the repository
  data/            small files the figures are drawn from (trends, series), one per
                   figure / variable / dataset; written by "<script>.py compute"
  processed_data/  caches of monthly fields read from the raw files (rebuilt if deleted)
  results/         figures (PNG) and summary tables (CSV)
  src/             this code
  doc/             notes
"""
from pathlib import Path

# ---------------------------------------------------------------- repository
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = ROOT / "processed_data"
CACHE = PROCESSED / "cache"
RESULTS = ROOT / "results"

# ---------------------------------------------------------------- raw data
HOME = Path("/soge-home/users/jesu4127")
SCRATCH = HOME / "scratch"
RAW = {
    "era5": SCRATCH / "era5",
    "merra2": SCRATCH / "merra_2",
    "jra3q": SCRATCH / "jra_3q",
    "clara": [HOME / "data/clara_a3", HOME / "Data_P1/clara_a3/data"],
    "sarah": HOME / "data/sarah_3",
    "chirts": HOME / "data/chirts",
    "noaa": HOME / "data/mlost/gridded",
    "crutem": HOME / "data/crutem/CRUTEM.5.1.0.0.anomalies.nc",
    "modis_terra": HOME / "data/modis/MOD08_M3",
    "modis_aqua": HOME / "data/modis/MYD08_M3",
    "ceres": HOME / "data/ceres",
    "geba": HOME / "data/geba",
    "jra3q_grib": Path("/soge-home/data/analysis/jra-q3"),       # 6-hourly GRIB analyses
    "era5_chi": HOME / "data/era5_vel_pot/chi_october_pl.nc",
    "era5_sst": HOME / "data/era5/global_era5_ssts.nc",             # monthly SST (sst, K)
    "cmip6_vp": HOME / "data/cmip6_vel_pot/velocity_potential_trends",
}

# JRA-3Q monthly means already computed by the old scripts (reused, so files are not read again)
OLD_JRA3Q_CACHES = [HOME / "Paper_1/radiation/global_trends/cache",
                    HOME / "Paper_1/radiation/qnet_budget/cache",
                    HOME / "Paper_1/detrended_corr/cache"]

# ---------------------------------------------------------------- years
# Every dataset uses exactly these years. A cell needs a value in every one of them
# (cells with any missing year are left out). A year missing from a whole dataset is an
# error (STRICT_YEARS), so a figure never silently changes its period.
YEARS = {
    "ERA5": (1980, 2025),
    "MERRA-2": (1980, 2025),
    "JRA-3Q": (1980, 2025),
    "CLARA-A3": (1982, 2024),
    "SARAH-3": (1983, 2024),
    "CHIRTS": (1983, 2016),
    "NOAA": (1980, 2025),        # kept on the reanalysis period, as in the original script
    "CRUTEM5": (1980, 2025),
    "MODIS Terra": (2002, 2024),
    "MODIS Aqua": (2002, 2024),
    "MODIS Terra+Aqua": (2002, 2024),
    "CERES": (2002, 2024),
}
# Months where a dataset has a whole month missing from the product (an empty file), so no
# cell has a complete record: that month uses these years instead. Shown in the panel title.
YEARS_BY_MONTH = {
    ("CLARA-A3", 2): (1986, 2024),   # February 1985 is empty in CLARA-A3 (SIS, SNS, SDL, SNL)
}


def years(name, month=None):
    """Years used for a dataset (and month): YEARS_BY_MONTH first, then YEARS."""
    return YEARS_BY_MONTH.get((name, month), YEARS[name])


STRICT_YEARS = True
BASELINE = (1991, 2020)          # anomalies are relative to this monthly climatology

# ---------------------------------------------------------------- statistics
ALPHA_FDR = 0.10                 # Benjamini-Hochberg, over the land cells of each map
LAND_MIN_FRACTION = 0.5          # a cell is land if at least half of it is land
LAND_SHAPEFILE = None            # None: Natural Earth 50 m via cartopy; or a local .shp
MASK_BELOW_GROUND = True         # pressure levels below the surface are left out
COLOUR_PERCENTILE = 98           # colour bar limit: this percentile of |field|

# ---------------------------------------------------------------- regions (degrees)
REGIONS = {
    "southern_africa": dict(lon=(10.0, 40.0), lat=(-35.0, -10.0)),
    "africa_south": dict(lon=(8.0, 52.0), lat=(-37.0, 0.0)),
    "sub_saharan": dict(lon=(-20.0, 55.0), lat=(-37.0, 18.0)),
    "africa_10_40_s": dict(lon=(10.0, 40.0), lat=(-35.0, 0.0)),   # the same, south of the equator
    "africa_10_40": dict(lon=(10.0, 40.0), lat=(-35.0, 18.0)),    # maps: the box's longitudes,
                                                                  # to 18N (no Madagascar)
    "cloud_maps": dict(lon=(-15.0, 50.0), lat=(-35.0, 20.0)),     # satellite cloud figure
    "cloud_bands": dict(lon=(10.0, 38.0), lat=(-35.0, 10.0)),     # its latitude-band region
    "omega": dict(lon=(-10.0, 60.0), lat=(-35.0, 20.0)),          # vertical velocity figure
    "omega_sfc": dict(lon=(10.0, 40.0), lat=(-35.0, 10.0)),       # omega300 correlations
    "tropics": dict(lon=(-120.0, 120.0), lat=(-40.0, 40.0)),      # global flux trends
    "tropics_band": dict(lon=(-180.0, 180.0), lat=(-23.5, 23.5)),  # temperature by region
    "composite": dict(lon=(-20.0, 60.0), lat=(-40.0, 20.0)),     # chi200 composite maps
    "sst_band": dict(lon=(-180.0, 180.0), lat=(-60.0, 60.0)),    # their SST maps
    "global": dict(lon=(-180.0, 180.0), lat=(-90.0, 90.0)),
}

# ---------------------------------------------------------------- datasets
DATASETS = {   # label, colour (the same in every figure)
    "ERA5": dict(label="ERA5", colour="#FF7C7E"),
    "MERRA-2": dict(label="MERRA-2", colour="#FEC300"),
    "JRA-3Q": dict(label="JRA-3Q", colour="#016FF2"),
    "CLARA-A3": dict(label="CLARA-A3", colour="#4C9A2A"),
    "SARAH-3": dict(label="SARAH-3", colour="#8C510A"),
    "MODIS Terra": dict(label="MODIS Terra", colour="#004D40"),
    "MODIS Aqua": dict(label="MODIS Aqua", colour="#1E88E5"),
    "MODIS Terra+Aqua": dict(label="MODIS Terra+Aqua", colour="#FFC107"),
    "CERES": dict(label="CERES EBAF", colour="#5E35B1"),
    "GEBA": dict(label="GEBA", colour="#333333"),
    "CHIRTS": dict(label="CHIRTS", colour="#9E3F71"),
    "NOAA": dict(label="NOAA", colour="#31BCBA"),
    "CRUTEM5": dict(label="CRUTEM5", colour="#2F5597"),
}

# ---------------------------------------------------------------- figures (AMS)
# AMS widths: 2 columns 33 picas = 5.5 in (14 cm); more than 2 columns 39 picas = 6.5 in
# (16.5 cm). Figures are drawn at their printed size, so the font sizes are the printed sizes.
WIDTH = {"two": 5.5, "full": 6.5}
DPI = 600
FONT = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]   # first one installed
FS = dict(tick=8.5, label=9, title=10, letter=10, legend=8.5, small=8, suptitle=11)
                                 # pt at printed size; AMS asks for about 8-12 pt
COASTLINE = "50m"
DOT_TARGET = 400                 # significance dots: about this many per map at most
DOT_SIZE = 1.5                   # points^2
DOT_COLOUR = "black"             # significance dots in every figure
SECTION_DOTS = (14, 12)          # cross sections: at most this many dot columns x rows per panel

# ---------------------------------------------------------------- omega regimes
# omega > 0 is subsidence. A regime per cell from the October omega series (stats.regimes).
MIN_PERSIST = 10                 # years of one sign after a switch for a "new" regime
REGIMES = ["str_sub", "wk_sub", "new_sub", "wk_asc", "str_asc", "new_asc"]
REGIME_LABELS = {"str_sub": "Strengthening subsidence", "wk_sub": "Weakening subsidence",
                 "new_sub": "New subsidence", "wk_asc": "Weakening ascent",
                 "str_asc": "Strengthening ascent", "new_asc": "New ascent"}
REGIME_COLOURS = {"str_sub": "#DC2671", "wk_sub": "#FE6100", "new_sub": "#FFB000",
                  "wk_asc": "#4DB5FA", "str_asc": "#785EF0", "new_asc": "#60D691"}
C_OTHER = "#E6E6E6"              # cells shown in the other row/column of a category figure
