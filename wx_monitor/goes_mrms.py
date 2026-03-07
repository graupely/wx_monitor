"""
wx_monitor.goes_mrms
====================
Download GOES-19 CONUS **GeoColor** composite + MRMS composite reflectivity
and save a single PNG with reflectivity overlaid on the satellite image.

GeoColor algorithm  (Miller et al. 2020, J. Atmos. Oceanic Technol., 37, 429-448)
----------------------------------------------------------------------------------
Daytime true color (µ0 >= 0.3, i.e. SZA <= 72.5 deg)
    R = C02  (0.64 µm, Red,    0.5 km)
    G = 0.48358168·C02 + 0.45706946·C01 + 0.06038137·C03  (hybrid simulated green)
    B = C01  (0.47 µm, Blue,   1.0 km)
    Scaling: truncate reflectance to [0.025, 1.20], apply log10,
             normalise over [-1.6, 0.176] (Miller et al. §5a, p.438-439)

Nighttime  (µ0 <= 0.1, i.e. SZA >= 84.3 deg)  -- three-layer nested stack (Eq. 3)
    L1 high cloud : N_IR  = 1 - N(C13)[IRmin(lat), 280]   white on dark
    L2 low  cloud : N_LC  = N(BTD=C13-C07)[1.0, 4.5]      white on dark
    L3 background : fixed dark nightscape (city lights/terrain omitted)
    C_night = N_IR + (1-N_IR)*[N_LC + (1-N_LC)*BG]
    IRmin varies with latitude per Miller et al. Eq. 9:
        200 K (lat<30), 200+20*(lat-30)/30 (30<=lat<=60), 220 K (lat>60)
    BTD zero-masked where C13 < 230 K to suppress deep-convection noise.

Twilight blend  (0.1 < µ0 < 0.3, SZA 72.5-84.3 deg)
    N_µ0 = clip((µ0 - 0.1)/(0.3 - 0.1), 0, 1)^1.5  [Miller et al. Eq. 5]
    C = N_µ0 · day_RGB + (1 - N_µ0) · night_RGB
    N_µ0 is computed **per pixel** using lat/lon derived from ABI fixed grid.

Divergences from full V2.0
    - Rayleigh scattering correction omitted (paper labels it optional preprocessing).
    - City lights/terrain background omitted (requires static VIIRS-DNB + DEM datasets).
    - Land/sea mask for BTD thresholds simplified (uniform land bounds used).

References
----------
Miller et al. 2020  https://doi.org/10.1175/JTECH-D-19-0134.1
GOES-R ABI L1b Product User Guide (PUG)
"""

from __future__ import annotations

import gzip
import math
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import s3fs
import xarray as xr
from matplotlib.colors import BoundaryNorm
from pyproj import Transformer
from scipy.ndimage import zoom as nd_zoom

from wx_monitor.config import (
    CONUS_EXTENT,
    GOES_BUCKET,
    GOES_PRODUCT,
    MRMS_URL,
    OUTPUT_DPI,
    VALID_TIME,
)
from wx_monitor.utils import download_bytes, parse_goes_scan_time, parse_mrms_time


# ---------------------------------------------------------------------------
# GeoColor band configuration
# ---------------------------------------------------------------------------

# µ0 = cos(SZA) thresholds from Miller et al. Eq. 5 / §4
# µ0 = 0.3  <=>  SZA = 72.5 deg  (fade START — fully day above)
# µ0 = 0.1  <=>  SZA = 84.3 deg  (fade END   — fully night below)
MU0_DAY   = 0.3    # fully daytime
MU0_NIGHT = 0.1    # fully nighttime

# Nighttime IR scaling bounds (latitude-independent parts; lat-varying in code)
IR_MAX = 280.0   # K — warm (surface) anchor, always fixed

# Dark nightscape background colour for the surface layer (simplified, no city lights)
# R, G, B in [0, 1]
BG_R, BG_G, BG_B = 0.03, 0.03, 0.07   # very dark blue

# Colour of the liquid-water cloud layer (Layer 2 of the nighttime stack).
# Operational GeoColor: fog/stratus = medium blue; ice clouds (Layer 1) = white.
LC_R, LC_G, LC_B = 0.60, 0.72, 0.90   # pale blue-white — liquid water clouds / fog / stratus

# ---------------------------------------------------------------------------
# Radar colormap
# ---------------------------------------------------------------------------
RADAR_LEVELS = [-10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
RADAR_COLORS = [
    "#00ecec", "#01a0f6", "#0000f6", "#00ff00",
    "#00c800", "#009000", "#ffff00", "#e7c000",
    "#ff9000", "#ff0000", "#d60000", "#c00000",
    "#ff00ff", "#9955cc", "#ffffff", "#aaaaaa",
]
radar_cmap = mcolors.ListedColormap(RADAR_COLORS)
radar_norm = BoundaryNorm(RADAR_LEVELS, ncolors=len(RADAR_COLORS))


# ---------------------------------------------------------------------------
# Solar geometry
# ---------------------------------------------------------------------------

def _solar_cos_zenith(
    dt: datetime,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """
    Vectorised computation of µ0 = cos(SZA) for 2D lat/lon arrays.
    Accurate to ~1 deg — sufficient for the [0.1, 0.3] twilight window.
    Returns float32 array matching lats/lons shape, values in [-1, 1].
    """
    doy      = dt.timetuple().tm_yday
    decl_rad = np.float32(math.radians(23.45 * math.sin(math.radians(360.0 / 365.0 * (doy - 81)))))
    solar_h  = dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + lons / 15.0
    ha_rad   = np.radians(15.0 * (solar_h - 12.0)).astype(np.float32)
    lat_rad  = np.radians(lats).astype(np.float32)
    mu0 = (np.sin(lat_rad) * math.sin(decl_rad)
           + np.cos(lat_rad) * math.cos(decl_rad) * np.cos(ha_rad))
    return np.clip(mu0, -1.0, 1.0).astype(np.float32)


def _terminator_alpha(mu0: np.ndarray) -> np.ndarray:
    """
    Per-pixel day/night blend weight from Miller et al. Eq. 5:
        N_µ0 = clip((µ0 - 0.1)/(0.3 - 0.1), 0, 1)^1.5
    Returns 1.0 for full daytime, 0.0 for full nighttime.
    """
    N = np.clip((mu0 - MU0_NIGHT) / (MU0_DAY - MU0_NIGHT), 0.0, 1.0)
    return (N ** 1.5).astype(np.float32)


def _conus_mu0_range(dt: datetime) -> tuple[float, float]:
    """Quick 5×5 CONUS-bounding-box µ0 scan for band selection."""
    lats = np.linspace(24, 50, 5, dtype=np.float32)
    lons = np.linspace(-125, -66, 5, dtype=np.float32)
    lg, la = np.meshgrid(lons, lats)
    mu0 = _solar_cos_zenith(dt, la, lg)
    return float(mu0.min()), float(mu0.max())


# ---------------------------------------------------------------------------
# S3 band discovery
# ---------------------------------------------------------------------------

def _list_hour_prefix(
    fs: s3fs.S3FileSystem,
    target: datetime,
) -> list[str]:
    """
    Return all file paths in the S3 prefix for the UTC hour of *target*.
    Falls back to the previous hour if the target-hour prefix is empty.
    """
    for offset in (0, -1):
        t      = target.replace(minute=0, second=0, microsecond=0) + timedelta(hours=offset)
        doy    = t.timetuple().tm_yday
        prefix = f"{GOES_BUCKET}/{GOES_PRODUCT}/{t.year}/{doy:03d}/{t.hour:02d}/"
        try:
            files = fs.ls(prefix)
        except Exception as exc:
            print(f"  Cannot list {prefix}: {exc}")
            continue
        if files:
            print(f"  GOES S3 prefix   : {prefix}  ({len(files)} files)")
            return files

    raise FileNotFoundError(
        f"No GOES-19 {GOES_PRODUCT} files found near {target.strftime('%Y-%m-%d %H UTC')}"
    )


def _find_band_file(
    all_files: list[str],
    band: str,
    target: datetime,
) -> tuple[str, datetime]:
    candidates = [
        (parse_goes_scan_time(f), f)
        for f in all_files
        if f"{band}_" in f and f.endswith(".nc")
    ]
    candidates = [(dt, f) for dt, f in candidates if dt is not None]
    if not candidates:
        raise FileNotFoundError(f"No {band} file found in S3 directory listing.")
    candidates.sort(key=lambda x: abs((x[0] - target).total_seconds()))
    best_dt, best_path = candidates[0]
    delta = abs((best_dt - target).total_seconds()) / 60
    print(f"  {band:<3}  {best_path.split('/')[-1][:55]}  Δ={delta:.1f} min")
    return best_path, best_dt


# ---------------------------------------------------------------------------
# Band loading
# ---------------------------------------------------------------------------

def _extract_meta(ds: xr.Dataset) -> dict:
    p = ds["goes_imager_projection"]
    return dict(
        sat_height = float(p.attrs["perspective_point_height"]),
        lon_origin = float(p.attrs["longitude_of_projection_origin"]),
        sweep_axis = p.attrs["sweep_angle_axis"],
        t_attr     = ds.attrs.get("time_coverage_start", "Unknown"),
    )


def _load_reflectance(
    fs: s3fs.S3FileSystem,
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load a reflective ABI L1b band.
    Returns (refl_factor [0,1.2], x_metres, y_metres, meta).
    Upper bound 1.20 to preserve super-unit reflectances from 3-D cloud scattering.
    """
    with fs.open(path, "rb") as fh:
        ds = xr.open_dataset(fh, engine="h5netcdf", decode_times=False)
        ds.load()
    refl = np.clip(
        ds["Rad"].values.astype(np.float32) * float(ds["kappa0"]),
        0.0, 1.20,     # 1.20 intentional — log10 scaling handles it
    )
    meta = _extract_meta(ds)
    x = ds["x"].values * meta["sat_height"]
    y = ds["y"].values * meta["sat_height"]
    return refl, x, y, meta


def _load_brightness_temp(
    fs: s3fs.S3FileSystem,
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load an emissive ABI L1b band.
    Returns (bt_kelvin, x_metres, y_metres, meta).
    Planck inversion uses fk1/fk2/bc1/bc2 coefficients per ABI L1b PUG §4.2.8.
    """
    with fs.open(path, "rb") as fh:
        ds = xr.open_dataset(fh, engine="h5netcdf", decode_times=False)
        ds.load()
    rad  = ds["Rad"].values.astype(np.float64)
    fk1  = float(ds["planck_fk1"])
    fk2  = float(ds["planck_fk2"])
    bc1  = float(ds["planck_bc1"])
    bc2  = float(ds["planck_bc2"])
    T_raw = fk2 / np.log(fk1 / rad + 1.0)
    bt    = ((T_raw - bc1) / bc2).astype(np.float32)
    meta  = _extract_meta(ds)
    x = ds["x"].values * meta["sat_height"]
    y = ds["y"].values * meta["sat_height"]
    return bt, x, y, meta


# ---------------------------------------------------------------------------
# Pixel geolocation (ABI fixed grid → lat/lon)
# ---------------------------------------------------------------------------

def _pixel_latlon(
    x_metres: np.ndarray,
    y_metres: np.ndarray,
    meta: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert ABI fixed-grid coordinates (x, y in metres) to geographic
    lat/lon via pyproj geostationary CRS.

    Parameters
    ----------
    x_metres, y_metres : 1-D arrays of scan-angle coordinates (radians × H)
    meta               : dict with sat_height, lon_origin, sweep_axis

    Returns
    -------
    lats_2d, lons_2d : float32 2-D arrays
    """
    crs_str = (
        f"+proj=geos +lon_0={meta['lon_origin']} "
        f"+h={meta['sat_height']} "
        f"+x_0=0 +y_0=0 +datum=WGS84 "
        f"+sweep={meta['sweep_axis']}"
    )
    transformer = Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)

    Xg, Yg = np.meshgrid(x_metres, y_metres)
    lons_flat, lats_flat = transformer.transform(Xg.ravel(), Yg.ravel())
    lats = lats_flat.reshape(Xg.shape).astype(np.float32)
    lons = lons_flat.reshape(Xg.shape).astype(np.float32)

    # Outside earth disk → NaN
    lats[~np.isfinite(lats)] = np.nan
    lons[~np.isfinite(lons)] = np.nan
    return lats, lons


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------

def _resample_to(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Bilinear zoom to target_shape (nrows, ncols).
    Works for both 2D arrays (H, W) and 3D arrays (H, W, C);
    the channel axis is never zoomed.
    """
    if arr.shape[:2] == tuple(target_shape):
        return arr
    factors: tuple = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
    if arr.ndim == 3:
        factors = (*factors, 1.0)   # leave channel axis unchanged
    return nd_zoom(arr, factors, order=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Daytime composition
# ---------------------------------------------------------------------------

def _scale_log(refl: np.ndarray) -> np.ndarray:
    """
    Miller et al. §5a daytime reflectance scaling:
    1. Truncate to [0.025, 1.20]
    2. Apply log10
    3. Normalise over [-1.6, 0.176]
    Returns float32 in [0, 1].
    """
    refl_c   = np.clip(refl, 0.025, 1.20)
    log_refl = np.log10(refl_c).astype(np.float32)
    scaled   = (log_refl - (-1.6)) / (0.176 - (-1.6))
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _compose_day_rgb(
    c01: np.ndarray,
    c02: np.ndarray,
    c03: np.ndarray,
) -> np.ndarray:
    """
    True Color RGB on a common spatial grid (all inputs same shape).

    Hybrid simulated green (Miller et al. §5a, published GOES-R AWG coefficients):
        G = 0.48358168·C02 + 0.45706946·C01 + 0.06038137·C03

    Each channel is log10-scaled per Miller et al. Eq. (1) / §5a before forming RGB.
    Returns float32 H×W×3 in [0, 1].
    """
    G_sim = 0.48358168 * c02 + 0.45706946 * c01 + 0.06038137 * c03
    R = _scale_log(c02)
    G = _scale_log(G_sim)
    B = _scale_log(c01)
    return np.dstack([R, G, B]).astype(np.float32)


# ---------------------------------------------------------------------------
# Nighttime composition  (Miller et al. V2.0, §5a(2), three-layer nested stack)
# ---------------------------------------------------------------------------

def _irmin_from_lat(lats_2d: np.ndarray) -> np.ndarray:
    """
    Latitude-varying lower bound for IR scaling, Miller et al. Eq. (9):
        IRmin = 200           if lat < 30
              = 200 + 20*(lat-30)/30   if 30 <= lat <= 60
              = 220           if lat > 60
    """
    irmin = np.where(
        lats_2d < 30.0,
        200.0,
        np.where(
            lats_2d > 60.0,
            220.0,
            200.0 + 20.0 * (lats_2d - 30.0) / 30.0,
        ),
    ).astype(np.float32)
    # NaN pixels (space, off-disk) → 200 K default
    irmin = np.where(np.isfinite(irmin), irmin, 200.0)
    return irmin


def _compose_night_rgb(
    c07:     np.ndarray,
    c13:     np.ndarray,
    lats_2d: np.ndarray | None = None,
) -> np.ndarray:
    """
    Nighttime GeoColor via three-layer nested stack (Miller et al. Eq. 3):
        C = N_IR·white + (1−N_IR)·[N_LC·LC_blue + (1−N_LC)·background]

    Layer 1 — high cloud (C13):
        N_IR = 1 − clip((C13 − IRmin) / (280 − IRmin), 0, 1)
        IRmin latitude-varying per Eq. 9; cold tops → N_IR → 1 (opaque white).
        High ice clouds / deep convection appear gray-to-white.

    Layer 2 — low cloud (BTD = C13 − C07):
        BTD isolates liquid-phase boundary-layer clouds via emissivity difference.
        BTD zeroed where C13 < 230 K to avoid noise on deep-convective tops.
        N_LC = clip((BTD − 1.0) / (4.5 − 1.0), 0, 1)   [land scaling bounds]
        Liquid water clouds (fog, stratus) appear medium blue per operational GeoColor.

    Layer 3 — background:
        Dark blue-gray nightscape (city lights / terrain omitted; requires
        static VIIRS-DNB + DEM datasets not downloaded in this pipeline).

    Returns float32 H×W×3 in [0, 1].
    """
    h, w = c13.shape

    # Layer 1: high cloud transparency from C13
    if lats_2d is not None:
        irmin = _resample_to(_irmin_from_lat(lats_2d), c13.shape)
    else:
        irmin = np.full(c13.shape, 210.0, dtype=np.float32)   # mid-latitude fallback

    N_IR = 1.0 - np.clip((c13 - irmin) / (IR_MAX - irmin), 0.0, 1.0)

    # Layer 2: low cloud BTD = C13 - C07 (Mie emissivity differential)
    c07_r = _resample_to(c07, c13.shape)
    btd   = c13 - c07_r
    btd   = np.where(c13 < 230.0, 0.0, btd)     # mask deep-convection noise
    N_LC  = np.clip((btd - 1.0) / (4.5 - 1.0), 0.0, 1.0).astype(np.float32)

    # Layer 3: background (static dark nightscape)
    BG = np.array([BG_R, BG_G, BG_B], dtype=np.float32)

    # Nested blend: Eq. (3)
    # inner = N_LC·LC_blue + (1−N_LC)·BG
    # Using medium blue for liquid-water clouds (fog/stratus) instead of white,
    # consistent with operational GeoColor: blue = liquid cloud, white = ice cloud.
    LC_COLOR = np.array([LC_R, LC_G, LC_B], dtype=np.float32)
    inner = N_LC[:, :, np.newaxis] * LC_COLOR + (1.0 - N_LC[:, :, np.newaxis]) * BG

    # C = N_IR·white + (1−N_IR)·inner
    C = N_IR[:, :, np.newaxis] + (1.0 - N_IR[:, :, np.newaxis]) * inner

    return np.clip(C, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Master GOES fetch and compose
# ---------------------------------------------------------------------------

def _get_geocolor(fs: s3fs.S3FileSystem, target: datetime) -> dict:
    """
    Fetch the required GOES ABI bands and compose a per-pixel GeoColor RGB.

    Returns
    -------
    dict:
        rgb     : ndarray (H, W, 3) float32
        X, Y    : 1-D scan-angle coordinate arrays (metres)
        meta    : dict (sat_height, lon_origin, sweep_axis, t_attr)
        scan_dt : datetime of C02 scan (reference band)
    """
    # Determine band requirements from CONUS µ0 range (no downloads needed yet)
    mu0_min, mu0_max = _conus_mu0_range(target)
    needs_day   = mu0_max > MU0_NIGHT + 0.02   # any CONUS pixel could be daytime
    needs_night = mu0_min < MU0_DAY   - 0.02   # any CONUS pixel could be nighttime
    print(f"  CONUS µ0 range   : [{mu0_min:.3f}, {mu0_max:.3f}]"
          f"  =>  day={needs_day}  night={needs_night}")

    # List hour directory once — all bands live at the same S3 prefix
    all_files = _list_hour_prefix(fs, target)

    # C02 is always the reference / scan-time anchor
    c02_path, scan_dt = _find_band_file(all_files, "C02", target)
    c02_raw, c02_x, c02_y, meta = _load_reflectance(fs, c02_path)

    day_rgb   = None
    night_rgb = None
    ref_x, ref_y = c02_x, c02_y    # updated below if we use a different grid

    # ── Daytime: C01 (1 km) + C02 (0.5 km) + C03 (1 km) ─────────────────
    if needs_day:
        c01_path, _ = _find_band_file(all_files, "C01", scan_dt)
        c03_path, _ = _find_band_file(all_files, "C03", scan_dt)
        c01, c01_x, c01_y, _ = _load_reflectance(fs, c01_path)
        c03, *_               = _load_reflectance(fs, c03_path)

        # Work at C01 native 1-km grid; downsample C02 (0.5 km) to match
        c02_ds = _resample_to(c02_raw, c01.shape)
        c03_ds = _resample_to(c03,     c01.shape)

        day_rgb = _compose_day_rgb(c01, c02_ds, c03_ds)
        ref_x, ref_y = c01_x, c01_y   # use 1-km grid as reference

    # ── Nighttime: C07 (2 km) + C13 (2 km) ──────────────────────────────
    c13_lats = None
    if needs_night:
        c07_path, _ = _find_band_file(all_files, "C07", scan_dt)
        c13_path, _ = _find_band_file(all_files, "C13", scan_dt)
        c07, *_               = _load_brightness_temp(fs, c07_path)
        c13, c13_x, c13_y, _ = _load_brightness_temp(fs, c13_path)

        # Compute pixel latitudes on C13 grid for latitude-varying IRmin
        c13_lats, _ = _pixel_latlon(c13_x, c13_y, meta)

        night_rgb = _compose_night_rgb(c07, c13, c13_lats)

        if not needs_day:
            ref_x, ref_y = c13_x, c13_y

    # ── Per-pixel µ0 blending across the terminator ──────────────────────
    if needs_day and needs_night:
        # Compute µ0 on the reference (daytime, 1-km) grid
        ref_lats, ref_lons = _pixel_latlon(ref_x, ref_y, meta)
        mu0   = _solar_cos_zenith(scan_dt, ref_lats, ref_lons)
        alpha = _terminator_alpha(mu0)          # 1=day, 0=night, shape (H,W)
        print(f"  Terminator blend : alpha range [{alpha.min():.3f}, {alpha.max():.3f}]  "
              f"({(alpha > 0.99).sum():,} day / {(alpha < 0.01).sum():,} night pixels)")

        # Resample night_rgb to daytime grid before blending
        night_r = _resample_to(night_rgb, day_rgb.shape[:2])

        a = alpha[:, :, np.newaxis]
        rgb = (a * day_rgb + (1.0 - a) * night_r).astype(np.float32)

    elif needs_day:
        rgb = day_rgb
    else:
        rgb = night_rgb

    print(f"  GeoColor shape   : {rgb.shape}  "
          f"range=[{np.nanmin(rgb):.3f}, {np.nanmax(rgb):.3f}]")

    return dict(rgb=rgb, X=ref_x, Y=ref_y, meta=meta, scan_dt=scan_dt,
                needs_day=needs_day, needs_night=needs_night)


# ---------------------------------------------------------------------------
# MRMS helpers
# ---------------------------------------------------------------------------

def _get_closest_mrms_url(target: datetime) -> tuple[str, datetime]:
    import requests
    print(f"  Fetching MRMS listing from {MRMS_URL} ...")
    resp = requests.get(MRMS_URL, timeout=30)
    resp.raise_for_status()
    filenames = re.findall(
        r'href="(MRMS_MergedReflectivityQCComposite[^"]+\.grib2\.gz)"',
        resp.text,
    )
    if not filenames:
        raise FileNotFoundError("No MRMS grib2.gz files in directory listing.")
    print(f"  Found {len(filenames)} MRMS file(s)")
    candidates = [(parse_mrms_time(f), f) for f in filenames]
    candidates = [(dt, f) for dt, f in candidates if dt is not None]
    candidates.sort(key=lambda x: abs((x[0] - target).total_seconds()))
    best_dt, best_fname = candidates[0]
    delta = abs((best_dt - target).total_seconds()) / 60
    print(f"  MRMS file        : {best_fname}  Δ={delta:.1f} min")
    return MRMS_URL + best_fname, best_dt


def _load_mrms(grib2_bytes: bytes, dbz_threshold: float = 5.0):
    import cfgrib
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(grib2_bytes)
        tmp_path = tmp.name
    try:
        ds_mrms  = cfgrib.open_dataset(tmp_path)
        refl_var = next(
            (v for v in ["unknown", "MergedReflectivityQCComposite", "refl"]
             if v in ds_mrms), None,
        )
        if refl_var is None:
            refl_var = list(ds_mrms.data_vars)[0]
        dbz  = ds_mrms[refl_var].values.astype(np.float32)
        lats = ds_mrms["latitude"].values
        lons = ds_mrms["longitude"].values
    finally:
        os.unlink(tmp_path)
    print(f"  MRMS variable    : '{refl_var}'  shape={dbz.shape}  "
          f"range=[{np.nanmin(dbz):.1f}, {np.nanmax(dbz):.1f}] dBZ")
    return lats, lons, np.ma.masked_where((dbz < dbz_threshold) | np.isnan(dbz), dbz)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(output_path: str = "goes19_conus_geocolor_mrms.png") -> str:
    """
    Fetch GOES-19 GeoColor + MRMS data for config.VALID_TIME and save a PNG.
    Returns the absolute path of the saved file.
    """
    target = VALID_TIME
    print(f"\n{'─'*70}")
    print(f"  wx_monitor.goes_mrms  ->  {target.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'─'*70}")

    print("\n-- GOES-19 GeoColor -------------------------------------------------------")
    fs  = s3fs.S3FileSystem(anon=True)
    geo = _get_geocolor(fs, target)

    print("\n-- MRMS -------------------------------------------------------------------")
    mrms_url, mrms_dt = _get_closest_mrms_url(target)
    print("  Downloading MRMS ...")
    mrms_lats, mrms_lons, dbz_masked = _load_mrms(gzip.decompress(download_bytes(mrms_url, timeout=60)))

    print("\n-- Plotting ---------------------------------------------------------------")
    meta = geo["meta"]

    crs_geos = ccrs.Geostationary(
        central_longitude = meta["lon_origin"],
        satellite_height  = meta["sat_height"],
        sweep_axis        = meta["sweep_axis"],
    )
    crs_lcc = ccrs.LambertConformal(
        central_longitude  = -97.5,
        central_latitude   = 38.5,
        standard_parallels = (38.5, 38.5),
    )

    fig, ax = plt.subplots(
        figsize=(16, 9), subplot_kw={"projection": crs_lcc}, facecolor="black"
    )
    ax.set_facecolor("black")

    X, Y = geo["X"], geo["Y"]

    ax.imshow(
        geo["rgb"],
        origin="upper",
        extent=(X.min(), X.max(), Y.min(), Y.max()),
        transform=crs_geos,
        interpolation="bilinear",
        regrid_shape=2000,
        zorder=1,
    )

    radar_im = ax.pcolormesh(
        mrms_lons, mrms_lats, dbz_masked,
        cmap=radar_cmap, norm=radar_norm,
        transform=ccrs.PlateCarree(),
        alpha=0.70, zorder=2,
    )

    ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, edgecolor="white", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),   linewidth=0.6, edgecolor="white", zorder=3)
    ax.add_feature(cfeature.STATES.with_scale("50m"),    linewidth=0.4, edgecolor="white", zorder=3)
    ax.add_feature(cfeature.LAKES.with_scale("50m"),     linewidth=0.3, edgecolor="white",
                   facecolor="none", zorder=3)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.4, color="white", alpha=0.4, linestyle="--", zorder=3)
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator     = mticker.MultipleLocator(10)
    gl.ylocator     = mticker.MultipleLocator(10)
    gl.xlabel_style = {"color": "white", "fontsize": 9}
    gl.ylabel_style = {"color": "white", "fontsize": 9}

    cbar = fig.colorbar(radar_im, ax=ax, orientation="vertical",
                        fraction=0.025, pad=0.02, shrink=0.80)
    cbar.set_label("Reflectivity (dBZ)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    mode_str = ("True Color"  if (geo["needs_day"]   and not geo["needs_night"]) else
                "Nighttime IR" if (geo["needs_night"] and not geo["needs_day"])   else
                "Day/Night blend")
    ax.set_title(
        f"GOES-19 GeoColor ({mode_str}) + MRMS Composite Reflectivity\n"
        f"GOES: {meta['t_attr']}    "
        f"MRMS: {mrms_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}    "
        f"[valid: {target.strftime('%Y-%m-%d %H:%M UTC')}]",
        color="white", fontsize=12, pad=10,
    )
    fig.patch.set_facecolor("black")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {output_path}")
    return os.path.abspath(output_path)
