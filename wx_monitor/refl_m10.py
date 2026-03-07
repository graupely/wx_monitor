"""
wx_monitor.refl_m10
===================
Compare MRMS observed reflectivity at the −10 °C isotherm against four
HRRR forecast hours (F01, F03, F06, F12), producing one figure per
forecast hour, all valid at ``config.VALID_TIME``.

Data sources
------------
MRMS : https://mrms.ncep.noaa.gov/2D/Reflectivity_-10C/
    2-minute, 0.01° grid, GRIB2 gzip-compressed, units dBZ.
    A single file whose filename timestamp is closest to VALID_TIME is
    fetched once and reused across all four figures.

HRRR : noaa-hrrr-bdp-pds.s3.amazonaws.com
    ``wrfsfcf`` surface file, field ``REFD:263 K above mean sea level``
    (263 K = −10 °C).  For each forecast hour *fxx* ∈ {1, 3, 6, 12} the
    model run initialised at ``VALID_TIME − fxx h`` is used, so every
    figure verifies at the same valid time.

Output filenames
----------------
``refl_m10_f01_<YYYYMMDD_HH>.png``
``refl_m10_f03_<YYYYMMDD_HH>.png``
``refl_m10_f06_<YYYYMMDD_HH>.png``
``refl_m10_f12_<YYYYMMDD_HH>.png``

Colormap
--------
Standard NWS radar colormap in dBZ, identical to the one used by
``goes_mrms`` for the reflectivity overlay.
"""

from __future__ import annotations

import gzip
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cfgrib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import requests
from matplotlib.colors import BoundaryNorm
from scipy.stats import gaussian_kde

from wx_monitor.config import (
    CACHE_DIR,
    HRRR_BASE,
    OUTPUT_DPI,
    VALID_TIME,
)
from wx_monitor.utils import ensure_dir, parse_mrms_time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MRMS_REFL_M10_URL    = "https://mrms.ncep.noaa.gov/2D/Reflectivity_-10C/"
MRMS_REFL_M10_PREFIX = "MRMS_Reflectivity_-10C_"

# HRRR idx search pattern for reflectivity at the −10 °C isotherm (263 K)
# Index lines look like:
#   N:REFD:263 K above mean sea level:1 hour fcst:
_HRRR_REFD_PATTERNS = [
    r"REFD:263 K",          # primary — temperature-level reflectivity
    r"REFD:-10 C",          # alternate phrasing some HRRR versions use
]

# NWS radar colormap (dBZ)
_RADAR_LEVELS = [-10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
_RADAR_COLORS = [
    "#00ecec", "#01a0f6", "#0000f6", "#00ff00",
    "#00c800", "#009000", "#ffff00", "#e7c000",
    "#ff9000", "#ff0000", "#d60000", "#c00000",
    "#ff00ff", "#9955cc", "#ffffff", "#aaaaaa",
]
_radar_cmap = mcolors.ListedColormap(_RADAR_COLORS)
_radar_norm = BoundaryNorm(_RADAR_LEVELS, ncolors=len(_RADAR_COLORS))

DBZ_THRESHOLD = 5.0   # values below this are masked (noise floor)
DBZ_KDE_FLOOR = 0.0   # KDE distribution threshold (include all echoes ≥ 0 dBZ)

FXX_LIST = [1, 3, 6, 12]   # forecast hours rendered (one figure each)

EXTENT = [-120, -72, 23, 50]
PROJ   = ccrs.LambertConformal(central_longitude=-96, central_latitude=39)


# ---------------------------------------------------------------------------
# MRMS helpers
# ---------------------------------------------------------------------------

def _get_mrms_refl_url(target: datetime) -> tuple[str, datetime]:
    """
    Scrape the MRMS Reflectivity_-10C directory listing and return
    (url, valid_time) for the file closest to *target*.
    """
    print(f"    Fetching directory listing: {MRMS_REFL_M10_URL}")
    resp = requests.get(MRMS_REFL_M10_URL, timeout=30)
    resp.raise_for_status()

    pattern   = re.compile(
        rf'href="({re.escape(MRMS_REFL_M10_PREFIX)}[^"]+\.grib2\.gz)"'
    )
    filenames = pattern.findall(resp.text)
    if not filenames:
        raise FileNotFoundError(
            f"No MRMS Reflectivity_-10C files found in {MRMS_REFL_M10_URL}.\n"
            f"Looked for prefix '{MRMS_REFL_M10_PREFIX}'."
        )
    print(f"    Found {len(filenames)} file(s) in listing")

    candidates = [
        (dt, fname)
        for fname in filenames
        if (dt := parse_mrms_time(fname)) is not None
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not parse timestamps from any filename in {MRMS_REFL_M10_URL}."
        )

    candidates.sort(key=lambda x: abs((x[0] - target).total_seconds()))
    best_dt, best_fname = candidates[0]
    delta = abs((best_dt - target).total_seconds()) / 60
    print(f"    ✓ Selected: {best_fname}")
    print(f"      Valid time: {best_dt.strftime('%Y-%m-%d %H:%M UTC')}  "
          f"(Δ{delta:.1f} min from target)")
    return MRMS_REFL_M10_URL + best_fname, best_dt


def _download_mrms_refl(url: str) -> str:
    """
    Download and decompress a MRMS Reflectivity_-10C .grib2.gz file.
    Returns the path to the local decompressed .grib2 file (cached).
    """
    ensure_dir(CACHE_DIR)
    gz_basename   = url.split("/")[-1]
    grib_basename = gz_basename[:-3]                 # strip .gz
    local         = os.path.join(CACHE_DIR, grib_basename)

    if os.path.exists(local) and os.path.getsize(local) > 0:
        print(f"    Using cached: {grib_basename}  "
              f"({os.path.getsize(local) // 1024} KB)")
        return local

    print(f"    Downloading {gz_basename} …", end=" ", flush=True)
    resp = requests.get(url, timeout=90)
    resp.raise_for_status()
    compressed = resp.content
    print(f"done ({len(compressed) // 1024} KB compressed)")

    if compressed[:2] != b"\x1f\x8b":
        raise RuntimeError(
            f"Expected gzip bytes but got: {compressed[:8]!r}\nURL: {url}"
        )

    raw = gzip.decompress(compressed)
    with open(local, "wb") as fh:
        fh.write(raw)
    print(f"    Decompressed → {grib_basename}  ({len(raw) // 1024} KB)")
    return local


def _load_mrms_refl(grib_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MRMS Reflectivity_-10C GRIB2 and return (lons_2d, lats_2d, dbz).
    Values below DBZ_THRESHOLD are masked.  Missing / fill values → NaN.
    """
    ds       = cfgrib.open_dataset(grib_path)
    var_name = next(
        (v for v in ["unknown", "refd", "REFD", "Reflectivity_-10C"]
         if v in ds),
        list(ds.data_vars)[0],
    )
    dbz  = ds[var_name].values.astype(np.float32)
    lats = ds["latitude"].values
    lons = ds["longitude"].values

    # MRMS uses regular lat/lon; cfgrib may return 1-D coords
    if lats.ndim == 1 and lons.ndim == 1:
        lons, lats = np.meshgrid(lons, lats)

    # Normalise longitudes to [-180, 180]
    lons = np.where(lons > 180, lons - 360, lons).astype(np.float32)

    # Mask fill values and below-threshold
    FILL = -99.0
    dbz  = np.where((dbz < FILL + 1) | np.isnan(dbz), np.nan, dbz)
    dbz_masked = np.ma.masked_where(
        (dbz < DBZ_THRESHOLD) | np.isnan(dbz), dbz
    )
    print(f"    MRMS variable: '{var_name}'  shape={dbz_masked.shape}  "
          f"valid range=[{np.nanmin(dbz):.1f}, {np.nanmax(dbz):.1f}] dBZ")
    return lons, lats, dbz_masked


# ---------------------------------------------------------------------------
# HRRR helpers
# ---------------------------------------------------------------------------

def _hrrr_url(run_dt: datetime, fxx: int) -> tuple[str, str]:
    date_str = run_dt.strftime("%Y%m%d")
    hour_str = run_dt.strftime("%H")
    fname    = f"hrrr.t{hour_str}z.wrfsfcf{fxx:02d}.grib2"
    base     = f"{HRRR_BASE}/hrrr.{date_str}/conus/{fname}"
    return base, base + ".idx"


def _find_refd_byte_range(idx_text: str, fxx: int) -> tuple[int, int | str]:
    """
    Search the HRRR .idx for the REFD at −10 °C (263 K) byte range.
    Tries multiple patterns; raises RuntimeError if none match.
    """
    lines = idx_text.strip().splitlines()

    for pat in _HRRR_REFD_PATTERNS:
        compiled = re.compile(pat, re.IGNORECASE)
        for i, line in enumerate(lines):
            if compiled.search(line):
                start = int(line.split(":")[1])
                end   = (int(lines[i + 1].split(":")[1]) - 1
                         if i + 1 < len(lines) else "")
                print(f"    Matched HRRR field: {line.split(':')[3]}:{line.split(':')[4]}")
                return start, end

    # Helpful diagnostics if no match
    refd_lines = [l for l in lines if "REFD" in l.upper()]
    raise RuntimeError(
        f"REFD at −10 °C not found in HRRR idx (tried: {_HRRR_REFD_PATTERNS}).\n"
        f"All REFD lines present:\n" +
        "\n".join(f"  {l}" for l in refd_lines)
    )


def _download_hrrr_refd(run_dt: datetime, fxx: int = 1) -> str:
    """
    Download the REFD:263 K field from the HRRR wrfsfcf grib2 via byte-range
    request.  Returns path to a local temp/cache .grib2 file.
    """
    ensure_dir(CACHE_DIR)
    local = os.path.join(
        CACHE_DIR,
        f"hrrr_{run_dt.strftime('%Y%m%d_%Hz')}_f{fxx:02d}_refd263k.grib2",
    )
    if os.path.exists(local) and os.path.getsize(local) > 0:
        print(f"    Using cached HRRR REFD file ({os.path.getsize(local) // 1024} KB)")
        return local

    grib_url, idx_url = _hrrr_url(run_dt, fxx)

    print(f"    Fetching HRRR idx: {idx_url.split('/')[-1]} …", end=" ", flush=True)
    r = requests.get(idx_url, timeout=30)
    r.raise_for_status()
    print("done")

    start, end = _find_refd_byte_range(r.text, fxx)
    byte_range  = f"{start}-{end}" if end != "" else f"{start}-"
    print(f"    Downloading REFD bytes {byte_range} …", end=" ", flush=True)
    r2 = requests.get(
        grib_url, headers={"Range": f"bytes={byte_range}"}, timeout=60
    )
    r2.raise_for_status()
    with open(local, "wb") as fh:
        fh.write(r2.content)
    print(f"done ({len(r2.content) // 1024} KB)")
    return local


def _load_hrrr_refd(grib_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load HRRR REFD GRIB2 and return (lons_2d, lats_2d, dbz_masked).
    """
    datasets = cfgrib.open_datasets(grib_path)
    for ds in datasets:
        for var in ds.data_vars:
            if "refd" in var.lower() or var.lower() == "unknown":
                dbz  = ds[var].values.astype(np.float32)
                lats = ds["latitude"].values
                lons = ds["longitude"].values
                # Normalise longitudes
                lons = np.where(lons > 180, lons - 360, lons).astype(np.float32)
                dbz  = np.where(np.isnan(dbz), np.nan, dbz)
                dbz_masked = np.ma.masked_where(
                    (dbz < DBZ_THRESHOLD) | np.isnan(dbz), dbz
                )
                print(f"    HRRR variable: '{var}'  shape={dbz_masked.shape}  "
                      f"valid range=[{np.nanmin(dbz):.1f}, {np.nanmax(dbz):.1f}] dBZ")
                return lons, lats, dbz_masked

    raise RuntimeError(
        f"Could not find REFD variable in {grib_path}.\n"
        f"Variables present: {[v for ds in datasets for v in ds.data_vars]}"
    )


# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------

def _add_map_features(ax) -> None:
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_facecolor("#cde8f5")
    ax.add_feature(cfeature.LAND,      facecolor="white",   zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#cde8f5", zorder=0)
    ax.add_feature(cfeature.LAKES,     facecolor="#cde8f5", zorder=1, linewidth=0.3)
    ax.add_feature(cfeature.STATES,    edgecolor="#888888", linewidth=0.4, zorder=3)
    ax.add_feature(cfeature.BORDERS,   edgecolor="#555555", linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#555555", linewidth=0.6, zorder=3)



# ---------------------------------------------------------------------------
# KDE distribution panel
# ---------------------------------------------------------------------------

def _plot_kde(ax, mrms_vals: np.ndarray, hrrr_vals: np.ndarray, fxx: int) -> None:
    """
    Plot normalised KDE distributions of dBZ values ≥ DBZ_KDE_FLOOR for
    MRMS (observed) and HRRR (forecast) on a shared log-linear axis.
    Mirrors the style used in mrms_hrrr for QPE distributions.
    """
    mrms_v = mrms_vals[mrms_vals >= DBZ_KDE_FLOOR]
    hrrr_v = hrrr_vals[hrrr_vals >= DBZ_KDE_FLOOR]

    if len(mrms_v) < 2 or len(hrrr_v) < 2:
        ax.text(0.5, 0.5, "Insufficient data\n(above threshold)",
                ha="center", va="center", transform=ax.transAxes, fontsize=13)
        return

    x_min  = max(DBZ_KDE_FLOOR, min(mrms_v.min(), hrrr_v.min()))
    x_max  = max(mrms_v.max(),  hrrr_v.max())
    x_grid = np.linspace(x_min, x_max, 400)

    KDE_BW = 0.15   # bandwidth in dBZ units (on linear scale)

    for vals, color, label in [
        (mrms_v, "#1464b4", f"MRMS obs  (n={len(mrms_v):,})"),
        (hrrr_v, "#b40000", f"HRRR F{fxx:02d}  (n={len(hrrr_v):,})"),
    ]:
        kde     = gaussian_kde(vals, bw_method=KDE_BW)
        density = kde(x_grid)
        density = density / density.max()
        ax.plot(x_grid, density, color=color, linewidth=2, label=label)
        ax.fill_between(x_grid, density, alpha=0.18, color=color)
        ax.axvline(np.median(vals), color=color, linewidth=1.2,
                   linestyle="--", alpha=0.8)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0)
    ax.set_xlabel("Reflectivity (dBZ)", fontsize=14)
    ax.set_ylabel("Normalised density",  fontsize=14)
    ax.legend(fontsize=12, framealpha=0.85)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.tick_params(labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _make_figure(
    target: datetime,
    mrms_vt: datetime,
    m_lons: np.ndarray,
    m_lats: np.ndarray,
    m_dbz: np.ndarray,
    h_lons: np.ndarray,
    h_lats: np.ndarray,
    h_dbz: np.ndarray,
    fxx: int,
    output_dir: str,
) -> str:
    """
    Render and save one 4-panel comparison figure for a given *fxx*.
    Returns the absolute path of the saved PNG.
    """
    import matplotlib.gridspec as gridspec

    run_time  = target - timedelta(hours=fxx)

    mrms_vals = m_dbz.compressed().ravel()
    hrrr_vals = h_dbz.compressed().ravel()

    fig = plt.figure(figsize=(22, 9), facecolor="white")
    gs  = gridspec.GridSpec(
        1, 4,
        width_ratios=[5, 5, 0.8, 2.8],
        hspace=0.0, wspace=0.04,
        left=0.02, right=0.99, top=0.88, bottom=0.18,
    )
    ax_mrms = fig.add_subplot(gs[0, 0], projection=PROJ)
    ax_hrrr = fig.add_subplot(gs[0, 1], projection=PROJ)
    ax_kde  = fig.add_subplot(gs[0, 3])
    cax     = fig.add_axes([0.12, 0.13, 0.45, 0.03])

    _add_map_features(ax_mrms)
    _add_map_features(ax_hrrr)

    kw = dict(cmap=_radar_cmap, norm=_radar_norm,
              transform=ccrs.PlateCarree(), zorder=2, shading="auto")
    ax_mrms.pcolormesh(m_lons, m_lats, m_dbz, **kw)
    ax_hrrr.pcolormesh(h_lons, h_lats, h_dbz, **kw)

    ax_mrms.set_title(
        f"MRMS Reflectivity at −10 °C\n"
        f"Valid: {mrms_vt.strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=14, fontweight="bold", pad=6,
    )
    ax_hrrr.set_title(
        f"HRRR F{fxx:02d} Reflectivity at −10 °C (263 K)\n"
        f"Init: {run_time.strftime('%Hz %d %b %Y UTC')}  "
        f"Valid: {target.strftime('%Hz %d %b %Y UTC')}",
        fontsize=14, fontweight="bold", pad=6,
    )

    _plot_kde(ax_kde, mrms_vals, hrrr_vals, fxx=fxx)
    ax_kde.set_box_aspect(1)
    ax_kde.set_title("Reflectivity ≥ 0 dBZ", fontsize=15, fontweight="bold")

    # Horizontal colorbar spanning both map panels
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=_radar_norm, cmap=_radar_cmap),
        cax=cax, extend="both", orientation="horizontal",
    )
    cb.ax.tick_params(labelsize=10)
    cb.set_ticks(_RADAR_LEVELS)
    cb.ax.set_xticklabels([str(v) for v in _RADAR_LEVELS],
                          fontsize=10, rotation=45, ha="right")
    cb.set_label("Reflectivity (dBZ)", fontsize=12, labelpad=4)

    fig.suptitle(
        f"Reflectivity at −10 °C — MRMS vs HRRR F{fxx:02d}\n"
        f"Valid: {target.strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=20, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.01,
        "Obs: MRMS (mrms.ncep.noaa.gov/2D/Reflectivity_-10C/)  |  "
        f"Fcst: HRRR F{fxx:02d} REFD:263 K (noaa-hrrr-bdp-pds.s3.amazonaws.com)",
        ha="center", color="#666666", fontsize=11,
    )

    valid_str = target.strftime("%Y%m%d_%H%z")
    fname     = f"refl_m10_f{fxx:02d}_{valid_str}.png"
    out_path  = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return os.path.abspath(out_path)


def run(output_dir: str = ".") -> list[str]:
    """
    Fetch MRMS Reflectivity at −10 °C once, then compare against HRRR
    forecasts for each of F01, F03, F06, F12 (all valid at ``VALID_TIME``).
    Saves one PNG per forecast hour.

    Returns a list of absolute paths of the saved files.
    """
    target = VALID_TIME

    print(f"\n{'─'*70}")
    print(f"  wx_monitor.refl_m10  →  valid time: {target.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Forecast hours: {FXX_LIST}")
    print(f"{'─'*70}")

    ensure_dir(output_dir)

    # ── MRMS — fetched once, shared across all forecast hours ─────────────────
    print("\n-- MRMS Reflectivity −10 °C -----------------------------------------------")
    mrms_url, mrms_vt = _get_mrms_refl_url(target)
    mrms_grib         = _download_mrms_refl(mrms_url)
    m_lons, m_lats, m_dbz = _load_mrms_refl(mrms_grib)

    # ── HRRR — one download + figure per forecast hour ────────────────────────
    output_files: list[str] = []

    for fxx in FXX_LIST:
        run_time = target - timedelta(hours=fxx)
        print(f"\n-- HRRR F{fxx:02d}  run: {run_time.strftime('%Hz %d %b %Y UTC')} "
              f"──────────────────────────────")
        hrrr_grib             = _download_hrrr_refd(run_time, fxx=fxx)
        h_lons, h_lats, h_dbz = _load_hrrr_refd(hrrr_grib)

        print(f"\n-- Plotting F{fxx:02d} -------------------------------------------------------")
        path = _make_figure(
            target, mrms_vt,
            m_lons, m_lats, m_dbz,
            h_lons, h_lats, h_dbz,
            fxx=fxx, output_dir=output_dir,
        )
        output_files.append(path)

    return output_files
