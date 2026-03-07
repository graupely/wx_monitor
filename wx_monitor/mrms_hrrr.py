"""
wx_monitor.mrms_hrrr
====================
Compare MRMS Multi-Sensor QPE observed precipitation against HRRR
accumulated precipitation forecasts for four periods: 1 h, 3 h, 6 h, 12 h.

Observation source
------------------
NOAA MRMS Multi-Sensor QPE Pass 2, served as gzip-compressed GRIB2 files:

    1-hour  : https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_01H_Pass2/
    3-hour  : https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_03H_Pass2/
    6-hour  : https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_06H_Pass2/
    12-hour : https://mrms.ncep.noaa.gov/2D/MultiSensor_QPE_12H_Pass2/

The valid time is encoded in the filename, e.g.::

    MRMS_MultiSensor_QPE_12H_Pass2_00.00_20260228-230000.grib2.gz
                                         ^^^^^^^^^^^^^^^^
                                         valid at 2026-02-28 23:00 UTC

The file whose filename-encoded valid time is closest to ``config.VALID_TIME``
is selected automatically for each accumulation period.

HRRR forecast
-------------
Run initialised at ``VALID_TIME − fxx`` hours, forecast hour ``fxx``,
so that the HRRR valid time equals ``VALID_TIME``.

Units
-----
MRMS QPE GRIB2 values are in mm (kg m⁻²).  All precipitation is converted
to inches before plotting and comparison.
"""

from __future__ import annotations

import gzip
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import cfgrib
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import requests
from scipy.stats import gaussian_kde
from shapely.ops import unary_union
from shapely.vectorized import contains

from wx_monitor.config import (
    ACCUM_HOURS,
    CACHE_DIR,
    HRRR_BASE,
    MRMS_QPE_BASE,
    MRMS_QPE_DIRS,
    MRMS_QPE_PREFIXES,
    NWS_COLORS,
    NWS_LEVELS,
    OUTPUT_DPI,
    VALID_TIME,
)
from wx_monitor.utils import ensure_dir, make_nws_precip_cmap

EXTENT = [-120, -72, 23, 50]
PROJ   = ccrs.LambertConformal(central_longitude=-96, central_latitude=39)

# MRMS QPE file timestamp regex — matches  20260228-230000  in the filename
_MRMS_QPE_TS_RE = re.compile(r"(\d{8})-(\d{6})\.grib2\.gz$")


# ---------------------------------------------------------------------------
# MRMS QPE directory listing helpers
# ---------------------------------------------------------------------------

def _parse_mrms_qpe_time(filename: str) -> Optional[datetime]:
    """
    Extract the valid time from a MRMS QPE filename.

    Example filename::

        MRMS_MultiSensor_QPE_12H_Pass2_00.00_20260228-230000.grib2.gz

    Returns a timezone-aware UTC datetime, or None if the pattern is not found.
    """
    m = _MRMS_QPE_TS_RE.search(filename)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def _get_mrms_qpe_url(hours: int, target: datetime) -> tuple[str, datetime]:
    """
    Scrape the MRMS QPE directory listing for *hours*-hour accumulation and
    return the URL + valid time of the file closest to *target*.

    Parameters
    ----------
    hours:
        Accumulation period in hours (1, 3, 6, or 12).
    target:
        Desired valid time (usually ``config.VALID_TIME``).

    Returns
    -------
    (url, valid_time)
    """
    subdir   = MRMS_QPE_DIRS[hours]
    prefix   = MRMS_QPE_PREFIXES[hours]
    base_url = f"{MRMS_QPE_BASE}{subdir}/"

    print(f"    Fetching directory listing: {base_url}")
    resp = requests.get(base_url, timeout=30)
    resp.raise_for_status()

    # Pull all matching .grib2.gz filenames from the HTML index
    pattern  = re.compile(rf'href="({re.escape(prefix)}[^"]+\.grib2\.gz)"')
    filenames = pattern.findall(resp.text)

    if not filenames:
        raise FileNotFoundError(
            f"No MRMS QPE files found in {base_url}.\n"
            f"Looked for prefix '{prefix}' in the directory listing."
        )

    print(f"    Found {len(filenames)} file(s) in listing")

    candidates = [
        (dt, fname)
        for fname in filenames
        if (dt := _parse_mrms_qpe_time(fname)) is not None
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Could not parse valid times from any filename in {base_url}.\n"
            f"First file found: {filenames[0]}"
        )

    candidates.sort(key=lambda x: abs((x[0] - target).total_seconds()))
    best_dt, best_fname = candidates[0]
    delta_min = abs((best_dt - target).total_seconds()) / 60

    print(f"    ✓ Selected: {best_fname}")
    print(f"      Valid time: {best_dt.strftime('%Y-%m-%d %H:%M UTC')}  "
          f"(Δ{delta_min:.1f} min from target)")

    return base_url + best_fname, best_dt


def _download_mrms_qpe(url: str, hours: int) -> str:
    """
    Download a ``.grib2.gz`` MRMS QPE file, decompress it in memory,
    and write the raw GRIB2 bytes to a cache file.  Returns the local path.

    The cache key includes the full URL so different valid times are stored
    separately.  The decompressed file (no ``.gz``) is what is cached.
    """
    ensure_dir(CACHE_DIR)

    # Build a stable cache filename from the URL's basename minus .gz
    gz_basename   = url.split("/")[-1]              # e.g. MRMS_…_20260228-230000.grib2.gz
    grib_basename = gz_basename[:-3]                # strip .gz  → …20260228-230000.grib2
    local         = os.path.join(CACHE_DIR, grib_basename)

    if os.path.exists(local) and os.path.getsize(local) > 0:
        print(f"    Using cached: {grib_basename}  "
              f"({os.path.getsize(local)//1024} KB)")
        return local

    print(f"    Downloading {gz_basename} …", end=" ", flush=True)
    resp = requests.get(url, timeout=90)
    resp.raise_for_status()

    compressed = resp.content
    print(f"done ({len(compressed)//1024} KB compressed)")

    # Validate magic bytes: gzip starts with \x1f\x8b
    if compressed[:2] != b"\x1f\x8b":
        snippet = compressed[:200].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"MRMS QPE download did not return a gzip file.\n"
            f"URL: {url}\n"
            f"First bytes: {compressed[:8]!r}\n"
            f"Snippet: {snippet}"
        )

    print(f"    Decompressing …", end=" ", flush=True)
    raw = gzip.decompress(compressed)
    print(f"done ({len(raw)//1024} KB)")

    with open(local, "wb") as fh:
        fh.write(raw)

    return local


# ---------------------------------------------------------------------------
# MRMS QPE GRIB2 loader
# ---------------------------------------------------------------------------

def _load_mrms_qpe(grib_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a MRMS QPE GRIB2 file and return (lons_2d, lats_2d, precip_inches).

    MRMS QPE is on a regular lat/lon grid.  cfgrib returns 1-D coordinate
    vectors for such grids (lats shape (ny,), lons shape (nx,)).  This
    function always returns fully 2-D arrays matching the data shape (ny, nx)
    by calling np.meshgrid when the coordinates are 1-D.

    Values are in mm (kg m⁻²) on the wire; converted to inches on return.
    Negative fill values are masked to NaN.

    Returns
    -------
    lons : np.ndarray  shape (ny, nx), degrees east  (-180..180)
    lats : np.ndarray  shape (ny, nx), degrees north
    data : np.ndarray  shape (ny, nx), precipitation in inches
    """
    datasets = cfgrib.open_datasets(grib_path)

    for ds in datasets:
        for vname in ds.data_vars:
            arr = ds[vname].values.astype(np.float32)
            if arr.ndim < 2:
                continue

            data = arr if arr.ndim == 2 else arr[0]   # (ny, nx)

            lats_raw = ds["latitude"].values
            lons_raw = ds["longitude"].values

            # cfgrib gives 1-D vectors for regular lat/lon grids.
            # Meshgrid them into (ny, nx) arrays that match `data`.
            if lats_raw.ndim == 1 and lons_raw.ndim == 1:
                # Verify the 1-D sizes are consistent with data shape
                ny, nx = data.shape
                if lats_raw.size != ny or lons_raw.size != nx:
                    # Some cfgrib versions return (nx,) for both; try both orderings
                    if lats_raw.size == nx and lons_raw.size == ny:
                        lats_raw, lons_raw = lons_raw, lats_raw
                    else:
                        raise RuntimeError(
                            f"Cannot align coordinate vectors "
                            f"lats={lats_raw.shape} lons={lons_raw.shape} "
                            f"with data shape {data.shape}"
                        )
                # np.meshgrid(lons_1d, lats_1d) → each (ny, nx)
                lons_2d, lats_2d = np.meshgrid(lons_raw, lats_raw)
            elif lats_raw.ndim == 2 and lons_raw.ndim == 2:
                lats_2d, lons_2d = lats_raw, lons_raw
            else:
                raise RuntimeError(
                    f"Unexpected coordinate shapes: "
                    f"lats={lats_raw.shape}, lons={lons_raw.shape}"
                )

            # Unwrap 0–360 longitudes to -180..180
            lons_2d = np.where(lons_2d > 180, lons_2d - 360.0, lons_2d)

            # mm → inches;  MRMS fill values are typically -3 or -99999
            data /= 25.4
            data[data < 0] = np.nan

            print(f"    MRMS QPE variable : '{vname}'  shape={data.shape}  "
                  f"range=[{np.nanmin(data):.3f}, {np.nanmax(data):.3f}] in")
            return lons_2d, lats_2d, data

    raise RuntimeError(
        f"No 2-D precipitation field found in MRMS QPE GRIB2 file: {grib_path}\n"
        f"Variables available: "
        + ", ".join(str(v) for ds in datasets for v in ds.data_vars)
    )


# ---------------------------------------------------------------------------
# HRRR helpers
# ---------------------------------------------------------------------------

def _hrrr_url(run_dt: datetime, fxx: int) -> tuple[str, str]:
    date_str = run_dt.strftime("%Y%m%d")
    hour_str = run_dt.strftime("%H")
    fname    = f"hrrr.t{hour_str}z.wrfsfcf{fxx:02d}.grib2"
    base     = f"{HRRR_BASE}/hrrr.{date_str}/conus/{fname}"
    return base, base + ".idx"


def _find_apcp_byte_range(idx_text: str, fxx: int) -> tuple[int, int | str]:
    lines   = idx_text.strip().splitlines()
    pattern = re.compile(rf"APCP:surface:0-{fxx} hour acc fcst", re.IGNORECASE)
    for i, line in enumerate(lines):
        if pattern.search(line):
            start = int(line.split(":")[1])
            end   = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else ""
            return start, end
    raise RuntimeError(
        f"APCP 0-{fxx}h not found in HRRR index.\nAvailable APCP lines:\n"
        + "\n".join(l for l in lines if "APCP" in l)
    )


def _download_hrrr_apcp(run_dt: datetime, fxx: int) -> str:
    """Byte-range download of HRRR APCP field, cached locally."""
    ensure_dir(CACHE_DIR)
    local = os.path.join(
        CACHE_DIR,
        f"hrrr_{run_dt.strftime('%Y%m%d_%Hz')}_f{fxx:02d}_apcp.grib2",
    )

    if os.path.exists(local) and os.path.getsize(local) > 0:
        print(f"    Using cached HRRR: {Path(local).name}")
        return local

    grib_url, idx_url = _hrrr_url(run_dt, fxx)
    print(f"    Fetching HRRR idx: {idx_url.split('/')[-1]} …", end=" ", flush=True)
    idx_resp = requests.get(idx_url, timeout=30)
    idx_resp.raise_for_status()
    print("done")

    start, end   = _find_apcp_byte_range(idx_resp.text, fxx)
    byte_range   = f"{start}-{end}" if end != "" else f"{start}-"
    print(f"    Downloading HRRR APCP bytes {byte_range} …", end=" ", flush=True)
    data_resp = requests.get(
        grib_url, headers={"Range": f"bytes={byte_range}"}, timeout=60
    )
    data_resp.raise_for_status()

    with open(local, "wb") as fh:
        fh.write(data_resp.content)
    print(f"done ({len(data_resp.content)//1024} KB)")
    return local


def _load_hrrr_apcp(grib_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load HRRR APCP; return (lons, lats, precip_inches)."""
    for ds in cfgrib.open_datasets(grib_path):
        if "tp" in ds:
            tp       = ds["tp"].values.astype(float)
            lats     = ds["latitude"].values
            lons_raw = ds["longitude"].values
            lons     = np.where(lons_raw > 180, lons_raw - 360.0, lons_raw)
            tp      /= 25.4       # mm → inches
            tp[tp < 0] = np.nan
            return lons, lats, tp
    raise RuntimeError(f"'tp' not found in HRRR GRIB2 file: {grib_path}")


# ---------------------------------------------------------------------------
# CONUS mask (built once, reused for all accumulation periods)
# ---------------------------------------------------------------------------

def _build_conus_mask_fn():
    """Return a callable ``mask(lons, lats) → bool array``."""
    shpfile = shpreader.natural_earth(
        resolution="110m", category="cultural", name="admin_1_states_provinces"
    )
    reader = shpreader.Reader(shpfile)
    conus_states = [
        rec.geometry for rec in reader.records()
        if rec.attributes["admin"] == "United States of America"
        and rec.attributes["name"] not in ("Alaska", "Hawaii")
    ]
    conus_shape = unary_union(conus_states)

    def _mask(lons_arr: np.ndarray, lats_arr: np.ndarray) -> np.ndarray:
        return contains(conus_shape, lons_arr.ravel(),
                        lats_arr.ravel()).reshape(lons_arr.shape)

    return _mask


# ---------------------------------------------------------------------------
# Plotting helpers
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


def _plot_kde(
    ax,
    mrms_vals: np.ndarray,
    hrrr_vals: np.ndarray,
    hours: int,
) -> None:
    if len(mrms_vals) < 2 or len(hrrr_vals) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=13)
        return

    # Floor at 0.05" so trace-amount noise doesn't dominate the x-axis.
    # Silverman's bandwidth is too narrow for the heavy low-value tail of
    # precipitation distributions; a fixed bw=0.35 on the log scale gives
    # smooth, readable curves across all four accumulation periods.
    X_FLOOR   = 0.05
    KDE_BW    = 0.35

    mrms_vals = mrms_vals[mrms_vals >= X_FLOOR]
    hrrr_vals = hrrr_vals[hrrr_vals >= X_FLOOR]

    if len(mrms_vals) < 2 or len(hrrr_vals) < 2:
        ax.text(0.5, 0.5, "Insufficient data\n(above floor)", ha="center",
                va="center", transform=ax.transAxes, fontsize=13)
        return

    x_max  = max(mrms_vals.max(), hrrr_vals.max())
    x_grid = np.logspace(np.log10(X_FLOOR), np.log10(max(x_max, X_FLOOR * 2)), 300)

    for vals, color, label in [
        (mrms_vals, "#1464b4", f"MRMS QPE  (n={len(mrms_vals):,})"),
        (hrrr_vals, "#b40000", f"HRRR F{hours:02d}  (n={len(hrrr_vals):,})"),
    ]:
        log_vals = np.log10(vals)
        kde      = gaussian_kde(log_vals, bw_method=KDE_BW)
        density  = kde(np.log10(x_grid))
        density /= density.max()
        ax.plot(x_grid, density, color=color, linewidth=2, label=label)
        ax.fill_between(x_grid, density, alpha=0.18, color=color)
        ax.axvline(np.median(vals), color=color, linewidth=1.2,
                   linestyle="--", alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlim(X_FLOOR, x_max)
    ax.set_ylim(0)
    ax.set_xlabel("Precipitation (inches)", fontsize=14)
    ax.set_ylabel("Normalized density",     fontsize=14)
    ax.legend(fontsize=13, framealpha=0.85)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.tick_params(labelsize=13)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(output_dir: str = ".") -> list[str]:
    """
    Fetch MRMS QPE + HRRR data for each accumulation period and save PNGs.

    For each period (1 h, 3 h, 6 h, 12 h):
      - The MRMS QPE file whose valid time is closest to ``VALID_TIME``
        is selected from the appropriate directory listing.
      - The HRRR run initialised at ``VALID_TIME − fxx`` with forecast
        hour ``fxx`` is used, so that both datasets verify at ``VALID_TIME``.

    Parameters
    ----------
    output_dir:
        Directory where PNG files are written.

    Returns
    -------
    list[str]
        Absolute paths of saved PNG files (one per accumulation period).
    """
    target = VALID_TIME
    print(f"\n{'─'*70}")
    print(f"  wx_monitor.mrms_hrrr  →  valid time: {target.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'─'*70}")

    ensure_dir(output_dir)
    cmap, norm = make_nws_precip_cmap()

    print("\nBuilding CONUS mask …")
    mask_fn = _build_conus_mask_fn()

    # HRRR grid / mask are identical for all accumulation periods; built once
    h_lons: Optional[np.ndarray] = None
    h_lats: Optional[np.ndarray] = None
    h_mask: Optional[np.ndarray] = None
    m_mask: Optional[np.ndarray] = None  # MRMS QPE mask (same for all periods)

    rows: list[dict] = []

    for hours in ACCUM_HOURS:
        label    = f"{hours}-Hour"
        run_time = target - timedelta(hours=hours)

        print(f"\n{'='*60}")
        print(f"  Period : {label}  |  valid: {target.strftime('%Hz %d %b %Y UTC')}"
              f"  |  HRRR run: {run_time.strftime('%Hz %d %b %Y UTC')}")
        print(f"{'='*60}")

        # ── MRMS QPE ──────────────────────────────────────────────────────────
        print(f"  MRMS QPE ({hours}-hour accumulation):")
        mrms_url, mrms_vt = _get_mrms_qpe_url(hours, target)
        mrms_grib         = _download_mrms_qpe(mrms_url, hours)
        m_lons, m_lats, m_raw = _load_mrms_qpe(mrms_grib)

        delta_min = abs((mrms_vt - target).total_seconds()) / 60
        if delta_min > 30:
            print(f"  ⚠  MRMS QPE valid time ({mrms_vt.strftime('%H:%M UTC')}) "
                  f"differs from target by {delta_min:.0f} min")

        m_raw[m_raw < 0.01] = np.nan  # treat trace as missing

        # ── HRRR ─────────────────────────────────────────────────────────────
        print(f"  HRRR F{hours:02d}:")
        hrrr_grib          = _download_hrrr_apcp(run_time, fxx=hours)
        h_lons_, h_lats_, h_raw = _load_hrrr_apcp(hrrr_grib)
        h_raw[h_raw < 0.01] = np.nan

        # Build masks on first iteration (grids are constant across periods)
        if h_lons is None:
            h_lons, h_lats = h_lons_, h_lats_
            h_mask = mask_fn(h_lons, h_lats)
            m_mask = mask_fn(m_lons, m_lats)

        # Apply CONUS mask
        m_masked = np.where(m_mask, m_raw,  np.nan)
        h_masked = np.where(h_mask, h_raw,  np.nan)

        mrms_vals = m_masked[np.isfinite(m_masked) & (m_masked >= 0.01)].ravel()
        hrrr_vals = h_masked[np.isfinite(h_masked) & (h_masked >= 0.01)].ravel()

        rows.append(dict(
            hours=hours, label=label,
            m_lons=m_lons, m_lats=m_lats, m_masked=m_masked,
            h_masked=h_masked,
            mrms_vals=mrms_vals, hrrr_vals=hrrr_vals,
            valid_time=target, run_time=run_time,
            mrms_vt=mrms_vt,
            period_str=(
                f"{run_time.strftime('%Hz')}–{target.strftime('%Hz %d %b %Y UTC')}"
            ),
        ))

    # ── Save one figure per accumulation period ────────────────────────────
    output_files: list[str] = []

    for row in rows:
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

        ax_mrms.pcolormesh(
            row["m_lons"], row["m_lats"], row["m_masked"],
            cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
            zorder=2, shading="auto",
        )
        ax_hrrr.pcolormesh(
            h_lons, h_lats, row["h_masked"],
            cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
            zorder=2, shading="auto",
        )

        ax_mrms.set_title(
            f"MRMS MultiSensor QPE — {row['label']}\n"
            f"Valid: {row['mrms_vt'].strftime('%Hz %d %b %Y UTC')}  "
            f"(Pass 2)",
            fontsize=14, fontweight="bold", pad=6,
        )
        ax_hrrr.set_title(
            f"HRRR F{row['hours']:02d} Forecast — {row['label']}\n"
            f"Init: {row['run_time'].strftime('%Hz %d %b %Y UTC')}  "
            f"Valid: {row['valid_time'].strftime('%Hz %d %b %Y UTC')}",
            fontsize=14, fontweight="bold", pad=6,
        )

        _plot_kde(ax_kde, row["mrms_vals"], row["hrrr_vals"], row["hours"])
        ax_kde.set_box_aspect(1)
        ax_kde.set_title(
            f'{row["label"]} Precip > 0.01"',
            fontsize=15, fontweight="bold",
        )

        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax, extend="max", orientation="horizontal",
        )
        cb.ax.tick_params(labelsize=11)
        cb.set_ticks(NWS_LEVELS)
        cb.ax.set_xticklabels(
            [str(v) for v in NWS_LEVELS], fontsize=11, rotation=45, ha="right"
        )
        cb.set_label("Precipitation (inches)", fontsize=13, labelpad=4)

        fig.suptitle(
            f"Accumulated Precipitation — MRMS QPE vs HRRR — {row['label']}\n"
            f"Valid: {row['valid_time'].strftime('%Y-%m-%d %H:%M UTC')}",
            fontsize=20, fontweight="bold", y=0.97,
        )
        fig.text(
            0.5, 0.01,
            "Obs: MRMS MultiSensor QPE Pass 2 (mrms.ncep.noaa.gov)  |  "
            "Fcst: HRRR (NOAA/ESRL, noaa-hrrr-bdp-pds.s3.amazonaws.com)",
            ha="center", color="#666666", fontsize=11,
        )

        valid_str = row["valid_time"].strftime("%Y%m%d_%H%z")
        fname     = f"mrms_qpe_vs_hrrr_{row['hours']:02d}hr_{valid_str}.png"
        out_path  = os.path.join(output_dir, fname)
        plt.savefig(out_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved → {out_path}")
        output_files.append(os.path.abspath(out_path))

    return output_files
